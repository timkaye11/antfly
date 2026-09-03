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
const builtin = @import("builtin");
const platform = @import("antfly_platform");
const ml = @import("ml");
const finetune = @import("gemma4.zig");
const gemma4_real = @import("gemma4_real_autodiff.zig");
const gemma4_mm_real = @import("gemma4_multimodal_real_autodiff.zig");
const real_autodiff = @import("real_autodiff_trainer.zig");
const safetensors_checkpoint = @import("safetensors_checkpoint.zig");
const graph_bridge = @import("graph_bridge.zig");
const gemma_graph = @import("../architectures/gemma_graph.zig");
const training_executor_policy = @import("../graph/training_executor_policy.zig");
const manifest_mod = @import("../models/manifest.zig");
const safetensors = @import("../models/safetensors.zig");
const gguf = @import("../gguf/root.zig");
const c_file = @import("../util/c_file.zig");
const build_options = @import("build_options");
const run_contract = @import("../run/contract.zig");
const artifact_writer = @import("../run/artifact_writer.zig");
const artifact_publication = @import("artifact_publication.zig");
const path_isolation = @import("path_isolation.zig");
const ops_mod = @import("../ops/ops.zig");
const ComputeBackend = ops_mod.ComputeBackend;
const pjrt_mod = if (build_options.enable_pjrt) @import("pjrt") else struct {
    pub const pjrt = struct {
        pub const Client = void;
    };
};

const TrainerMode = enum { auto, surrogate, autodiff };

pub const BackendKind = gemma4_real.BackendKind;

/// Typed production contract for Gemma4 LoRA training.  Public callers use
/// this instead of manufacturing a process argv vector and invoking `main`.
/// The legacy CLI parser below is intentionally only a compatibility adapter.
pub const TrainOptions = struct {
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    train_prepared_inputs_path: []const u8,
    eval_prepared_inputs_path: []const u8,
    out_dir: []const u8,
    learning_rate: f32 = 0.001,
    max_examples: usize = 32,
    eval_max_examples: usize = 0,
    epochs: usize = 1,
    max_grad_norm: f32 = 1.0,
    grad_accum_steps: u32 = 1,
    activation_checkpoint_interval: u32 = 0,
    /// Zero preserves fixed prepared-artifact padding. A nonzero value rounds
    /// each independent text row up to this many tokens.
    sequence_length_bucket_quantum: u32 = 0,
    /// Zero selects one bucket quantum when bucketing is enabled.
    sequence_length_bucket_min: u32 = 0,
    /// Zero selects the policy default (1 fixed-shape, 4 bucketed).
    graph_cache_capacity: u8 = 0,
    seed: u64 = 42,
    checkpoint_path: ?[]const u8 = null,
    checkpoint_every_epochs: usize = 0,
    resume_from_checkpoint: bool = false,
    backend_kind: BackendKind,
    /// Optional fail-closed benchmark evidence contract. Both paths must be
    /// present together; normal typed callers leave them null.
    benchmark_request_path: ?[]const u8 = null,
    benchmark_telemetry_out_path: ?[]const u8 = null,
    /// Optional fail-closed numerical-oracle capture contract. Both paths
    /// must be supplied together; the capture directory is immutable and is
    /// intentionally separate from the candidate adapter bundle.
    oracle_request_path: ?[]const u8 = null,
    oracle_capture_out_dir: ?[]const u8 = null,
};

pub const EvalOptions = struct {
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    prepared_inputs_path: []const u8,
    out_path: []const u8,
    backend_kind: BackendKind,
    max_examples: usize = 0,
    max_grad_norm: f32 = 1.0,
};

pub const AdapterValidateOptions = struct {
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
};

pub const AdapterValidationResult = struct {
    valid: bool = true,
    target_module_count: usize,
    rank: u32,
    alpha: f32,
    recursive_lora: bool,
};

const AutodiffEpochTimingSummary = struct {
    schema_version: []const u8 = "antfly.gemma4.epoch-timing/v1",
    clock: []const u8 = "CLOCK_MONOTONIC",
    synchronization: []const u8 = "trainer-step-contract-no-extra-sync",
    wall_time_ns: u64 = 0,
    examples_per_second: f64 = 0,
    logical_input_tokens_per_second: f64 = 0,
    scheduled_input_tokens_per_second: f64 = 0,
    supervised_tokens_per_second: f64 = 0,
    optimizer_steps_per_second: f64 = 0,
};

const AutodiffPhaseTimingSummary = struct {
    schema_version: []const u8 = "antfly.gemma4.phase-timing/v1",
    clock: []const u8 = "CLOCK_MONOTONIC",
    synchronization: []const u8 = "trainer-step-contract-no-extra-sync",
    initial_evaluation_placement: []const u8,
    initialization_and_restore_wall_time_ns: u64,
    initial_evaluation_wall_time_ns: u64,
    training_epochs_wall_time_ns: u64,
    adapter_save_wall_time_ns: u64,
    final_evaluation_wall_time_ns: u64,
    through_final_evaluation_wall_time_ns: u64,
};

const AutodiffEpochGraphCacheDelta = struct {
    builds: u64 = 0,
    hits: u64 = 0,
    active_reuses: u64 = 0,
    evictions: u64 = 0,
    resident_signatures: usize = 0,
    peak_resident_signatures: usize = 0,
    reuse_only: bool = false,
};

const AutodiffEpochSummary = struct {
    epoch: usize = 0,
    timing: AutodiffEpochTimingSummary = .{},
    graph_cache_delta: AutodiffEpochGraphCacheDelta = .{},
    examples_seen: usize = 0,
    supervised_tokens_seen: usize = 0,
    logical_input_tokens_seen: usize = 0,
    scheduled_input_tokens_seen: usize = 0,
    fixed_shape_input_tokens_seen: usize = 0,
    padding_tokens_avoided: usize = 0,
    length_bucketed_examples: usize = 0,
    min_scheduled_sequence_length: u32 = 0,
    max_scheduled_sequence_length: u32 = 0,
    teacher_examples_seen: usize = 0,
    teacher_supervised_tokens_seen: usize = 0,
    mean_teacher_temperature: f64 = 0,
    average_loss: f64 = 0,
    mean_grad_norm: f64 = 0,
    optimizer_steps: usize = 0,
    graph_executor_steps: u64 = 0,
    graph_executor_fallback_steps: u64 = 0,
    graph_executor_partitions: u64 = 0,
    graph_executor_command_dispatches: u64 = 0,
    graph_executor_native_partitions: u64 = 0,
    graph_executor_unsupported_ops: u64 = 0,
    graph_executor_interpreter_fallbacks: u64 = 0,
    graph_executor_runtime_region_dispatches: u64 = 0,
    graph_executor_true_host_outputs: u64 = 0,
    runtime_input_uploads: u64 = 0,
    runtime_input_upload_bytes: u64 = 0,
    runtime_input_h2d_bytes: u64 = 0,
    runtime_input_d2h_bytes: u64 = 0,
    declared_runtime_input_uploads: u64 = 0,
    declared_runtime_input_upload_bytes: u64 = 0,
    declared_runtime_input_h2d_bytes: u64 = 0,
    graph_execution_h2d_bytes: u64 = 0,
    graph_execution_d2h_bytes: u64 = 0,
    training_runtime_h2d_bytes: u64 = 0,
    training_runtime_d2h_bytes: u64 = 0,
    metal_optimizer_steps: u64 = 0,
};

const CliOptions = struct {
    learning_rate: f32 = 0.001,
    max_examples: usize = 32,
    eval_max_examples: usize = 0,
    epochs: usize = 1,
    layer_name: ?[]const u8 = null,
    max_grad_norm: f32 = 1.0,
    grad_accum_steps: u32 = 1,
    activation_checkpoint_interval: u32 = 0,
    sequence_length_bucket_quantum: u32 = 0,
    sequence_length_bucket_min: u32 = 0,
    graph_cache_capacity: u8 = 0,
    llrd_decay: f32 = 1.0,
    use_schedule_free: bool = false,
    trainer_mode: TrainerMode = .autodiff,
    backend_kind: ?gemma4_real.BackendKind = null,
    gguf_projector_path: ?[]const u8 = null,
    eval_prepared_inputs_path: ?[]const u8 = null,
    seed: u64 = 42,
    checkpoint_path: ?[]const u8 = null,
    checkpoint_every_epochs: usize = 0,
    resume_from_checkpoint: bool = false,
    benchmark_request_path: ?[]const u8 = null,
    benchmark_telemetry_out_path: ?[]const u8 = null,
    oracle_request_path: ?[]const u8 = null,
    oracle_capture_out_dir: ?[]const u8 = null,

    fn effectiveEvalMaxExamples(self: CliOptions) usize {
        return if (self.eval_max_examples > 0) self.eval_max_examples else self.max_examples;
    }
};

pub const AutodiffExecutionPolicy = struct {
    engine: real_autodiff.TrainingExecutionEngine,
    compiled_required: bool,
    strict_metal_execution: bool,
};

pub fn autodiffExecutionPolicy(backend_kind: gemma4_real.BackendKind) AutodiffExecutionPolicy {
    const metal = backend_kind == .metal;
    return .{
        .engine = if (metal) .compiled_device else .interpreter,
        .compiled_required = metal,
        .strict_metal_execution = metal,
    };
}

const MultimodalPreparedStats = struct {
    examples_with_media: usize = 0,
    total_image_inputs: usize = 0,
    total_audio_inputs: usize = 0,
    total_image_soft_tokens: usize = 0,
    total_audio_soft_tokens: usize = 0,
};

const ReportContext = struct {
    prepared_inputs_path: []const u8,
    eval_prepared_inputs_path: ?[]const u8 = null,
    learning_rate: f32,
    max_examples: usize,
    eval_max_examples: usize,
    epochs: usize,
    layer_name: ?[]const u8,
    max_grad_norm: f32,
    grad_accum_steps: u32,
    activation_checkpoint_interval: u32,
    sequence_length_bucket_quantum: u32 = 0,
    sequence_length_bucket_min: u32 = 0,
    graph_cache_capacity: u8 = 1,
    llrd_decay: f32,
    use_schedule_free: bool,
    backend_label: []const u8,
    seed: u64 = 42,
    checkpoint_path: ?[]const u8 = null,
    checkpoint_every_epochs: usize = 0,
    resumed_from_epoch: ?usize = null,
    run_fingerprint: ?[]const u8 = null,
};

const benchmark_request_schema_v1 = "antfly_gemma4_lora_benchmark_request/v2";
const benchmark_telemetry_schema_v1 = "antfly_gemma4_lora_benchmark_telemetry/v4";
const oracle_request_schema_v1 = "antfly_gemma4_lora_zig_oracle_request/v1";
const oracle_capture_schema_v1 = "antfly_gemma4_lora_zig_oracle_capture/v1";
const benchmark_timed_unit_v1 = "optimizer-step-including-grad-accumulation";
const benchmark_sync_point_v1 = "after-optimizer-update-before-timer-stop-every-window";
const benchmark_peak_memory_source_v1 = "darwin-proc-pid-rusage-v4-lifetime-max-phys-footprint";
const benchmark_command_digest_env = "ANTFLY_GEMMA4_BENCHMARK_COMMAND_SHA256";
const benchmark_control_fd_env = "ANTFLY_GEMMA4_BENCHMARK_CONTROL_FD";
const benchmark_ack_fd_env = "ANTFLY_GEMMA4_BENCHMARK_ACK_FD";
const benchmark_initial_adapter_digest_domain_v1 = "antfly_gemma4_initial_adapter_semantics/v1";
const benchmark_target_inventory_digest_domain_v1 = "antfly_gemma4_target_inventory/v1";
const benchmark_workload_digest_domain_v1 = "antfly_gemma4_benchmark_workload/v1";
const benchmark_cold_steps_v1: usize = 1;
const benchmark_first_steps_v1: usize = 1;
const benchmark_warmup_steps_v1: usize = 3;
const benchmark_measured_steps_v1: usize = 20;
const benchmark_optimizer_steps_v1 = benchmark_cold_steps_v1 + benchmark_first_steps_v1 + benchmark_warmup_steps_v1 + benchmark_measured_steps_v1;

const BenchmarkImplementationV1 = struct {
    version: []const u8,
    executable_sha256: []const u8,
    source_revision: []const u8,
    metal_device: []const u8,
};

const BenchmarkBindingsV1 = struct {
    oracle_lock_sha256: []const u8,
    model_key: []const u8,
    model_revision: []const u8,
    local_artifact_sha256: []const u8,
    initial_adapter_semantic_sha256: []const u8,
    target_inventory_sha256: []const u8,
    target_count: usize,
    semantic_contract_sha256: []const u8,
    train_prepared_sha256: []const u8,
    eval_prepared_sha256: []const u8,
    workload_sha256: []const u8,
    prepared_example_index: usize,
    target_preset: []const u8,
    rank: usize,
    alpha: f64,
    sequence_length: usize,
    grad_accum: usize,
    microbatch: usize,
    supervised_tokens: usize,
};

const BenchmarkProtocolV1 = struct {
    fresh_process: bool,
    cold_optimizer_steps: usize,
    cold_step_mutates_optimizer_state: bool,
    first_steady_steps: usize,
    warmup_steps: usize,
    measured_steps: usize,
    explicit_device_sync: bool,
    sync_point: []const u8,
    timed_unit: []const u8,
};

const BenchmarkRuntimeV1 = struct {
    attention_kv_cache: bool,
    activation_checkpointing: bool,
    training_checkpoint_io: []const u8,
    compiled_graph_cache: []const u8,
    compile_policy: []const u8,
    filesystem_cache_policy: []const u8,
    per_step_device_sync: []const u8,
};

const BenchmarkRequestV1 = struct {
    schema_version: []const u8,
    implementation: BenchmarkImplementationV1,
    bindings: BenchmarkBindingsV1,
    protocol: BenchmarkProtocolV1,
    runtime: BenchmarkRuntimeV1,
    measurement_control: BenchmarkMeasurementControlV1,
};

const BenchmarkMeasurementControlV1 = struct {
    schema_version: []const u8,
    transport: []const u8,
    signal_fd_environment: []const u8,
    ack_fd_environment: []const u8,
    before_measured_signal: []const u8,
    before_measured_ack: []const u8,
    after_measured_signal: []const u8,
    after_measured_ack: []const u8,
};

const BenchmarkAdmission = struct {
    parsed: std.json.Parsed(BenchmarkRequestV1),
    telemetry_out_path: []const u8,
    request_sha256: []const u8,
    command_sha256: []const u8,
    control_signal_fd: std.posix.fd_t,
    control_ack_fd: std.posix.fd_t,

    fn deinit(self: *BenchmarkAdmission, allocator: std.mem.Allocator) void {
        self.parsed.deinit();
        allocator.free(self.request_sha256);
        allocator.free(self.command_sha256);
        self.* = undefined;
    }

    fn request(self: *const BenchmarkAdmission) *const BenchmarkRequestV1 {
        return &self.parsed.value;
    }
};

const OracleImplementationV1 = struct {
    version: []const u8,
    executable_sha256: []const u8,
    source_revision: []const u8,
    backend: []const u8,
    metal_device: ?[]const u8 = null,
};

const OracleBindingsV1 = struct {
    oracle_lock_sha256: []const u8,
    model_key: []const u8,
    model_revision: []const u8,
    local_artifact_sha256: []const u8,
    base_model_sha256: []const u8,
    initial_adapter_sha256: []const u8,
    train_prepared_sha256: []const u8,
    source_dataset_sha256: []const u8,
    example_index: usize,
    target_preset: []const u8,
    rank: usize,
    alpha: f64,
    target_count: usize,
};

const OracleTrainingV1 = struct {
    optimizer: []const u8,
    seed: u64,
    steps: usize,
    learning_rate: f64,
    betas: [2]f64,
    eps: f64,
    weight_decay: f64,
    max_grad_norm: f64,
    grad_accum_steps: u32,
    supervised_token_normalization: []const u8,
    dropout: f64,
    use_cache: bool,
};

const OracleRequestV1 = struct {
    schema_version: []const u8,
    implementation: OracleImplementationV1,
    bindings: OracleBindingsV1,
    training: OracleTrainingV1,
};

const OracleAdmission = struct {
    parsed: std.json.Parsed(OracleRequestV1),
    capture_out_dir: []const u8,
    request_sha256: []const u8,

    fn deinit(self: *OracleAdmission, allocator: std.mem.Allocator) void {
        self.parsed.deinit();
        allocator.free(self.request_sha256);
        self.* = undefined;
    }

    fn request(self: *const OracleAdmission) *const OracleRequestV1 {
        return &self.parsed.value;
    }
};

const CapturedOracleGradient = struct {
    name: []u8,
    values: []f32,
};

const OracleExecutionEvidence = struct {
    optimizer_steps: u64 = 0,
    micro_batch_steps: u64 = 0,
    metal_optimizer_steps: u64 = 0,
    graph_executor_steps: u64 = 0,
    graph_executor_fallback_steps: u64 = 0,
    graph_executor_native_partitions: u64 = 0,
    graph_executor_unsupported_ops: u64 = 0,
    graph_executor_interpreter_fallbacks: u64 = 0,
    graph_executor_true_host_outputs: u64 = 0,
};

const OracleCapture = struct {
    allocator: std.mem.Allocator,
    expected_steps: usize,
    gradient_hook_calls: usize = 0,
    gradients: std.ArrayListUnmanaged(CapturedOracleGradient) = .empty,
    loss_history: std.ArrayListUnmanaged(f64) = .empty,
    probes: ?gemma4_real.SupervisedLogitProbes = null,
    execution: OracleExecutionEvidence = .{},

    fn deinit(self: *OracleCapture) void {
        for (self.gradients.items) |gradient| {
            self.allocator.free(gradient.name);
            self.allocator.free(gradient.values);
        }
        self.gradients.deinit(self.allocator);
        self.loss_history.deinit(self.allocator);
        if (self.probes) |*probes| probes.deinit();
        self.* = undefined;
    }

    fn reduce(context: *anyopaque, blocks: []const real_autodiff.GradBlock) anyerror!void {
        const self: *OracleCapture = @ptrCast(@alignCast(context));
        if (blocks.len == 0) return error.InvalidOracleGradientCapture;
        for (blocks) |block| {
            for (block.data) |value| if (!std.math.isFinite(value)) return error.NonFiniteOracleGradient;
        }
        if (!try self.beginGradientHook()) return;
        try self.gradients.ensureTotalCapacity(self.allocator, blocks.len);
        for (blocks) |block| {
            const name = try self.allocator.dupe(u8, block.name);
            errdefer self.allocator.free(name);
            const values = try self.allocator.dupe(f32, block.data);
            errdefer self.allocator.free(values);
            self.gradients.appendAssumeCapacity(.{ .name = name, .values = values });
        }
    }

    /// Metal keeps the normal direct-device AdamW trajectory and pays a
    /// single bounded D2H readback only for the final raw-gradient surface.
    /// Earlier hooks are deliberately no-ops; merely installing this hook
    /// must not materialize every trajectory step on the host.
    fn observeDirectDevice(
        context: *anyopaque,
        compute_backend: *const ComputeBackend,
        blocks: []const real_autodiff.DeviceGradBlock,
    ) anyerror!void {
        const self: *OracleCapture = @ptrCast(@alignCast(context));
        if (blocks.len == 0) return error.InvalidOracleGradientCapture;
        if (!try self.beginGradientHook()) return;
        try self.gradients.ensureTotalCapacity(self.allocator, blocks.len);
        for (blocks) |block| {
            const values = try compute_backend.toFloat32(block.data, self.allocator);
            errdefer self.allocator.free(values);
            if (values.len != block.elem_count) return error.InvalidOracleGradientCapture;
            for (values) |value| if (!std.math.isFinite(value)) return error.NonFiniteOracleGradient;
            const name = try self.allocator.dupe(u8, block.name);
            errdefer self.allocator.free(name);
            self.gradients.appendAssumeCapacity(.{ .name = name, .values = values });
        }
    }

    fn beginGradientHook(self: *OracleCapture) !bool {
        self.gradient_hook_calls = try std.math.add(usize, self.gradient_hook_calls, 1);
        if (self.gradient_hook_calls > self.expected_steps) return error.OracleGradientHookCallOverflow;
        if (self.gradient_hook_calls != self.expected_steps) return false;
        if (self.gradients.items.len != 0) return error.InvalidOracleGradientCapture;
        return true;
    }

    fn recordStep(self: *OracleCapture, metrics: gemma4_real.CausalLmMetrics) !void {
        if (metrics.examples_seen != 1 or metrics.optimizer_steps != 1 or
            metrics.supervised_tokens_seen == 0 or !std.math.isFinite(metrics.average_loss))
        {
            return error.InvalidOracleTrainingStep;
        }
        try self.loss_history.append(self.allocator, metrics.average_loss);
        self.execution.optimizer_steps = try std.math.add(u64, self.execution.optimizer_steps, metrics.optimizer_steps);
        self.execution.micro_batch_steps = try std.math.add(u64, self.execution.micro_batch_steps, metrics.examples_seen);
        self.execution.metal_optimizer_steps = try std.math.add(u64, self.execution.metal_optimizer_steps, metrics.metal_optimizer_steps);
        self.execution.graph_executor_steps = try std.math.add(u64, self.execution.graph_executor_steps, metrics.graph_executor_steps);
        self.execution.graph_executor_fallback_steps = try std.math.add(u64, self.execution.graph_executor_fallback_steps, metrics.graph_executor_fallback_steps);
        self.execution.graph_executor_native_partitions = try std.math.add(u64, self.execution.graph_executor_native_partitions, metrics.graph_executor_native_partitions);
        self.execution.graph_executor_unsupported_ops = try std.math.add(u64, self.execution.graph_executor_unsupported_ops, metrics.graph_executor_unsupported_ops);
        self.execution.graph_executor_interpreter_fallbacks = try std.math.add(u64, self.execution.graph_executor_interpreter_fallbacks, metrics.graph_executor_interpreter_fallbacks);
        self.execution.graph_executor_true_host_outputs = try std.math.add(u64, self.execution.graph_executor_true_host_outputs, metrics.graph_executor_true_host_outputs);
    }

    fn validateComplete(self: *const OracleCapture, backend_kind: BackendKind, target_count: usize) !void {
        if (self.gradient_hook_calls != self.expected_steps or
            self.loss_history.items.len != self.expected_steps or
            self.gradients.items.len != target_count * 2 or
            self.probes == null or self.probes.?.rows.len == 0 or
            self.execution.optimizer_steps != self.expected_steps or
            self.execution.micro_batch_steps != self.expected_steps)
        {
            return error.IncompleteOracleCapture;
        }
        if (backend_kind == .metal and
            (self.execution.metal_optimizer_steps != self.expected_steps or
                self.execution.graph_executor_steps != self.expected_steps or
                self.execution.graph_executor_fallback_steps != 0 or
                self.execution.graph_executor_native_partitions != 0 or
                self.execution.graph_executor_unsupported_ops != 0 or
                self.execution.graph_executor_interpreter_fallbacks != 0 or
                self.execution.graph_executor_true_host_outputs != 0))
        {
            return error.InvalidOracleStrictMetalEvidence;
        }
    }
};

const OracleCaptureArtifactV1 = struct {
    path: []const u8,
    sha256: []const u8,
    size_bytes: u64,
};

const OracleCandidateAdapterArtifactV1 = struct {
    sha256: []const u8,
    size_bytes: u64,
};

const OracleCapturedTargetV1 = struct {
    source_name: []const u8,
    trainer_slot_name: []const u8,
    shape: []const usize,
    gradient_storage_key: []const u8,
    checkpoint_weight_storage_key: []const u8,
    checkpoint_m_storage_key: []const u8,
    checkpoint_v_storage_key: []const u8,
};

const BenchmarkStrictMetalEvidence = struct {
    optimizer_backend: []const u8 = "metal",
    metal_optimizer_steps: u64 = 0,
    graph_executor_steps: u64 = 0,
    graph_executor_fallback_steps: u64 = 0,
    native_partitions: u64 = 0,
    unsupported_ops: u64 = 0,
    interpreter_fallbacks: u64 = 0,
    runtime_region_fallbacks: u64 = 0,
    true_host_outputs: u64 = 0,
    host_gradient_tensors: u64 = 0,
};

/// Existing trainer timers, accumulated across every microbatch in one
/// optimizer window. These phases may overlap (for example `train_step_ns`
/// contains several child phases), so consumers must not add them together to
/// manufacture wall time.
const BenchmarkPhaseEvidence = struct {
    graph_build_ns: u64 = 0,
    runtime_input_ns: u64 = 0,
    train_step_ns: u64 = 0,
    compile_ns: u64 = 0,
    autodiff_ns: u64 = 0,
    execute_ns: u64 = 0,
    extract_ns: u64 = 0,
    optimizer_update_ns: u64 = 0,
    device_optimizer_ns: u64 = 0,
    total_ns: u64 = 0,
    metal_frame_wait_ns: u64 = 0,
    metal_frame_gpu_ns: u64 = 0,
    graph_executor_plan_build_ns: u64 = 0,
    graph_executor_buffer_plan_build_ns: u64 = 0,
};

/// Static/dynamic command-plan counters already emitted by the compiled
/// trainer, accumulated over one complete optimizer window. This makes graph
/// cache churn and dispatch-shape changes visible in the immutable benchmark
/// evidence instead of relying on diagnostic stderr.
const BenchmarkCommandPlanEvidence = struct {
    graph_executor_partitions: u64 = 0,
    graph_executor_command_dispatches: u64 = 0,
    graph_executor_planned_dispatches: u64 = 0,
    graph_executor_runtime_region_dispatches: u64 = 0,
    graph_executor_runtime_region_active_regions: u64 = 0,
    graph_executor_runtime_region_covered_nodes: u64 = 0,
    graph_executor_runtime_region_elided_nodes: u64 = 0,
    graph_executor_runtime_region_plan_compiles: u64 = 0,
    graph_executor_runtime_region_plan_reuses: u64 = 0,
    graph_executor_plan_cache_hits: u64 = 0,
    graph_executor_plan_cache_misses: u64 = 0,
    metal_lora_backward_regions: u64 = 0,
    metal_low_rank_lora_backward_regions: u64 = 0,
    metal_rank_adapter_backward_regions: u64 = 0,
    metal_ffn_gelu_backward_regions: u64 = 0,
    metal_head_mlp_forward_regions: u64 = 0,
    metal_head_mlp_backward_regions: u64 = 0,
    metal_gemma4_bf16_gate_up_fused_calls: u64 = 0,
    metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls: u64 = 0,
    metal_linear_cce_forward_calls: u64 = 0,
    metal_linear_cce_backward_calls: u64 = 0,
    metal_linear_cce_forward_state_hits: u64 = 0,
    metal_linear_cce_forward_state_misses: u64 = 0,
    metal_linear_cce_peak_scratch_bytes: u64 = 0,
    metal_command_dot_general_dispatches: u64 = 0,
    metal_command_head_dot_dispatches: u64 = 0,
    metal_command_transpose_dispatches: u64 = 0,
    metal_command_gather_dispatches: u64 = 0,
    metal_command_reduce_dispatches: u64 = 0,
    metal_command_elementwise_dispatches: u64 = 0,
    metal_command_activation_dispatches: u64 = 0,
    metal_command_activation_backward_dispatches: u64 = 0,
    metal_command_other_dispatches: u64 = 0,
    metal_last_frame_compute_encoders: u64 = 0,
    metal_last_frame_blit_encoders: u64 = 0,
    metal_last_frame_planned_scopes: u64 = 0,
    metal_last_frame_planned_barriers: u64 = 0,
    metal_last_frame_planned_command_ops: u64 = 0,
};

const BenchmarkOptimizerStep = struct {
    index: usize,
    phase: []const u8,
    duration_ns: u64,
    input_tokens: usize,
    supervised_tokens: usize,
    optimizer_stepped: bool = true,
    explicit_device_sync: bool = true,
    strict_metal_evidence: BenchmarkStrictMetalEvidence,
    phase_evidence: BenchmarkPhaseEvidence,
    command_plan_evidence: BenchmarkCommandPlanEvidence,
};

/// Aggregates microstep observations into complete optimizer windows. It is
/// deliberately independent of the epoch loop so a future request version can
/// classify a cold compile-mutating window separately without changing the
/// trainer observer contract.
const BenchmarkCapture = struct {
    allocator: std.mem.Allocator,
    request: *const BenchmarkRequestV1,
    io: std.Io,
    control_signal_fd: ?std.posix.fd_t = null,
    control_ack_fd: ?std.posix.fd_t = null,
    optimizer_steps: std.ArrayListUnmanaged(BenchmarkOptimizerStep) = .empty,
    load_ns: u64 = 0,
    compile_ns: u64 = 0,
    cold_step_was_first_graph_execution: bool = false,
    window_started_ns: ?u64 = null,
    window_microsteps: usize = 0,
    window_input_tokens: usize = 0,
    window_supervised_tokens: usize = 0,
    window_compile_ns: u64 = 0,
    window_evidence: BenchmarkStrictMetalEvidence = .{},
    window_phase_evidence: BenchmarkPhaseEvidence = .{},
    window_command_plan_evidence: BenchmarkCommandPlanEvidence = .{},

    fn deinit(self: *BenchmarkCapture) void {
        self.optimizer_steps.deinit(self.allocator);
        self.* = undefined;
    }

    fn observe(context: *anyopaque, observation: gemma4_real.PreparedMicrostepObservation) anyerror!void {
        const self: *BenchmarkCapture = @ptrCast(@alignCast(context));
        try self.record(observation);
    }

    fn record(self: *BenchmarkCapture, observation: gemma4_real.PreparedMicrostepObservation) !void {
        if (observation.started_ns == 0 or observation.finished_ns <= observation.started_ns) {
            return error.BenchmarkInvalidMonotonicInterval;
        }
        if (self.window_started_ns == null) self.window_started_ns = observation.started_ns;
        if (observation.started_ns < self.window_started_ns.?) return error.BenchmarkNonMonotonicMicrostep;
        self.window_microsteps = try std.math.add(usize, self.window_microsteps, 1);
        self.window_input_tokens = try std.math.add(usize, self.window_input_tokens, observation.input_tokens);
        self.window_supervised_tokens = try std.math.add(usize, self.window_supervised_tokens, observation.supervised_tokens);
        self.window_compile_ns = try std.math.add(u64, self.window_compile_ns, observation.step.profile.compile_ns);
        self.window_evidence.graph_executor_steps = try std.math.add(u64, self.window_evidence.graph_executor_steps, 1);
        if (observation.step.profile.graph_executor_fallback_reason != null) {
            self.window_evidence.graph_executor_fallback_steps = try std.math.add(u64, self.window_evidence.graph_executor_fallback_steps, 1);
        }
        self.window_evidence.native_partitions = try std.math.add(u64, self.window_evidence.native_partitions, observation.step.profile.graph_executor_native_partitions);
        self.window_evidence.unsupported_ops = try std.math.add(u64, self.window_evidence.unsupported_ops, observation.step.profile.graph_executor_unsupported_ops);
        self.window_evidence.interpreter_fallbacks = try std.math.add(u64, self.window_evidence.interpreter_fallbacks, observation.step.profile.graph_executor_interpreter_fallbacks);
        self.window_evidence.runtime_region_fallbacks = try std.math.add(u64, self.window_evidence.runtime_region_fallbacks, observation.step.profile.graph_executor_runtime_region_fallbacks);
        self.window_evidence.true_host_outputs = try std.math.add(u64, self.window_evidence.true_host_outputs, observation.step.profile.graph_executor_true_host_outputs);
        self.window_evidence.host_gradient_tensors = try std.math.add(u64, self.window_evidence.host_gradient_tensors, observation.step.profile.host_gradient_tensors);
        try addBenchmarkPhaseEvidence(&self.window_phase_evidence, observation.step.profile);
        try addBenchmarkCommandPlanEvidence(&self.window_command_plan_evidence, observation.step.profile);

        if (!observation.step.optimizer_stepped) {
            if (observation.explicit_device_sync) return error.BenchmarkUnexpectedDeviceSynchronization;
            if (self.window_microsteps >= self.request.bindings.grad_accum) return error.BenchmarkOptimizerWindowDidNotClose;
            return;
        }
        if (!observation.explicit_device_sync) return error.BenchmarkMissingDeviceSynchronization;
        if (observation.step.profile.optimizer_backend != .metal) return error.BenchmarkMetalOptimizerRequired;
        self.window_evidence.metal_optimizer_steps = 1;
        try self.finishWindow(observation.finished_ns);
    }

    fn finishWindow(self: *BenchmarkCapture, finished_ns: u64) !void {
        if (self.optimizer_steps.items.len >= benchmark_optimizer_steps_v1) return error.BenchmarkTooManyOptimizerSteps;
        if (self.window_microsteps != self.request.bindings.grad_accum) return error.BenchmarkIncompleteOptimizerWindow;
        const expected_input_tokens = try std.math.mul(
            usize,
            try std.math.mul(usize, self.request.bindings.sequence_length, self.request.bindings.grad_accum),
            self.request.bindings.microbatch,
        );
        if (self.window_input_tokens != expected_input_tokens or
            self.window_supervised_tokens != self.request.bindings.supervised_tokens)
        {
            return error.BenchmarkWorkloadTokenMismatch;
        }
        try validateBenchmarkStrictMetalEvidence(self.window_evidence, self.request.bindings.grad_accum);
        try validateBenchmarkCommandPlanEvidence(
            self.window_command_plan_evidence,
            self.request.bindings.grad_accum,
            self.optimizer_steps.items.len == 0,
        );
        if (self.window_phase_evidence.compile_ns != self.window_compile_ns) {
            return error.BenchmarkPhaseCompileMismatch;
        }
        const plan_was_built = self.window_command_plan_evidence.graph_executor_plan_cache_misses != 0;
        if ((self.window_phase_evidence.graph_executor_plan_build_ns != 0) != plan_was_built or
            (self.window_phase_evidence.graph_executor_buffer_plan_build_ns != 0) != plan_was_built)
        {
            return error.BenchmarkPlanBuildEvidenceMismatch;
        }
        if (self.optimizer_steps.items.len == 0) {
            self.compile_ns = self.window_compile_ns;
        } else if (self.window_compile_ns != 0) {
            return error.BenchmarkUnexpectedRecompile;
        }
        const started_ns = self.window_started_ns orelse return error.BenchmarkIncompleteOptimizerWindow;
        if (finished_ns <= started_ns) return error.BenchmarkInvalidMonotonicInterval;
        const index = self.optimizer_steps.items.len;
        const phase = if (index == 0)
            "cold"
        else if (index == 1)
            "first"
        else if (index < benchmark_cold_steps_v1 + benchmark_first_steps_v1 + benchmark_warmup_steps_v1)
            "warmup"
        else
            "measured";
        try self.optimizer_steps.append(self.allocator, .{
            .index = index,
            .phase = phase,
            .duration_ns = finished_ns - started_ns,
            .input_tokens = self.window_input_tokens,
            .supervised_tokens = self.window_supervised_tokens,
            .strict_metal_evidence = self.window_evidence,
            .phase_evidence = self.window_phase_evidence,
            .command_plan_evidence = self.window_command_plan_evidence,
        });
        if (self.control_signal_fd) |signal_fd| {
            const ack_fd = self.control_ack_fd orelse return error.MissingBenchmarkControlAckFd;
            const completed = self.optimizer_steps.items.len;
            const measured_start = benchmark_cold_steps_v1 + benchmark_first_steps_v1 + benchmark_warmup_steps_v1;
            if (completed == measured_start) {
                try exchangeBenchmarkControlSignal(self.io, signal_fd, ack_fd, 'B', 'b');
            }
            if (completed == benchmark_optimizer_steps_v1) {
                try exchangeBenchmarkControlSignal(self.io, signal_fd, ack_fd, 'A', 'a');
            }
        }
        self.window_started_ns = null;
        self.window_microsteps = 0;
        self.window_input_tokens = 0;
        self.window_supervised_tokens = 0;
        self.window_compile_ns = 0;
        self.window_evidence = .{};
        self.window_phase_evidence = .{};
        self.window_command_plan_evidence = .{};
    }

    fn validateComplete(self: *const BenchmarkCapture) !void {
        if (self.window_started_ns != null or self.window_microsteps != 0) return error.BenchmarkIncompleteOptimizerWindow;
        if (self.optimizer_steps.items.len != benchmark_optimizer_steps_v1) return error.BenchmarkOptimizerStepCountMismatch;
        if (self.compile_ns == 0) return error.BenchmarkColdCompileMissing;
        if (!self.cold_step_was_first_graph_execution) return error.BenchmarkColdStepNotFirstGraphExecution;
    }
};

fn addU64(target: *u64, value: u64) !void {
    target.* = try std.math.add(u64, target.*, value);
}

fn addBenchmarkPhaseEvidence(target: *BenchmarkPhaseEvidence, profile: real_autodiff.StepProfile) !void {
    try addU64(&target.graph_build_ns, profile.graph_build_ns);
    try addU64(&target.runtime_input_ns, profile.runtime_input_ns);
    try addU64(&target.train_step_ns, profile.train_step_ns);
    try addU64(&target.compile_ns, profile.compile_ns);
    try addU64(&target.autodiff_ns, profile.autodiff_ns);
    try addU64(&target.execute_ns, profile.execute_ns);
    try addU64(&target.extract_ns, profile.extract_ns);
    try addU64(&target.optimizer_update_ns, profile.optimizer_update_ns);
    try addU64(&target.device_optimizer_ns, profile.device_optimizer_ns);
    try addU64(&target.total_ns, profile.total_ns);
    try addU64(&target.metal_frame_wait_ns, profile.metal_frame_wait_ns);
    try addU64(&target.metal_frame_gpu_ns, profile.metal_frame_gpu_ns);
    try addU64(&target.graph_executor_plan_build_ns, profile.graph_executor_plan_build_ns);
    try addU64(&target.graph_executor_buffer_plan_build_ns, profile.graph_executor_buffer_plan_build_ns);
}

fn addBenchmarkCommandPlanEvidence(target: *BenchmarkCommandPlanEvidence, profile: real_autodiff.StepProfile) !void {
    try addU64(&target.graph_executor_partitions, profile.graph_executor_partitions);
    try addU64(&target.graph_executor_command_dispatches, profile.graph_executor_command_dispatches);
    try addU64(&target.graph_executor_planned_dispatches, profile.graph_executor_planned_dispatches);
    try addU64(&target.graph_executor_runtime_region_dispatches, profile.graph_executor_runtime_region_dispatches);
    try addU64(&target.graph_executor_runtime_region_active_regions, profile.graph_executor_runtime_region_active_regions);
    try addU64(&target.graph_executor_runtime_region_covered_nodes, profile.graph_executor_runtime_region_covered_nodes);
    try addU64(&target.graph_executor_runtime_region_elided_nodes, profile.graph_executor_runtime_region_elided_nodes);
    try addU64(&target.graph_executor_runtime_region_plan_compiles, profile.graph_executor_runtime_region_plan_compiles);
    try addU64(&target.graph_executor_runtime_region_plan_reuses, profile.graph_executor_runtime_region_plan_reuses);
    try addU64(&target.graph_executor_plan_cache_hits, profile.graph_executor_plan_cache_hits);
    try addU64(&target.graph_executor_plan_cache_misses, profile.graph_executor_plan_cache_misses);
    try addU64(&target.metal_lora_backward_regions, profile.metal_lora_backward_regions);
    try addU64(&target.metal_low_rank_lora_backward_regions, profile.metal_low_rank_lora_backward_regions);
    try addU64(&target.metal_rank_adapter_backward_regions, profile.metal_rank_adapter_backward_regions);
    try addU64(&target.metal_ffn_gelu_backward_regions, profile.metal_ffn_gelu_backward_regions);
    try addU64(&target.metal_head_mlp_forward_regions, profile.metal_head_mlp_forward_regions);
    try addU64(&target.metal_head_mlp_backward_regions, profile.metal_head_mlp_backward_regions);
    try addU64(&target.metal_gemma4_bf16_gate_up_fused_calls, profile.metal_gemma4_bf16_gate_up_fused_calls);
    try addU64(&target.metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls, profile.metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls);
    try addU64(&target.metal_linear_cce_forward_calls, profile.metal_linear_cce_forward_calls);
    try addU64(&target.metal_linear_cce_backward_calls, profile.metal_linear_cce_backward_calls);
    try addU64(&target.metal_linear_cce_forward_state_hits, profile.metal_linear_cce_forward_state_hits);
    try addU64(&target.metal_linear_cce_forward_state_misses, profile.metal_linear_cce_forward_state_misses);
    target.metal_linear_cce_peak_scratch_bytes = @max(
        target.metal_linear_cce_peak_scratch_bytes,
        profile.metal_linear_cce_peak_scratch_bytes,
    );
    try addU64(&target.metal_command_dot_general_dispatches, profile.metal_command_dot_general_dispatches);
    try addU64(&target.metal_command_head_dot_dispatches, profile.metal_command_head_dot_dispatches);
    try addU64(&target.metal_command_transpose_dispatches, profile.metal_command_transpose_dispatches);
    try addU64(&target.metal_command_gather_dispatches, profile.metal_command_gather_dispatches);
    try addU64(&target.metal_command_reduce_dispatches, profile.metal_command_reduce_dispatches);
    try addU64(&target.metal_command_elementwise_dispatches, profile.metal_command_elementwise_dispatches);
    try addU64(&target.metal_command_activation_dispatches, profile.metal_command_activation_dispatches);
    try addU64(&target.metal_command_activation_backward_dispatches, profile.metal_command_activation_backward_dispatches);
    try addU64(&target.metal_command_other_dispatches, profile.metal_command_other_dispatches);
    try addU64(&target.metal_last_frame_compute_encoders, profile.metal_last_frame_compute_encoders);
    try addU64(&target.metal_last_frame_blit_encoders, profile.metal_last_frame_blit_encoders);
    try addU64(&target.metal_last_frame_planned_scopes, profile.metal_last_frame_planned_scopes);
    try addU64(&target.metal_last_frame_planned_barriers, profile.metal_last_frame_planned_barriers);
    try addU64(&target.metal_last_frame_planned_command_ops, profile.metal_last_frame_planned_command_ops);
}

fn exchangeBenchmarkControlSignal(
    io: std.Io,
    signal_fd: std.posix.fd_t,
    ack_fd: std.posix.fd_t,
    signal: u8,
    expected_ack: u8,
) !void {
    const signal_file = std.Io.File{ .handle = signal_fd, .flags = .{ .nonblocking = false } };
    try signal_file.writeStreamingAll(io, &.{signal});
    const ack_file = std.Io.File{ .handle = ack_fd, .flags = .{ .nonblocking = false } };
    var ack: [1]u8 = undefined;
    const read = try ack_file.readStreaming(io, &.{ack[0..]});
    if (read != 1) return error.BenchmarkControlAckEof;
    if (ack[0] != expected_ack) return error.BenchmarkControlAckMismatch;
}

fn validateBenchmarkStrictMetalEvidence(evidence: BenchmarkStrictMetalEvidence, grad_accum: usize) !void {
    if (!std.mem.eql(u8, evidence.optimizer_backend, "metal") or evidence.metal_optimizer_steps != 1) {
        return error.BenchmarkMetalOptimizerRequired;
    }
    if (evidence.graph_executor_steps != grad_accum) return error.BenchmarkGraphExecutorStepCountMismatch;
    if (evidence.graph_executor_fallback_steps != 0 or
        evidence.native_partitions != 0 or
        evidence.unsupported_ops != 0 or
        evidence.interpreter_fallbacks != 0 or
        evidence.runtime_region_fallbacks != 0 or
        evidence.true_host_outputs != 0 or
        evidence.host_gradient_tensors != 0)
    {
        return error.BenchmarkStrictMetalFallbackObserved;
    }
}

fn validateBenchmarkCommandPlanEvidence(
    evidence: BenchmarkCommandPlanEvidence,
    grad_accum: usize,
    cold_window: bool,
) !void {
    if (evidence.graph_executor_partitions < grad_accum or evidence.graph_executor_command_dispatches == 0) {
        return error.BenchmarkCommandPlanMissingExecution;
    }
    if (evidence.graph_executor_planned_dispatches > evidence.graph_executor_command_dispatches) {
        return error.BenchmarkCommandPlanDispatchMismatch;
    }
    const attributed_counts = [_]u64{
        evidence.metal_command_dot_general_dispatches,
        evidence.metal_command_head_dot_dispatches,
        evidence.metal_command_transpose_dispatches,
        evidence.metal_command_gather_dispatches,
        evidence.metal_command_reduce_dispatches,
        evidence.metal_command_elementwise_dispatches,
        evidence.metal_command_activation_dispatches,
        evidence.metal_command_activation_backward_dispatches,
        evidence.metal_command_other_dispatches,
    };
    var attributed_dispatches: u64 = 0;
    for (attributed_counts) |count| attributed_dispatches = try std.math.add(u64, attributed_dispatches, count);
    if (attributed_dispatches > evidence.graph_executor_command_dispatches) {
        return error.BenchmarkCommandPlanAttributionMismatch;
    }
    const cce_state_events = try std.math.add(
        u64,
        evidence.metal_linear_cce_forward_state_hits,
        evidence.metal_linear_cce_forward_state_misses,
    );
    if (cce_state_events != evidence.metal_linear_cce_backward_calls or
        evidence.metal_linear_cce_backward_calls != evidence.metal_linear_cce_forward_calls or
        evidence.metal_linear_cce_forward_state_misses != 0)
    {
        return error.BenchmarkLinearCceEvidenceMismatch;
    }
    const cce_executed = evidence.metal_linear_cce_forward_calls != 0;
    if ((evidence.metal_linear_cce_peak_scratch_bytes != 0) != cce_executed) {
        return error.BenchmarkLinearCceScratchEvidenceMismatch;
    }
    const cache_lookups = try std.math.add(
        u64,
        evidence.graph_executor_plan_cache_hits,
        evidence.graph_executor_plan_cache_misses,
    );
    if (cache_lookups != grad_accum) return error.BenchmarkCommandPlanCacheLookupMismatch;
    if (cold_window) {
        if (evidence.graph_executor_plan_cache_misses != 1) return error.BenchmarkColdPlanCacheMissMismatch;
    } else if (evidence.graph_executor_plan_cache_misses != 0) {
        return error.BenchmarkUnexpectedPlanCacheMiss;
    }
}

test "gemma4 benchmark command-plan evidence rejects cache and attribution drift" {
    const cold = BenchmarkCommandPlanEvidence{
        .graph_executor_partitions = 4,
        .graph_executor_command_dispatches = 12,
        .graph_executor_planned_dispatches = 8,
        .graph_executor_plan_cache_hits = 3,
        .graph_executor_plan_cache_misses = 1,
        .metal_command_dot_general_dispatches = 4,
        .metal_command_elementwise_dispatches = 4,
        .metal_command_other_dispatches = 4,
    };
    try validateBenchmarkCommandPlanEvidence(cold, 4, true);

    var warm = cold;
    warm.graph_executor_plan_cache_hits = 4;
    warm.graph_executor_plan_cache_misses = 0;
    try validateBenchmarkCommandPlanEvidence(warm, 4, false);

    warm.graph_executor_plan_cache_hits = 3;
    warm.graph_executor_plan_cache_misses = 1;
    try std.testing.expectError(
        error.BenchmarkUnexpectedPlanCacheMiss,
        validateBenchmarkCommandPlanEvidence(warm, 4, false),
    );

    warm.graph_executor_plan_cache_hits = 4;
    warm.graph_executor_plan_cache_misses = 0;
    warm.metal_command_other_dispatches = 5;
    try std.testing.expectError(
        error.BenchmarkCommandPlanAttributionMismatch,
        validateBenchmarkCommandPlanEvidence(warm, 4, false),
    );
}

test "gemma4 benchmark capture separates cold steady warmup and measured optimizer windows" {
    const request = BenchmarkRequestV1{
        .schema_version = benchmark_request_schema_v1,
        .implementation = .{
            .version = "test-version",
            .executable_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            .source_revision = "0000000000000000000000000000000000000000",
            .metal_device = "test-metal",
        },
        .bindings = .{
            .oracle_lock_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            .model_key = "gemma-4-E2B-it",
            .model_revision = "0000000000000000000000000000000000000000",
            .local_artifact_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            .initial_adapter_semantic_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            .target_inventory_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            .target_count = 2,
            .semantic_contract_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            .train_prepared_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            .eval_prepared_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            .workload_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            .prepared_example_index = 0,
            .target_preset = "peft-qv",
            .rank = 16,
            .alpha = 32,
            .sequence_length = 8,
            .grad_accum = 2,
            .microbatch = 1,
            .supervised_tokens = 6,
        },
        .protocol = .{
            .fresh_process = true,
            .cold_optimizer_steps = 1,
            .cold_step_mutates_optimizer_state = true,
            .first_steady_steps = 1,
            .warmup_steps = 3,
            .measured_steps = 20,
            .explicit_device_sync = true,
            .sync_point = benchmark_sync_point_v1,
            .timed_unit = benchmark_timed_unit_v1,
        },
        .runtime = .{
            .attention_kv_cache = false,
            .activation_checkpointing = false,
            .training_checkpoint_io = "disabled",
            .compiled_graph_cache = "enabled-process-local-shape-keyed",
            .compile_policy = "lazy-first-full-optimizer-window-included-in-cold-metric",
            .filesystem_cache_policy = "not-flushed-between-alternating-fresh-processes",
            .per_step_device_sync = benchmark_sync_point_v1,
        },
        .measurement_control = .{
            .schema_version = "antfly_gemma4_benchmark_measurement_control/v1",
            .transport = "inherited-fd-byte-signals-with-ack",
            .signal_fd_environment = benchmark_control_fd_env,
            .ack_fd_environment = benchmark_ack_fd_env,
            .before_measured_signal = "B",
            .before_measured_ack = "b",
            .after_measured_signal = "A",
            .after_measured_ack = "a",
        },
    };
    var capture = BenchmarkCapture{
        .allocator = std.testing.allocator,
        .request = &request,
        .io = std.testing.io,
        .cold_step_was_first_graph_execution = true,
    };
    defer capture.deinit();
    var now: u64 = 1;
    for (0..benchmark_optimizer_steps_v1) |window_index| {
        for (0..request.bindings.grad_accum) |microstep| {
            const stepped = microstep + 1 == request.bindings.grad_accum;
            const started = now;
            now += 10;
            try capture.record(.{
                .step = .{
                    .loss = 1,
                    .grad_norm = 1,
                    .step = @intCast(window_index * request.bindings.grad_accum + microstep + 1),
                    .optimizer_stepped = stepped,
                    .profile = .{
                        .compile_ns = if (window_index == 0 and microstep == 0) 7 else 0,
                        .optimizer_backend = .metal,
                        .graph_executor_partitions = 1,
                        .graph_executor_command_dispatches = 3,
                        .graph_executor_planned_dispatches = 2,
                        .graph_executor_plan_cache_hits = if (window_index != 0 or microstep != 0) 1 else 0,
                        .graph_executor_plan_cache_misses = if (window_index == 0 and microstep == 0) 1 else 0,
                        .graph_executor_plan_build_ns = if (window_index == 0 and microstep == 0) 5 else 0,
                        .graph_executor_buffer_plan_build_ns = if (window_index == 0 and microstep == 0) 2 else 0,
                        .metal_command_dot_general_dispatches = 1,
                        .metal_command_elementwise_dispatches = 1,
                        .metal_command_other_dispatches = 1,
                        .metal_gemma4_bf16_gate_up_fused_calls = 1,
                        .metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls = 3,
                        .metal_linear_cce_forward_calls = 1,
                        .metal_linear_cce_backward_calls = 1,
                        .metal_linear_cce_forward_state_hits = 1,
                        .metal_linear_cce_peak_scratch_bytes = if (microstep == 0) 65_536 else 32_768,
                    },
                },
                .started_ns = started,
                .finished_ns = now,
                .input_tokens = request.bindings.sequence_length,
                .supervised_tokens = request.bindings.supervised_tokens / request.bindings.grad_accum,
                .explicit_device_sync = stepped,
            });
        }
    }
    try capture.validateComplete();
    try std.testing.expectEqualStrings("cold", capture.optimizer_steps.items[0].phase);
    try std.testing.expectEqualStrings("first", capture.optimizer_steps.items[1].phase);
    try std.testing.expectEqualStrings("warmup", capture.optimizer_steps.items[2].phase);
    try std.testing.expectEqualStrings("measured", capture.optimizer_steps.items[5].phase);
    try std.testing.expectEqual(@as(u64, 7), capture.compile_ns);
    try std.testing.expectEqual(@as(u64, 6), capture.optimizer_steps.items[0].command_plan_evidence.graph_executor_command_dispatches);
    try std.testing.expectEqual(@as(u64, 1), capture.optimizer_steps.items[0].command_plan_evidence.graph_executor_plan_cache_misses);
    try std.testing.expectEqual(@as(u64, 2), capture.optimizer_steps.items[0].command_plan_evidence.metal_gemma4_bf16_gate_up_fused_calls);
    try std.testing.expectEqual(@as(u64, 6), capture.optimizer_steps.items[0].command_plan_evidence.metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls);
    try std.testing.expectEqual(@as(u64, 2), capture.optimizer_steps.items[0].command_plan_evidence.metal_linear_cce_forward_calls);
    try std.testing.expectEqual(@as(u64, 2), capture.optimizer_steps.items[0].command_plan_evidence.metal_linear_cce_forward_state_hits);
    try std.testing.expectEqual(@as(u64, 65_536), capture.optimizer_steps.items[0].command_plan_evidence.metal_linear_cce_peak_scratch_bytes);
    try std.testing.expectEqual(@as(u64, 2), capture.optimizer_steps.items[0].phase_evidence.graph_executor_buffer_plan_build_ns);
}

const ImmutableRunPublication = artifact_publication.ImmutableDirectoryPublication;

test "gemma4 whole-run publication is immutable and cleans owned staging" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const existing_dir = try std.fs.path.join(allocator, &.{ root, "existing" });
    defer allocator.free(existing_dir);
    try cwd.createDirPath(io, existing_dir);
    const sentinel_path = try std.fs.path.join(allocator, &.{ existing_dir, "sentinel.txt" });
    defer allocator.free(sentinel_path);
    try cwd.writeFile(io, .{ .sub_path = sentinel_path, .data = "known-good" });

    try std.testing.expectError(
        error.Gemma4RunOutputAlreadyExists,
        ImmutableRunPublication.init(allocator, io, existing_dir),
    );
    const sentinel = try cwd.readFileAlloc(io, sentinel_path, allocator, .limited(32));
    defer allocator.free(sentinel);
    try std.testing.expectEqualStrings("known-good", sentinel);

    const failed_dir = try std.fs.path.join(allocator, &.{ root, "failed" });
    defer allocator.free(failed_dir);
    var failed_staging_copy: []u8 = undefined;
    {
        var failed = try ImmutableRunPublication.init(allocator, io, failed_dir);
        defer failed.deinit();
        failed_staging_copy = try allocator.dupe(u8, failed.staging_dir);
        try cwd.createDirPath(io, failed.staging_dir);
        failed.claimStaging();
        const partial_path = try std.fs.path.join(allocator, &.{ failed.staging_dir, "partial.json" });
        defer allocator.free(partial_path);
        try cwd.writeFile(io, .{ .sub_path = partial_path, .data = "partial" });
    }
    defer allocator.free(failed_staging_copy);
    try std.testing.expectError(error.FileNotFound, cwd.access(io, failed_staging_copy, .{}));
    try std.testing.expectError(error.FileNotFound, cwd.access(io, failed_dir, .{}));

    const fresh_dir = try std.fs.path.join(allocator, &.{ root, "fresh" });
    defer allocator.free(fresh_dir);
    var fresh = try ImmutableRunPublication.init(allocator, io, fresh_dir);
    defer fresh.deinit();
    try cwd.createDirPath(io, fresh.staging_dir);
    fresh.claimStaging();
    const complete_path = try std.fs.path.join(allocator, &.{ fresh.staging_dir, "training_report.json" });
    defer allocator.free(complete_path);
    try cwd.writeFile(io, .{ .sub_path = complete_path, .data = "complete" });
    try fresh.publish();

    const published_path = try std.fs.path.join(allocator, &.{ fresh_dir, "training_report.json" });
    defer allocator.free(published_path);
    const published = try cwd.readFileAlloc(io, published_path, allocator, .limited(32));
    defer allocator.free(published);
    try std.testing.expectEqualStrings("complete", published);

    const late_dir = try std.fs.path.join(allocator, &.{ root, "late-collision" });
    defer allocator.free(late_dir);
    var late = try ImmutableRunPublication.init(allocator, io, late_dir);
    defer late.deinit();
    try cwd.createDirPath(io, late.staging_dir);
    late.claimStaging();
    try cwd.createDirPath(io, late_dir);
    const late_sentinel_path = try std.fs.path.join(allocator, &.{ late_dir, "sentinel.txt" });
    defer allocator.free(late_sentinel_path);
    try cwd.writeFile(io, .{ .sub_path = late_sentinel_path, .data = "late-winner" });
    try std.testing.expectError(error.Gemma4RunOutputAlreadyExists, late.publish());
    const late_sentinel = try cwd.readFileAlloc(io, late_sentinel_path, allocator, .limited(32));
    defer allocator.free(late_sentinel);
    try std.testing.expectEqualStrings("late-winner", late_sentinel);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    var argv: std.ArrayListUnmanaged([]const u8) = .empty;
    defer argv.deinit(allocator);
    while (args.next()) |arg| try argv.append(allocator, arg);
    try runFromArgs(allocator, init.io, argv.items);
}

pub fn runFromArgs(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
    try runFromArgsWithSummary(allocator, io, argv, true);
}

/// Run the production parser and trainer without writing the terminal summary.
/// This is useful for embedders and for Zig's server-mode test runner, whose
/// stdout is reserved for the compiler protocol.
pub fn runFromArgsWithoutSummary(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
    try runFromArgsWithSummary(allocator, io, argv, false);
}

fn runFromArgsWithSummary(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8, emit_summary: bool) !void {
    if (argv.len == 1 and isHelpArg(argv[0])) {
        printTrainUsage();
        return;
    }
    const parsed = try parseTrainArgs(argv);
    try trainWithSummary(allocator, io, parsed, emit_summary);
}

/// Parse the canonical named-flag interface and the one-release positional
/// compatibility form into the same typed production operation.
pub fn parseTrainArgs(argv: []const []const u8) !TrainOptions {
    var base_model_dir: ?[]const u8 = null;
    var adapter_model_dir: ?[]const u8 = null;
    var prepared_inputs_path: ?[]const u8 = null;
    var out_dir: ?[]const u8 = null;
    var opts = CliOptions{};
    var legacy_option_count: usize = 0;
    var seed_seen = false;
    var checkpoint_interval_seen = false;
    var resume_seen = false;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= argv.len or base_model_dir != null) return usageError();
            base_model_dir = argv[i];
        } else if (std.mem.eql(u8, arg, "--adapter")) {
            i += 1;
            if (i >= argv.len or adapter_model_dir != null) return usageError();
            adapter_model_dir = argv[i];
        } else if (std.mem.eql(u8, arg, "--train-prepared")) {
            i += 1;
            if (i >= argv.len or prepared_inputs_path != null) return usageError();
            prepared_inputs_path = argv[i];
        } else if (std.mem.eql(u8, arg, "--out")) {
            i += 1;
            if (i >= argv.len or out_dir != null) return usageError();
            out_dir = argv[i];
        } else if (std.mem.eql(u8, arg, "--lr") or std.mem.eql(u8, arg, "--learning-rate")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.learning_rate = try std.fmt.parseFloat(f32, argv[i]);
        } else if (std.mem.eql(u8, arg, "--max-examples")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.max_examples = try std.fmt.parseUnsigned(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--eval-max-examples")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.eval_max_examples = try std.fmt.parseUnsigned(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--eval-prepared")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.eval_prepared_inputs_path = argv[i];
        } else if (std.mem.eql(u8, arg, "--epochs")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.epochs = try std.fmt.parseUnsigned(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--layer-name") or std.mem.eql(u8, arg, "--layer")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.layer_name = argv[i];
        } else if (std.mem.eql(u8, arg, "--max-grad-norm")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.max_grad_norm = try std.fmt.parseFloat(f32, argv[i]);
        } else if (std.mem.eql(u8, arg, "--grad-accum")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.grad_accum_steps = try std.fmt.parseUnsigned(u32, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--activation-checkpoint-interval")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.activation_checkpoint_interval = try std.fmt.parseUnsigned(u32, argv[i], 10);
            if (opts.activation_checkpoint_interval == 0) return usageError();
        } else if (std.mem.eql(u8, arg, "--sequence-length-bucket-quantum")) {
            i += 1;
            if (i >= argv.len or opts.sequence_length_bucket_quantum != 0) return usageError();
            opts.sequence_length_bucket_quantum = try std.fmt.parseUnsigned(u32, argv[i], 10);
            if (opts.sequence_length_bucket_quantum == 0) return usageError();
        } else if (std.mem.eql(u8, arg, "--sequence-length-bucket-min")) {
            i += 1;
            if (i >= argv.len or opts.sequence_length_bucket_min != 0) return usageError();
            opts.sequence_length_bucket_min = try std.fmt.parseUnsigned(u32, argv[i], 10);
            if (opts.sequence_length_bucket_min == 0) return usageError();
        } else if (std.mem.eql(u8, arg, "--graph-cache-capacity")) {
            i += 1;
            if (i >= argv.len or opts.graph_cache_capacity != 0) return usageError();
            opts.graph_cache_capacity = try std.fmt.parseUnsigned(u8, argv[i], 10);
            if (opts.graph_cache_capacity == 0) return usageError();
        } else if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= argv.len or seed_seen) return usageError();
            seed_seen = true;
            opts.seed = try std.fmt.parseUnsigned(u64, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--checkpoint-path")) {
            i += 1;
            if (i >= argv.len or opts.checkpoint_path != null) return usageError();
            opts.checkpoint_path = argv[i];
        } else if (std.mem.eql(u8, arg, "--checkpoint-every-epochs")) {
            i += 1;
            if (i >= argv.len or checkpoint_interval_seen) return usageError();
            checkpoint_interval_seen = true;
            opts.checkpoint_every_epochs = try std.fmt.parseUnsigned(usize, argv[i], 10);
            if (opts.checkpoint_every_epochs == 0) return usageError();
        } else if (std.mem.eql(u8, arg, "--resume")) {
            if (resume_seen) return usageError();
            resume_seen = true;
            opts.resume_from_checkpoint = true;
        } else if (std.mem.eql(u8, arg, "--benchmark-request")) {
            i += 1;
            if (i >= argv.len or opts.benchmark_request_path != null) return usageError();
            opts.benchmark_request_path = argv[i];
        } else if (std.mem.eql(u8, arg, "--benchmark-telemetry-out")) {
            i += 1;
            if (i >= argv.len or opts.benchmark_telemetry_out_path != null) return usageError();
            opts.benchmark_telemetry_out_path = argv[i];
        } else if (std.mem.eql(u8, arg, "--oracle-request")) {
            i += 1;
            if (i >= argv.len or opts.oracle_request_path != null) return usageError();
            opts.oracle_request_path = argv[i];
        } else if (std.mem.eql(u8, arg, "--oracle-capture-out")) {
            i += 1;
            if (i >= argv.len or opts.oracle_capture_out_dir != null) return usageError();
            opts.oracle_capture_out_dir = argv[i];
        } else if (std.mem.eql(u8, arg, "--llrd-decay")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.llrd_decay = try std.fmt.parseFloat(f32, argv[i]);
        } else if (std.mem.eql(u8, arg, "--schedule-free")) {
            opts.use_schedule_free = true;
        } else if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i >= argv.len) return usageError();
            const val = argv[i];
            if (std.mem.eql(u8, val, "native")) {
                opts.backend_kind = .native;
            } else if (std.mem.eql(u8, val, "metal")) {
                opts.backend_kind = .metal;
            } else return usageError();
        } else if (std.mem.eql(u8, arg, "--trainer")) {
            i += 1;
            if (i >= argv.len) return usageError();
            const val = argv[i];
            if (std.mem.eql(u8, val, "auto")) {
                opts.trainer_mode = .auto;
            } else if (std.mem.eql(u8, val, "surrogate")) {
                opts.trainer_mode = .surrogate;
            } else if (std.mem.eql(u8, val, "autodiff")) {
                opts.trainer_mode = .autodiff;
            } else return usageError();
        } else if (std.mem.eql(u8, arg, "--gguf-projector")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.gguf_projector_path = argv[i];
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return usageError();
        } else if (base_model_dir == null) {
            base_model_dir = arg;
        } else if (adapter_model_dir == null) {
            adapter_model_dir = arg;
        } else if (prepared_inputs_path == null) {
            prepared_inputs_path = arg;
        } else if (out_dir == null) {
            out_dir = arg;
        } else {
            switch (legacy_option_count) {
                0 => opts.learning_rate = try std.fmt.parseFloat(f32, arg),
                1 => opts.max_examples = try std.fmt.parseUnsigned(usize, arg, 10),
                2 => opts.epochs = try std.fmt.parseUnsigned(usize, arg, 10),
                3 => opts.layer_name = arg,
                else => return usageError(),
            }
            legacy_option_count += 1;
        }
    }

    const actual_mode = resolveTrainerMode(opts.trainer_mode);
    if (actual_mode == .surrogate) return error.Gemma4SurrogateTrainingNotSupported;
    if (opts.gguf_projector_path != null) return error.Gemma4MultimodalFinetuningNotSupported;
    if (opts.backend_kind == null) return error.MissingBackend;
    if (opts.eval_prepared_inputs_path == null) return error.MissingEvaluationPreparedInputs;
    try validateAutodiffTrainingOptions(opts);

    return .{
        .base_model_dir = base_model_dir orelse return usageError(),
        .adapter_model_dir = adapter_model_dir orelse return usageError(),
        .train_prepared_inputs_path = prepared_inputs_path orelse return usageError(),
        .eval_prepared_inputs_path = opts.eval_prepared_inputs_path.?,
        .out_dir = out_dir orelse return usageError(),
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.eval_max_examples,
        .epochs = opts.epochs,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .activation_checkpoint_interval = opts.activation_checkpoint_interval,
        .sequence_length_bucket_quantum = opts.sequence_length_bucket_quantum,
        .sequence_length_bucket_min = opts.sequence_length_bucket_min,
        .graph_cache_capacity = opts.graph_cache_capacity,
        .seed = opts.seed,
        .checkpoint_path = opts.checkpoint_path,
        .checkpoint_every_epochs = opts.checkpoint_every_epochs,
        .resume_from_checkpoint = opts.resume_from_checkpoint,
        .backend_kind = opts.backend_kind.?,
        .benchmark_request_path = opts.benchmark_request_path,
        .benchmark_telemetry_out_path = opts.benchmark_telemetry_out_path,
        .oracle_request_path = opts.oracle_request_path,
        .oracle_capture_out_dir = opts.oracle_capture_out_dir,
    };
}

pub fn train(allocator: std.mem.Allocator, io: std.Io, options: TrainOptions) !void {
    try trainWithSummary(allocator, io, options, true);
}

fn trainWithSummary(allocator: std.mem.Allocator, io: std.Io, options: TrainOptions, emit_summary: bool) !void {
    try validateTrainOptions(options);
    var benchmark_admission = try loadBenchmarkAdmission(allocator, io, options);
    defer if (benchmark_admission) |*admission| admission.deinit(allocator);
    var oracle_admission = try loadOracleAdmission(allocator, io, options);
    defer if (oracle_admission) |*admission| admission.deinit(allocator);
    var graph_executor_scope = try acquireMetalGraphExecutorScope(options.backend_kind);
    defer if (graph_executor_scope) |*scope| scope.deinit();
    try validateAutodiffBaseArtifact(allocator, options.base_model_dir, options.backend_kind);

    var prepared = try finetune.loadPreparedInputsSummary(allocator, options.train_prepared_inputs_path);
    defer finetune.freePreparedInputsSummary(allocator, &prepared);
    var eval_prepared = try finetune.loadPreparedInputsSummary(allocator, options.eval_prepared_inputs_path);
    defer finetune.freePreparedInputsSummary(allocator, &eval_prepared);

    if (options.checkpoint_path) |checkpoint_path| {
        var immutable_inputs: [6][]const u8 = undefined;
        var immutable_input_count: usize = 0;
        for ([_][]const u8{
            options.base_model_dir,
            options.adapter_model_dir,
            options.train_prepared_inputs_path,
            options.eval_prepared_inputs_path,
        }) |path| {
            immutable_inputs[immutable_input_count] = path;
            immutable_input_count += 1;
        }
        if (prepared.source_dataset_path) |path| {
            immutable_inputs[immutable_input_count] = path;
            immutable_input_count += 1;
        }
        if (eval_prepared.source_dataset_path) |path| {
            immutable_inputs[immutable_input_count] = path;
            immutable_input_count += 1;
        }
        try validateCheckpointPathIsolation(
            allocator,
            io,
            checkpoint_path,
            options.out_dir,
            immutable_inputs[0..immutable_input_count],
        );
    }

    if (countMultimodalExamples(prepared.examples) > 0 or countMultimodalExamples(eval_prepared.examples) > 0) {
        return error.Gemma4MultimodalFinetuningNotSupported;
    }

    try runAutodiff(
        io,
        allocator,
        options.base_model_dir,
        options.adapter_model_dir,
        options.train_prepared_inputs_path,
        options.eval_prepared_inputs_path,
        options.out_dir,
        prepared,
        eval_prepared,
        .{
            .learning_rate = options.learning_rate,
            .max_examples = options.max_examples,
            .eval_max_examples = options.eval_max_examples,
            .epochs = options.epochs,
            .max_grad_norm = options.max_grad_norm,
            .grad_accum_steps = options.grad_accum_steps,
            .activation_checkpoint_interval = options.activation_checkpoint_interval,
            .sequence_length_bucket_quantum = options.sequence_length_bucket_quantum,
            .sequence_length_bucket_min = options.sequence_length_bucket_min,
            .graph_cache_capacity = options.graph_cache_capacity,
            .seed = options.seed,
            .checkpoint_path = options.checkpoint_path,
            .checkpoint_every_epochs = options.checkpoint_every_epochs,
            .resume_from_checkpoint = options.resume_from_checkpoint,
            .trainer_mode = .autodiff,
            .backend_kind = options.backend_kind,
            .benchmark_request_path = options.benchmark_request_path,
            .benchmark_telemetry_out_path = options.benchmark_telemetry_out_path,
            .oracle_request_path = options.oracle_request_path,
            .oracle_capture_out_dir = options.oracle_capture_out_dir,
        },
        if (benchmark_admission) |*admission| admission else null,
        if (oracle_admission) |*admission| admission else null,
        emit_summary,
    );
}

fn validateTrainOptions(options: TrainOptions) !void {
    if (!std.math.isFinite(options.learning_rate) or options.learning_rate <= 0) return error.InvalidLearningRate;
    if (options.epochs == 0) return error.InvalidEpochCount;
    if (options.grad_accum_steps == 0) return error.InvalidGradientAccumulation;
    if (!std.math.isFinite(options.max_grad_norm) or options.max_grad_norm < 0) return error.InvalidMaxGradNorm;
    try validateSequenceLengthPolicy(
        options.sequence_length_bucket_quantum,
        options.sequence_length_bucket_min,
        options.graph_cache_capacity,
    );
    try validateBenchmarkOptionPair(options.benchmark_request_path, options.benchmark_telemetry_out_path);
    try validateOracleOptionPair(options.oracle_request_path, options.oracle_capture_out_dir);
    if (options.benchmark_request_path != null and options.oracle_request_path != null) {
        return error.IncompatibleEvidenceModes;
    }
    try validateCheckpointOptions(options.checkpoint_path, options.checkpoint_every_epochs, options.resume_from_checkpoint, options.epochs);
}

const max_sequence_bucket_graph_cache_capacity: u8 = 8;

fn validateSequenceLengthPolicy(quantum: u32, minimum: u32, graph_cache_capacity: u8) !void {
    if (quantum == 0 and (minimum != 0 or graph_cache_capacity != 0)) {
        return error.SequenceLengthBucketQuantumRequired;
    }
    if (graph_cache_capacity > max_sequence_bucket_graph_cache_capacity) {
        return error.GraphCacheCapacityTooLarge;
    }
}

fn sequenceLengthBuckets(opts: CliOptions) ?gemma4_real.SequenceLengthBuckets {
    if (opts.sequence_length_bucket_quantum == 0) return null;
    return .{
        .quantum = opts.sequence_length_bucket_quantum,
        .minimum = effectiveSequenceLengthBucketMinimum(opts),
    };
}

fn effectiveSequenceLengthBucketMinimum(opts: CliOptions) u32 {
    if (opts.sequence_length_bucket_quantum == 0) return 0;
    return if (opts.sequence_length_bucket_min == 0)
        opts.sequence_length_bucket_quantum
    else
        opts.sequence_length_bucket_min;
}

fn effectiveGraphCacheCapacity(opts: CliOptions) u8 {
    if (opts.graph_cache_capacity != 0) return opts.graph_cache_capacity;
    return if (opts.sequence_length_bucket_quantum != 0) 4 else 1;
}

fn validateBenchmarkOptionPair(request_path: ?[]const u8, telemetry_path: ?[]const u8) !void {
    if ((request_path == null) != (telemetry_path == null)) return error.IncompleteBenchmarkOptions;
    if (request_path) |path| if (path.len == 0) return error.InvalidBenchmarkRequestPath;
    if (telemetry_path) |path| if (path.len == 0) return error.InvalidBenchmarkTelemetryPath;
}

fn validateOracleOptionPair(request_path: ?[]const u8, capture_out_dir: ?[]const u8) !void {
    if ((request_path == null) != (capture_out_dir == null)) return error.IncompleteOracleOptions;
    if (request_path) |path| if (path.len == 0) return error.InvalidOracleRequestPath;
    if (capture_out_dir) |path| if (path.len == 0) return error.InvalidOracleCapturePath;
}

test "gemma4 oracle CLI options are paired and exclusive with benchmark mode" {
    try validateOracleOptionPair(null, null);
    try validateOracleOptionPair("request.json", "capture");
    try std.testing.expectError(error.IncompleteOracleOptions, validateOracleOptionPair("request.json", null));
    try std.testing.expectError(error.IncompleteOracleOptions, validateOracleOptionPair(null, "capture"));
    try std.testing.expectError(error.InvalidOracleRequestPath, validateOracleOptionPair("", "capture"));
    try std.testing.expectError(error.InvalidOracleCapturePath, validateOracleOptionPair("request.json", ""));

    var options = TrainOptions{
        .base_model_dir = "model",
        .adapter_model_dir = "adapter",
        .train_prepared_inputs_path = "train.json",
        .eval_prepared_inputs_path = "eval.json",
        .out_dir = "out",
        .backend_kind = .native,
    };
    options.benchmark_request_path = "benchmark.json";
    options.benchmark_telemetry_out_path = "benchmark-out.json";
    options.oracle_request_path = "oracle.json";
    options.oracle_capture_out_dir = "oracle-out";
    try std.testing.expectError(error.IncompatibleEvidenceModes, validateTrainOptions(options));
}

test "gemma4 benchmark CLI options are paired" {
    try validateBenchmarkOptionPair(null, null);
    try validateBenchmarkOptionPair("request.json", "telemetry.json");
    try std.testing.expectError(error.IncompleteBenchmarkOptions, validateBenchmarkOptionPair("request.json", null));
    try std.testing.expectError(error.IncompleteBenchmarkOptions, validateBenchmarkOptionPair(null, "telemetry.json"));
    try std.testing.expectError(error.InvalidBenchmarkRequestPath, validateBenchmarkOptionPair("", "telemetry.json"));
    try std.testing.expectError(error.InvalidBenchmarkTelemetryPath, validateBenchmarkOptionPair("request.json", ""));
}

fn loadBenchmarkAdmission(
    allocator: std.mem.Allocator,
    io: std.Io,
    options: TrainOptions,
) !?BenchmarkAdmission {
    const request_path = options.benchmark_request_path orelse return null;
    const telemetry_path = options.benchmark_telemetry_out_path orelse return error.IncompleteBenchmarkOptions;
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.BenchmarkRequiresDarwinArm64;

    const raw = try readFileAtPathLimited(io, allocator, request_path, 64 * 1024);
    defer allocator.free(raw);
    var parsed = try std.json.parseFromSlice(BenchmarkRequestV1, allocator, raw, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = false,
    });
    errdefer parsed.deinit();
    try validateBenchmarkRequestStatic(parsed.value, options);
    try validateBenchmarkPathIsolation(allocator, io, request_path, telemetry_path, options);
    try rejectExistingBenchmarkTelemetry(io, telemetry_path);

    const executable_path = try std.process.executablePathAlloc(io, allocator);
    defer allocator.free(executable_path);
    var executable_fingerprint = try finetune.fingerprintProjectorFile(allocator, executable_path);
    defer finetune.freeProjectorFingerprint(allocator, &executable_fingerprint);
    if (!matchesPrefixedSha256(parsed.value.implementation.executable_sha256, executable_fingerprint.sha256)) {
        return error.BenchmarkExecutableFingerprintMismatch;
    }
    try validateEmbeddedBenchmarkIdentity(
        build_options.inference_version,
        build_options.benchmark_source_revision,
        parsed.value.implementation.version,
        parsed.value.implementation.source_revision,
    );
    try validateBenchmarkStrictEnvironment();
    const command_digest = platform.env.getenv(benchmark_command_digest_env) orelse return error.MissingBenchmarkCommandDigest;
    if (!isPrefixedSha256(command_digest)) return error.InvalidBenchmarkCommandDigest;
    const control_fd_text = platform.env.getenv(benchmark_control_fd_env) orelse return error.MissingBenchmarkControlFd;
    const control_signal_fd = try std.fmt.parseInt(std.posix.fd_t, control_fd_text, 10);
    const ack_fd_text = platform.env.getenv(benchmark_ack_fd_env) orelse return error.MissingBenchmarkControlAckFd;
    const control_ack_fd = try std.fmt.parseInt(std.posix.fd_t, ack_fd_text, 10);
    if (control_signal_fd < 3 or control_ack_fd < 3 or control_signal_fd == control_ack_fd) {
        return error.InvalidBenchmarkControlFd;
    }
    for ([_]std.posix.fd_t{ control_signal_fd, control_ack_fd }) |fd| {
        const control_file = std.Io.File{ .handle = fd, .flags = .{ .nonblocking = false } };
        const control_stat = try control_file.stat(io);
        if (control_stat.kind != .named_pipe) return error.InvalidBenchmarkControlFd;
    }

    const request_sha256 = try sha256PrefixedAlloc(allocator, raw);
    errdefer allocator.free(request_sha256);
    const command_sha256 = try allocator.dupe(u8, command_digest);
    errdefer allocator.free(command_sha256);
    return .{
        .parsed = parsed,
        .telemetry_out_path = telemetry_path,
        .request_sha256 = request_sha256,
        .command_sha256 = command_sha256,
        .control_signal_fd = control_signal_fd,
        .control_ack_fd = control_ack_fd,
    };
}

fn validateEmbeddedOracleIdentity(
    embedded_version: []const u8,
    embedded_source_revision: []const u8,
    requested_version: []const u8,
    requested_source_revision: []const u8,
) !void {
    if (!isLowerHex(embedded_source_revision, 40)) return error.OracleSourceRevisionNotEmbedded;
    if (!std.mem.eql(u8, requested_version, embedded_version)) return error.OracleEmbeddedVersionMismatch;
    if (!std.mem.eql(u8, requested_source_revision, embedded_source_revision)) {
        return error.OracleEmbeddedSourceRevisionMismatch;
    }
}

fn loadOracleAdmission(
    allocator: std.mem.Allocator,
    io: std.Io,
    options: TrainOptions,
) !?OracleAdmission {
    const request_path = options.oracle_request_path orelse return null;
    const capture_out_dir = options.oracle_capture_out_dir orelse return error.IncompleteOracleOptions;
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.OracleRequiresDarwinArm64;

    const raw = try readFileAtPathLimited(io, allocator, request_path, 64 * 1024);
    defer allocator.free(raw);
    var parsed = try std.json.parseFromSlice(OracleRequestV1, allocator, raw, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = false,
    });
    errdefer parsed.deinit();
    try validateOracleRequestStatic(parsed.value, options);
    try validateOraclePathIsolation(allocator, io, request_path, capture_out_dir, options);

    const executable_path = try std.process.executablePathAlloc(io, allocator);
    defer allocator.free(executable_path);
    var executable_fingerprint = try finetune.fingerprintProjectorFile(allocator, executable_path);
    defer finetune.freeProjectorFingerprint(allocator, &executable_fingerprint);
    if (!matchesPrefixedSha256(parsed.value.implementation.executable_sha256, executable_fingerprint.sha256)) {
        return error.OracleExecutableFingerprintMismatch;
    }
    try validateEmbeddedOracleIdentity(
        build_options.inference_version,
        build_options.benchmark_source_revision,
        parsed.value.implementation.version,
        parsed.value.implementation.source_revision,
    );

    const request_sha256 = try sha256PrefixedAlloc(allocator, raw);
    errdefer allocator.free(request_sha256);
    return .{
        .parsed = parsed,
        .capture_out_dir = capture_out_dir,
        .request_sha256 = request_sha256,
    };
}

fn validateOracleRequestStatic(request: OracleRequestV1, options: TrainOptions) !void {
    if (!std.mem.eql(u8, request.schema_version, oracle_request_schema_v1)) {
        return error.UnsupportedOracleRequestVersion;
    }
    const metal_device_valid = if (request.implementation.metal_device) |device|
        device.len > 0 and std.unicode.utf8ValidateSlice(device) and
            std.mem.indexOfAny(u8, device, "\r\n\x00") == null
    else
        false;
    if (!isPrefixedSha256(request.implementation.executable_sha256) or
        !isLowerHex(request.implementation.source_revision, 40) or
        request.implementation.version.len == 0 or
        !std.unicode.utf8ValidateSlice(request.implementation.version) or
        std.mem.indexOfAny(u8, request.implementation.version, "\r\n\x00") != null or
        !std.mem.eql(u8, request.implementation.backend, options.backend_kind.label()) or
        (options.backend_kind == .metal and !metal_device_valid) or
        (options.backend_kind != .metal and request.implementation.metal_device != null))
    {
        return error.InvalidOracleImplementation;
    }
    inline for (.{
        request.bindings.oracle_lock_sha256,
        request.bindings.local_artifact_sha256,
        request.bindings.initial_adapter_sha256,
        request.bindings.train_prepared_sha256,
    }) |digest| if (!isPrefixedSha256(digest)) return error.InvalidOracleArtifactFingerprint;
    if ((!std.mem.eql(u8, request.bindings.model_key, "gemma-4-E2B-it") and
        !std.mem.eql(u8, request.bindings.model_key, "gemma-4-E4B-it")) or
        !isLowerHex(request.bindings.model_revision, 40) or
        !isLowerHex(request.bindings.base_model_sha256, 64) or
        !isLowerHex(request.bindings.source_dataset_sha256, 64) or
        finetune.parseGemma4LoRATargetPreset(request.bindings.target_preset) == null or
        request.bindings.rank == 0 or request.bindings.target_count == 0 or
        !std.math.isFinite(request.bindings.alpha) or request.bindings.alpha <= 0)
    {
        return error.InvalidOracleBindings;
    }
    const training = request.training;
    if (!std.mem.eql(u8, training.optimizer, "adamw") or
        training.seed != 42 or
        (training.steps != 1 and training.steps != 2 and training.steps != 8) or
        training.learning_rate != 0.001 or
        training.betas[0] != 0.9 or training.betas[1] != 0.999 or
        training.eps != 1e-8 or training.weight_decay != 0.01 or
        training.max_grad_norm != 1.0 or training.grad_accum_steps != 1 or
        !std.mem.eql(u8, training.supervised_token_normalization, "mean") or
        training.dropout != 0.0 or training.use_cache)
    {
        return error.InvalidOracleTrainingContract;
    }
    if (options.learning_rate != @as(f32, @floatCast(training.learning_rate)) or
        options.max_examples != 1 or options.eval_max_examples != 1 or
        options.epochs != training.steps or options.max_grad_norm != @as(f32, @floatCast(training.max_grad_norm)) or
        options.grad_accum_steps != training.grad_accum_steps or options.seed != training.seed or
        options.activation_checkpoint_interval != 0 or options.sequence_length_bucket_quantum != 0 or
        options.sequence_length_bucket_min != 0 or options.graph_cache_capacity != 0 or
        options.checkpoint_path != null or options.checkpoint_every_epochs != 0 or
        options.resume_from_checkpoint or options.benchmark_request_path != null)
    {
        return error.OracleCommandContractMismatch;
    }
}

fn oracleTestOptions(backend_kind: BackendKind) TrainOptions {
    return .{
        .base_model_dir = "model",
        .adapter_model_dir = "adapter",
        .train_prepared_inputs_path = "train.json",
        .eval_prepared_inputs_path = "train.json",
        .out_dir = "out",
        .learning_rate = 0.001,
        .max_examples = 1,
        .eval_max_examples = 1,
        .epochs = 2,
        .max_grad_norm = 1.0,
        .grad_accum_steps = 1,
        .seed = 42,
        .backend_kind = backend_kind,
        .oracle_request_path = "request.json",
        .oracle_capture_out_dir = "capture",
    };
}

fn oracleTestRequest(backend: []const u8, metal_device: ?[]const u8) OracleRequestV1 {
    return .{
        .schema_version = oracle_request_schema_v1,
        .implementation = .{
            .version = "1.2.3",
            .executable_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            .source_revision = "0123456789abcdef0123456789abcdef01234567",
            .backend = backend,
            .metal_device = metal_device,
        },
        .bindings = .{
            .oracle_lock_sha256 = "sha256:1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            .model_key = "gemma-4-E2B-it",
            .model_revision = "1123456789abcdef0123456789abcdef01234567",
            .local_artifact_sha256 = "sha256:2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            .base_model_sha256 = "3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            .initial_adapter_sha256 = "sha256:4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            .train_prepared_sha256 = "sha256:5123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            .source_dataset_sha256 = "6123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            .example_index = 0,
            .target_preset = "text-all-linear",
            .rank = 16,
            .alpha = 32,
            .target_count = 11,
        },
        .training = .{
            .optimizer = "adamw",
            .seed = 42,
            .steps = 2,
            .learning_rate = 0.001,
            .betas = .{ 0.9, 0.999 },
            .eps = 1e-8,
            .weight_decay = 0.01,
            .max_grad_norm = 1.0,
            .grad_accum_steps = 1,
            .supervised_token_normalization = "mean",
            .dropout = 0,
            .use_cache = false,
        },
    };
}

test "gemma4 oracle request pins implementation artifacts and optimizer trajectory" {
    const native_options = oracleTestOptions(.native);
    try validateOracleRequestStatic(oracleTestRequest("native", null), native_options);

    var invalid = oracleTestRequest("native", null);
    invalid.training.steps = 3;
    try std.testing.expectError(error.InvalidOracleTrainingContract, validateOracleRequestStatic(invalid, native_options));
    invalid = oracleTestRequest("native", null);
    invalid.bindings.train_prepared_sha256 = "not-a-digest";
    try std.testing.expectError(error.InvalidOracleArtifactFingerprint, validateOracleRequestStatic(invalid, native_options));
    invalid = oracleTestRequest("metal", "Apple M-series");
    try std.testing.expectError(error.InvalidOracleImplementation, validateOracleRequestStatic(invalid, native_options));
    invalid = oracleTestRequest("native", "unexpected");
    try std.testing.expectError(error.InvalidOracleImplementation, validateOracleRequestStatic(invalid, native_options));

    var metal_options = oracleTestOptions(.metal);
    try validateOracleRequestStatic(oracleTestRequest("metal", "Apple M-series"), metal_options);
    try std.testing.expectError(
        error.InvalidOracleImplementation,
        validateOracleRequestStatic(oracleTestRequest("metal", null), metal_options),
    );
    metal_options.epochs = 1;
    try std.testing.expectError(
        error.OracleCommandContractMismatch,
        validateOracleRequestStatic(oracleTestRequest("metal", "Apple M-series"), metal_options),
    );
}

test "gemma4 oracle host reducer retains only the final raw gradient surface" {
    var capture = OracleCapture{ .allocator = std.testing.allocator, .expected_steps = 2 };
    defer capture.deinit();
    var first = [_]f32{ 1, 2 };
    var second = [_]f32{ 3, 4 };
    const blocks = [_]real_autodiff.GradBlock{
        .{ .name = "weight.a.lora_A", .data = &first },
        .{ .name = "weight.a.lora_B", .data = &second },
    };
    try OracleCapture.reduce(@ptrCast(&capture), &blocks);
    try std.testing.expectEqual(@as(usize, 0), capture.gradients.items.len);
    first = .{ 5, 6 };
    second = .{ 7, 8 };
    try OracleCapture.reduce(@ptrCast(&capture), &blocks);
    try std.testing.expectEqual(@as(usize, 2), capture.gradients.items.len);
    try std.testing.expectEqualSlices(f32, &.{ 5, 6 }, capture.gradients.items[0].values);
    try std.testing.expectEqualSlices(f32, &.{ 7, 8 }, capture.gradients.items[1].values);
    try std.testing.expectError(error.OracleGradientHookCallOverflow, OracleCapture.reduce(@ptrCast(&capture), &blocks));
}

fn validateOraclePathIsolation(
    allocator: std.mem.Allocator,
    io: std.Io,
    request_path: []const u8,
    capture_out_dir: []const u8,
    options: TrainOptions,
) !void {
    const request_resolved = try std.Io.Dir.cwd().realPathFileAlloc(io, request_path, allocator);
    defer allocator.free(request_resolved);
    const capture_resolved = try path_isolation.resolveRequestedPath(allocator, io, capture_out_dir);
    defer allocator.free(capture_resolved);
    const output_resolved = try path_isolation.resolveRequestedPath(allocator, io, options.out_dir);
    defer allocator.free(output_resolved);
    if (path_isolation.pathsOverlap(request_resolved, capture_resolved) or
        path_isolation.pathsOverlap(request_resolved, output_resolved) or
        path_isolation.pathsOverlap(capture_resolved, output_resolved))
    {
        return error.OracleOutputPathOverlap;
    }
    for ([_][]const u8{
        options.base_model_dir,
        options.adapter_model_dir,
        options.train_prepared_inputs_path,
        options.eval_prepared_inputs_path,
    }) |immutable_input| {
        const input_resolved = try std.Io.Dir.cwd().realPathFileAlloc(io, immutable_input, allocator);
        defer allocator.free(input_resolved);
        if (path_isolation.pathsOverlap(input_resolved, capture_resolved) or
            path_isolation.pathsOverlap(input_resolved, output_resolved))
        {
            return error.OracleOutputOverlapsImmutableInput;
        }
    }
}

fn validateBenchmarkRequestStatic(request: BenchmarkRequestV1, options: TrainOptions) !void {
    if (!std.mem.eql(u8, request.schema_version, benchmark_request_schema_v1)) return error.UnsupportedBenchmarkRequestVersion;
    if (!isPrefixedSha256(request.implementation.executable_sha256)) return error.InvalidBenchmarkExecutableFingerprint;
    if (!isLowerHex(request.implementation.source_revision, 40)) return error.InvalidBenchmarkSourceRevision;
    if (request.implementation.version.len == 0 or
        !std.unicode.utf8ValidateSlice(request.implementation.version) or
        std.mem.indexOfAny(u8, request.implementation.version, "\r\n\x00") != null)
    {
        return error.InvalidBenchmarkVersion;
    }
    if (request.implementation.metal_device.len == 0 or
        !std.unicode.utf8ValidateSlice(request.implementation.metal_device) or
        std.mem.indexOfAny(u8, request.implementation.metal_device, "\r\n\x00") != null)
    {
        return error.InvalidBenchmarkMetalDevice;
    }
    inline for (.{
        request.bindings.oracle_lock_sha256,
        request.bindings.local_artifact_sha256,
        request.bindings.initial_adapter_semantic_sha256,
        request.bindings.target_inventory_sha256,
        request.bindings.semantic_contract_sha256,
        request.bindings.train_prepared_sha256,
        request.bindings.eval_prepared_sha256,
        request.bindings.workload_sha256,
    }) |digest| if (!isPrefixedSha256(digest)) return error.InvalidBenchmarkArtifactFingerprint;
    if ((!std.mem.eql(u8, request.bindings.model_key, "gemma-4-E2B-it") and
        !std.mem.eql(u8, request.bindings.model_key, "gemma-4-E4B-it")) or
        !isLowerHex(request.bindings.model_revision, 40))
    {
        return error.InvalidBenchmarkModelBinding;
    }
    if (finetune.parseGemma4LoRATargetPreset(request.bindings.target_preset) == null or
        request.bindings.rank == 0 or
        request.bindings.target_count == 0 or
        !std.math.isFinite(request.bindings.alpha) or
        request.bindings.alpha <= 0 or
        request.bindings.sequence_length == 0 or
        request.bindings.grad_accum == 0 or
        request.bindings.microbatch != 1 or
        request.bindings.supervised_tokens == 0)
    {
        return error.InvalidBenchmarkWorkloadBinding;
    }
    if (!request.protocol.fresh_process or
        request.protocol.cold_optimizer_steps != benchmark_cold_steps_v1 or
        !request.protocol.cold_step_mutates_optimizer_state or
        request.protocol.first_steady_steps != benchmark_first_steps_v1 or
        request.protocol.warmup_steps != benchmark_warmup_steps_v1 or
        request.protocol.measured_steps != benchmark_measured_steps_v1 or
        !request.protocol.explicit_device_sync or
        !std.mem.eql(u8, request.protocol.sync_point, benchmark_sync_point_v1) or
        !std.mem.eql(u8, request.protocol.timed_unit, benchmark_timed_unit_v1))
    {
        return error.InvalidBenchmarkProtocol;
    }
    if (options.backend_kind != .metal or
        options.learning_rate != @as(f32, 0.001) or
        options.max_grad_norm != @as(f32, 1.0) or
        options.seed != 42 or
        options.max_examples != request.bindings.grad_accum or
        options.eval_max_examples != 1 or
        options.epochs != benchmark_optimizer_steps_v1 or
        options.grad_accum_steps != request.bindings.grad_accum or
        options.activation_checkpoint_interval != 0 or
        options.sequence_length_bucket_quantum != 0 or
        options.sequence_length_bucket_min != 0 or
        options.graph_cache_capacity != 0 or
        options.checkpoint_path != null or
        options.checkpoint_every_epochs != 0 or
        options.resume_from_checkpoint)
    {
        return error.BenchmarkCommandContractMismatch;
    }
    try validateBenchmarkRuntime(request.runtime);
    if (!std.mem.eql(u8, request.measurement_control.schema_version, "antfly_gemma4_benchmark_measurement_control/v1") or
        !std.mem.eql(u8, request.measurement_control.transport, "inherited-fd-byte-signals-with-ack") or
        !std.mem.eql(u8, request.measurement_control.signal_fd_environment, benchmark_control_fd_env) or
        !std.mem.eql(u8, request.measurement_control.ack_fd_environment, benchmark_ack_fd_env) or
        !std.mem.eql(u8, request.measurement_control.before_measured_signal, "B") or
        !std.mem.eql(u8, request.measurement_control.before_measured_ack, "b") or
        !std.mem.eql(u8, request.measurement_control.after_measured_signal, "A") or
        !std.mem.eql(u8, request.measurement_control.after_measured_ack, "a"))
    {
        return error.InvalidBenchmarkMeasurementControl;
    }
}

fn validateEmbeddedBenchmarkIdentity(
    embedded_version: []const u8,
    embedded_source_revision: []const u8,
    requested_version: []const u8,
    requested_source_revision: []const u8,
) !void {
    if (!isLowerHex(embedded_source_revision, 40)) return error.BenchmarkSourceRevisionNotEmbedded;
    if (!std.mem.eql(u8, requested_version, embedded_version)) return error.BenchmarkEmbeddedVersionMismatch;
    if (!std.mem.eql(u8, requested_source_revision, embedded_source_revision)) {
        return error.BenchmarkEmbeddedSourceRevisionMismatch;
    }
}

test "gemma4 benchmark independently attests product version and source revision" {
    const revision = "0123456789abcdef0123456789abcdef01234567";
    try validateEmbeddedBenchmarkIdentity("1.2.3", revision, "1.2.3", revision);
    try std.testing.expectError(
        error.BenchmarkSourceRevisionNotEmbedded,
        validateEmbeddedBenchmarkIdentity("1.2.3", "dev", "1.2.3", revision),
    );
    try std.testing.expectError(
        error.BenchmarkEmbeddedVersionMismatch,
        validateEmbeddedBenchmarkIdentity("1.2.4", revision, "1.2.3", revision),
    );
    try std.testing.expectError(
        error.BenchmarkEmbeddedSourceRevisionMismatch,
        validateEmbeddedBenchmarkIdentity("1.2.3", revision, "1.2.3", "fedcba9876543210fedcba9876543210fedcba98"),
    );
}

fn validateBenchmarkRuntime(runtime: BenchmarkRuntimeV1) !void {
    if (runtime.attention_kv_cache or
        runtime.activation_checkpointing or
        !std.mem.eql(u8, runtime.training_checkpoint_io, "disabled") or
        !std.mem.eql(u8, runtime.compiled_graph_cache, "enabled-process-local-shape-keyed") or
        !std.mem.eql(u8, runtime.compile_policy, "lazy-first-full-optimizer-window-included-in-cold-metric") or
        !std.mem.eql(u8, runtime.filesystem_cache_policy, "not-flushed-between-alternating-fresh-processes") or
        !std.mem.eql(u8, runtime.per_step_device_sync, benchmark_sync_point_v1))
    {
        return error.BenchmarkRuntimeContractMismatch;
    }
}

fn validateBenchmarkStrictEnvironment() !void {
    const required = [_]struct { name: [:0]const u8, value: []const u8 }{
        .{ .name = "TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR", .value = "1" },
        .{ .name = "TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", .value = "0" },
        .{ .name = "TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK", .value = "0" },
        .{ .name = "TERMITE_DEBUG_DEVICE_GRAD_NORM", .value = "0" },
    };
    for (required) |entry| {
        const actual = platform.env.getenvSlice(entry.name) orelse return error.MissingBenchmarkEnvironmentBinding;
        if (!std.mem.eql(u8, actual, entry.value)) return error.BenchmarkEnvironmentBindingMismatch;
    }
    if (platform.env.getenv("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_NODE_IDS") != null) {
        return error.BenchmarkDiagnosticEnvironmentForbidden;
    }
}

fn validateBenchmarkPathIsolation(
    allocator: std.mem.Allocator,
    io: std.Io,
    request_path: []const u8,
    telemetry_path: []const u8,
    options: TrainOptions,
) !void {
    const request_resolved = try std.Io.Dir.cwd().realPathFileAlloc(io, request_path, allocator);
    defer allocator.free(request_resolved);
    const telemetry_resolved = try path_isolation.resolveRequestedPath(allocator, io, telemetry_path);
    defer allocator.free(telemetry_resolved);
    if (path_isolation.pathsOverlap(request_resolved, telemetry_resolved)) return error.BenchmarkTelemetryOverlapsRequest;

    const mutable_output = try path_isolation.resolveRequestedPath(allocator, io, options.out_dir);
    defer allocator.free(mutable_output);
    if (path_isolation.pathsOverlap(mutable_output, telemetry_resolved) or path_isolation.pathsOverlap(mutable_output, request_resolved)) {
        return error.BenchmarkPathsOverlapTrainingOutput;
    }
    for ([_][]const u8{
        options.base_model_dir,
        options.adapter_model_dir,
        options.train_prepared_inputs_path,
        options.eval_prepared_inputs_path,
    }) |immutable_input| {
        const input_resolved = try std.Io.Dir.cwd().realPathFileAlloc(io, immutable_input, allocator);
        defer allocator.free(input_resolved);
        if (path_isolation.pathsOverlap(input_resolved, telemetry_resolved)) return error.BenchmarkTelemetryOverlapsImmutableInput;
    }
}

fn rejectExistingBenchmarkTelemetry(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    return error.BenchmarkTelemetryAlreadyExists;
}

fn readFileAtPathLimited(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    max_bytes: usize,
) ![]u8 {
    if (!std.fs.path.isAbsolute(path)) {
        return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_bytes));
    }
    const parent = std.fs.path.dirname(path) orelse return error.BadPathName;
    var dir = try std.Io.Dir.openDirAbsolute(io, parent, .{});
    defer dir.close(io);
    return dir.readFileAlloc(io, std.fs.path.basename(path), allocator, .limited(max_bytes));
}

fn sha256PrefixedAlloc(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(bytes);
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    return std.fmt.allocPrint(allocator, "sha256:{s}", .{&hex});
}

fn isLowerHex(value: []const u8, expected_len: usize) bool {
    if (value.len != expected_len) return false;
    for (value) |byte| if (!std.ascii.isDigit(byte) and (byte < 'a' or byte > 'f')) return false;
    return true;
}

fn isPrefixedSha256(value: []const u8) bool {
    return value.len == "sha256:".len + 64 and
        std.mem.startsWith(u8, value, "sha256:") and
        isLowerHex(value["sha256:".len..], 64);
}

fn matchesPrefixedSha256(prefixed: []const u8, raw_hex: []const u8) bool {
    return isPrefixedSha256(prefixed) and std.mem.eql(u8, prefixed["sha256:".len..], raw_hex);
}

const BenchmarkAdapterSemanticIdentity = struct {
    semantic_sha256: []const u8,
    target_inventory_sha256: []const u8,
    tensor_count: usize,
    target_count: usize,

    fn deinit(self: *BenchmarkAdapterSemanticIdentity, allocator: std.mem.Allocator) void {
        allocator.free(self.semantic_sha256);
        allocator.free(self.target_inventory_sha256);
        self.* = undefined;
    }
};

const BenchmarkSemanticTensor = struct {
    module: []u8,
    role: []const u8,
    shape: []const i64,
    values: []const u8,
    data_start: u64,
    data_end: u64,
};

fn inspectBenchmarkAdapterSemantics(
    allocator: std.mem.Allocator,
    io: std.Io,
    adapter_checkpoint_path: []const u8,
) !BenchmarkAdapterSemanticIdentity {
    const stat = try std.Io.Dir.cwd().statFile(io, adapter_checkpoint_path, .{ .follow_symlinks = false });
    if (stat.kind != .file) return error.BenchmarkAdapterCheckpointNotRegular;
    var reader = try safetensors.MMapReader.openFileAbsolute(allocator, adapter_checkpoint_path);
    defer reader.deinit();
    if (reader.header.tensors.count() < 2) return error.BenchmarkAdapterTensorInventoryEmpty;

    const tensors = try allocator.alloc(BenchmarkSemanticTensor, reader.header.tensors.count());
    defer allocator.free(tensors);
    var tensor_count: usize = 0;
    defer for (tensors[0..tensor_count]) |tensor| allocator.free(tensor.module);
    const data_offset = std.math.cast(usize, reader.data_offset) orelse return error.InvalidTensorLayout;
    const data_bytes = reader.file_bytes[data_offset..];
    var tensor_it = reader.header.tensors.iterator();
    while (tensor_it.next()) |entry| {
        const canonical = try canonicalizeBenchmarkAdapterTensorName(allocator, entry.key_ptr.*);
        errdefer allocator.free(canonical.module);
        const meta = entry.value_ptr.*;
        if (meta.dtype != .f32 or meta.shape.len != 2) return error.BenchmarkAdapterTensorMustBeF32Rank2;
        var element_count: usize = 1;
        for (meta.shape) |dimension| {
            const positive = std.math.cast(usize, dimension) orelse return error.InvalidTensorShape;
            if (positive == 0) return error.InvalidTensorShape;
            element_count = try std.math.mul(usize, element_count, positive);
        }
        const expected_bytes = try std.math.mul(usize, element_count, @sizeOf(f32));
        if (meta.data_end < meta.data_start or
            meta.data_end - meta.data_start != expected_bytes or
            meta.data_end > data_bytes.len)
        {
            return error.InvalidTensorLayout;
        }
        const values = data_bytes[@intCast(meta.data_start)..@intCast(meta.data_end)];
        var value_offset: usize = 0;
        while (value_offset < values.len) : (value_offset += @sizeOf(f32)) {
            const bits = std.mem.readInt(u32, values[value_offset..][0..@sizeOf(f32)], .little);
            const value: f32 = @bitCast(bits);
            if (!std.math.isFinite(value)) return error.BenchmarkAdapterTensorNonFinite;
        }
        tensors[tensor_count] = .{
            .module = canonical.module,
            .role = canonical.role,
            .shape = meta.shape,
            .values = values,
            .data_start = meta.data_start,
            .data_end = meta.data_end,
        };
        tensor_count += 1;
    }
    std.mem.sort(BenchmarkSemanticTensor, tensors[0..tensor_count], {}, struct {
        fn lessThan(_: void, lhs: BenchmarkSemanticTensor, rhs: BenchmarkSemanticTensor) bool {
            const module_order = std.mem.order(u8, lhs.module, rhs.module);
            return if (module_order == .eq) std.mem.lessThan(u8, lhs.role, rhs.role) else module_order == .lt;
        }
    }.lessThan);
    for (tensors[1..tensor_count], tensors[0 .. tensor_count - 1]) |current, previous| {
        if (std.mem.eql(u8, current.module, previous.module) and std.mem.eql(u8, current.role, previous.role)) {
            return error.BenchmarkDuplicateCanonicalAdapterTensor;
        }
    }
    const by_offset = try allocator.dupe(BenchmarkSemanticTensor, tensors[0..tensor_count]);
    defer allocator.free(by_offset);
    std.mem.sort(BenchmarkSemanticTensor, by_offset, {}, struct {
        fn lessThan(_: void, lhs: BenchmarkSemanticTensor, rhs: BenchmarkSemanticTensor) bool {
            return lhs.data_start < rhs.data_start or (lhs.data_start == rhs.data_start and lhs.data_end < rhs.data_end);
        }
    }.lessThan);
    var covered_end: u64 = 0;
    for (by_offset) |tensor| {
        if (tensor.data_start != covered_end) return error.InvalidTensorLayout;
        covered_end = tensor.data_end;
    }
    if (covered_end != data_bytes.len) return error.InvalidTensorLayout;

    var module_count: usize = 0;
    var prior_module: ?[]const u8 = null;
    for (tensors[0..tensor_count]) |tensor| {
        if (prior_module == null or !std.mem.eql(u8, prior_module.?, tensor.module)) {
            if (!std.mem.eql(u8, tensor.role, "lora_A")) return error.BenchmarkAdapterTargetMissingPair;
            module_count += 1;
            prior_module = tensor.module;
        } else if (!std.mem.eql(u8, tensor.role, "lora_B")) {
            return error.BenchmarkAdapterTargetMissingPair;
        }
    }
    if (tensor_count != try std.math.mul(usize, module_count, 2)) return error.BenchmarkAdapterTargetMissingPair;

    var semantic_hasher = std.crypto.hash.sha2.Sha256.init(.{});
    semantic_hasher.update(benchmark_initial_adapter_digest_domain_v1);
    semantic_hasher.update(&.{0});
    hashBenchmarkU64(&semantic_hasher, tensor_count);
    for (tensors[0..tensor_count]) |tensor| {
        hashBenchmarkBytes(&semantic_hasher, tensor.module);
        hashBenchmarkBytes(&semantic_hasher, tensor.role);
        hashBenchmarkBytes(&semantic_hasher, "float32");
        hashBenchmarkU64(&semantic_hasher, tensor.shape.len);
        for (tensor.shape) |dimension| hashBenchmarkU64(
            &semantic_hasher,
            @as(u64, @intCast(dimension)),
        );
        hashBenchmarkU64(&semantic_hasher, tensor.values.len);
        semantic_hasher.update(tensor.values);
    }
    const semantic_sha256 = try finishPrefixedSha256(allocator, &semantic_hasher);
    errdefer allocator.free(semantic_sha256);

    var inventory_hasher = std.crypto.hash.sha2.Sha256.init(.{});
    inventory_hasher.update(benchmark_target_inventory_digest_domain_v1);
    inventory_hasher.update(&.{0});
    hashBenchmarkU64(&inventory_hasher, module_count);
    prior_module = null;
    for (tensors[0..tensor_count]) |tensor| {
        if (prior_module != null and std.mem.eql(u8, prior_module.?, tensor.module)) continue;
        if (!std.mem.startsWith(u8, tensor.module, "model.")) return error.BenchmarkAdapterModuleNotCanonical;
        hashBenchmarkBytes(&inventory_hasher, tensor.module);
        prior_module = tensor.module;
    }
    const target_inventory_sha256 = try finishPrefixedSha256(allocator, &inventory_hasher);
    errdefer allocator.free(target_inventory_sha256);
    return .{
        .semantic_sha256 = semantic_sha256,
        .target_inventory_sha256 = target_inventory_sha256,
        .tensor_count = tensor_count,
        .target_count = module_count,
    };
}

const CanonicalBenchmarkAdapterTensorName = struct {
    module: []u8,
    role: []const u8,
};

fn canonicalizeBenchmarkAdapterTensorName(
    allocator: std.mem.Allocator,
    source_name: []const u8,
) !CanonicalBenchmarkAdapterTensorName {
    if (!std.mem.endsWith(u8, source_name, ".weight")) return error.UnsupportedBenchmarkAdapterTensorName;
    const body = source_name[0 .. source_name.len - ".weight".len];
    const marker_a = ".lora_A";
    const marker_b = ".lora_B";
    const marker_pos, const marker, const role = if (std.mem.lastIndexOf(u8, body, marker_a)) |position|
        .{ position, marker_a, "lora_A" }
    else if (std.mem.lastIndexOf(u8, body, marker_b)) |position|
        .{ position, marker_b, "lora_B" }
    else
        return error.UnsupportedBenchmarkAdapterTensorName;
    const role_suffix = body[marker_pos + marker.len ..];
    if (role_suffix.len != 0 and
        (role_suffix[0] != '.' or role_suffix.len == 1 or std.mem.indexOfScalar(u8, role_suffix[1..], '.') != null))
    {
        return error.UnsupportedBenchmarkAdapterTensorName;
    }
    var module = body[0..marker_pos];
    if (std.mem.endsWith(u8, module, ".weight")) module = module[0 .. module.len - ".weight".len];
    const prefixes = [_][]const u8{
        "base_model.model.model.language_model.",
        "base_model.model.language_model.",
        "base_model.model.",
        "model.language_model.",
        "language_model.",
    };
    var changed = true;
    while (changed) {
        changed = false;
        for (prefixes) |prefix| {
            if (std.mem.startsWith(u8, module, prefix)) {
                module = module[prefix.len..];
                changed = true;
                break;
            }
        }
    }
    var owned = if (std.mem.startsWith(u8, module, "layers.") or std.mem.startsWith(u8, module, "per_layer_input."))
        try std.fmt.allocPrint(allocator, "model.{s}", .{module})
    else
        try allocator.dupe(u8, module);
    errdefer allocator.free(owned);
    const aliases = [_]struct { legacy: []const u8, canonical: []const u8 }{
        .{ .legacy = "per_layer_model_projection", .canonical = "per_layer_input.per_layer_model_proj" },
        .{ .legacy = "per_layer_input_gate", .canonical = "per_layer_input.inp_gate" },
        .{ .legacy = "per_layer_projection", .canonical = "per_layer_input.proj" },
    };
    for (aliases) |alias| {
        const replaced = try std.mem.replaceOwned(u8, allocator, owned, alias.legacy, alias.canonical);
        allocator.free(owned);
        owned = replaced;
    }
    if (owned.len == 0 or
        std.mem.startsWith(u8, owned, ".") or
        std.mem.endsWith(u8, owned, ".") or
        std.mem.indexOf(u8, owned, "..") != null)
    {
        return error.UnsupportedBenchmarkAdapterTensorName;
    }
    return .{ .module = owned, .role = role };
}

fn hashBenchmarkU64(hasher: *std.crypto.hash.sha2.Sha256, value: anytype) void {
    var encoded: [8]u8 = undefined;
    std.mem.writeInt(u64, &encoded, @intCast(value), .little);
    hasher.update(&encoded);
}

fn hashBenchmarkBytes(hasher: *std.crypto.hash.sha2.Sha256, bytes: []const u8) void {
    hashBenchmarkU64(hasher, bytes.len);
    hasher.update(bytes);
}

fn finishPrefixedSha256(
    allocator: std.mem.Allocator,
    hasher: *std.crypto.hash.sha2.Sha256,
) ![]const u8 {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    return std.fmt.allocPrint(allocator, "sha256:{s}", .{&hex});
}

const BenchmarkWorkloadIdentity = struct {
    sha256: []const u8,
    input_tokens: usize,
    supervised_tokens: usize,

    fn deinit(self: *BenchmarkWorkloadIdentity, allocator: std.mem.Allocator) void {
        allocator.free(self.sha256);
        self.* = undefined;
    }
};

fn benchmarkWorkloadIdentity(
    allocator: std.mem.Allocator,
    example: *const finetune.PreparedExampleInput,
    sequence_length: usize,
    grad_accum: usize,
) !BenchmarkWorkloadIdentity {
    if (example.input_ids.len != sequence_length or example.labels.len != sequence_length) {
        return error.BenchmarkPreparedRowLengthMismatch;
    }
    var supervised_per_microstep: usize = 0;
    for (example.input_ids) |token_id| if (token_id < 0) return error.BenchmarkInputTokenOutOfRange;
    for (example.labels, 0..) |label, index| {
        if (label < 0 and label != -100) return error.BenchmarkInvalidPreparedLabel;
        if (index > 0 and label != -100) supervised_per_microstep += 1;
    }
    if (supervised_per_microstep == 0 or supervised_per_microstep != example.num_supervised_tokens) {
        return error.BenchmarkSupervisedTokenMismatch;
    }
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(benchmark_workload_digest_domain_v1);
    hasher.update(&.{0});
    hashBenchmarkU64(&hasher, grad_accum);
    for (0..grad_accum) |_| {
        hashBenchmarkU64(&hasher, sequence_length);
        for (example.input_ids) |token_id| hashBenchmarkI64(&hasher, token_id);
        for (example.labels) |label| hashBenchmarkI64(&hasher, label);
        for (0..sequence_length) |_| hashBenchmarkI64(&hasher, 1);
    }
    return .{
        .sha256 = try finishPrefixedSha256(allocator, &hasher),
        .input_tokens = try std.math.mul(usize, sequence_length, grad_accum),
        .supervised_tokens = try std.math.mul(usize, supervised_per_microstep, grad_accum),
    };
}

fn hashBenchmarkI64(hasher: *std.crypto.hash.sha2.Sha256, value: anytype) void {
    var encoded: [8]u8 = undefined;
    std.mem.writeInt(i64, &encoded, @intCast(value), .little);
    hasher.update(&encoded);
}

fn monotonicNowNs() u64 {
    var ts: std.posix.timespec = undefined;
    switch (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts))) {
        .SUCCESS => return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec),
        else => return 0,
    }
}

fn monotonicElapsedNs(started_ns: u64, finished_ns: u64) u64 {
    if (started_ns == 0 or finished_ns <= started_ns) return 0;
    return finished_ns - started_ns;
}

fn countPerSecond(count: anytype, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0;
    return @as(f64, @floatFromInt(count)) *
        @as(f64, @floatFromInt(std.time.ns_per_s)) /
        @as(f64, @floatFromInt(elapsed_ns));
}

fn summarizeEpochTiming(
    started_ns: u64,
    finished_ns: u64,
    metrics: gemma4_real.CausalLmMetrics,
) AutodiffEpochTimingSummary {
    const elapsed_ns = monotonicElapsedNs(started_ns, finished_ns);
    return .{
        .wall_time_ns = elapsed_ns,
        .examples_per_second = countPerSecond(metrics.examples_seen, elapsed_ns),
        .logical_input_tokens_per_second = countPerSecond(metrics.logical_input_tokens_seen, elapsed_ns),
        .scheduled_input_tokens_per_second = countPerSecond(metrics.scheduled_input_tokens_seen, elapsed_ns),
        .supervised_tokens_per_second = countPerSecond(metrics.supervised_tokens_seen, elapsed_ns),
        .optimizer_steps_per_second = countPerSecond(metrics.optimizer_steps, elapsed_ns),
    };
}

fn summarizeEpochGraphCacheDelta(
    before: real_autodiff.RealAutodiffTrainer.GraphCacheStats,
    after: real_autodiff.RealAutodiffTrainer.GraphCacheStats,
) AutodiffEpochGraphCacheDelta {
    const builds = after.builds -| before.builds;
    const hits = after.hits -| before.hits;
    const active_reuses = after.active_reuses -| before.active_reuses;
    const evictions = after.evictions -| before.evictions;
    return .{
        .builds = builds,
        .hits = hits,
        .active_reuses = active_reuses,
        .evictions = evictions,
        .resident_signatures = after.resident_signatures,
        .peak_resident_signatures = after.peak_resident_signatures,
        .reuse_only = builds == 0 and evictions == 0 and (hits != 0 or active_reuses != 0),
    };
}

fn validateBenchmarkArtifactBindings(
    allocator: std.mem.Allocator,
    io: std.Io,
    admission: *const BenchmarkAdmission,
    adapter_inspect: finetune.InspectionSummary,
    validated_adapter: ValidatedAdapterConfig,
    prepared_inputs_path: []const u8,
    eval_prepared_inputs_path: []const u8,
    prepared: finetune.PreparedInputsSummary,
) !void {
    const request = admission.request();
    if (!std.mem.eql(u8, prepared.schema_version, finetune.prepared_schema_v6) or
        prepared.max_seq_len != request.bindings.sequence_length or
        request.bindings.prepared_example_index >= prepared.examples.len)
    {
        return error.BenchmarkPreparedArtifactMismatch;
    }
    var train_fingerprint = try finetune.fingerprintProjectorFile(allocator, prepared_inputs_path);
    defer finetune.freeProjectorFingerprint(allocator, &train_fingerprint);
    if (!matchesPrefixedSha256(request.bindings.train_prepared_sha256, train_fingerprint.sha256)) {
        return error.BenchmarkPreparedArtifactMismatch;
    }
    var eval_fingerprint = try finetune.fingerprintProjectorFile(allocator, eval_prepared_inputs_path);
    defer finetune.freeProjectorFingerprint(allocator, &eval_fingerprint);
    if (!matchesPrefixedSha256(request.bindings.eval_prepared_sha256, eval_fingerprint.sha256)) {
        return error.BenchmarkEvaluationArtifactMismatch;
    }
    var workload = try benchmarkWorkloadIdentity(
        allocator,
        &prepared.examples[request.bindings.prepared_example_index],
        request.bindings.sequence_length,
        request.bindings.grad_accum,
    );
    defer workload.deinit(allocator);
    if (!std.mem.eql(u8, workload.sha256, request.bindings.workload_sha256) or
        workload.supervised_tokens != request.bindings.supervised_tokens)
    {
        return error.BenchmarkWorkloadBindingMismatch;
    }
    const checkpoint_path = adapter_inspect.adapter_checkpoint_path orelse return error.MissingAdapterCheckpoint;
    var semantic_identity = try inspectBenchmarkAdapterSemantics(allocator, io, checkpoint_path);
    defer semantic_identity.deinit(allocator);
    if (!std.mem.eql(u8, semantic_identity.semantic_sha256, request.bindings.initial_adapter_semantic_sha256) or
        !std.mem.eql(u8, semantic_identity.target_inventory_sha256, request.bindings.target_inventory_sha256) or
        semantic_identity.target_count != request.bindings.target_count or
        semantic_identity.tensor_count != try std.math.mul(usize, request.bindings.target_count, 2))
    {
        return error.BenchmarkInitialAdapterSemanticMismatch;
    }
    if (validated_adapter.rank != request.bindings.rank or
        @as(f64, @floatCast(validated_adapter.alpha)) != request.bindings.alpha or
        adapter_inspect.target_module_count != request.bindings.target_count)
    {
        return error.BenchmarkAdapterConfigMismatch;
    }
    const target_preset = adapter_inspect.target_preset orelse return error.MissingAdapterTargetPreset;
    if (!std.mem.eql(u8, target_preset, request.bindings.target_preset)) {
        return error.BenchmarkAdapterConfigMismatch;
    }
}

fn validateOracleArtifactBindings(
    allocator: std.mem.Allocator,
    admission: *const OracleAdmission,
    adapter_inspect: finetune.InspectionSummary,
    validated_adapter: ValidatedAdapterConfig,
    prepared_inputs_path: []const u8,
    prepared: finetune.PreparedInputsSummary,
    provenance: finetune.ModelProvenance,
) !void {
    const request = admission.request();
    if (!std.mem.eql(u8, prepared.schema_version, finetune.prepared_schema_v6) or
        request.bindings.example_index >= prepared.examples.len or
        !std.mem.eql(u8, request.bindings.source_dataset_sha256, prepared.source_dataset_sha256 orelse return error.PreparedInputsProvenanceRequired))
    {
        return error.OraclePreparedArtifactMismatch;
    }
    const example = prepared.examples[request.bindings.example_index];
    if (example.num_input_tokens == 0 or example.num_supervised_tokens == 0 or
        example.teacher_top_k != 0 or example.teacher_top_k_token_ids.len != 0 or
        example.teacher_top_k_probs.len != 0 or example.image_paths.len != 0 or
        example.audio_paths.len != 0)
    {
        return error.OraclePreparedExampleNotAdmitted;
    }
    var prepared_fingerprint = try finetune.fingerprintProjectorFile(allocator, prepared_inputs_path);
    defer finetune.freeProjectorFingerprint(allocator, &prepared_fingerprint);
    if (!matchesPrefixedSha256(request.bindings.train_prepared_sha256, prepared_fingerprint.sha256)) {
        return error.OraclePreparedArtifactMismatch;
    }
    if (!std.mem.eql(u8, request.bindings.base_model_sha256, provenance.base_model_sha256)) {
        return error.OracleModelArtifactMismatch;
    }
    const checkpoint_path = adapter_inspect.adapter_checkpoint_path orelse return error.MissingAdapterCheckpoint;
    var adapter_fingerprint = try finetune.fingerprintProjectorFile(allocator, checkpoint_path);
    defer finetune.freeProjectorFingerprint(allocator, &adapter_fingerprint);
    if (!matchesPrefixedSha256(request.bindings.initial_adapter_sha256, adapter_fingerprint.sha256)) {
        return error.OracleInitialAdapterMismatch;
    }
    if (validated_adapter.rank != request.bindings.rank or
        @as(f64, @floatCast(validated_adapter.alpha)) != request.bindings.alpha or
        adapter_inspect.target_module_count != request.bindings.target_count)
    {
        return error.OracleAdapterConfigMismatch;
    }
    const target_preset = adapter_inspect.target_preset orelse return error.MissingAdapterTargetPreset;
    if (!std.mem.eql(u8, target_preset, request.bindings.target_preset)) {
        return error.OracleAdapterConfigMismatch;
    }
}

pub fn runEvalFromArgs(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
    if (argv.len == 1 and isHelpArg(argv[0])) {
        printEvalUsage();
        return;
    }
    var model: ?[]const u8 = null;
    var adapter: ?[]const u8 = null;
    var prepared: ?[]const u8 = null;
    var out: ?[]const u8 = null;
    var backend: ?BackendKind = null;
    var max_examples: usize = 0;
    var max_grad_norm: f32 = 1.0;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= argv.len or model != null) return evalUsageError();
            model = argv[i];
        } else if (std.mem.eql(u8, arg, "--adapter")) {
            i += 1;
            if (i >= argv.len or adapter != null) return evalUsageError();
            adapter = argv[i];
        } else if (std.mem.eql(u8, arg, "--prepared")) {
            i += 1;
            if (i >= argv.len or prepared != null) return evalUsageError();
            prepared = argv[i];
        } else if (std.mem.eql(u8, arg, "--out")) {
            i += 1;
            if (i >= argv.len or out != null) return evalUsageError();
            out = argv[i];
        } else if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i >= argv.len or backend != null) return evalUsageError();
            backend = parseBackend(argv[i]) orelse return evalUsageError();
        } else if (std.mem.eql(u8, arg, "--max-examples")) {
            i += 1;
            if (i >= argv.len) return evalUsageError();
            max_examples = try std.fmt.parseUnsigned(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--max-grad-norm")) {
            i += 1;
            if (i >= argv.len) return evalUsageError();
            max_grad_norm = try std.fmt.parseFloat(f32, argv[i]);
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return evalUsageError();
        } else if (model == null) {
            model = arg;
        } else if (adapter == null) {
            adapter = arg;
        } else if (prepared == null) {
            prepared = arg;
        } else if (out == null) {
            out = arg;
        } else {
            return evalUsageError();
        }
    }
    try evaluate(allocator, io, .{
        .base_model_dir = model orelse return evalUsageError(),
        .adapter_model_dir = adapter orelse return evalUsageError(),
        .prepared_inputs_path = prepared orelse return evalUsageError(),
        .out_path = out orelse return evalUsageError(),
        .backend_kind = backend orelse return error.MissingBackend,
        .max_examples = max_examples,
        .max_grad_norm = max_grad_norm,
    });
}

pub fn evaluate(allocator: std.mem.Allocator, io: std.Io, options: EvalOptions) !void {
    if (!std.math.isFinite(options.max_grad_norm) or options.max_grad_norm < 0) return error.InvalidMaxGradNorm;
    var graph_executor_scope = try acquireMetalGraphExecutorScope(options.backend_kind);
    defer if (graph_executor_scope) |*scope| scope.deinit();
    try validateAutodiffBaseArtifact(allocator, options.base_model_dir, options.backend_kind);

    var prepared = try finetune.loadPreparedInputsSummary(allocator, options.prepared_inputs_path);
    defer finetune.freePreparedInputsSummary(allocator, &prepared);
    if (countMultimodalExamples(prepared.examples) > 0) return error.Gemma4MultimodalFinetuningNotSupported;

    var bundle_inspection = try finetune.inspectLoRABundle(allocator, options.base_model_dir, options.adapter_model_dir);
    defer finetune.freeLoRABundleInspectionSummary(allocator, &bundle_inspection);
    try finetune.validateLoRABundleInspection(bundle_inspection);
    var adapter_inspect = try finetune.inspectCheckpoint(allocator, options.adapter_model_dir);
    defer finetune.freeInspectionSummary(allocator, &adapter_inspect);
    const validated_adapter = try validateAutodiffAdapterConfig(adapter_inspect);
    const graph_config = try gemma4_real.loadGraphConfig(allocator, options.base_model_dir);
    _ = try finetune.validatePreparedSequenceAdmission(prepared, graph_config.max_position_embeddings);
    try finetune.validatePreparedVocabularyAdmission(prepared, graph_config.vocab_size);
    try finetune.validatePreparedSourceDatasetProvenance(allocator, prepared);
    var provenance = try finetune.fingerprintGemma4Model(allocator, options.base_model_dir);
    defer provenance.deinit(allocator);
    try finetune.validatePreparedModelProvenance(prepared, provenance);
    try finetune.validateAdapterModelProvenance(adapter_inspect, provenance);
    const target_modules = adapter_inspect.target_modules orelse return error.MissingAdapterTargetInventory;
    const metrics = try evaluateAutodiff(
        allocator,
        options.base_model_dir,
        options.adapter_model_dir,
        prepared.examples,
        options.max_examples,
        graph_config,
        .{
            .rank = validated_adapter.rank,
            .alpha = validated_adapter.alpha,
            .target_patterns = target_modules,
            .strict_target_patterns = true,
            .sharing = if (adapter_inspect.recursive_lora_enabled) .by_use else .by_weight,
        },
        options.backend_kind,
        options.max_grad_norm,
        null,
        1,
        null,
        null,
    );
    if (metrics.examples_seen == 0 or metrics.supervised_tokens_seen == 0) return error.NoEvaluationData;
    if (!std.math.isFinite(metrics.average_loss)) return error.NonFiniteEvaluationLoss;

    const payload = .{
        .artifact_family_version = finetune.artifact_family_version,
        .task = "gemma4_lora_eval",
        .base_model_dir = options.base_model_dir,
        .adapter_model_dir = options.adapter_model_dir,
        .base_model_sha256 = provenance.base_model_sha256,
        .tokenizer_sha256 = provenance.tokenizer_sha256,
        .chat_template_sha256 = provenance.chat_template_sha256,
        .prepared_inputs_path = options.prepared_inputs_path,
        .prepared_schema_version = prepared.schema_version,
        .prepared_examples_sha256 = prepared.prepared_examples_sha256,
        .source_dataset_path = prepared.source_dataset_path,
        .source_dataset_sha256 = prepared.source_dataset_sha256,
        .source_split = prepared.source_split,
        .source_revision = prepared.source_revision,
        .backend_kind = options.backend_kind,
        .max_examples = options.max_examples,
        .metrics = metrics,
    };
    const rendered = try std.json.Stringify.valueAlloc(allocator, payload, .{ .whitespace = .indent_2 });
    defer allocator.free(rendered);
    try artifact_publication.writeFileImmutable(allocator, io, options.out_path, rendered);

    const stdout = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var writer = stdout.writer(io, &buf);
    try std.json.Stringify.value(payload, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}

pub fn runValidateAdapterFromArgs(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
    if (argv.len == 1 and isHelpArg(argv[0])) {
        printValidateAdapterUsage();
        return;
    }
    var model: ?[]const u8 = null;
    var adapter: ?[]const u8 = null;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= argv.len or model != null) return validateAdapterUsageError();
            model = argv[i];
        } else if (std.mem.eql(u8, arg, "--adapter")) {
            i += 1;
            if (i >= argv.len or adapter != null) return validateAdapterUsageError();
            adapter = argv[i];
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return validateAdapterUsageError();
        } else if (model == null) {
            model = arg;
        } else if (adapter == null) {
            adapter = arg;
        } else {
            return validateAdapterUsageError();
        }
    }
    const result = try validateAdapter(allocator, .{
        .base_model_dir = model orelse return validateAdapterUsageError(),
        .adapter_model_dir = adapter orelse return validateAdapterUsageError(),
    });
    const stdout = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var writer = stdout.writer(io, &buf);
    try std.json.Stringify.value(result, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}

pub fn validateAdapter(allocator: std.mem.Allocator, options: AdapterValidateOptions) !AdapterValidationResult {
    var bundle_inspection = try finetune.inspectLoRABundle(allocator, options.base_model_dir, options.adapter_model_dir);
    defer finetune.freeLoRABundleInspectionSummary(allocator, &bundle_inspection);
    try finetune.validateLoRABundleInspection(bundle_inspection);
    var inspected = try finetune.inspectCheckpoint(allocator, options.adapter_model_dir);
    defer finetune.freeInspectionSummary(allocator, &inspected);
    const config = try validateAutodiffAdapterConfig(inspected);
    var provenance = try finetune.fingerprintGemma4Model(allocator, options.base_model_dir);
    defer provenance.deinit(allocator);
    try finetune.validateAdapterModelProvenance(inspected, provenance);
    return .{
        .target_module_count = inspected.target_module_count,
        .rank = config.rank,
        .alpha = config.alpha,
        .recursive_lora = inspected.recursive_lora_enabled,
    };
}

fn parseBackend(value: []const u8) ?BackendKind {
    if (std.mem.eql(u8, value, "native")) return .native;
    if (std.mem.eql(u8, value, "metal")) return .metal;
    return null;
}

fn resolveTrainerMode(requested: TrainerMode) TrainerMode {
    return switch (requested) {
        .surrogate, .autodiff => requested,
        // `auto` is retained as a compatibility spelling, but production
        // training must never turn a missing real model into a successful
        // surrogate run.
        .auto => .autodiff,
    };
}

fn validateAutodiffTrainingOptions(opts: CliOptions) !void {
    if (!std.math.isFinite(opts.learning_rate) or opts.learning_rate <= 0) return error.InvalidLearningRate;
    if (opts.epochs == 0) return error.InvalidEpochCount;
    if (opts.grad_accum_steps == 0) return error.InvalidGradientAccumulation;
    if (!std.math.isFinite(opts.max_grad_norm) or opts.max_grad_norm < 0) return error.InvalidMaxGradNorm;
    try validateSequenceLengthPolicy(
        opts.sequence_length_bucket_quantum,
        opts.sequence_length_bucket_min,
        opts.graph_cache_capacity,
    );
    if (opts.layer_name != null) return error.LayerScopedAutodiffNotYetSupported;
    if (!std.math.approxEqAbs(f32, opts.llrd_decay, 1.0, 1e-6)) return error.LayerWiseDecayNotYetSupportedForAutodiff;
    if (opts.use_schedule_free) return error.ScheduleFreeNotYetSupportedForAutodiff;
    try validateCheckpointOptions(opts.checkpoint_path, opts.checkpoint_every_epochs, opts.resume_from_checkpoint, opts.epochs);
}

fn validateCheckpointOptions(
    checkpoint_path: ?[]const u8,
    checkpoint_every_epochs: usize,
    resume_from_checkpoint: bool,
    epochs: usize,
) !void {
    if (checkpoint_path) |path| {
        if (path.len == 0) return error.InvalidCheckpointPath;
        if (!resume_from_checkpoint and checkpoint_every_epochs == 0) return error.CheckpointIntervalRequired;
    } else if (resume_from_checkpoint or checkpoint_every_epochs != 0) {
        return error.CheckpointPathRequired;
    }
    if (checkpoint_every_epochs > epochs) return error.CheckpointIntervalExceedsEpochCount;
}

fn validateCheckpointPathIsolation(
    allocator: std.mem.Allocator,
    io: std.Io,
    checkpoint_path: []const u8,
    out_dir: []const u8,
    immutable_input_paths: []const []const u8,
) !void {
    const checkpoint_resolved = try path_isolation.resolveRequestedPath(allocator, io, checkpoint_path);
    defer allocator.free(checkpoint_resolved);
    const output_resolved = try path_isolation.resolveRequestedPath(allocator, io, out_dir);
    defer allocator.free(output_resolved);
    if (path_isolation.pathsOverlap(output_resolved, checkpoint_resolved)) {
        return error.CheckpointPathOverlapsOutput;
    }
    for (immutable_input_paths) |input_path| {
        const input_resolved = try std.Io.Dir.cwd().realPathFileAlloc(io, input_path, allocator);
        defer allocator.free(input_resolved);
        if (path_isolation.pathsOverlap(input_resolved, checkpoint_resolved)) {
            return error.CheckpointPathOverlapsImmutableInput;
        }
    }
}

pub fn acquireMetalGraphExecutorScope(backend_kind: gemma4_real.BackendKind) !?training_executor_policy.ProductEnableScope {
    if (backend_kind != .metal) return null;
    try validateMetalGraphExecutorFlags(
        platform.env.getenvBoolDefault("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", false),
        platform.env.getenv("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_NODE_IDS") != null,
        platform.env.getenvBoolDefault("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK", false),
        platform.env.getenvBoolDefault("TERMITE_DEBUG_DEVICE_GRAD_NORM", false),
    );
    return training_executor_policy.ProductEnableScope.acquire();
}

fn validateMetalGraphExecutorFlags(
    disabled: bool,
    parity_node_ids: bool,
    parity_check: bool,
    debug_device_grad_norm: bool,
) !void {
    if (disabled) return error.Gemma4MetalTrainingGraphExecutorDisabled;
    if (parity_node_ids) return error.Gemma4MetalTrainingParityDiagnosticForbidden;
    if (parity_check) return error.Gemma4MetalTrainingParityCheckForbidden;
    if (debug_device_grad_norm) return error.Gemma4MetalDebugGradientReadbackForbidden;
}

const ValidatedAdapterConfig = struct {
    rank: u32,
    alpha: f32,
};

fn validateAutodiffAdapterConfig(inspected: finetune.InspectionSummary) !ValidatedAdapterConfig {
    const peft_type = inspected.peft_type orelse return error.MissingAdapterPeftType;
    if (!std.ascii.eqlIgnoreCase(peft_type, "LORA")) return error.UnsupportedAdapterPeftType;
    const task_type = inspected.task_type orelse return error.MissingAdapterTaskType;
    if (!std.ascii.eqlIgnoreCase(task_type, "CAUSAL_LM")) return error.UnsupportedAdapterTaskType;
    const inference_mode = inspected.inference_mode orelse return error.MissingAdapterInferenceMode;
    if (inference_mode) return error.InferenceOnlyAdapterNotTrainable;
    if (inspected.use_dora orelse false) return error.DoRAAutodiffNotYetSupported;
    if (inspected.use_rslora orelse false) return error.RSLoRAAutodiffNotYetSupported;
    if ((inspected.lora_dropout orelse 0) != 0) return error.LoRADropoutAutodiffNotYetSupported;
    if (inspected.bias) |bias| {
        if (!std.ascii.eqlIgnoreCase(bias, "none")) return error.LoRABiasAutodiffNotYetSupported;
    }
    if (inspected.fan_in_fan_out orelse false) return error.LoRAFanInFanOutAutodiffNotYetSupported;
    if (inspected.modules_to_save_count != 0) return error.LoRAModulesToSaveAutodiffNotYetSupported;
    try finetune.validateLoRAInitializerBaseCompatibility(inspected.init_lora_weights);

    const rank = inspected.lora_rank orelse return error.MissingAdapterConfig;
    if (rank == 0 or rank > std.math.maxInt(u32)) return error.InvalidLoRARank;

    const alpha = inspected.lora_alpha orelse return error.MissingAdapterConfig;
    if (!std.math.isFinite(alpha) or alpha <= 0 or alpha > std.math.floatMax(f32)) return error.InvalidLoRAAlpha;
    return .{ .rank = @intCast(rank), .alpha = @floatCast(alpha) };
}

pub fn validateAutodiffBaseArtifact(
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    backend_kind: gemma4_real.BackendKind,
) !void {
    const experimental_gguf = platform.env.getenvBoolDefault("ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA", false);
    return validateAutodiffBaseArtifactWithAdmission(
        allocator,
        base_model_dir,
        backend_kind,
        experimental_gguf,
    );
}

/// Recipe entry points use a typed admission bit rather than ambient process
/// state so direct-GGUF preference runs are reproducible and fingerprinted.
pub fn validateAutodiffBaseArtifactForRecipe(
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    backend_kind: gemma4_real.BackendKind,
    allow_direct_gguf_training: bool,
) !void {
    return validateAutodiffBaseArtifactWithAdmission(
        allocator,
        base_model_dir,
        backend_kind,
        allow_direct_gguf_training,
    );
}

fn validateAutodiffBaseArtifactWithAdmission(
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    backend_kind: gemma4_real.BackendKind,
    allow_direct_gguf_training: bool,
) !void {
    var manifest = try manifest_mod.loadFromDir(allocator, base_model_dir);
    defer manifest.deinit();

    const artifact_kind = manifest.nativeWeightArtifactKind() orelse return;
    try validateAutodiffBaseCapabilities(artifact_kind == .gguf, backend_kind, allow_direct_gguf_training);
    if (artifact_kind == .gguf) {
        try validateGgufAutodiffInventory(allocator, manifest.gguf_path.?);
        return;
    }
    if (backend_kind != .metal) return;

    switch (artifact_kind) {
        .gguf => unreachable,
        .safetensors => try validateMetalStoredWeightBackward(
            allocator,
            manifest.safetensors_path,
            null,
        ),
        .sharded_safetensors => try validateMetalStoredWeightBackward(
            allocator,
            null,
            manifest.safetensors_index_path,
        ),
    }
}

/// Returns whether the artifact selected by the native model manifest is GGUF.
/// A directory that also contains a deployment GGUF still returns false when
/// its selected training weights are safetensors.
pub fn autodiffBaseUsesGguf(allocator: std.mem.Allocator, base_model_dir: []const u8) !bool {
    var manifest = try manifest_mod.loadFromDir(allocator, base_model_dir);
    defer manifest.deinit();
    return manifest.nativeWeightArtifactKind() == .gguf;
}

fn validateMetalStoredWeightBackward(
    allocator: std.mem.Allocator,
    single_path: ?[]const u8,
    index_path: ?[]const u8,
) !void {
    var dependencies = try safetensors.inspectArtifactDependencies(allocator, single_path, index_path);
    defer dependencies.deinit();

    const first_tensor_path: usize = if (single_path != null) 0 else 1;
    for (dependencies.paths[first_tensor_path..]) |path| {
        var reader = try safetensors.MMapReader.openFileAbsolute(allocator, path);
        defer reader.deinit();
        var tensor_it = reader.header.tensors.iterator();
        while (tensor_it.next()) |entry| {
            const meta = entry.value_ptr.*;
            // Every rank-2 frozen weight must have an exact device-resident
            // backward-input representation. F16 and BF16 retain their native
            // 16-bit bytes; F32 uses the ordinary dense Metal path.
            if (meta.shape.len == 2 and switch (meta.dtype) {
                .f32, .f16, .bf16 => false,
                else => true,
            }) {
                return error.MetalStoredWeightBackwardDTypeNotYetSupported;
            }
        }
    }
}

fn validateAutodiffBaseCapabilities(
    has_gguf_weights: bool,
    backend_kind: gemma4_real.BackendKind,
    experimental_gguf: bool,
) !void {
    // Q4_0, Q4_K, and Q6_K now have device-only frozen-linear dX kernels, but
    // direct GGUF admission stays closed until a complete model step proves
    // format coverage, strict no-host telemetry, bounded memory, optimizer
    // mutation, and checkpoint/artifact behavior. Never substitute host
    // dequantization for those missing end-to-end QLoRA gates.
    if (!has_gguf_weights) return;
    if (backend_kind != .metal) return error.GgufAutodiffRequiresMetal;
    if (!experimental_gguf) return error.GgufAutodiffBackwardNotYetSupported;
}

fn validateGgufAutodiffInventory(allocator: std.mem.Allocator, gguf_path: []const u8) !void {
    var region = try c_file.MmapRegion.init(allocator, gguf_path);
    defer region.deinit();
    var file = try gguf.format.parseStructure(allocator, region.data);
    defer file.deinit(allocator);
    try gguf.format.validateTensorDataRanges(&file, region.data.len);
    if (file.tensors.len == 0) return error.EmptyGgufTensorInventory;

    var saw_packed_linear = false;
    for (file.tensors) |tensor| {
        if (gguf.tensor_types.byteLen(tensor.tensor_type, tensor.dimensions) == null) {
            return error.UnsupportedGgufQloraTensorLayout;
        }
        switch (tensor.tensor_type) {
            .known => |known| switch (known) {
                .F32 => {
                    // The current exact F32 GGUF route is for scalar/vector
                    // metadata weights. Frozen rank-2 dX must retain F16/BF16
                    // or one of the qualified packed formats below.
                    if (tensor.dimensions.len >= 2) return error.UnsupportedGgufQloraRank2F32;
                },
                .F16, .BF16 => {},
                .Q4_0, .Q4_K, .Q6_K => saw_packed_linear = true,
                else => return error.UnsupportedGgufQloraTensorType,
            },
            else => return error.UnsupportedGgufQloraTensorType,
        }
    }
    if (!saw_packed_linear) return error.GgufQloraRequiresPackedLinearWeights;
}

test "gemma4 auto trainer is fail-closed real autodiff" {
    try std.testing.expectEqual(TrainerMode.autodiff, resolveTrainerMode(.auto));
    try std.testing.expectEqual(TrainerMode.autodiff, resolveTrainerMode(.autodiff));
    try std.testing.expectEqual(TrainerMode.surrogate, resolveTrainerMode(.surrogate));
}

test "gemma4 autodiff CLI requires an explicit backend before opening artifacts" {
    try std.testing.expectError(
        error.MissingBackend,
        runFromArgs(std.testing.allocator, std.testing.io, &.{ "missing-base", "missing-adapter", "missing-prepared", "missing-out" }),
    );
}

test "gemma4 train CLI canonical flags and legacy positional form share one typed contract" {
    const named = try parseTrainArgs(&.{
        "--model", "base", "--adapter", "adapter", "--train-prepared", "train.json", "--eval-prepared", "eval.json", "--out", "out", "--backend", "metal", "--lr", "0.0003", "--max-examples", "17", "--epochs", "4", "--sequence-length-bucket-quantum", "16", "--sequence-length-bucket-min", "32", "--graph-cache-capacity", "4", "--seed", "991", "--checkpoint-path", "state.safetensors", "--checkpoint-every-epochs", "2", "--resume",
    });
    try std.testing.expectEqualStrings("base", named.base_model_dir);
    try std.testing.expectEqualStrings("adapter", named.adapter_model_dir);
    try std.testing.expectEqualStrings("train.json", named.train_prepared_inputs_path);
    try std.testing.expectEqualStrings("eval.json", named.eval_prepared_inputs_path);
    try std.testing.expectEqualStrings("out", named.out_dir);
    try std.testing.expectEqual(BackendKind.metal, named.backend_kind);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0003), named.learning_rate, 1e-8);
    try std.testing.expectEqual(@as(usize, 17), named.max_examples);
    try std.testing.expectEqual(@as(u32, 16), named.sequence_length_bucket_quantum);
    try std.testing.expectEqual(@as(u32, 32), named.sequence_length_bucket_min);
    try std.testing.expectEqual(@as(u8, 4), named.graph_cache_capacity);
    try std.testing.expectEqual(@as(u64, 991), named.seed);
    try std.testing.expectEqualStrings("state.safetensors", named.checkpoint_path.?);
    try std.testing.expectEqual(@as(usize, 2), named.checkpoint_every_epochs);
    try std.testing.expect(named.resume_from_checkpoint);

    const legacy = try parseTrainArgs(&.{ "base", "adapter", "train.json", "out", "--eval-prepared", "eval.json", "--backend", "native" });
    try std.testing.expectEqualStrings(named.base_model_dir, legacy.base_model_dir);
    try std.testing.expectEqualStrings(named.adapter_model_dir, legacy.adapter_model_dir);
    try std.testing.expectEqualStrings(named.train_prepared_inputs_path, legacy.train_prepared_inputs_path);
    try std.testing.expectEqualStrings(named.eval_prepared_inputs_path, legacy.eval_prepared_inputs_path);
    try std.testing.expectEqual(BackendKind.native, legacy.backend_kind);
}

test "gemma4 checkpoint parser fails closed on incomplete contracts" {
    const prefix = [_][]const u8{ "base", "adapter", "train.json", "out", "--eval-prepared", "eval.json", "--backend", "native" };
    try std.testing.expectError(error.CheckpointPathRequired, parseTrainArgs(&(prefix ++ .{"--resume"})));
    try std.testing.expectError(
        error.CheckpointPathRequired,
        parseTrainArgs(&(prefix ++ .{ "--checkpoint-every-epochs", "1" })),
    );
    try std.testing.expectError(
        error.CheckpointIntervalRequired,
        parseTrainArgs(&(prefix ++ .{ "--checkpoint-path", "state.safetensors" })),
    );
    try std.testing.expectError(
        error.CheckpointIntervalExceedsEpochCount,
        parseTrainArgs(&(prefix ++ .{ "--epochs", "2", "--checkpoint-path", "state.safetensors", "--checkpoint-every-epochs", "3" })),
    );
}

test "gemma4 checkpoint path is isolated from immutable inputs and output" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const base_dir = try std.fs.path.join(allocator, &.{ root, "base" });
    defer allocator.free(base_dir);
    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    try cwd.createDirPath(io, base_dir);
    try cwd.createDirPath(io, adapter_dir);

    const train_prepared = try std.fs.path.join(allocator, &.{ root, "train-prepared.json" });
    defer allocator.free(train_prepared);
    const eval_prepared = try std.fs.path.join(allocator, &.{ root, "eval-prepared.json" });
    defer allocator.free(eval_prepared);
    const train_source = try std.fs.path.join(allocator, &.{ root, "train-source.jsonl" });
    defer allocator.free(train_source);
    const eval_source = try std.fs.path.join(allocator, &.{ root, "eval-source.jsonl" });
    defer allocator.free(eval_source);
    for ([_][]const u8{ train_prepared, eval_prepared, train_source, eval_source }) |path| {
        try cwd.writeFile(io, .{ .sub_path = path, .data = "fixture" });
    }
    const immutable_inputs = [_][]const u8{
        base_dir,
        adapter_dir,
        train_prepared,
        eval_prepared,
        train_source,
        eval_source,
    };
    const out_dir = try std.fs.path.join(allocator, &.{ root, "out" });
    defer allocator.free(out_dir);

    const base_collision = try std.fs.path.join(allocator, &.{ base_dir, "model.safetensors" });
    defer allocator.free(base_collision);
    try std.testing.expectError(
        error.CheckpointPathOverlapsImmutableInput,
        validateCheckpointPathIsolation(allocator, io, base_collision, out_dir, &immutable_inputs),
    );
    try std.testing.expectError(
        error.CheckpointPathOverlapsImmutableInput,
        validateCheckpointPathIsolation(allocator, io, train_source, out_dir, &immutable_inputs),
    );

    const output_collision = try std.fs.path.join(allocator, &.{ out_dir, "state.safetensors" });
    defer allocator.free(output_collision);
    try std.testing.expectError(
        error.CheckpointPathOverlapsOutput,
        validateCheckpointPathIsolation(allocator, io, output_collision, out_dir, &immutable_inputs),
    );

    const safe_checkpoint = try std.fs.path.join(allocator, &.{ root, "state.safetensors" });
    defer allocator.free(safe_checkpoint);
    try validateCheckpointPathIsolation(allocator, io, safe_checkpoint, out_dir, &immutable_inputs);
}

test "gemma4 public train parser rejects surrogate and multimodal modes before artifact IO" {
    const prefix = [_][]const u8{ "base", "adapter", "train.json", "out", "--eval-prepared", "eval.json", "--backend", "native" };
    try std.testing.expectError(
        error.Gemma4SurrogateTrainingNotSupported,
        parseTrainArgs(&(prefix ++ .{ "--trainer", "surrogate" })),
    );
    try std.testing.expectError(
        error.Gemma4MultimodalFinetuningNotSupported,
        parseTrainArgs(&(prefix ++ .{ "--gguf-projector", "projector.gguf" })),
    );
}

test "gemma4 Metal execution policy is strict for train and eval" {
    const metal = autodiffExecutionPolicy(.metal);
    try std.testing.expectEqual(real_autodiff.TrainingExecutionEngine.compiled_device, metal.engine);
    try std.testing.expect(metal.compiled_required);
    try std.testing.expect(metal.strict_metal_execution);

    const native = autodiffExecutionPolicy(.native);
    try std.testing.expectEqual(real_autodiff.TrainingExecutionEngine.interpreter, native.engine);
    try std.testing.expect(!native.compiled_required);
    try std.testing.expect(!native.strict_metal_execution);

    try std.testing.expectError(
        error.Gemma4MetalTrainingGraphExecutorDisabled,
        validateMetalGraphExecutorFlags(true, false, false, false),
    );
    try std.testing.expectError(
        error.Gemma4MetalTrainingParityDiagnosticForbidden,
        validateMetalGraphExecutorFlags(false, true, false, false),
    );
    try std.testing.expectError(
        error.Gemma4MetalTrainingParityCheckForbidden,
        validateMetalGraphExecutorFlags(false, false, true, false),
    );
    try std.testing.expectError(
        error.Gemma4MetalDebugGradientReadbackForbidden,
        validateMetalGraphExecutorFlags(false, false, false, true),
    );
    try validateMetalGraphExecutorFlags(false, false, false, false);
}

test "gemma4 strict Metal step evidence rejects every fallback surface" {
    const valid = real_autodiff.StepProfile{
        .optimizer_backend = .metal,
        .graph_executor_partitions = 1,
        .graph_executor_command_dispatches = 1,
        .graph_executor_device_outputs = 3,
    };
    try real_autodiff.validateStrictMetalStepEvidence(valid, 0, 2, 2, .train);
    try real_autodiff.validateStrictMetalStepEvidence(valid, 0, 0, 0, .eval);
    try std.testing.expectError(error.StrictMetalGradientNotDeviceResident, real_autodiff.validateStrictMetalStepEvidence(valid, 0, 1, 0, .eval));
    var evidence: gemma4_real.CausalLmMetrics = .{};
    gemma4_real.recordStepExecutionEvidence(&evidence, .{
        .loss = 1,
        .grad_norm = 1,
        .step = 1,
        .optimizer_stepped = true,
        .profile = valid,
    });
    try std.testing.expectEqual(@as(u64, 1), evidence.graph_executor_steps);
    try std.testing.expectEqual(@as(u64, 1), evidence.graph_executor_partitions);
    try std.testing.expectEqual(@as(u64, 1), evidence.graph_executor_command_dispatches);
    try std.testing.expectEqual(@as(u64, 1), evidence.metal_optimizer_steps);

    var profile = valid;
    profile.graph_executor_fallback_reason = "fallback";
    try std.testing.expectError(error.StrictMetalGraphExecutorFallback, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_partitions = 0;
    try std.testing.expectError(error.StrictMetalGraphExecutorDidNotRun, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_command_dispatches = 0;
    try std.testing.expectError(error.StrictMetalGraphExecutorDidNotDispatch, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_native_partitions = 1;
    try std.testing.expectError(error.StrictMetalNativePartition, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_unsupported_ops = 1;
    try std.testing.expectError(error.StrictMetalUnsupportedOperation, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_interpreter_fallbacks = 1;
    try std.testing.expectError(error.StrictMetalInterpreterFallback, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_runtime_region_dispatches = 1;
    profile.graph_executor_runtime_region_active_regions = 1;
    try real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train);
    profile.graph_executor_runtime_region_fallbacks = 1;
    try std.testing.expectError(error.StrictMetalRuntimeRegion, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_host_output_runtime_region = 1;
    try std.testing.expectError(error.StrictMetalRuntimeRegion, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_true_host_outputs = 1;
    try std.testing.expectError(error.StrictMetalHostOutput, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    try std.testing.expectError(error.StrictMetalGradientNotDeviceResident, real_autodiff.validateStrictMetalStepEvidence(valid, 1, 2, 2, .train));
    profile = valid;
    profile.optimizer_backend = .host;
    try real_autodiff.validateStrictMetalStepEvidence(profile, 0, 0, 0, .eval);
    try std.testing.expectError(error.StrictMetalOptimizerRequired, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
}

test "gemma4 autodiff training options reject zero-update configurations" {
    try validateAutodiffTrainingOptions(.{ .backend_kind = .native });
    try std.testing.expectEqual(
        @as(u32, 0),
        effectiveSequenceLengthBucketMinimum(.{ .backend_kind = .native }),
    );
    try std.testing.expectEqual(
        @as(u32, 16),
        effectiveSequenceLengthBucketMinimum(.{
            .backend_kind = .native,
            .sequence_length_bucket_quantum = 16,
        }),
    );
    try std.testing.expectEqual(
        @as(u32, 32),
        effectiveSequenceLengthBucketMinimum(.{
            .backend_kind = .native,
            .sequence_length_bucket_quantum = 16,
            .sequence_length_bucket_min = 32,
        }),
    );
    try std.testing.expectError(error.InvalidLearningRate, validateAutodiffTrainingOptions(.{ .backend_kind = .native, .learning_rate = 0 }));
    try std.testing.expectError(error.InvalidEpochCount, validateAutodiffTrainingOptions(.{ .backend_kind = .native, .epochs = 0 }));
    try std.testing.expectError(error.InvalidGradientAccumulation, validateAutodiffTrainingOptions(.{ .backend_kind = .native, .grad_accum_steps = 0 }));
    try std.testing.expectError(error.InvalidMaxGradNorm, validateAutodiffTrainingOptions(.{ .backend_kind = .native, .max_grad_norm = -1 }));
    try std.testing.expectError(
        error.SequenceLengthBucketQuantumRequired,
        validateAutodiffTrainingOptions(.{ .backend_kind = .native, .sequence_length_bucket_min = 32 }),
    );
    try std.testing.expectError(
        error.SequenceLengthBucketQuantumRequired,
        validateAutodiffTrainingOptions(.{ .backend_kind = .native, .graph_cache_capacity = 4 }),
    );
    try std.testing.expectError(
        error.GraphCacheCapacityTooLarge,
        validateAutodiffTrainingOptions(.{
            .backend_kind = .native,
            .sequence_length_bucket_quantum = 16,
            .graph_cache_capacity = 9,
        }),
    );
    try validateAutodiffTrainingOptions(.{
        .backend_kind = .native,
        .sequence_length_bucket_quantum = 16,
        .sequence_length_bucket_min = 32,
        .graph_cache_capacity = 4,
    });
}

test "gemma4 unsupported autodiff options fail before artifact IO" {
    const prefix = [_][]const u8{ "missing-base", "missing-adapter", "missing-prepared", "missing-out", "--eval-prepared", "missing-eval-prepared", "--backend", "native" };
    try std.testing.expectError(
        error.LayerScopedAutodiffNotYetSupported,
        runFromArgs(std.testing.allocator, std.testing.io, &(prefix ++ .{ "--layer-name", "model.layers.0" })),
    );
    try std.testing.expectError(
        error.LayerWiseDecayNotYetSupportedForAutodiff,
        runFromArgs(std.testing.allocator, std.testing.io, &(prefix ++ .{ "--llrd-decay", "0.9" })),
    );
    try std.testing.expectError(
        error.ScheduleFreeNotYetSupportedForAutodiff,
        runFromArgs(std.testing.allocator, std.testing.io, &(prefix ++ .{"--schedule-free"})),
    );
}

test "gemma4 autodiff adapter admission rejects unsupported and zero-update configs" {
    var inspected = finetune.InspectionSummary{
        .artifact_family_version = finetune.artifact_family_version,
        .variant = .adapter_only,
        .model_dir = "adapter",
        .peft_type = "LORA",
        .task_type = "CAUSAL_LM",
        .inference_mode = false,
        .lora_rank = 8,
        .lora_alpha = 16,
    };
    const valid = try validateAutodiffAdapterConfig(inspected);
    try std.testing.expectEqual(@as(u32, 8), valid.rank);
    try std.testing.expectEqual(@as(f32, 16), valid.alpha);

    inspected.peft_type = "IA3";
    try std.testing.expectError(error.UnsupportedAdapterPeftType, validateAutodiffAdapterConfig(inspected));
    inspected.peft_type = "LORA";
    inspected.task_type = "SEQ_CLS";
    try std.testing.expectError(error.UnsupportedAdapterTaskType, validateAutodiffAdapterConfig(inspected));
    inspected.task_type = "CAUSAL_LM";
    inspected.inference_mode = null;
    try std.testing.expectError(error.MissingAdapterInferenceMode, validateAutodiffAdapterConfig(inspected));
    inspected.inference_mode = true;
    try std.testing.expectError(error.InferenceOnlyAdapterNotTrainable, validateAutodiffAdapterConfig(inspected));
    inspected.inference_mode = false;

    inspected.use_dora = true;
    try std.testing.expectError(error.DoRAAutodiffNotYetSupported, validateAutodiffAdapterConfig(inspected));
    inspected.use_dora = false;

    inspected.use_rslora = true;
    try std.testing.expectError(error.RSLoRAAutodiffNotYetSupported, validateAutodiffAdapterConfig(inspected));
    inspected.use_rslora = false;
    inspected.lora_dropout = 0.05;
    try std.testing.expectError(error.LoRADropoutAutodiffNotYetSupported, validateAutodiffAdapterConfig(inspected));
    inspected.lora_dropout = 0;
    inspected.bias = "all";
    try std.testing.expectError(error.LoRABiasAutodiffNotYetSupported, validateAutodiffAdapterConfig(inspected));
    inspected.bias = "none";
    inspected.fan_in_fan_out = true;
    try std.testing.expectError(error.LoRAFanInFanOutAutodiffNotYetSupported, validateAutodiffAdapterConfig(inspected));
    inspected.fan_in_fan_out = false;
    inspected.modules_to_save_count = 1;
    try std.testing.expectError(error.LoRAModulesToSaveAutodiffNotYetSupported, validateAutodiffAdapterConfig(inspected));
    inspected.modules_to_save_count = 0;

    inspected.init_lora_weights = "pissa";
    try std.testing.expectError(error.LoRAInitializerRequiresAdjustedBase, validateAutodiffAdapterConfig(inspected));
    inspected.init_lora_weights = "loftq";
    try std.testing.expectError(error.LoRAInitializerRequiresAdjustedBase, validateAutodiffAdapterConfig(inspected));
    inspected.init_lora_weights = "eva";

    inspected.lora_rank = 0;
    try std.testing.expectError(error.InvalidLoRARank, validateAutodiffAdapterConfig(inspected));
    if (comptime @bitSizeOf(usize) > @bitSizeOf(u32)) {
        inspected.lora_rank = @as(usize, std.math.maxInt(u32)) + 1;
        try std.testing.expectError(error.InvalidLoRARank, validateAutodiffAdapterConfig(inspected));
    }
    inspected.lora_rank = 8;

    inspected.lora_alpha = 0;
    try std.testing.expectError(error.InvalidLoRAAlpha, validateAutodiffAdapterConfig(inspected));
    inspected.lora_alpha = std.math.nan(f64);
    try std.testing.expectError(error.InvalidLoRAAlpha, validateAutodiffAdapterConfig(inspected));
    inspected.lora_alpha = std.math.inf(f64);
    try std.testing.expectError(error.InvalidLoRAAlpha, validateAutodiffAdapterConfig(inspected));
    inspected.lora_alpha = @as(f64, std.math.floatMax(f32)) * 2;
    try std.testing.expectError(error.InvalidLoRAAlpha, validateAutodiffAdapterConfig(inspected));
}

test "gemma4 GGUF autodiff requires explicit typed Metal admission" {
    try std.testing.expectError(
        error.GgufAutodiffBackwardNotYetSupported,
        validateAutodiffBaseCapabilities(true, .metal, false),
    );
    try std.testing.expectError(
        error.GgufAutodiffRequiresMetal,
        validateAutodiffBaseCapabilities(true, .native, true),
    );
    try validateAutodiffBaseCapabilities(true, .metal, true);
    try validateAutodiffBaseCapabilities(false, .native, false);
}

test "gemma4 Metal autodiff admits native-dense BF16 stored weights from headers only" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const header =
        \\{"weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}
    ;
    var bytes: [8 + header.len + 8]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &bytes });

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);
    try validateAutodiffBaseArtifact(allocator, model_dir, .native);
    try validateAutodiffBaseArtifact(allocator, model_dir, .metal);
}

test "gemma4 Metal autodiff admits native-dense F16 stored weights from headers only" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const header =
        \\{"weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}}
    ;
    var bytes: [8 + header.len + 8]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &bytes });

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);
    try validateAutodiffBaseArtifact(allocator, model_dir, .metal);
}

test "gemma4 Metal autodiff admits mixed BF16 and F16 stored weights from headers only" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const header =
        \\{"bf16_weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"f16_weight":{"dtype":"F16","shape":[2,2],"data_offsets":[8,16]}}
    ;
    var bytes: [8 + header.len + 16]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &bytes });

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);
    try validateAutodiffBaseArtifact(allocator, model_dir, .metal);
}

test "gemma4 autodiff admits selected safetensors when a deployment GGUF is colocated" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);
    const checkpoint_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(checkpoint_path);
    const values = [_]f32{ 1, 0, 0, 1 };
    try safetensors_checkpoint.save(allocator, checkpoint_path, &.{.{
        .name = "weight",
        .shape = &.{ 2, 2 },
        .data = &values,
    }});
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "export.gguf", .data = "GGUFstub" });

    var manifest = try manifest_mod.loadFromDir(allocator, model_dir);
    defer manifest.deinit();

    try std.testing.expectEqual(
        manifest_mod.NativeWeightArtifactKind.safetensors,
        manifest.nativeWeightArtifactKind().?,
    );
    try std.testing.expect(manifest.gguf_path != null);
    try std.testing.expect(!(try autodiffBaseUsesGguf(allocator, model_dir)));
    try std.testing.expect(try autodiffBaseUsesGguf(allocator, manifest.gguf_path.?));
    try validateAutodiffBaseArtifact(allocator, model_dir, .native);
    try validateAutodiffBaseArtifact(allocator, model_dir, .metal);

    try std.testing.expectError(
        error.GgufAutodiffBackwardNotYetSupported,
        validateAutodiffBaseArtifact(allocator, manifest.gguf_path.?, .metal),
    );
}

test "gemma4 autodiff rejects a real DoRA bootstrap before backend and output mutation" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const checkpoint_path = try std.fs.path.join(allocator, &.{ root, finetune.checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    const prepared_path = try std.fs.path.join(allocator, &.{ root, "prepared.json" });
    defer allocator.free(prepared_path);
    const out_dir = try std.fs.path.join(allocator, &.{ root, "out" });
    defer allocator.free(out_dir);

    const base_values = [_]f32{ 1, 0, 0, 1 };
    try tmp.dir.writeFile(io, .{ .sub_path = finetune.hf_config_file_name, .data =
        \\{"model_type":"gemma4_text","max_position_embeddings":32}
    });
    try safetensors_checkpoint.save(allocator, checkpoint_path, &.{.{
        .name = "model.layers.0.self_attn.q_proj.weight",
        .shape = &.{ 2, 2 },
        .data = &base_values,
    }});
    var bootstrap = try finetune.bootstrapLoRABundle(allocator, root, adapter_dir, .{
        .rank = 1,
        .alpha = 1,
        .target_modules = &.{"q_proj"},
        .use_dora = true,
    });
    defer finetune.freeBootstrapSummary(allocator, &bootstrap);

    var prompt_ids = [_]i32{1};
    var response_ids = [_]i32{2};
    var input_ids = [_]i32{ 1, 2 };
    var labels = [_]i32{ -100, 2 };
    var examples = [_]finetune.PreparedExampleInput{.{
        .mode = .instruction,
        .prompt_input_ids = &prompt_ids,
        .response_input_ids = &response_ids,
        .num_prompt_tokens = 1,
        .num_response_tokens = 1,
        .input_ids = &input_ids,
        .labels = &labels,
        .num_input_tokens = 2,
        .num_supervised_tokens = 1,
    }};
    var provenance = try finetune.fingerprintGemma4Model(allocator, root);
    defer provenance.deinit(allocator);
    const examples_digest = try finetune.fingerprintPreparedExamplesAlloc(allocator, &examples);
    defer allocator.free(examples_digest);
    try finetune.savePreparedInputsSummary(allocator, prepared_path, .{
        .artifact_family_version = finetune.artifact_family_version,
        .model_dir = root,
        .schema_version = finetune.prepared_schema_v4,
        .base_model_sha256 = provenance.base_model_sha256,
        .tokenizer_sha256 = provenance.tokenizer_sha256,
        .chat_template_sha256 = provenance.chat_template_sha256,
        .prepared_examples_sha256 = examples_digest,
        .max_examples = 1,
        .examples_seen = 1,
        .max_seq_len = 2,
        .max_input_tokens = 2,
        .max_supervised_tokens = 1,
        .examples = &examples,
    });

    try tmp.dir.createDirPath(io, "out");
    try tmp.dir.writeFile(io, .{ .sub_path = "out/sentinel.txt", .data = "preserve" });
    try std.testing.expectError(
        error.DoRAAutodiffNotYetSupported,
        runFromArgs(allocator, io, &.{ root, adapter_dir, prepared_path, out_dir, "--backend", "native", "--eval-prepared", prepared_path }),
    );

    const sentinel = try tmp.dir.readFileAlloc(io, "out/sentinel.txt", allocator, .limited(16));
    defer allocator.free(sentinel);
    try std.testing.expectEqualStrings("preserve", sentinel);
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "out/adapter_model.safetensors", .{}));
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "out/training_report.json", .{}));
}

const AutodiffRunFingerprintInputs = struct {
    base_model_sha256: []const u8,
    tokenizer_sha256: []const u8,
    chat_template_sha256: []const u8,
    adapter_checkpoint_sha256: []const u8,
    train_prepared_sha256: []const u8,
    eval_prepared_sha256: []const u8,
    train_schema_version: []const u8,
    eval_schema_version: []const u8,
    train_source_sha256: []const u8,
    eval_source_sha256: []const u8,
    train_source_split: ?[]const u8,
    eval_source_split: ?[]const u8,
    train_source_revision: []const u8,
    eval_source_revision: []const u8,
    target_modules: []const []const u8,
    rank: u32,
    alpha: f32,
    recursive_lora: bool,
    recursive_source_num_layers: ?usize,
    recursive_shared_block_size: ?usize,
    recursive_loop_count: ?usize,
    learning_rate: f32,
    max_examples: usize,
    eval_max_examples: usize,
    epochs: usize,
    max_grad_norm: f32,
    grad_accum_steps: u32,
    activation_checkpoint_interval: u32,
    sequence_length_bucket_quantum: u32,
    sequence_length_bucket_min: u32,
    graph_cache_capacity: u8,
    seed: u64,
    backend_kind: BackendKind,
    linear_cross_entropy_mode: []const u8,
    projector_sha256: ?[]const u8,
};

fn autodiffRunFingerprint(inputs: AutodiffRunFingerprintInputs) [std.crypto.hash.sha2.Sha256.digest_length]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    // Preserve the established v2 identity for the unchanged fixed-shape
    // policy so compatible in-flight checkpoints remain resumable. Only the
    // opt-in bucketed policy extends the fingerprint domain.
    hashRunField(&hasher, "antfly.gemma4.lora.autodiff.run.v2");
    hashRunField(&hasher, inputs.base_model_sha256);
    hashRunField(&hasher, inputs.tokenizer_sha256);
    hashRunField(&hasher, inputs.chat_template_sha256);
    hashRunField(&hasher, inputs.adapter_checkpoint_sha256);
    hashRunField(&hasher, inputs.train_prepared_sha256);
    hashRunField(&hasher, inputs.eval_prepared_sha256);
    hashRunField(&hasher, inputs.train_schema_version);
    hashRunField(&hasher, inputs.eval_schema_version);
    hashRunField(&hasher, inputs.train_source_sha256);
    hashRunField(&hasher, inputs.eval_source_sha256);
    hashRunOptionalField(&hasher, inputs.train_source_split);
    hashRunOptionalField(&hasher, inputs.eval_source_split);
    hashRunField(&hasher, inputs.train_source_revision);
    hashRunField(&hasher, inputs.eval_source_revision);
    hashRunU64(&hasher, inputs.target_modules.len);
    for (inputs.target_modules) |module| hashRunField(&hasher, module);
    hashRunU64(&hasher, inputs.rank);
    hashRunU64(&hasher, @as(u32, @bitCast(inputs.alpha)));
    hashRunU64(&hasher, @intFromBool(inputs.recursive_lora));
    hashRunOptionalU64(&hasher, inputs.recursive_source_num_layers);
    hashRunOptionalU64(&hasher, inputs.recursive_shared_block_size);
    hashRunOptionalU64(&hasher, inputs.recursive_loop_count);
    hashRunU64(&hasher, @as(u32, @bitCast(inputs.learning_rate)));
    hashRunU64(&hasher, inputs.max_examples);
    hashRunU64(&hasher, inputs.eval_max_examples);
    hashRunU64(&hasher, inputs.epochs);
    hashRunU64(&hasher, @as(u32, @bitCast(inputs.max_grad_norm)));
    hashRunU64(&hasher, inputs.grad_accum_steps);
    hashRunU64(&hasher, inputs.activation_checkpoint_interval);
    if (inputs.sequence_length_bucket_quantum != 0 or
        inputs.sequence_length_bucket_min != 0 or
        inputs.graph_cache_capacity != 1)
    {
        hashRunField(&hasher, "independent-row-buckets-v1");
        hashRunU64(&hasher, inputs.sequence_length_bucket_quantum);
        hashRunU64(&hasher, inputs.sequence_length_bucket_min);
        hashRunU64(&hasher, inputs.graph_cache_capacity);
    }
    hashRunU64(&hasher, inputs.seed);
    hashRunField(&hasher, inputs.backend_kind.label());
    hashRunField(&hasher, inputs.linear_cross_entropy_mode);
    hashRunOptionalField(&hasher, inputs.projector_sha256);
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return digest;
}

fn hashRunField(hasher: *std.crypto.hash.sha2.Sha256, value: []const u8) void {
    hashRunU64(hasher, value.len);
    hasher.update(value);
}

fn hashRunOptionalField(hasher: *std.crypto.hash.sha2.Sha256, value: ?[]const u8) void {
    hashRunU64(hasher, @intFromBool(value != null));
    if (value) |present| hashRunField(hasher, present);
}

fn hashRunOptionalU64(hasher: *std.crypto.hash.sha2.Sha256, value: ?usize) void {
    hashRunU64(hasher, @intFromBool(value != null));
    if (value) |present| hashRunU64(hasher, present);
}

fn hashRunU64(hasher: *std.crypto.hash.sha2.Sha256, value: anytype) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, @intCast(value), .little);
    hasher.update(&bytes);
}

fn trainingProgressForEpoch(seed: u64, epoch_index: usize, examples_seen: u64) real_autodiff.TrainingProgress {
    var state = seed ^ (@as(u64, @intCast(epoch_index)) *% 0x9e3779b97f4a7c15);
    const order_seed = splitMix64(&state);
    return .{
        .epoch_index = @intCast(epoch_index),
        .examples_seen = examples_seen,
        .order_seed = order_seed,
        .rng_state = .{ splitMix64(&state), splitMix64(&state), splitMix64(&state), splitMix64(&state) },
    };
}

fn splitMix64(state: *u64) u64 {
    state.* +%= 0x9e3779b97f4a7c15;
    var value = state.*;
    value = (value ^ (value >> 30)) *% 0xbf58476d1ce4e5b9;
    value = (value ^ (value >> 27)) *% 0x94d049bb133111eb;
    return value ^ (value >> 31);
}

fn validateEpochBoundaryResume(
    restored: real_autodiff.RestoredTrainingCheckpoint,
    seed: u64,
    total_epochs: usize,
) !usize {
    if (restored.accumulation_micro_batches != 0 or
        restored.progress.next_example_index != 0 or
        restored.progress.order_cursor != 0)
    {
        return error.CheckpointNotAtEpochBoundary;
    }
    const epoch_index = std.math.cast(usize, restored.progress.epoch_index) orelse return error.InvalidTrainingCheckpointState;
    // Equality is a valid recovery point: a crash can happen after the final
    // checkpoint is durable but before the immutable output directory is
    // published. Reloading that state must still be able to export/report.
    if (epoch_index > total_epochs) return error.CheckpointBeyondRequestedEpochCount;
    const expected = trainingProgressForEpoch(seed, epoch_index, restored.progress.examples_seen);
    if (restored.progress.order_seed != expected.order_seed or
        !std.mem.eql(u64, &restored.progress.rng_state, &expected.rng_state))
    {
        return error.TrainingCheckpointOrderStateMismatch;
    }
    return epoch_index;
}

test "gemma4 full run fingerprint binds trajectory inputs" {
    const base = AutodiffRunFingerprintInputs{
        .base_model_sha256 = "base",
        .tokenizer_sha256 = "tokenizer",
        .chat_template_sha256 = "chat",
        .adapter_checkpoint_sha256 = "adapter",
        .train_prepared_sha256 = "train",
        .eval_prepared_sha256 = "eval",
        .train_schema_version = finetune.prepared_schema_v6,
        .eval_schema_version = finetune.prepared_schema_v6,
        .train_source_sha256 = "train-source",
        .eval_source_sha256 = "eval-source",
        .train_source_split = "train",
        .eval_source_split = "validation",
        .train_source_revision = "revision-train",
        .eval_source_revision = "revision-eval",
        .target_modules = &.{ "q_proj", "v_proj" },
        .rank = 8,
        .alpha = 16,
        .recursive_lora = false,
        .recursive_source_num_layers = null,
        .recursive_shared_block_size = null,
        .recursive_loop_count = null,
        .learning_rate = 1e-4,
        .max_examples = 32,
        .eval_max_examples = 16,
        .epochs = 3,
        .max_grad_norm = 1,
        .grad_accum_steps = 2,
        .activation_checkpoint_interval = 4,
        .sequence_length_bucket_quantum = 0,
        .sequence_length_bucket_min = 0,
        .graph_cache_capacity = 1,
        .seed = 42,
        .backend_kind = .metal,
        .linear_cross_entropy_mode = "fused-linear-cce",
        .projector_sha256 = null,
    };
    const expected = autodiffRunFingerprint(base);
    const repeated = autodiffRunFingerprint(base);
    try std.testing.expectEqualSlices(u8, &expected, &repeated);
    var changed = base;
    changed.seed += 1;
    const changed_seed = autodiffRunFingerprint(changed);
    try std.testing.expect(!std.mem.eql(u8, &expected, &changed_seed));
    changed = base;
    changed.eval_prepared_sha256 = "different-eval";
    const changed_eval = autodiffRunFingerprint(changed);
    try std.testing.expect(!std.mem.eql(u8, &expected, &changed_eval));
    changed = base;
    changed.eval_source_split = null;
    const changed_split = autodiffRunFingerprint(changed);
    try std.testing.expect(!std.mem.eql(u8, &expected, &changed_split));
    changed = base;
    changed.linear_cross_entropy_mode = "decomposed-gguf";
    const changed_loss_graph = autodiffRunFingerprint(changed);
    try std.testing.expect(!std.mem.eql(u8, &expected, &changed_loss_graph));
    changed = base;
    changed.sequence_length_bucket_quantum = 16;
    changed.graph_cache_capacity = 4;
    const changed_length_policy = autodiffRunFingerprint(changed);
    try std.testing.expect(!std.mem.eql(u8, &expected, &changed_length_policy));
}

test "gemma4 epoch-boundary resume admits final export recovery and rejects partial windows" {
    const progress = trainingProgressForEpoch(42, 3, 19);
    const restored = real_autodiff.RestoredTrainingCheckpoint{
        .micro_batch_steps = 19,
        .optimizer_steps = 7,
        .accumulation_micro_batches = 0,
        .configured_accumulation_steps = 2,
        .stochastic_steps = 19,
        .progress = progress,
    };
    try std.testing.expectEqual(@as(usize, 3), try validateEpochBoundaryResume(restored, 42, 3));
    var partial = restored;
    partial.accumulation_micro_batches = 1;
    try std.testing.expectError(error.CheckpointNotAtEpochBoundary, validateEpochBoundaryResume(partial, 42, 3));
}

const InitialEvaluationPlacement = enum { before_training, after_benchmark_training };

fn initialEvaluationPlacement(benchmark_enabled: bool) InitialEvaluationPlacement {
    return if (benchmark_enabled) .after_benchmark_training else .before_training;
}

test "gemma4 benchmark defers initial evaluation until after timed training" {
    try std.testing.expectEqual(InitialEvaluationPlacement.before_training, initialEvaluationPlacement(false));
    try std.testing.expectEqual(InitialEvaluationPlacement.after_benchmark_training, initialEvaluationPlacement(true));
}

fn runAutodiff(
    io: std.Io,
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    prepared_inputs_path: []const u8,
    eval_prepared_inputs_path: []const u8,
    out_dir: []const u8,
    prepared: finetune.PreparedInputsSummary,
    eval_prepared: finetune.PreparedInputsSummary,
    opts: CliOptions,
    benchmark_admission: ?*BenchmarkAdmission,
    oracle_admission: ?*OracleAdmission,
    emit_summary: bool,
) !void {
    const operation_started_ns = monotonicNowNs();
    var bundle_inspection = try finetune.inspectLoRABundle(allocator, base_model_dir, adapter_model_dir);
    defer finetune.freeLoRABundleInspectionSummary(allocator, &bundle_inspection);
    try finetune.validateLoRABundleInspection(bundle_inspection);

    var adapter_inspect = try finetune.inspectCheckpoint(allocator, adapter_model_dir);
    defer finetune.freeInspectionSummary(allocator, &adapter_inspect);
    const validated_adapter = try validateAutodiffAdapterConfig(adapter_inspect);
    const recursive_shared_block_size = adapter_inspect.recursive_shared_block_size;

    const graph_config = try gemma4_real.loadGraphConfig(allocator, base_model_dir);
    const direct_gguf_base = try autodiffBaseUsesGguf(allocator, base_model_dir);
    _ = try finetune.validatePreparedSequenceAdmission(prepared, graph_config.max_position_embeddings);
    _ = try finetune.validatePreparedSequenceAdmission(eval_prepared, graph_config.max_position_embeddings);
    try finetune.validatePreparedVocabularyAdmission(prepared, graph_config.vocab_size);
    try finetune.validatePreparedVocabularyAdmission(eval_prepared, graph_config.vocab_size);
    try finetune.validatePreparedSourceDatasetProvenance(allocator, prepared);
    try finetune.validatePreparedSourceDatasetProvenance(allocator, eval_prepared);
    if (oracle_admission == null) {
        try finetune.validatePreparedEvalDisjoint(allocator, prepared.examples, eval_prepared.examples);
    }
    var provenance = try finetune.fingerprintGemma4Model(allocator, base_model_dir);
    defer provenance.deinit(allocator);
    try finetune.validatePreparedModelProvenance(prepared, provenance);
    try finetune.validatePreparedModelProvenance(eval_prepared, provenance);
    try finetune.validateAdapterModelProvenance(adapter_inspect, provenance);
    if (benchmark_admission) |admission| {
        try validateBenchmarkArtifactBindings(
            allocator,
            io,
            admission,
            adapter_inspect,
            validated_adapter,
            prepared_inputs_path,
            eval_prepared_inputs_path,
            prepared,
        );
    }
    if (oracle_admission) |admission| {
        try validateOracleArtifactBindings(
            allocator,
            admission,
            adapter_inspect,
            validated_adapter,
            prepared_inputs_path,
            prepared,
            provenance,
        );
    }

    var publication = try ImmutableRunPublication.init(allocator, io, out_dir);
    defer publication.deinit();
    var oracle_publication: ?ImmutableRunPublication = if (oracle_admission) |admission|
        try ImmutableRunPublication.init(allocator, io, admission.capture_out_dir)
    else
        null;
    defer if (oracle_publication) |*capture_publication| capture_publication.deinit();
    if (oracle_publication) |*capture_publication| try capture_publication.createStaging();

    const bootstrap = if (oracle_admission) |admission|
        &prepared.examples[admission.request().bindings.example_index]
    else
        gemma4_real.findFirstSupervisedExample(prepared.examples) orelse return error.NoTrainingData;
    const is_multimodal = prepared.examples_with_images > 0 or prepared.examples_with_audio > 0;
    const eval_is_multimodal = eval_prepared.examples_with_images > 0 or eval_prepared.examples_with_audio > 0;
    if ((is_multimodal or eval_is_multimodal) and opts.gguf_projector_path == null) return error.MissingGgufProjector;
    var maybe_projector_fingerprint: ?finetune.ProjectorFingerprint = null;
    defer if (maybe_projector_fingerprint) |*fp| finetune.freeProjectorFingerprint(allocator, fp);
    if (is_multimodal or eval_is_multimodal) {
        maybe_projector_fingerprint = try finetune.fingerprintProjectorFile(allocator, opts.gguf_projector_path.?);
        if (is_multimodal) try validatePreparedProjectorFingerprint(prepared, maybe_projector_fingerprint.?);
        if (eval_is_multimodal) try validatePreparedProjectorFingerprint(eval_prepared, maybe_projector_fingerprint.?);
    }
    const mm_stats = summarizeMultimodalPrepared(prepared.examples);
    const backend_kind = opts.backend_kind orelse return error.MissingBackend;
    const execution_policy = autodiffExecutionPolicy(backend_kind);
    const length_buckets = sequenceLengthBuckets(opts);
    const bucket_minimum = effectiveSequenceLengthBucketMinimum(opts);
    if (is_multimodal and length_buckets != null) return error.Gemma4MultimodalLengthBucketsNotSupported;
    const graph_cache_capacity = effectiveGraphCacheCapacity(opts);
    const fused_linear_cross_entropy_enabled = backend_kind == .metal and
        !direct_gguf_base and
        !platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_LINEAR_CCE", false);
    const linear_cross_entropy_mode: []const u8 = if (is_multimodal)
        "multimodal"
    else if (fused_linear_cross_entropy_enabled)
        "fused-linear-cce"
    else if (direct_gguf_base)
        "decomposed-gguf"
    else
        "decomposed";
    const target_modules = adapter_inspect.target_modules orelse finetune.default_lora_target_modules[0..];
    const lora_config = ml.graph.lora.LoRAConfig{
        .rank = validated_adapter.rank,
        .alpha = validated_adapter.alpha,
        .target_patterns = target_modules,
        .strict_target_patterns = true,
        .sharing = if (adapter_inspect.recursive_lora_enabled) .by_use else .by_weight,
    };
    if ((opts.resume_from_checkpoint or opts.checkpoint_every_epochs != 0) and is_multimodal) {
        return error.Gemma4MultimodalCheckpointResumeNotSupported;
    }
    const adapter_checkpoint_path = adapter_inspect.adapter_checkpoint_path orelse return error.MissingAdapterCheckpoint;
    var adapter_checkpoint_fingerprint = try finetune.fingerprintProjectorFile(allocator, adapter_checkpoint_path);
    defer finetune.freeProjectorFingerprint(allocator, &adapter_checkpoint_fingerprint);
    const run_fingerprint = autodiffRunFingerprint(.{
        .base_model_sha256 = provenance.base_model_sha256,
        .tokenizer_sha256 = provenance.tokenizer_sha256,
        .chat_template_sha256 = provenance.chat_template_sha256,
        .adapter_checkpoint_sha256 = adapter_checkpoint_fingerprint.sha256,
        .train_prepared_sha256 = prepared.prepared_examples_sha256 orelse return error.PreparedInputsProvenanceRequired,
        .eval_prepared_sha256 = eval_prepared.prepared_examples_sha256 orelse return error.PreparedInputsProvenanceRequired,
        .train_schema_version = prepared.schema_version,
        .eval_schema_version = eval_prepared.schema_version,
        .train_source_sha256 = prepared.source_dataset_sha256 orelse return error.PreparedInputsProvenanceRequired,
        .eval_source_sha256 = eval_prepared.source_dataset_sha256 orelse return error.PreparedInputsProvenanceRequired,
        .train_source_split = prepared.source_split,
        .eval_source_split = eval_prepared.source_split,
        .train_source_revision = prepared.source_revision orelse return error.PreparedInputsProvenanceRequired,
        .eval_source_revision = eval_prepared.source_revision orelse return error.PreparedInputsProvenanceRequired,
        .target_modules = target_modules,
        .rank = validated_adapter.rank,
        .alpha = validated_adapter.alpha,
        .recursive_lora = adapter_inspect.recursive_lora_enabled,
        .recursive_source_num_layers = adapter_inspect.recursive_source_num_layers,
        .recursive_shared_block_size = recursive_shared_block_size,
        .recursive_loop_count = adapter_inspect.recursive_loop_count,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .activation_checkpoint_interval = opts.activation_checkpoint_interval,
        .sequence_length_bucket_quantum = opts.sequence_length_bucket_quantum,
        .sequence_length_bucket_min = bucket_minimum,
        .graph_cache_capacity = graph_cache_capacity,
        .seed = opts.seed,
        .backend_kind = backend_kind,
        .linear_cross_entropy_mode = linear_cross_entropy_mode,
        .projector_sha256 = if (maybe_projector_fingerprint) |fp| fp.sha256 else null,
    });
    const run_fingerprint_hex_array = std.fmt.bytesToHex(run_fingerprint, .lower);
    const run_fingerprint_hex: []const u8 = &run_fingerprint_hex_array;

    var benchmark_capture: ?BenchmarkCapture = if (benchmark_admission) |admission| .{
        .allocator = allocator,
        .request = admission.request(),
        .io = io,
        .control_signal_fd = admission.control_signal_fd,
        .control_ack_fd = admission.control_ack_fd,
    } else null;
    defer if (benchmark_capture) |*capture| capture.deinit();
    var oracle_capture: ?OracleCapture = if (oracle_admission) |admission| .{
        .allocator = allocator,
        .expected_steps = admission.request().training.steps,
    } else null;
    defer if (oracle_capture) |*capture| capture.deinit();
    const benchmark_load_started_ns = if (benchmark_capture != null) monotonicNowNs() else 0;
    var backend = try gemma4_real.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();

    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = if (oracle_admission) |admission| .{
            .beta1 = @floatCast(admission.request().training.betas[0]),
            .beta2 = @floatCast(admission.request().training.betas[1]),
            .eps = @floatCast(admission.request().training.eps),
            .weight_decay = @floatCast(admission.request().training.weight_decay),
        } else .{},
        .lr_schedule = .{ .constant = opts.learning_rate },
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .seed = opts.seed,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
        .graph_cache_capacity = graph_cache_capacity,
        .reduce_grads = if (oracle_capture != null and backend_kind != .metal) &OracleCapture.reduce else null,
        .reduce_grads_ctx = if (oracle_capture != null and backend_kind != .metal)
            @ptrCast(&oracle_capture.?)
        else
            null,
        .observe_direct_device_grads = if (oracle_capture != null and backend_kind == .metal) &OracleCapture.observeDirectDevice else null,
        .observe_direct_device_grads_ctx = if (oracle_capture != null and backend_kind == .metal)
            @ptrCast(&oracle_capture.?)
        else
            null,
        .checkpoint_config = if (opts.activation_checkpoint_interval > 0) .{
            .strategy = .every_n_layers,
            .layer_interval = opts.activation_checkpoint_interval,
        } else null,
    });
    defer trainer.deinit();
    var maybe_text_ctx: ?gemma4_real.GemmaAutodiffCtx = null;
    var maybe_mm_ctx: ?gemma4_mm_real.MultimodalCtx = null;
    const training_max_seq_len: u32 = if (oracle_admission) |admission|
        @intCast(prepared.examples[admission.request().bindings.example_index].num_input_tokens)
    else
        @intCast(prepared.max_seq_len);
    const bootstrap_seq_len = try gemma4_real.sequenceLengthForExample(
        bootstrap.num_input_tokens,
        training_max_seq_len,
        length_buckets,
    );
    if (is_multimodal) {
        const tokenizer = try gemma4_mm_real.loadTokenizerForModelDir(allocator, base_model_dir);
        maybe_mm_ctx = if (recursive_shared_block_size) |shared_block_size|
            gemma4_mm_real.MultimodalCtx.initRecursive(allocator, backend.backendPtr(), graph_config, opts.gguf_projector_path.?, maybe_projector_fingerprint.?.sha256, tokenizer, shared_block_size)
        else
            gemma4_mm_real.MultimodalCtx.init(allocator, backend.backendPtr(), graph_config, opts.gguf_projector_path.?, maybe_projector_fingerprint.?.sha256, tokenizer);
        try gemma4_mm_real.initializeTrainerFromAdapterDir(
            allocator,
            &trainer,
            &maybe_mm_ctx.?,
            adapter_model_dir,
            bootstrap,
            bootstrap_seq_len,
        );
    } else {
        maybe_text_ctx = if (recursive_shared_block_size) |shared_block_size|
            gemma4_real.GemmaAutodiffCtx.initRecursive(graph_config, shared_block_size)
        else
            gemma4_real.GemmaAutodiffCtx.init(graph_config);
        // This optimization is admitted only on the strict Metal product
        // lane. Native keeps the decomposed RMSNorm VJP as the independent
        // reference implementation.
        maybe_text_ctx.?.enable_fused_rms_norm_backward = backend_kind == .metal;
        maybe_text_ctx.?.enable_fused_gqa_attention_backward = backend_kind == .metal and gemma4_real.fusedGqaAttentionExperimentEnabled(graph_config);
        // The builder-level fused loss internally hides the quantized
        // vocabulary projection and its backward-input projection from the
        // graph dependency planner. Exact paired E2B GGUF runs proved that
        // route nondeterministic; the equivalent decomposed graph is exact.
        // Keep dense safetensors on the qualified fast path and fail closed to
        // the decomposed loss whenever the selected training artifact is GGUF.
        maybe_text_ctx.?.enable_fused_linear_cross_entropy = fused_linear_cross_entropy_enabled;
        try gemma4_real.initializeTrainerFromAdapterDir(
            allocator,
            &trainer,
            &maybe_text_ctx.?,
            adapter_model_dir,
            bootstrap,
            bootstrap_seq_len,
        );
    }
    defer if (maybe_mm_ctx) |*ctx| ctx.deinit();
    if (benchmark_capture) |*capture| {
        const finished_ns = monotonicNowNs();
        if (benchmark_load_started_ns == 0 or finished_ns <= benchmark_load_started_ns) {
            return error.BenchmarkInvalidMonotonicInterval;
        }
        capture.load_ns = finished_ns - benchmark_load_started_ns;
    }

    var start_epoch: usize = 0;
    if (opts.resume_from_checkpoint) {
        const checkpoint_path = opts.checkpoint_path orelse return error.CheckpointPathRequired;
        const restored = try trainer.loadTrainingCheckpoint(checkpoint_path, &run_fingerprint);
        start_epoch = try validateEpochBoundaryResume(restored, opts.seed, opts.epochs);
    } else {
        trainer.setTrainingProgress(trainingProgressForEpoch(opts.seed, 0, 0));
    }
    const initialization_finished_ns = monotonicNowNs();

    // Normal training preserves the historical before/train/after evaluation
    // order. Benchmark mode defers this read-only initial-adapter evaluation
    // until after the timed optimizer windows so the cold window is the
    // process's first graph execution, matching the MLX reference protocol.
    var before: gemma4_real.CausalLmMetrics = .{};
    var initial_evaluation_wall_time_ns: u64 = 0;
    if (oracle_admission == null and initialEvaluationPlacement(benchmark_admission != null) == .before_training) {
        const initial_evaluation_started_ns = monotonicNowNs();
        before = if (is_multimodal)
            try gemma4_mm_real.evaluatePreparedExamples(
                allocator,
                &trainer,
                &maybe_mm_ctx.?,
                eval_prepared.examples,
                opts.effectiveEvalMaxExamples(),
                @intCast(eval_prepared.max_seq_len),
            )
        else
            (try gemma4_real.evaluatePreparedExamplesRange(
                allocator,
                &trainer,
                &maybe_text_ctx.?,
                eval_prepared.examples,
                .{
                    .max_examples = opts.effectiveEvalMaxExamples(),
                    .seq_len = @intCast(eval_prepared.max_seq_len),
                    .sequence_length_buckets = length_buckets,
                },
            )).metrics;
        initial_evaluation_wall_time_ns = monotonicElapsedNs(initial_evaluation_started_ns, monotonicNowNs());
    }

    const epoch_history = try allocator.alloc(AutodiffEpochSummary, opts.epochs - start_epoch);
    defer allocator.free(epoch_history);
    var repeated_benchmark_examples: ?[]finetune.PreparedExampleInput = null;
    defer if (repeated_benchmark_examples) |examples| allocator.free(examples);
    if (benchmark_admission) |admission| {
        const bindings = admission.request().bindings;
        const examples = try allocator.alloc(finetune.PreparedExampleInput, bindings.grad_accum);
        for (examples) |*example| example.* = prepared.examples[bindings.prepared_example_index];
        repeated_benchmark_examples = examples;
    }
    if (benchmark_capture) |*capture| {
        if (trainer.benchmarkHasExecutedGraph()) return error.BenchmarkColdStepNotFirstGraphExecution;
        capture.cold_step_was_first_graph_execution = true;
    }
    const benchmark_optimizer_steps_before = trainer.optimizerSteps();
    var training_epochs_wall_time_ns: u64 = 0;
    for (start_epoch..opts.epochs, 0..) |epoch_idx, history_idx| {
        const prior_examples_seen = trainer.trainingProgress().examples_seen;
        trainer.setTrainingProgress(trainingProgressForEpoch(opts.seed, epoch_idx, prior_examples_seen));
        if (oracle_admission) |admission| {
            if (epoch_idx + 1 == opts.epochs) {
                const capture = &oracle_capture.?;
                if (capture.probes != null) return error.DuplicateOracleLogitCapture;
                const example = &prepared.examples[admission.request().bindings.example_index];
                capture.probes = try gemma4_real.captureSupervisedLogitProbes(
                    allocator,
                    &trainer,
                    &maybe_text_ctx.?,
                    example,
                    training_max_seq_len,
                    opts.seed,
                );
            }
        }
        const graph_cache_before = trainer.graphCacheStats();
        const epoch_started_ns = monotonicNowNs();
        const metrics = if (is_multimodal)
            try gemma4_mm_real.trainPreparedExamples(
                allocator,
                &trainer,
                &maybe_mm_ctx.?,
                prepared.examples,
                opts.max_examples,
                @intCast(prepared.max_seq_len),
            )
        else if (benchmark_capture) |*capture|
            (try gemma4_real.trainPreparedExamplesRange(
                allocator,
                &trainer,
                &maybe_text_ctx.?,
                repeated_benchmark_examples.?,
                .{
                    .max_examples = repeated_benchmark_examples.?.len,
                    .seq_len = @intCast(prepared.max_seq_len),
                    .flush_at_end = true,
                    .benchmark_observer_context = capture,
                    .benchmark_observer = &BenchmarkCapture.observe,
                },
            )).metrics
        else if (oracle_admission) |admission|
            (try gemma4_real.trainPreparedExamplesRange(
                allocator,
                &trainer,
                &maybe_text_ctx.?,
                prepared.examples[admission.request().bindings.example_index .. admission.request().bindings.example_index + 1],
                .{
                    .max_examples = 1,
                    .seq_len = training_max_seq_len,
                    .flush_at_end = true,
                },
            )).metrics
        else
            (try gemma4_real.trainPreparedExamplesRange(
                allocator,
                &trainer,
                &maybe_text_ctx.?,
                prepared.examples,
                .{
                    .max_examples = opts.max_examples,
                    .seq_len = training_max_seq_len,
                    .sequence_length_buckets = length_buckets,
                    .flush_at_end = true,
                },
            )).metrics;
        const epoch_finished_ns = monotonicNowNs();
        const epoch_timing = summarizeEpochTiming(epoch_started_ns, epoch_finished_ns, metrics);
        const graph_cache_delta = summarizeEpochGraphCacheDelta(graph_cache_before, trainer.graphCacheStats());
        training_epochs_wall_time_ns +|= epoch_timing.wall_time_ns;
        try validateAutodiffEpoch(metrics);
        if (oracle_capture) |*capture| try capture.recordStep(metrics);
        epoch_history[history_idx] = .{
            .epoch = epoch_idx + 1,
            .timing = epoch_timing,
            .graph_cache_delta = graph_cache_delta,
            .examples_seen = metrics.examples_seen,
            .supervised_tokens_seen = metrics.supervised_tokens_seen,
            .logical_input_tokens_seen = metrics.logical_input_tokens_seen,
            .scheduled_input_tokens_seen = metrics.scheduled_input_tokens_seen,
            .fixed_shape_input_tokens_seen = metrics.fixed_shape_input_tokens_seen,
            .padding_tokens_avoided = metrics.padding_tokens_avoided,
            .length_bucketed_examples = metrics.length_bucketed_examples,
            .min_scheduled_sequence_length = metrics.min_scheduled_sequence_length,
            .max_scheduled_sequence_length = metrics.max_scheduled_sequence_length,
            .teacher_examples_seen = metrics.teacher_examples_seen,
            .teacher_supervised_tokens_seen = metrics.teacher_supervised_tokens_seen,
            .mean_teacher_temperature = metrics.mean_teacher_temperature,
            .average_loss = metrics.average_loss,
            .mean_grad_norm = metrics.mean_grad_norm,
            .optimizer_steps = metrics.optimizer_steps,
            .graph_executor_steps = metrics.graph_executor_steps,
            .graph_executor_fallback_steps = metrics.graph_executor_fallback_steps,
            .graph_executor_partitions = metrics.graph_executor_partitions,
            .graph_executor_command_dispatches = metrics.graph_executor_command_dispatches,
            .graph_executor_native_partitions = metrics.graph_executor_native_partitions,
            .graph_executor_unsupported_ops = metrics.graph_executor_unsupported_ops,
            .graph_executor_interpreter_fallbacks = metrics.graph_executor_interpreter_fallbacks,
            .graph_executor_runtime_region_dispatches = metrics.graph_executor_runtime_region_dispatches,
            .graph_executor_true_host_outputs = metrics.graph_executor_true_host_outputs,
            .runtime_input_uploads = metrics.runtime_input_uploads,
            .runtime_input_upload_bytes = metrics.runtime_input_upload_bytes,
            .runtime_input_h2d_bytes = metrics.runtime_input_h2d_bytes,
            .runtime_input_d2h_bytes = metrics.runtime_input_d2h_bytes,
            .declared_runtime_input_uploads = metrics.declared_runtime_input_uploads,
            .declared_runtime_input_upload_bytes = metrics.declared_runtime_input_upload_bytes,
            .declared_runtime_input_h2d_bytes = metrics.declared_runtime_input_h2d_bytes,
            .graph_execution_h2d_bytes = metrics.graph_execution_h2d_bytes,
            .graph_execution_d2h_bytes = metrics.graph_execution_d2h_bytes,
            .training_runtime_h2d_bytes = metrics.training_runtime_h2d_bytes,
            .training_runtime_d2h_bytes = metrics.training_runtime_d2h_bytes,
            .metal_optimizer_steps = metrics.metal_optimizer_steps,
        };
        const completed_epochs = epoch_idx + 1;
        trainer.setTrainingProgress(trainingProgressForEpoch(
            opts.seed,
            completed_epochs,
            trainer.trainingProgress().examples_seen,
        ));
        if (opts.checkpoint_every_epochs > 0 and
            (completed_epochs % opts.checkpoint_every_epochs == 0 or completed_epochs == opts.epochs))
        {
            try trainer.saveTrainingCheckpoint(opts.checkpoint_path.?, &run_fingerprint, null);
        }
        std.log.info(
            "gemma4 autodiff: epoch={d}/{d} loss={d:.4} examples={d} tokens={d} updates={d} wall_s={d:.3} logical_tok_s={d:.2} cache_builds={d} cache_evictions={d}",
            .{
                epoch_idx + 1,
                opts.epochs,
                metrics.average_loss,
                metrics.examples_seen,
                metrics.supervised_tokens_seen,
                metrics.optimizer_steps,
                @as(f64, @floatFromInt(epoch_timing.wall_time_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s)),
                epoch_timing.logical_input_tokens_per_second,
                graph_cache_delta.builds,
                graph_cache_delta.evictions,
            },
        );
    }
    if (benchmark_capture) |*capture| {
        try capture.validateComplete();
        if (trainer.optimizerSteps() -| benchmark_optimizer_steps_before != benchmark_optimizer_steps_v1) {
            return error.BenchmarkColdStepDidNotMutateOptimizerState;
        }
    }
    if (oracle_capture) |*capture| {
        try capture.validateComplete(backend_kind, oracle_admission.?.request().bindings.target_count);
        if (trainer.optimizerSteps() != oracle_admission.?.request().training.steps or
            trainer.microBatchSteps() != oracle_admission.?.request().training.steps or
            trainer.accumulatedMicroBatches() != 0)
        {
            return error.OracleTrainerCounterMismatch;
        }
    }

    if (oracle_admission == null and initialEvaluationPlacement(benchmark_admission != null) == .after_benchmark_training) {
        const initial_evaluation_started_ns = monotonicNowNs();
        before = try evaluateAutodiff(
            allocator,
            base_model_dir,
            adapter_model_dir,
            eval_prepared.examples,
            opts.effectiveEvalMaxExamples(),
            graph_config,
            lora_config,
            backend_kind,
            opts.max_grad_norm,
            length_buckets,
            graph_cache_capacity,
            opts.gguf_projector_path,
            if (maybe_projector_fingerprint) |fp| fp.sha256 else null,
        );
        initial_evaluation_wall_time_ns = monotonicElapsedNs(initial_evaluation_started_ns, monotonicNowNs());
    }

    const adapter_save_started_ns = monotonicNowNs();
    try gemma4_real.saveTrainerAsGemmaBundle(allocator, &trainer, base_model_dir, adapter_model_dir, publication.staging_dir);
    publication.claimStaging();
    const adapter_save_wall_time_ns = monotonicElapsedNs(adapter_save_started_ns, monotonicNowNs());
    const final_evaluation_started_ns = monotonicNowNs();
    const after = if (oracle_admission == null)
        try evaluateAutodiff(
            allocator,
            base_model_dir,
            publication.staging_dir,
            eval_prepared.examples,
            opts.effectiveEvalMaxExamples(),
            graph_config,
            lora_config,
            backend_kind,
            opts.max_grad_norm,
            length_buckets,
            graph_cache_capacity,
            opts.gguf_projector_path,
            if (maybe_projector_fingerprint) |fp| fp.sha256 else null,
        )
    else
        gemma4_real.CausalLmMetrics{};
    const final_evaluation_finished_ns = monotonicNowNs();
    const final_evaluation_wall_time_ns = monotonicElapsedNs(final_evaluation_started_ns, final_evaluation_finished_ns);

    const report_payload = .{
        .artifact_family_version = finetune.artifact_family_version,
        .trainer_kind = if (is_multimodal) "real_autodiff_multimodal_causal_lm_v1" else "real_autodiff_causal_lm_v1",
        .prepared_inputs_path = prepared_inputs_path,
        .eval_prepared_inputs_path = eval_prepared_inputs_path,
        .saved_adapter_checkpoint = finetune.adapter_checkpoint_file_name,
        .run_fingerprint_sha256 = run_fingerprint_hex,
        .seed = opts.seed,
        .checkpoint_resume = .{
            .enabled = opts.resume_from_checkpoint,
            .start_epoch = start_epoch,
            .checkpoint_path = opts.checkpoint_path,
            .checkpoint_every_epochs = opts.checkpoint_every_epochs,
        },
        .numerical_oracle_capture = .{
            .enabled = oracle_admission != null,
            .request_sha256 = if (oracle_admission) |admission| admission.request_sha256 else null,
            .capture_schema_version = if (oracle_admission != null) oracle_capture_schema_v1 else null,
        },
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .activation_checkpoint_interval = opts.activation_checkpoint_interval,
        .sequence_length_policy = .{
            .mode = if (length_buckets != null) "independent-row-buckets-v1" else "fixed-prepared-maximum",
            .bucket_quantum = opts.sequence_length_bucket_quantum,
            .bucket_minimum = bucket_minimum,
            .graph_cache_capacity = graph_cache_capacity,
        },
        .graph_cache = trainer.graphCacheStats(),
        .phase_timing = AutodiffPhaseTimingSummary{
            .initial_evaluation_placement = @tagName(initialEvaluationPlacement(benchmark_admission != null)),
            .initialization_and_restore_wall_time_ns = monotonicElapsedNs(operation_started_ns, initialization_finished_ns),
            .initial_evaluation_wall_time_ns = initial_evaluation_wall_time_ns,
            .training_epochs_wall_time_ns = training_epochs_wall_time_ns,
            .adapter_save_wall_time_ns = adapter_save_wall_time_ns,
            .final_evaluation_wall_time_ns = final_evaluation_wall_time_ns,
            .through_final_evaluation_wall_time_ns = monotonicElapsedNs(operation_started_ns, final_evaluation_finished_ns),
        },
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
        .backend_kind = backend_kind,
        .base_weight_artifact_kind = if (direct_gguf_base) "gguf" else "safetensors",
        .linear_cross_entropy_mode = linear_cross_entropy_mode,
        .adapter = .{
            .rank = validated_adapter.rank,
            .alpha = validated_adapter.alpha,
            .target_preset = adapter_inspect.target_preset,
            .target_modules = target_modules,
            .recursive_lora = adapter_inspect.recursive_lora_enabled,
            .recursive_shared_block_size = recursive_shared_block_size,
        },
        .multimodal = .{
            .enabled = is_multimodal,
            .gguf_projector_path = opts.gguf_projector_path,
            .gguf_projector_sha256 = if (maybe_projector_fingerprint) |fp| fp.sha256 else null,
            .gguf_projector_size_bytes = if (maybe_projector_fingerprint) |fp| fp.size_bytes else null,
            .projected_media_cache_entries = if (maybe_mm_ctx) |*ctx| ctx.projectedMediaCacheEntries() else 0,
            .projected_media_cache_hits = if (maybe_mm_ctx) |*ctx| ctx.projected_media_cache_hits else 0,
            .projected_media_cache_misses = if (maybe_mm_ctx) |*ctx| ctx.projected_media_cache_misses else 0,
            .examples_with_media = mm_stats.examples_with_media,
            .total_image_inputs = mm_stats.total_image_inputs,
            .total_audio_inputs = mm_stats.total_audio_inputs,
            .total_image_soft_tokens = mm_stats.total_image_soft_tokens,
            .total_audio_soft_tokens = mm_stats.total_audio_soft_tokens,
        },
        .prepared_dataset = .{
            .schema_version = prepared.schema_version,
            .base_model_sha256 = prepared.base_model_sha256,
            .tokenizer_sha256 = prepared.tokenizer_sha256,
            .chat_template_sha256 = prepared.chat_template_sha256,
            .prepared_examples_sha256 = prepared.prepared_examples_sha256,
            .source_dataset_path = prepared.source_dataset_path,
            .source_dataset_sha256 = prepared.source_dataset_sha256,
            .source_split = prepared.source_split,
            .source_revision = prepared.source_revision,
            .examples_seen = prepared.examples_seen,
            .max_seq_len = prepared.max_seq_len,
            .max_input_tokens = prepared.max_input_tokens,
            .max_supervised_tokens = prepared.max_supervised_tokens,
            .examples_with_tool_calls = prepared.examples_with_tool_calls,
            .examples_with_tool_results = prepared.examples_with_tool_messages,
            .examples_with_multiturn = prepared.examples_with_multiturn,
            .examples_with_images = prepared.examples_with_images,
            .examples_with_audio = prepared.examples_with_audio,
            .examples_truncated = prepared.examples_truncated,
            .max_turns_dropped = prepared.max_turns_dropped,
        },
        .evaluation_dataset = .{
            .schema_version = eval_prepared.schema_version,
            .base_model_sha256 = eval_prepared.base_model_sha256,
            .tokenizer_sha256 = eval_prepared.tokenizer_sha256,
            .chat_template_sha256 = eval_prepared.chat_template_sha256,
            .examples_seen = eval_prepared.examples_seen,
            .max_seq_len = eval_prepared.max_seq_len,
            .max_input_tokens = eval_prepared.max_input_tokens,
            .max_supervised_tokens = eval_prepared.max_supervised_tokens,
            .prepared_examples_sha256 = eval_prepared.prepared_examples_sha256,
            .source_dataset_path = eval_prepared.source_dataset_path,
            .source_dataset_sha256 = eval_prepared.source_dataset_sha256,
            .source_split = eval_prepared.source_split,
            .source_revision = eval_prepared.source_revision,
        },
        .before = before,
        .epoch_history = epoch_history,
        .after = after,
    };
    try writeRunOutputs(io, allocator, publication.staging_dir, base_model_dir, adapter_model_dir, if (is_multimodal) "autodiff_multimodal" else "autodiff", report_payload, .{
        .prepared_inputs_path = prepared_inputs_path,
        .eval_prepared_inputs_path = eval_prepared_inputs_path,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .activation_checkpoint_interval = opts.activation_checkpoint_interval,
        .sequence_length_bucket_quantum = opts.sequence_length_bucket_quantum,
        .sequence_length_bucket_min = bucket_minimum,
        .graph_cache_capacity = graph_cache_capacity,
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
        .backend_label = backend_kind.label(),
        .seed = opts.seed,
        .checkpoint_path = opts.checkpoint_path,
        .checkpoint_every_epochs = opts.checkpoint_every_epochs,
        .resumed_from_epoch = if (opts.resume_from_checkpoint) start_epoch else null,
        .run_fingerprint = run_fingerprint_hex,
    }, emit_summary);
    try writeRunCompletionManifest(io, allocator, publication.staging_dir);
    if (oracle_admission) |admission| {
        const capture_publication = &oracle_publication.?;
        try writeOracleCapture(
            allocator,
            io,
            admission,
            &oracle_capture.?,
            &trainer,
            publication.staging_dir,
            capture_publication.staging_dir,
            backend_kind,
            prepared,
        );
        try writeRunCompletionManifest(io, allocator, capture_publication.staging_dir);
    }
    try publication.publish();
    if (oracle_publication) |*capture_publication| try capture_publication.publish();
    if (benchmark_admission) |admission| {
        try publishBenchmarkTelemetry(allocator, io, admission, &benchmark_capture.?);
    }
}

fn publishBenchmarkTelemetry(
    allocator: std.mem.Allocator,
    io: std.Io,
    admission: *const BenchmarkAdmission,
    capture: *const BenchmarkCapture,
) !void {
    try capture.validateComplete();
    const memory = platform.process_memory.snapshot();
    if (!memory.available or memory.peak_footprint_bytes == 0) return error.BenchmarkPeakMemoryUnavailable;
    const steps = capture.optimizer_steps.items;
    const warmup_start = benchmark_cold_steps_v1 + benchmark_first_steps_v1;
    const measured_start = warmup_start + benchmark_warmup_steps_v1;
    const payload = .{
        .schema_version = benchmark_telemetry_schema_v1,
        .producer = .{
            .pid = @as(u64, @intCast(std.posix.system.getpid())),
            .backend = "metal",
            .strict_metal_execution = true,
            .version = build_options.inference_version,
            .metal_device = admission.request().implementation.metal_device,
            .executable_sha256 = admission.request().implementation.executable_sha256,
            .source_revision = build_options.benchmark_source_revision,
            .request_sha256 = admission.request_sha256,
            .command_sha256 = admission.command_sha256,
        },
        .bindings = admission.request().bindings,
        .protocol = admission.request().protocol,
        .runtime = admission.request().runtime,
        .measurement_control = admission.request().measurement_control,
        .timings = .{
            .load_ns = capture.load_ns,
            .cold_step_was_first_graph_execution = capture.cold_step_was_first_graph_execution,
            .cold_compile_and_step_ns = steps[0].duration_ns,
            .cold_compile_ns = capture.compile_ns,
            .first_steady_step_ns = steps[1].duration_ns,
            .warmup_step_ns = [_]u64{
                steps[warmup_start].duration_ns,
                steps[warmup_start + 1].duration_ns,
                steps[warmup_start + 2].duration_ns,
            },
            .measured_step_ns = blk: {
                var values: [benchmark_measured_steps_v1]u64 = undefined;
                for (&values, steps[measured_start..]) |*value, step| value.* = step.duration_ns;
                break :blk values;
            },
            .optimizer_steps = steps,
        },
        .memory = .{
            .peak_bytes = memory.peak_footprint_bytes,
            .source = benchmark_peak_memory_source_v1,
        },
    };
    const rendered = try std.json.Stringify.valueAlloc(allocator, payload, .{ .whitespace = .indent_2 });
    defer allocator.free(rendered);
    try artifact_publication.writeFileImmutable(allocator, io, admission.telemetry_out_path, rendered);
}

fn writeOracleCapture(
    allocator: std.mem.Allocator,
    io: std.Io,
    admission: *const OracleAdmission,
    capture: *const OracleCapture,
    trainer: *real_autodiff.RealAutodiffTrainer,
    candidate_adapter_dir: []const u8,
    capture_staging_dir: []const u8,
    backend_kind: BackendKind,
    prepared: finetune.PreparedInputsSummary,
) !void {
    const request = admission.request();
    try capture.validateComplete(backend_kind, request.bindings.target_count);
    if (trainer.lora_params.items.len != capture.gradients.items.len) {
        return error.OracleTargetInventoryMismatch;
    }

    const gradient_file_name = "raw_gradients.safetensors";
    const checkpoint_file_name = "trainer_checkpoint.safetensors";
    const capture_file_name = "capture.json";
    const gradient_path = try std.fs.path.join(allocator, &.{ capture_staging_dir, gradient_file_name });
    defer allocator.free(gradient_path);
    const checkpoint_path = try std.fs.path.join(allocator, &.{ capture_staging_dir, checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    const capture_path = try std.fs.path.join(allocator, &.{ capture_staging_dir, capture_file_name });
    defer allocator.free(capture_path);

    const named_gradients = try allocator.alloc(safetensors_checkpoint.NamedTensor, capture.gradients.items.len);
    defer allocator.free(named_gradients);
    const targets = try allocator.alloc(OracleCapturedTargetV1, capture.gradients.items.len);
    defer allocator.free(targets);
    var owned_names = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (owned_names.items) |name| allocator.free(name);
        owned_names.deinit(allocator);
    }
    var owned_shapes = std.ArrayListUnmanaged([]usize).empty;
    defer {
        for (owned_shapes.items) |shape| allocator.free(shape);
        owned_shapes.deinit(allocator);
    }
    try owned_names.ensureTotalCapacity(allocator, capture.gradients.items.len * 5);
    try owned_shapes.ensureTotalCapacity(allocator, capture.gradients.items.len);

    var gradient_sq: f64 = 0;
    for (capture.gradients.items, trainer.lora_params.items, 0..) |gradient, slot, index| {
        if (!std.mem.eql(u8, gradient.name, slot.name) or gradient.values.len != slot.weights.len) {
            return error.OracleTargetInventoryMismatch;
        }
        const source_name = try gemma4_real.mapTrainerSlotNameToGemmaAdapterTensor(allocator, slot.name);
        errdefer allocator.free(source_name);
        const gradient_key = try std.fmt.allocPrint(allocator, "gradient::{s}", .{slot.name});
        errdefer allocator.free(gradient_key);
        const checkpoint_weight_key = try std.fmt.allocPrint(allocator, "weight::{s}", .{slot.name});
        errdefer allocator.free(checkpoint_weight_key);
        const checkpoint_m_key = try std.fmt.allocPrint(allocator, "adam_m::{s}", .{slot.name});
        errdefer allocator.free(checkpoint_m_key);
        const checkpoint_v_key = try std.fmt.allocPrint(allocator, "adam_v::{s}", .{slot.name});
        errdefer allocator.free(checkpoint_v_key);
        const shape = try dimsToUsize(allocator, slot.dims);
        errdefer allocator.free(shape);
        owned_names.appendAssumeCapacity(source_name);
        owned_names.appendAssumeCapacity(gradient_key);
        owned_names.appendAssumeCapacity(checkpoint_weight_key);
        owned_names.appendAssumeCapacity(checkpoint_m_key);
        owned_names.appendAssumeCapacity(checkpoint_v_key);
        try owned_shapes.append(allocator, shape);
        named_gradients[index] = .{
            .name = gradient_key,
            .data = gradient.values,
            .shape = shape,
        };
        targets[index] = .{
            .source_name = source_name,
            .trainer_slot_name = slot.name,
            .shape = shape,
            .gradient_storage_key = gradient_key,
            .checkpoint_weight_storage_key = checkpoint_weight_key,
            .checkpoint_m_storage_key = checkpoint_m_key,
            .checkpoint_v_storage_key = checkpoint_v_key,
        };
        for (gradient.values) |value| {
            const widened: f64 = value;
            gradient_sq += widened * widened;
        }
    }
    const raw_gradient_norm = @sqrt(gradient_sq);
    if (!std.math.isFinite(raw_gradient_norm) or raw_gradient_norm <= 0) {
        return error.InvalidOracleGradientNorm;
    }

    try safetensors_checkpoint.save(allocator, gradient_path, named_gradients);
    try trainer.saveTrainingCheckpoint(checkpoint_path, null, null);
    var gradient_fingerprint = try finetune.fingerprintProjectorFile(allocator, gradient_path);
    defer finetune.freeProjectorFingerprint(allocator, &gradient_fingerprint);
    var checkpoint_fingerprint = try finetune.fingerprintProjectorFile(allocator, checkpoint_path);
    defer finetune.freeProjectorFingerprint(allocator, &checkpoint_fingerprint);
    const candidate_adapter_path = try std.fs.path.join(allocator, &.{ candidate_adapter_dir, finetune.adapter_checkpoint_file_name });
    defer allocator.free(candidate_adapter_path);
    var candidate_adapter_fingerprint = try finetune.fingerprintProjectorFile(allocator, candidate_adapter_path);
    defer finetune.freeProjectorFingerprint(allocator, &candidate_adapter_fingerprint);
    const gradient_sha256 = try std.fmt.allocPrint(allocator, "sha256:{s}", .{gradient_fingerprint.sha256});
    defer allocator.free(gradient_sha256);
    const checkpoint_sha256 = try std.fmt.allocPrint(allocator, "sha256:{s}", .{checkpoint_fingerprint.sha256});
    defer allocator.free(checkpoint_sha256);
    const candidate_adapter_sha256 = try std.fmt.allocPrint(allocator, "sha256:{s}", .{candidate_adapter_fingerprint.sha256});
    defer allocator.free(candidate_adapter_sha256);

    const example = prepared.examples[request.bindings.example_index];
    try artifact_writer.writeJsonFile(allocator, capture_path, .{
        .schema_version = oracle_capture_schema_v1,
        .request_sha256 = admission.request_sha256,
        .implementation = request.implementation,
        .bindings = request.bindings,
        .training = request.training,
        .result = .{
            .loss_history = capture.loss_history.items,
            .raw_gradient_norm = raw_gradient_norm,
            .supervised_tokens = example.num_supervised_tokens,
            .logit_probes = capture.probes.?.rows,
            .targets = targets,
            .execution = capture.execution,
        },
        .artifacts = .{
            .raw_gradients = OracleCaptureArtifactV1{
                .path = gradient_file_name,
                .sha256 = gradient_sha256,
                .size_bytes = gradient_fingerprint.size_bytes,
            },
            .trainer_checkpoint = OracleCaptureArtifactV1{
                .path = checkpoint_file_name,
                .sha256 = checkpoint_sha256,
                .size_bytes = checkpoint_fingerprint.size_bytes,
            },
            .candidate_adapter_model = OracleCandidateAdapterArtifactV1{
                .sha256 = candidate_adapter_sha256,
                .size_bytes = candidate_adapter_fingerprint.size_bytes,
            },
        },
    });
    _ = io;
}

fn dimsToUsize(allocator: std.mem.Allocator, dims: []const i32) ![]usize {
    const shape = try allocator.alloc(usize, dims.len);
    errdefer allocator.free(shape);
    for (dims, shape) |dim, *value| {
        if (dim <= 0) return error.InvalidOracleTensorShape;
        value.* = @intCast(dim);
    }
    return shape;
}

fn validateAutodiffEpoch(metrics: gemma4_real.CausalLmMetrics) !void {
    if (metrics.examples_seen == 0 or metrics.supervised_tokens_seen == 0) return error.NoTrainingData;
    if (!std.math.isFinite(metrics.average_loss)) return error.NonFiniteTrainingLoss;
    if (metrics.optimizer_steps == 0) return error.NoOptimizerSteps;
}

test "gemma4 epoch timing and graph cache deltas separate steady-state work" {
    const metrics = gemma4_real.CausalLmMetrics{
        .examples_seen = 4,
        .supervised_tokens_seen = 20,
        .logical_input_tokens_seen = 200,
        .scheduled_input_tokens_seen = 256,
        .optimizer_steps = 2,
    };
    const timing = summarizeEpochTiming(10, 10 + 2 * std.time.ns_per_s, metrics);
    try std.testing.expectEqual(@as(u64, 2 * std.time.ns_per_s), timing.wall_time_ns);
    try std.testing.expectApproxEqAbs(@as(f64, 2), timing.examples_per_second, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 100), timing.logical_input_tokens_per_second, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 128), timing.scheduled_input_tokens_per_second, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10), timing.supervised_tokens_per_second, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), timing.optimizer_steps_per_second, 1e-12);
    try std.testing.expectEqual(@as(u64, 0), summarizeEpochTiming(0, 10, metrics).wall_time_ns);
    try std.testing.expectEqual(@as(u64, 0), summarizeEpochTiming(10, 10, metrics).wall_time_ns);

    const before = real_autodiff.RealAutodiffTrainer.GraphCacheStats{
        .capacity = 4,
        .build_reserve_bytes = 0,
        .builds = 2,
        .hits = 3,
        .active_reuses = 4,
        .evictions = 1,
        .resident_signatures = 2,
        .peak_resident_signatures = 3,
    };
    const after_build = real_autodiff.RealAutodiffTrainer.GraphCacheStats{
        .capacity = 4,
        .build_reserve_bytes = 0,
        .builds = 3,
        .hits = 8,
        .active_reuses = 10,
        .evictions = 1,
        .resident_signatures = 3,
        .peak_resident_signatures = 4,
    };
    const built = summarizeEpochGraphCacheDelta(before, after_build);
    try std.testing.expectEqual(@as(u64, 1), built.builds);
    try std.testing.expectEqual(@as(u64, 5), built.hits);
    try std.testing.expectEqual(@as(u64, 6), built.active_reuses);
    try std.testing.expect(!built.reuse_only);

    var after_reuse = after_build;
    after_reuse.builds = before.builds;
    const reused = summarizeEpochGraphCacheDelta(before, after_reuse);
    try std.testing.expect(reused.reuse_only);
}

test "gemma4 autodiff epochs require a finite loss and an optimizer update" {
    try validateAutodiffEpoch(.{
        .examples_seen = 1,
        .supervised_tokens_seen = 2,
        .average_loss = 1.0,
        .optimizer_steps = 1,
    });
    try std.testing.expectError(error.NoTrainingData, validateAutodiffEpoch(.{}));
    try std.testing.expectError(error.NoOptimizerSteps, validateAutodiffEpoch(.{
        .examples_seen = 1,
        .supervised_tokens_seen = 2,
        .average_loss = 1.0,
    }));
}

fn evaluateAutodiff(
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    examples: []const finetune.PreparedExampleInput,
    max_examples: usize,
    graph_config: gemma_graph.Config,
    lora_config: ml.graph.lora.LoRAConfig,
    backend_kind: gemma4_real.BackendKind,
    max_grad_norm: f32,
    length_buckets: ?gemma4_real.SequenceLengthBuckets,
    graph_cache_capacity: u8,
    gguf_projector_path: ?[]const u8,
    gguf_projector_sha256: ?[]const u8,
) !gemma4_real.CausalLmMetrics {
    const execution_policy = autodiffExecutionPolicy(backend_kind);
    const direct_gguf_base = try autodiffBaseUsesGguf(allocator, base_model_dir);
    const bootstrap = gemma4_real.findFirstSupervisedExample(examples) orelse return error.NoTrainingData;
    var backend = try gemma4_real.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();
    const is_multimodal = countMultimodalExamples(examples) > 0;
    var adapter_inspect = try finetune.inspectCheckpoint(allocator, adapter_model_dir);
    defer finetune.freeInspectionSummary(allocator, &adapter_inspect);
    const recursive_shared_block_size = adapter_inspect.recursive_shared_block_size;

    const limit = if (max_examples > 0 and max_examples < examples.len) max_examples else examples.len;
    const eval_accum_steps: u32 = @intCast(@min(limit + 1, @as(usize, std.math.maxInt(u32))));
    const eval_max_seq_len: u32 = @intCast(limitExampleSeqLen(examples, max_examples, graph_config));
    const bootstrap_seq_len = try gemma4_real.sequenceLengthForExample(
        bootstrap.num_input_tokens,
        eval_max_seq_len,
        length_buckets,
    );
    if (is_multimodal and length_buckets != null) return error.Gemma4MultimodalLengthBucketsNotSupported;

    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = 0.0 },
        .max_grad_norm = max_grad_norm,
        .grad_accum_steps = eval_accum_steps,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
        .graph_cache_capacity = graph_cache_capacity,
    });
    defer trainer.deinit();
    if (is_multimodal) {
        const projector_path = gguf_projector_path orelse return error.MissingGgufProjector;
        const projector_sha256 = gguf_projector_sha256 orelse return error.MissingPreparedProjectorFingerprint;
        const tokenizer = try gemma4_mm_real.loadTokenizerForModelDir(allocator, base_model_dir);
        var ctx = if (recursive_shared_block_size) |shared_block_size|
            gemma4_mm_real.MultimodalCtx.initRecursive(allocator, backend.backendPtr(), graph_config, projector_path, projector_sha256, tokenizer, shared_block_size)
        else
            gemma4_mm_real.MultimodalCtx.init(allocator, backend.backendPtr(), graph_config, projector_path, projector_sha256, tokenizer);
        defer ctx.deinit();
        try gemma4_mm_real.initializeTrainerFromAdapterDir(
            allocator,
            &trainer,
            &ctx,
            adapter_model_dir,
            bootstrap,
            bootstrap_seq_len,
        );
        return gemma4_mm_real.evaluatePreparedExamples(
            allocator,
            &trainer,
            &ctx,
            examples,
            max_examples,
            eval_max_seq_len,
        );
    } else {
        var ctx = if (recursive_shared_block_size) |shared_block_size|
            gemma4_real.GemmaAutodiffCtx.initRecursive(graph_config, shared_block_size)
        else
            gemma4_real.GemmaAutodiffCtx.init(graph_config);
        ctx.enable_fused_rms_norm_backward = backend_kind == .metal;
        ctx.enable_fused_gqa_attention_backward = backend_kind == .metal and gemma4_real.fusedGqaAttentionExperimentEnabled(graph_config);
        ctx.enable_fused_linear_cross_entropy = backend_kind == .metal and
            !direct_gguf_base and
            !platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_LINEAR_CCE", false);
        try gemma4_real.initializeTrainerFromAdapterDir(
            allocator,
            &trainer,
            &ctx,
            adapter_model_dir,
            bootstrap,
            bootstrap_seq_len,
        );
        return (try gemma4_real.evaluatePreparedExamplesRange(
            allocator,
            &trainer,
            &ctx,
            examples,
            .{
                .max_examples = max_examples,
                .seq_len = eval_max_seq_len,
                .sequence_length_buckets = length_buckets,
            },
        )).metrics;
    }
}

fn limitExampleSeqLen(
    examples: []const finetune.PreparedExampleInput,
    max_examples: usize,
    graph_config: gemma_graph.Config,
) usize {
    _ = graph_config;
    var max_len: usize = 1;
    const limit = if (max_examples > 0 and max_examples < examples.len) max_examples else examples.len;
    for (examples[0..limit]) |example| {
        if (example.num_input_tokens > max_len) max_len = example.num_input_tokens;
    }
    return max_len;
}

fn runSurrogate(
    io: std.Io,
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    prepared_inputs_path: []const u8,
    out_dir: []const u8,
    prepared: finetune.PreparedInputsSummary,
    opts: CliOptions,
) !void {
    if (prepared.examples_with_images > 0 or prepared.examples_with_audio > 0) return error.MultimodalRequiresAutodiffTrainer;

    const backend_ptr: ?*const ComputeBackend = null;

    var bundle = try finetune.loadLoRABundleScoped(allocator, base_model_dir, adapter_model_dir, opts.layer_name);
    defer bundle.deinit();
    var publication = try ImmutableRunPublication.init(allocator, io, out_dir);
    defer publication.deinit();

    const PjrtClientT = if (build_options.enable_pjrt) ?pjrt_mod.pjrt.Client else void;
    var pjrt_client_storage: PjrtClientT = if (comptime build_options.enable_pjrt) null else {};
    if (comptime build_options.enable_pjrt) {
        pjrt_client_storage = pjrt_mod.pjrt.Client.initFromEnv(allocator) catch |err| blk: {
            std.log.warn("PJRT client init failed ({s}); LoRA gradients will use CPU", .{@errorName(err)});
            break :blk null;
        };
    }
    defer if (comptime build_options.enable_pjrt) {
        if (pjrt_client_storage) |*client| client.deinit();
    };

    const PjrtStepsT = if (build_options.enable_pjrt) ?[]?graph_bridge.LoRAPjrtTrainStep else void;
    var pjrt_lora_steps: PjrtStepsT = if (comptime build_options.enable_pjrt) null else {};
    if (comptime build_options.enable_pjrt) {
        if (pjrt_client_storage) |*pjrt_client| {
            const steps = try allocator.alloc(?graph_bridge.LoRAPjrtTrainStep, bundle.layers.len);
            @memset(steps, null);
            var compiled_count: usize = 0;
            for (bundle.layers, 0..) |*layer, li| {
                var layer_graph = graph_bridge.LoRALinearGraph.init(
                    allocator,
                    3,
                    layer.input_dim,
                    layer.output_dim,
                    layer.rank,
                    bundle.lora_alpha,
                ) catch continue;
                steps[li] = graph_bridge.compileLoRALinearPjrtStep(allocator, &layer_graph, pjrt_client) catch blk: {
                    layer_graph.deinit();
                    break :blk null;
                };
                if (steps[li] != null) {
                    layer_graph.deinit();
                    compiled_count += 1;
                }
            }
            std.log.info("PJRT: compiled {d}/{d} LoRA layers", .{ compiled_count, bundle.layers.len });
            pjrt_lora_steps = steps;
        }
    }
    defer if (comptime build_options.enable_pjrt) {
        if (pjrt_lora_steps) |steps| {
            for (steps) |*step_opt| if (step_opt.*) |*step| step.deinit();
            allocator.free(steps);
        }
    };

    const before = try finetune.evaluatePreparedExamples(allocator, &bundle, prepared.examples, .{
        .max_examples = opts.effectiveEvalMaxExamples(),
        .layer_name = opts.layer_name,
    });

    const epoch_history = try allocator.alloc(finetune.TrainEpochSummary, opts.epochs);
    defer allocator.free(epoch_history);
    for (0..opts.epochs) |epoch_idx| {
        epoch_history[epoch_idx] = try finetune.trainPreparedExamplesEpoch(allocator, &bundle, prepared.examples, .{
            .learning_rate = opts.learning_rate,
            .max_examples = opts.max_examples,
            .layer_name = opts.layer_name,
            .max_grad_norm = opts.max_grad_norm,
            .grad_accum_steps = opts.grad_accum_steps,
            .llrd_decay = opts.llrd_decay,
            .use_schedule_free = opts.use_schedule_free,
            .compute_backend = backend_ptr,
            .pjrt_lora_steps = if (comptime build_options.enable_pjrt) pjrt_lora_steps else {},
        });
    }
    const after = try finetune.evaluatePreparedExamples(allocator, &bundle, prepared.examples, .{
        .max_examples = opts.effectiveEvalMaxExamples(),
        .layer_name = opts.layer_name,
    });

    try publication.createStaging();
    try finetune.saveLoRABundleToStaging(&bundle, publication.staging_dir);

    const report_payload = .{
        .artifact_family_version = finetune.artifact_family_version,
        .trainer_kind = "surrogate_lora_turn_aware_v2",
        .prepared_inputs_path = prepared_inputs_path,
        .saved_adapter_checkpoint = finetune.adapter_checkpoint_file_name,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .activation_checkpoint_interval = opts.activation_checkpoint_interval,
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
        .multimodal = .{
            .enabled = false,
            .gguf_projector_path = @as(?[]const u8, null),
            .examples_with_media = @as(usize, 0),
            .total_image_inputs = @as(usize, 0),
            .total_audio_inputs = @as(usize, 0),
            .total_image_soft_tokens = @as(usize, 0),
            .total_audio_soft_tokens = @as(usize, 0),
        },
        .prepared_dataset = .{
            .schema_version = prepared.schema_version,
            .examples_seen = prepared.examples_seen,
            .max_seq_len = prepared.max_seq_len,
            .max_input_tokens = prepared.max_input_tokens,
            .max_supervised_tokens = prepared.max_supervised_tokens,
            .examples_with_tool_calls = prepared.examples_with_tool_calls,
            .examples_with_tool_results = prepared.examples_with_tool_messages,
            .examples_with_multiturn = prepared.examples_with_multiturn,
            .examples_with_images = prepared.examples_with_images,
            .examples_with_audio = prepared.examples_with_audio,
            .examples_truncated = prepared.examples_truncated,
            .max_turns_dropped = prepared.max_turns_dropped,
        },
        .before = before,
        .epoch_history = epoch_history,
        .after = after,
    };
    try writeRunOutputs(io, allocator, publication.staging_dir, base_model_dir, adapter_model_dir, "surrogate", report_payload, .{
        .prepared_inputs_path = prepared_inputs_path,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .activation_checkpoint_interval = opts.activation_checkpoint_interval,
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
        .backend_label = "surrogate",
    }, true);
    try writeRunCompletionManifest(io, allocator, publication.staging_dir);
    try publication.publish();
}

const RunCompletionArtifact = struct {
    name: []const u8,
    sha256: []const u8,
    size_bytes: u64,
};

fn writeRunCompletionManifest(io: std.Io, allocator: std.mem.Allocator, staging_dir: []const u8) !void {
    const manifest_name = "run_manifest.json";
    var dir = if (std.fs.path.isAbsolute(staging_dir))
        try std.Io.Dir.openDirAbsolute(io, staging_dir, .{ .iterate = true })
    else
        try std.Io.Dir.cwd().openDir(io, staging_dir, .{ .iterate = true });
    defer dir.close(io);

    var names = std.ArrayList([]const u8).empty;
    defer {
        for (names.items) |name| allocator.free(name);
        names.deinit(allocator);
    }
    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| switch (entry.kind) {
        .file => {
            if (std.mem.eql(u8, entry.name, manifest_name)) continue;
            const owned_name = try allocator.dupe(u8, entry.name);
            names.append(allocator, owned_name) catch |err| {
                allocator.free(owned_name);
                return err;
            };
        },
        else => return error.UnsupportedArtifactEntry,
    };
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);

    const fingerprints = try allocator.alloc(finetune.ProjectorFingerprint, names.items.len);
    defer allocator.free(fingerprints);
    var completed: usize = 0;
    defer for (fingerprints[0..completed]) |*fingerprint| finetune.freeProjectorFingerprint(allocator, fingerprint);
    const artifacts = try allocator.alloc(RunCompletionArtifact, names.items.len);
    defer allocator.free(artifacts);
    for (names.items, 0..) |name, index| {
        const path = try std.fs.path.join(allocator, &.{ staging_dir, name });
        defer allocator.free(path);
        fingerprints[index] = try finetune.fingerprintProjectorFile(allocator, path);
        completed += 1;
        artifacts[index] = .{
            .name = name,
            .sha256 = fingerprints[index].sha256,
            .size_bytes = fingerprints[index].size_bytes,
        };
    }
    const manifest_path = try std.fs.path.join(allocator, &.{ staging_dir, manifest_name });
    defer allocator.free(manifest_path);
    try artifact_writer.writeJsonFile(allocator, manifest_path, .{
        .schema_version = "antfly.gemma4.finetune.run-manifest.v1",
        .status = "complete",
        .artifact_family_version = finetune.artifact_family_version,
        .artifacts = artifacts,
    });
}

test "gemma4 run completion manifest is written after every required artifact" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const staging_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "staging" });
    defer allocator.free(staging_dir);
    try std.Io.Dir.cwd().createDirPath(io, staging_dir);
    const names = [_][]const u8{
        finetune.adapter_checkpoint_file_name,
        finetune.adapter_config_file_name,
        finetune.adapter_manifest_file_name,
        "training_config.json",
        "train_eval_report.json",
        "training_report.json",
        finetune.tokenizer_config_file_name,
        finetune.tokenizer_file_name,
        finetune.special_tokens_map_file_name,
    };
    for (names) |name| {
        const path = try std.fs.path.join(allocator, &.{ staging_dir, name });
        defer allocator.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = name });
    }
    try writeRunCompletionManifest(io, allocator, staging_dir);
    const manifest_path = try std.fs.path.join(allocator, &.{ staging_dir, "run_manifest.json" });
    defer allocator.free(manifest_path);
    const manifest = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .limited(32 * 1024));
    defer allocator.free(manifest);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"complete\"") != null);
    for (names) |name| try std.testing.expect(std.mem.indexOf(u8, manifest, name) != null);
    var sorted_names = names;
    std.mem.sort([]const u8, &sorted_names, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);
    var previous_index: usize = 0;
    for (sorted_names) |name| {
        const index = std.mem.indexOf(u8, manifest, name).?;
        try std.testing.expect(index >= previous_index);
        previous_index = index;
    }
}

test "gemma4 run completion manifest rejects unexpected staging entries" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const staging_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "staging" });
    defer allocator.free(staging_dir);
    try std.Io.Dir.cwd().createDirPath(io, staging_dir);
    const support_dir = try std.fs.path.join(allocator, &.{ staging_dir, "unexpected" });
    defer allocator.free(support_dir);
    try std.Io.Dir.cwd().createDir(io, support_dir, .default_dir);
    try std.testing.expectError(error.UnsupportedArtifactEntry, writeRunCompletionManifest(io, allocator, staging_dir));
}

fn writeRunOutputs(
    io: std.Io,
    allocator: std.mem.Allocator,
    out_dir: []const u8,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    trainer_name: []const u8,
    report_payload: anytype,
    ctx: ReportContext,
    emit_summary: bool,
) !void {
    const training_config_path = try std.fs.path.join(allocator, &.{ out_dir, "training_config.json" });
    defer allocator.free(training_config_path);
    try artifact_writer.writeJsonFile(allocator, training_config_path, .{
        .contract_version = run_contract.training_config_version,
        .artifact_family_version = finetune.artifact_family_version,
        .task = "gemma4_lora_train_eval",
        .inputs = .{
            .base_model_dir = base_model_dir,
            .adapter_model_dir = adapter_model_dir,
            .prepared_inputs_path = ctx.prepared_inputs_path,
            .eval_prepared_inputs_path = ctx.eval_prepared_inputs_path,
        },
        .training = .{
            .trainer = trainer_name,
            .learning_rate = ctx.learning_rate,
            .max_examples = ctx.max_examples,
            .eval_max_examples = ctx.eval_max_examples,
            .epochs = ctx.epochs,
            .layer_name = ctx.layer_name,
            .max_grad_norm = ctx.max_grad_norm,
            .grad_accum_steps = ctx.grad_accum_steps,
            .activation_checkpoint_interval = ctx.activation_checkpoint_interval,
            .sequence_length_policy = .{
                .mode = if (ctx.sequence_length_bucket_quantum != 0) "independent-row-buckets-v1" else "fixed-prepared-maximum",
                .bucket_quantum = ctx.sequence_length_bucket_quantum,
                .bucket_minimum = ctx.sequence_length_bucket_min,
                .graph_cache_capacity = ctx.graph_cache_capacity,
            },
            .llrd_decay = ctx.llrd_decay,
            .use_schedule_free = ctx.use_schedule_free,
            .seed = ctx.seed,
            .checkpoint_path = ctx.checkpoint_path,
            .checkpoint_every_epochs = ctx.checkpoint_every_epochs,
            .resumed_from_epoch = ctx.resumed_from_epoch,
            .run_fingerprint_sha256 = ctx.run_fingerprint,
        },
        .backend_policy = .{
            .selected = ctx.backend_label,
            .preferred = ctx.backend_label,
        },
        .distributed = .{
            .enabled = false,
            .backend = ctx.backend_label,
            .rank = 0,
            .world_size = 1,
            .primary_rank = 0,
        },
    });

    const report_path = try std.fs.path.join(allocator, &.{ out_dir, "train_eval_report.json" });
    defer allocator.free(report_path);
    try artifact_writer.writeJsonFile(allocator, report_path, report_payload);

    const training_report_path = try std.fs.path.join(allocator, &.{ out_dir, "training_report.json" });
    defer allocator.free(training_report_path);
    try artifact_writer.writeJsonFile(allocator, training_report_path, .{
        .contract_version = run_contract.training_report_version,
        .artifact_family_version = finetune.artifact_family_version,
        .task = "gemma4_lora_train_eval",
        .backend_policy = .{
            .selected = ctx.backend_label,
            .preferred = ctx.backend_label,
        },
        .distributed = .{
            .enabled = false,
            .backend = ctx.backend_label,
            .rank = 0,
            .world_size = 1,
            .primary_rank = 0,
        },
        .report = report_payload,
    });

    if (emit_summary) {
        const stdout = std.Io.File.stdout();
        var buf: [4096]u8 = undefined;
        var writer = stdout.writer(io, &buf);
        try std.json.Stringify.value(.{
            .before = report_payload.before,
            .epoch_history = report_payload.epoch_history,
            .after = report_payload.after,
        }, .{ .whitespace = .indent_2 }, &writer.interface);
        try writer.interface.writeByte('\n');
        try writer.interface.flush();
    }
}

fn usageError() error{InvalidArguments} {
    printTrainUsage();
    return error.InvalidArguments;
}

pub fn printTrainUsage() void {
    std.debug.print(
        \\usage: antfly inference finetune train gemma4-lora --model <dir> --adapter <dir> --train-prepared <json> \\
        \\       --eval-prepared <json> --out <dir> --backend native|metal [options]
        \\
        \\Legacy positional form (deprecated for one release):
        \\  antfly inference finetune train gemma4-lora <model> <adapter> <train-prepared> <out> [options]
        \\
        \\Flags:
        \\  --trainer auto|autodiff             Compatibility spelling; both select production autodiff
        \\  --lr, --learning-rate <f32>         Learning rate (default: 0.001)
        \\  --max-examples <usize>              Max examples per epoch (default: 32)
        \\  --eval-prepared <path>              Required disjoint prepared evaluation artifact
        \\  --eval-max-examples <usize>         Max examples for before/after eval (default: --max-examples)
        \\  --epochs <usize>                    Number of epochs (default: 1)
        \\  --max-grad-norm <f32>               Gradient norm clipping threshold (default: 1.0, 0=disabled)
        \\  --grad-accum <u32>                  Gradient accumulation steps (default: 1)
        \\  --activation-checkpoint-interval N  Recompute between every N layer boundaries
        \\  --sequence-length-bucket-quantum N  Opt in to independent-row length buckets
        \\  --sequence-length-bucket-min N      Minimum scheduled row length (default: one quantum)
        \\  --graph-cache-capacity N            Retained shape graphs (auto: fixed=1, bucketed=4; max=8)
        \\  --seed <u64>                        Deterministic adapter/dropout/order seed (default: 42)
        \\  --checkpoint-path <file>            Mutable full trainer state outside --out and all inputs
        \\  --checkpoint-every-epochs <N>        Save at each N-epoch boundary and at the final epoch
        \\  --resume                            Resume the bound run from --checkpoint-path
        \\  --benchmark-request <json>          Internal locked benchmark contract (paired flag)
        \\  --benchmark-telemetry-out <json>    Internal no-replace benchmark telemetry (paired flag)
        \\  --oracle-request <json>             Internal locked numerical-oracle contract (paired flag)
        \\  --oracle-capture-out <dir>          Immutable raw oracle capture directory (paired flag)
        \\  --backend native|metal               Required compute backend; no implicit fallback
        \\
        \\Gemma4 production finetuning is text-only BF16 LoRA. Surrogate, media, DoRA,
        \\and packed-weight QLoRA inputs are rejected before output publication.
        \\
    , .{});
}

fn evalUsageError() error{InvalidArguments} {
    printEvalUsage();
    return error.InvalidArguments;
}

pub fn printEvalUsage() void {
    std.debug.print(
        \\usage: antfly inference finetune eval gemma4-lora --model <dir> --adapter <dir> \\
        \\       --prepared <json> --out <report.json> --backend native|metal [--max-examples N]
        \\
        \\Legacy positional form: <model> <adapter> <prepared> <out> --backend native|metal
        \\
    , .{});
}

fn validateAdapterUsageError() error{InvalidArguments} {
    printValidateAdapterUsage();
    return error.InvalidArguments;
}

pub fn printValidateAdapterUsage() void {
    std.debug.print(
        \\usage: antfly inference finetune adapter validate gemma4 --model <dir> --adapter <dir>
        \\Legacy positional form: <model> <adapter>
        \\
    , .{});
}

fn isHelpArg(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help");
}

fn countMultimodalExamples(examples: []const finetune.PreparedExampleInput) usize {
    var count: usize = 0;
    for (examples) |example| {
        if (example.image_paths.len > 0 or example.audio_paths.len > 0) count += 1;
    }
    return count;
}

fn validatePreparedProjectorFingerprint(
    prepared: finetune.PreparedInputsSummary,
    actual: finetune.ProjectorFingerprint,
) !void {
    const expected_sha256 = prepared.gguf_projector_sha256 orelse return error.MissingPreparedProjectorFingerprint;
    const expected_size = prepared.gguf_projector_size_bytes orelse return error.MissingPreparedProjectorFingerprint;
    if (!std.mem.eql(u8, expected_sha256, actual.sha256)) return error.ProjectorFingerprintMismatch;
    if (expected_size != actual.size_bytes) return error.ProjectorFingerprintMismatch;
}

fn summarizeMultimodalPrepared(examples: []const finetune.PreparedExampleInput) MultimodalPreparedStats {
    var stats = MultimodalPreparedStats{};
    for (examples) |example| {
        if (example.image_paths.len > 0 or example.audio_paths.len > 0) {
            stats.examples_with_media += 1;
        }
        stats.total_image_inputs += example.image_paths.len;
        stats.total_audio_inputs += example.audio_paths.len;
        for (example.image_token_counts) |count| stats.total_image_soft_tokens += count;
        for (example.audio_token_counts) |count| stats.total_audio_soft_tokens += count;
    }
    return stats;
}
