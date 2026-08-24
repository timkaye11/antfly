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
const platform = @import("antfly_platform");

const grpo = @import("grpo.zig");
const preference_loss = @import("preference_loss.zig");
const gemma4 = @import("gemma4.zig");
const gemma4_real_autodiff = @import("gemma4_real_autodiff.zig");
const gemma4_mm_real_autodiff = @import("gemma4_multimodal_real_autodiff.zig");
const qwen2_real_autodiff = @import("qwen2_real_autodiff.zig");
const gemma_chat_data = @import("gemma_chat_data.zig");
const real_autodiff = @import("real_autodiff_trainer.zig");
const layoutlmv3 = @import("layoutlmv3.zig");
const colqwen2 = @import("colqwen2.zig");
const reranker_data = @import("reranker_data.zig");
const reranker_head = @import("reranker_head.zig");
const reranker_lora = @import("reranker_lora.zig");
const reranker = @import("reranker.zig");
const preference_harness = @import("preference_harness.zig");
const train_eval_gemma4_lora_bundle = @import("gemma4_train_command.zig");
const train_eval_layoutlmv3_lora_sequence = @import("train/train_eval_layoutlmv3_lora_sequence.zig");
const train_eval_layoutlmv3_lora_token = @import("train/train_eval_layoutlmv3_lora_token.zig");
const train_eval_colqwen2_lora_bundle = @import("train/train_eval_colqwen2_lora_bundle.zig");
const train_eval_reranker_lora_top_layer_cached_surrogate = @import("train/train_eval_reranker_lora_top_layer_cached_surrogate.zig");
const generation = @import("../pipelines/generation.zig");
const model_manager_mod = @import("../server/model_manager.zig");
const backends = @import("../backends/backends.zig");
const session_factory = @import("../architectures/session_factory.zig");
const gpt_arch = @import("../architectures/gpt.zig");
const native_backend_choice = @import("../native_backend_choice.zig");
const tokenizer_mod = @import("inference_tokenizer");
const compat = @import("../io/compat.zig");
const c_file = @import("../util/c_file.zig");
const command_registry = @import("command_registry.zig");
const ml = @import("ml");
const peft = @import("peft.zig");
const artifact_publication = @import("artifact_publication.zig");
const gemma_preference_environment = @import("gemma4_preference_environment.zig");
const path_isolation = @import("path_isolation.zig");
const metal_compute_mod = @import("../ops/metal_compute.zig");
const metal_partition_executor = @import("../graph/metal_partition_executor.zig");

const print = std.debug.print;

const default_lora_rank: usize = 16;
const default_policy_lora_rank: usize = 8;
const default_lora_alpha: f32 = 32.0;
const default_lora_target_preset = "all-linear";
const default_gemma4_lora_target_preset = "text-all-linear";

const qwen_attention_lora_target_modules = [_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" };
const qwen_mlp_lora_target_modules = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

pub const RecipeKind = enum {
    sft,
    lora_sft,
    qlora_sft,
    dpo,
    grpo,
    reranker,
    vlm_retrieval,
};

const PreferenceExecutionMode = enum {
    train,
    score,
};

const PreferenceTask = enum {
    dpo,
    grpo,
};

pub const ModelConfig = struct {
    path: ?[]const u8 = null,
    reference_path: ?[]const u8 = null,
    name: ?[]const u8 = null,
    family: ?[]const u8 = null,
    projector_path: ?[]const u8 = null,
    /// Explicitly admits a model whose selected training weights are a direct
    /// GGUF artifact. Preference training otherwise rejects GGUF before any
    /// optimizer or output mutation.
    allow_direct_gguf_training: ?bool = null,
};

pub const DatasetConfig = struct {
    path: ?[]const u8 = null,
    train_path: ?[]const u8 = null,
    eval_path: ?[]const u8 = null,
    train_split: ?[]const u8 = "train",
    eval_split: ?[]const u8 = null,
    prepared_path: ?[]const u8 = null,
    cache_path: ?[]const u8 = null,
    train_cache_path: ?[]const u8 = null,
    eval_cache_path: ?[]const u8 = null,
    format: ?[]const u8 = null,
    labels: ?[]const u8 = null,
    max_examples: ?usize = null,
    eval_max_examples: ?usize = null,
    max_seq_len: ?usize = null,
};

pub const AdapterConfig = struct {
    path: ?[]const u8 = null,
    rank: ?usize = null,
    alpha: ?f32 = null,
    dropout: ?f32 = null,
    layer_name: ?[]const u8 = null,
    base_model_name_or_path: ?[]const u8 = null,
    quantization: ?[]const u8 = null,
    target_preset: ?[]const u8 = null,
    target_modules: ?[]const []const u8 = null,
    init_lora_weights: ?[]const u8 = null,
    initialization_seed: ?u64 = null,
    use_dora: ?bool = null,
    scaling: ?[]const u8 = null,
};

pub const OptimizerConfig = struct {
    /// Deterministic trainer/RNG seed. This is currently admitted by the
    /// optimizer-backed Gemma4 DPO/GRPO recipe paths and is fingerprinted into
    /// their durable checkpoint identity.
    seed: ?u64 = null,
    learning_rate: ?f32 = null,
    weight_decay: ?f32 = null,
    lr_scheduler: ?[]const u8 = null,
    warmup_ratio: ?f32 = null,
    warmup_steps: ?u32 = null,
    num_cycles: ?f32 = null,
    max_steps: ?usize = null,
    epochs: ?usize = null,
    micro_batch_size: ?usize = null,
    gradient_accumulation_steps: ?u32 = null,
    max_grad_norm: ?f32 = null,
    schedule_free: ?bool = null,
    llrd_decay: ?f32 = null,
};

pub const PreferenceConfig = struct {
    beta: ?f32 = null,
    simpo_gamma: ?f32 = null,
    sft_lambda: ?f32 = null,
    ipo_tau: ?f32 = null,
};

pub const GrpoConfig = struct {
    group_size: ?usize = null,
    clip_epsilon: ?f32 = null,
    kl_coef: ?f32 = null,
    /// Fail before optimizer mutation when the unweighted mean token K3
    /// divergence for a sampled group exceeds this bound. Defaults to 0.1.
    train_max_kl: ?f32 = null,
    /// Enables a proportional controller for `kl_coef`. The target and
    /// horizon are required when this is true; coefficient bounds are
    /// optional and default to [0.001, 1.0].
    adaptive_kl: ?bool = null,
    target_kl: ?f32 = null,
    kl_horizon: ?f32 = null,
    min_kl_coef: ?f32 = null,
    max_kl_coef: ?f32 = null,
    advantage_eps: ?f32 = null,
    normalize_advantage: ?bool = null,
    max_completion_tokens: ?usize = null,
    reward_mode: ?[]const u8 = null,
};

const ResolvedGrpoKlControl = struct {
    train_max_kl: f32,
    adaptive: bool,
    target_kl: ?f32,
    kl_horizon: ?f32,
    min_kl_coef: ?f32,
    max_kl_coef: ?f32,
};

fn resolveGrpoKlControl(config: GrpoConfig) !ResolvedGrpoKlControl {
    const train_max_kl = config.train_max_kl orelse 0.1;
    if (!std.math.isFinite(train_max_kl) or train_max_kl <= 0.0) {
        return error.InvalidGrpoTrainKlBudget;
    }

    const adaptive = config.adaptive_kl orelse false;
    if (!adaptive) {
        if (config.target_kl != null or config.kl_horizon != null or
            config.min_kl_coef != null or config.max_kl_coef != null)
        {
            return error.IncompleteGrpoAdaptiveKlConfig;
        }
        return .{
            .train_max_kl = train_max_kl,
            .adaptive = false,
            .target_kl = null,
            .kl_horizon = null,
            .min_kl_coef = null,
            .max_kl_coef = null,
        };
    }

    const target_kl = config.target_kl orelse return error.IncompleteGrpoAdaptiveKlConfig;
    const kl_horizon = config.kl_horizon orelse return error.IncompleteGrpoAdaptiveKlConfig;
    const min_kl_coef = config.min_kl_coef orelse 0.001;
    const max_kl_coef = config.max_kl_coef orelse 1.0;
    const initial_kl_coef = config.kl_coef orelse 0.04;
    if (!std.math.isFinite(target_kl) or target_kl <= 0.0 or target_kl >= train_max_kl or
        !std.math.isFinite(kl_horizon) or kl_horizon < 1.0 or
        !std.math.isFinite(min_kl_coef) or min_kl_coef < 0.0 or
        !std.math.isFinite(max_kl_coef) or max_kl_coef < min_kl_coef or
        initial_kl_coef <= 0.0 or initial_kl_coef < min_kl_coef or initial_kl_coef > max_kl_coef)
    {
        return error.InvalidGrpoAdaptiveKlConfig;
    }
    return .{
        .train_max_kl = train_max_kl,
        .adaptive = true,
        .target_kl = target_kl,
        .kl_horizon = kl_horizon,
        .min_kl_coef = min_kl_coef,
        .max_kl_coef = max_kl_coef,
    };
}

/// A GRPO reward is either a deterministic built-in verifier, a pinned generic
/// executable, or a pinned model-backed executable. External providers receive
/// a versioned JSON request path as their final argument and must print one JSON
/// response to stdout. Model-backed providers additionally bind every model
/// input and attest the identity and token count used for each score.
pub const RewardProviderConfig = struct {
    name: []const u8,
    kind: []const u8,
    mode: ?[]const u8 = null,
    weight: f32 = 1.0,
    executable_path: ?[]const u8 = null,
    executable_sha256: ?[]const u8 = null,
    args: ?[]const []const u8 = null,
    timeout_ms: ?u32 = null,
    min_reward: ?f32 = null,
    max_reward: ?f32 = null,
    model_path: ?[]const u8 = null,
    model_sha256: ?[]const u8 = null,
    tokenizer_path: ?[]const u8 = null,
    tokenizer_sha256: ?[]const u8 = null,
    chat_template_path: ?[]const u8 = null,
    chat_template_sha256: ?[]const u8 = null,
    calibration_dataset_path: ?[]const u8 = null,
    calibration_dataset_sha256: ?[]const u8 = null,
    max_input_tokens: ?usize = null,
    max_batch_size: ?usize = null,
};

pub const RewardConfig = struct {
    aggregation: ?[]const u8 = null,
    failure_policy: ?[]const u8 = null,
    providers: ?[]const RewardProviderConfig = null,
    trace_path: ?[]const u8 = null,
    evaluation_trace_path: ?[]const u8 = null,
    exchange_dir: ?[]const u8 = null,
    max_trace_bytes: ?usize = null,
};

pub const EntityEvalMinimums = struct {
    precision: ?f64 = null,
    recall: ?f64 = null,
    f1: f64,
    exact_match: f64,
};

/// Required quality gates for every structured task scored by the native
/// GLiNER2 total-loss evaluator. Keeping these fields non-optional makes a
/// partially specified gate set invalid at recipe parse time.
pub const FullTaskEvalMinimums = struct {
    classifications_micro_f1: f64,
    classifications_exact_match: f64,
    json_structures_micro_f1: f64,
    json_structures_exact_match: f64,
    relations_micro_f1: f64,
    relations_exact_match: f64,
    count_accuracy: f64,
};

pub const DpoEvalMinimums = struct {
    accuracy: f64,
    max_loss: f64,
    min_accuracy_improvement: ?f64 = null,
    min_reward_margin_improvement: ?f64 = null,
    min_loss_improvement: ?f64 = null,
};

pub const GrpoEvalMinimums = struct {
    mean_reward: f64,
    top_rank_mean_reward: f64,
    positive_reward_group_rate: f64,
    max_kl_loss: f64,
    min_mean_reward_improvement: ?f64 = null,
    min_top_rank_mean_reward_improvement: ?f64 = null,
    min_positive_reward_group_rate_improvement: ?f64 = null,
};

pub const EvalConfig = struct {
    path: ?[]const u8 = null,
    max_examples: ?usize = null,
    split: ?[]const u8 = null,
    every_epochs: ?u32 = null,
    batch_size: ?u32 = null,
    early_stopping_patience: ?u32 = null,
    improvement_threshold: ?f64 = null,
    /// Full-task structured scoring currently requires the Zig native
    /// evaluator even when training itself runs through the Metal runtime.
    backend: ?[]const u8 = null,
    entity_minimums: ?EntityEvalMinimums = null,
    full_task_minimums: ?FullTaskEvalMinimums = null,
    dpo_minimums: ?DpoEvalMinimums = null,
    grpo_minimums: ?GrpoEvalMinimums = null,
};

pub const CheckpointConfig = struct {
    every_epochs: ?u32 = null,
    keep_last: ?u32 = null,
    resume_path: ?[]const u8 = null,
};

pub const RuntimeConfig = struct {
    compiled_required: ?bool = null,
    graph_cache_capacity: ?u8 = null,
    /// Gemma4 SFT independently rounds each causal row. Gemma4 DPO rounds the
    /// maximum chosen/rejected row so both halves of a preference pair retain
    /// one compiled signature. Null preserves the fixed prepared maximum.
    sequence_length_bucket_quantum: ?u32 = null,
    /// Optional minimum row length; the CLI defaults to one quantum.
    sequence_length_bucket_min: ?u32 = null,
    /// Exact paged-KV token-selection lane for multi-token Gemma4 GRPO.
    /// Cumulative sampler state is included in preference checkpoints; live
    /// pages must be quiescent at every durable boundary.
    grpo_incremental_kv: ?bool = null,
    /// Batches active candidates at the same decode position.
    grpo_incremental_kv_batch_active: ?bool = null,
    /// Fans out the final segmented prompt page on device.
    grpo_incremental_kv_clone_prompt_tail: ?bool = null,
    /// Runs one exact legacy shadow group before optimizer mutation.
    grpo_incremental_kv_shadow_exact: ?bool = null,
};

/// DPO and GRPO deliberately expose both optimizer-backed training and
/// metrics-only scoring. Requiring the caller to name that intent prevents a
/// missing dataset format or adapter field from silently changing the job's
/// semantics.
pub const ExecutionConfig = struct {
    mode: ?[]const u8 = null,
};

pub const ArtifactConfig = struct {
    root: ?[]const u8 = null,
    manifest_path: ?[]const u8 = null,
    prepared_path: ?[]const u8 = null,
    adapter_dir: ?[]const u8 = null,
    trained_adapter_dir: ?[]const u8 = null,
    materialized_dir: ?[]const u8 = null,
    validation_report_path: ?[]const u8 = null,
    evaluation_report_path: ?[]const u8 = null,
    reload_report_path: ?[]const u8 = null,
    report_path: ?[]const u8 = null,
};

pub const Recipe = struct {
    recipe: ?[]const u8 = null,
    kind: ?[]const u8 = null,
    model: ModelConfig = .{},
    dataset: DatasetConfig = .{},
    adapter: ?AdapterConfig = null,
    optimizer: OptimizerConfig = .{},
    preference: PreferenceConfig = .{},
    grpo: GrpoConfig = .{},
    reward: ?RewardConfig = null,
    eval: ?EvalConfig = null,
    checkpoint: ?CheckpointConfig = null,
    runtime: ?RuntimeConfig = null,
    execution: ExecutionConfig = .{},
    artifacts: ArtifactConfig = .{},
    backend: ?[]const u8 = null,
    trainer: ?[]const u8 = null,
};

pub const Step = struct {
    kind: StepKind = .command,
    name: []const u8,
    argv: []const []const u8,
};

pub const StepKind = enum {
    command,
    direct_sft,
    direct_dpo,
    direct_grpo,
};

pub const Plan = struct {
    steps: []Step,
};

const RunStatus = enum {
    planned,
    running,
    succeeded,
    failed,
};

const StepManifest = struct {
    index: usize,
    name: []const u8,
    argv: []const []const u8,
    status: RunStatus = .planned,
    exit_code: ?u8 = null,
    stdout_bytes: ?usize = null,
    stderr_bytes: ?usize = null,
};

const RunManifest = struct {
    schema_version: []const u8 = "antfly_inference_finetune_recipe_run/v1",
    status: RunStatus,
    recipe: Recipe,
    artifact_root: ?[]const u8,
    steps: []const StepManifest,
};

const TrainingConfigFile = struct {
    schema_version: []const u8 = "antfly_inference_finetune_training_config/v1",
    recipe: Recipe,
    steps: []const StepManifest,
    metadata: StaticMetadata,
};

const TrainingReportFile = struct {
    schema_version: []const u8 = "antfly_inference_finetune_training_report/v1",
    status: RunStatus,
    recipe: Recipe,
    artifact_root: ?[]const u8,
    steps: []const StepManifest,
    metadata: ReportMetadata,
};

const PathFingerprint = struct {
    label: []const u8,
    path: []const u8,
    exists: bool,
    kind: ?[]const u8 = null,
    size_bytes: ?u64 = null,
    entries: ?usize = null,
    digest: ?[]const u8 = null,
};

const BackendBuildInfo = struct {
    inference_version: []const u8,
    enable_native: bool,
    enable_onnx: bool,
    enable_mlx: bool,
    enable_pjrt: bool,
    skip_openapi: bool,
};

const BackendMetadata = struct {
    requested: ?[]const u8,
    build: BackendBuildInfo,
};

const OptimizerSummary = struct {
    seed: ?u64,
    learning_rate: ?f32,
    weight_decay: ?f32,
    lr_scheduler: ?[]const u8,
    warmup_ratio: ?f32,
    warmup_steps: ?u32,
    num_cycles: ?f32,
    max_steps: ?usize,
    epochs: ?usize,
    micro_batch_size: ?usize,
    gradient_accumulation_steps: ?u32,
    max_grad_norm: ?f32,
    schedule_free: ?bool,
    llrd_decay: ?f32,
};

const StaticMetadata = struct {
    dataset_fingerprints: []const PathFingerprint,
    backend: BackendMetadata,
    optimizer: OptimizerSummary,
};

const ReportMetadata = struct {
    dataset_fingerprints: []const PathFingerprint,
    backend: BackendMetadata,
    optimizer: OptimizerSummary,
    artifact_checksums: ?[]const PathFingerprint = null,
};

const PlannedPath = struct {
    label: []const u8,
    path: []const u8,
};

const DirectoryDigestEntry = struct {
    relative_path: []const u8,
    size_bytes: u64,
    digest: []const u8,
};

const DirectoryDigest = struct {
    digest: []const u8,
    size_bytes: u64,
    entries: usize,
};

const canonical_preference_evaluation_policy =
    "terminal-device-drained-host-weight-snapshot-fresh-backend-private-buffer-reuse-disabled";

/// Resolved trajectory-affecting Metal arithmetic policy. Several qualified
/// kernels retain environment rollback switches; persisting and fingerprinting
/// the resolved values prevents two numerically different runs from sharing a
/// checkpoint identity merely because their JSON recipes are identical.
const GemmaMetalNumericalPolicy = struct {
    schema_version: []const u8 = "antfly_gemma4_metal_numerical_policy/v2",
    fingerprint_flags: u64,
    sparse_loss_chunk_rows: u32,
    linear_cce_tile_vocab: usize,
    fused_rms_norm_backward: bool,
    fused_gqa_attention_backward: bool,
    fused_linear_cross_entropy: bool,
    sparse_logits_cross_entropy: bool,
    bf16_tiled32_m16: bool,
    bf16_simdgroup_mm: bool,
    bf16_simdgroup_m64: bool,
    bf16_forward_simdgroup_m64_packed: bool,
    bf16_simdgroup_m64_prefix_tail: bool,
    bf16_backward_tiled32_m16: bool,
    bf16_backward_small_rows: bool,
    bf16_backward_simdgroup_mm: bool,
    bf16_backward_simdgroup_m64: bool,
    bf16_backward_simdgroup_m64_coalesced: bool,
    bf16_backward_simdgroup_m64_packed: bool,
    rms_norm_backward_simdgroup: bool,
    rms_norm_backward_residual_add: bool,
    rms_norm_generated: bool,
    linear_cce_f16_grad: bool,
    linear_cce_logit_cache: bool,
    linear_cce_f16_mps_backward: bool,
    dense_mps_linear: bool,
    gemma4_bf16_mlp_fusion: bool,
    gemma4_gate_up_backward_input_sum: bool,
    q4_0_linear_rms_add_sumsq: bool,
    eager_rank1_dot_specialization: bool,
    dense_device_dot_general: bool,
    lora_forward_fused_branch: bool,
    lora_forward_generic_rank16: bool,
    lora_forward_rank1_fused: bool,
    reference_quant_linear: bool,
    quant_backward_force_barriers: bool,
    contiguous_slice_device_view: bool,
    partition_fused_patterns: bool,
    partition_runtime_commands: bool,
    runtime_region_plan: bool,
    grouped_mps_dot: bool,
    gather_promote_input: bool,
    reduce_promote_input: bool,
    lora_backward_runtime_region: bool,
    low_rank_lora_backward_runtime_region: bool,
    rank_adapter_backward_runtime_region: bool,
    ffn_gelu_backward_runtime_region: bool,
    gated_gelu_backward_runtime_region: bool,
    gated_gelu_forward_fusion: bool,
    masked_softmax_runtime_region: bool,
    softmax_backward_runtime_region: bool,
    graph_rank1_dot_specialization: bool,
    raw_linear_bias_pair_runtime_region: bool,
    raw_linear_runtime_regions_suppressed: bool,
    gated_ffn_graph_fusion: bool,
    gemma_gated_mlp_training_graph_fusion: bool,
    attention_output_residual_graph_fusion: bool,
    grouped_lora_a_r16: bool,
    add3_fusion: bool,
};

const DpoReport = struct {
    schema_version: []const u8 = "antfly_inference_finetune_dpo_report/v6",
    execution_mode: []const u8,
    dataset_format: []const u8,
    examples: usize,
    loss: f32,
    mean_reward_margin: f32,
    accuracy: f32,
    beta: f32,
    training_seed: u64 = 42,
    policy_backend: ?[]const u8 = null,
    optimizer_steps: ?u64 = null,
    micro_batch_steps: ?u64 = null,
    policy_scoring_mode: ?[]const u8 = null,
    training_microbatch_mode: ?[]const u8 = null,
    device_gradient_snapshot_mode: ?[]const u8 = null,
    activation_checkpointing_mode: ?[]const u8 = null,
    activation_checkpointing_layer_interval: ?u32 = null,
    metal_buffer_reuse_mode: ?[]const u8 = null,
    metal_completion_cache: ?DpoMetalCompletionCacheTelemetry = null,
    reference_mode: ?[]const u8 = null,
    reference_precompute_seconds: ?f64 = null,
    initial_logprob_parity: ?DpoInitialLogprobParity = null,
    initial_bucket_signature_parity: ?DpoInitialBucketSignatureParity = null,
    sequence_length_policy: ?DpoPairLengthPolicyTelemetry = null,
    graph_cache: ?DpoGraphCacheTelemetry = null,
    benchmark: ?DpoBenchmarkTelemetry = null,
    checkpoint_resume: ?PreferenceCheckpointResumeSummary = null,
    metal_numerical_policy: ?GemmaMetalNumericalPolicy = null,
    evaluation_execution_policy: ?[]const u8 = null,
    baseline_evaluation: ?DpoEvaluationSummary = null,
    baseline_relative: ?DpoBaselineRelativeSummary = null,
    evaluation: ?DpoEvaluationSummary = null,
    trained_adapter_dir: ?[]const u8 = null,
};

const DpoEvaluationSummary = struct {
    report_path: []const u8,
    examples: usize,
    loss: f32,
    mean_reward_margin: f32,
    accuracy: f32,
    passed: bool,
};

const DpoBaselineRelativeSummary = struct {
    accuracy_improvement: f32,
    reward_margin_improvement: f32,
    loss_improvement: f32,
    passed: bool,
};

fn compareDpoToBaseline(
    baseline: DpoEvaluationSummary,
    evaluation: DpoEvaluationSummary,
    minimums: DpoEvalMinimums,
) DpoBaselineRelativeSummary {
    const accuracy_improvement = evaluation.accuracy - baseline.accuracy;
    const reward_margin_improvement = evaluation.mean_reward_margin - baseline.mean_reward_margin;
    const loss_improvement = baseline.loss - evaluation.loss;
    return .{
        .accuracy_improvement = accuracy_improvement,
        .reward_margin_improvement = reward_margin_improvement,
        .loss_improvement = loss_improvement,
        .passed = accuracy_improvement >= minimums.min_accuracy_improvement.? and
            reward_margin_improvement >= minimums.min_reward_margin_improvement.? and
            loss_improvement >= minimums.min_loss_improvement.?,
    };
}

const DpoEvaluationReport = struct {
    schema_version: []const u8 = "antfly_inference_finetune_dpo_evaluation/v2",
    status: []const u8,
    dataset_path: []const u8,
    dataset_fingerprint: PathFingerprint,
    policy_adapter_digest: []const u8,
    policy_backend: []const u8,
    execution_policy: []const u8 = canonical_preference_evaluation_policy,
    metal_numerical_policy: ?GemmaMetalNumericalPolicy = null,
    examples: usize,
    prompt_overlap_count: usize,
    loss: f32,
    mean_reward_margin: f32,
    accuracy: f32,
    minimums: DpoEvalMinimums,
    reference_mode: []const u8,
    sequence_length_policy: ?DpoPairLengthPolicyTelemetry = null,
};

const DpoPairLengthPolicyTelemetry = struct {
    mode: []const u8,
    scope: []const u8,
    maximum_sequence_length: u32,
    bucket_quantum: ?u32 = null,
    bucket_minimum: ?u32 = null,
    graph_cache_capacity: usize,
    pairs: usize,
    logical_branch_rows: usize,
    scheduled_branch_rows: usize,
    fixed_shape_branch_rows: usize,
    padding_rows_avoided: usize,
    padding_reduction_fraction: f64,
    minimum_pair_sequence_length: u32,
    maximum_pair_sequence_length: u32,
    unique_pair_sequence_lengths: usize,
    unique_pair_graph_signatures: ?usize = null,
    weighted_target_row_policy: []const u8,
};

const DpoGraphCacheTelemetry = struct {
    after_initialization: real_autodiff.RealAutodiffTrainer.GraphCacheStats,
    after_reference_precompute: real_autodiff.RealAutodiffTrainer.GraphCacheStats,
    after_initial_bucket_signature_parity: real_autodiff.RealAutodiffTrainer.GraphCacheStats,
    after_training: real_autodiff.RealAutodiffTrainer.GraphCacheStats,
    after_evaluation: real_autodiff.RealAutodiffTrainer.GraphCacheStats,
};

const DpoMetalCompletionCacheTelemetry = struct {
    budget_policy: []const u8 = "metal-recommended-working-set-9/16;env-overridable",
    enabled: bool,
    max_bytes: u64,
    available_bytes: u64,
    available_slots: u64,
    peak_bytes: u64,
    peak_slots: u64,
    requests: u64,
    hits: u64,
    misses: u64,
    retired: u64,
    evictions: u64,
    completed_generation: u64,
};

const DpoInitialLogprobParity = struct {
    policy_chosen_logp: f32,
    policy_rejected_logp: f32,
    reference_chosen_logp: f32,
    reference_rejected_logp: f32,
    max_abs_error: f32,
    base_equivalent_policy: bool,
};

const DpoInitialBucketSignatureParity = struct {
    graph_signatures_checked: usize,
    representative_pair_index: usize,
    policy_chosen_logp: f32,
    policy_rejected_logp: f32,
    reference_chosen_logp: f32,
    reference_rejected_logp: f32,
    max_abs_error: f32,
    base_equivalent_policy: bool,
};

const DpoBenchmarkProtocol = struct {
    cold: usize,
    first: usize,
    warmup: usize,
    measured: usize,
};

const DpoBenchmarkTelemetry = struct {
    protocol: DpoBenchmarkProtocol,
    cold_seconds: f64,
    cold_loss: f32,
    first_seconds: f64,
    first_loss: f32,
    warmup_seconds: []const f64,
    warmup_losses: []const f32,
    measured_seconds: []const f64,
    measured_losses: []const f32,
    median_seconds: f64,
    mean_seconds: f64,
};

const SftReport = struct {
    schema_version: []const u8 = "antfly_inference_finetune_sft_report/v1",
    examples: usize,
    supervised_tokens: usize,
    loss: f32,
    epochs: usize,
    trained_adapter_dir: []const u8,
};

const GrpoReport = struct {
    schema_version: []const u8 = "antfly_inference_finetune_grpo_report/v6",
    execution_mode: []const u8,
    dataset_format: []const u8,
    completions: usize,
    tokens: usize,
    groups: usize,
    loss: f32,
    pg_loss: f32,
    kl_loss: f32,
    mean_kl: ?f32 = null,
    clip_fraction: f32,
    mean_reward: ?f32 = null,
    reward_stddev: ?f32 = null,
    training_seed: u64 = 42,
    policy_backend: ?[]const u8 = null,
    optimizer_steps: ?u64 = null,
    micro_batch_steps: ?u64 = null,
    sampling_mode: ?[]const u8 = null,
    policy_logprob_mode: ?[]const u8 = null,
    policy_rescore_completions: ?usize = null,
    training_microbatch_mode: ?[]const u8 = null,
    training_microbatch_batch_size: ?usize = null,
    training_physical_micro_batches_per_group: ?usize = null,
    sampling_seconds: ?f64 = null,
    incremental_kv: ?gemma4_real_autodiff.GrpoIncrementalKvTelemetry = null,
    policy_rescore_seconds: ?f64 = null,
    backward_update_seconds: ?f64 = null,
    reference_mode: ?[]const u8 = null,
    reference_scoring_seconds: ?f64 = null,
    reference_cache: ?GrpoReferenceCacheTelemetry = null,
    initial_logprob_parity: ?GrpoInitialLogprobParity = null,
    kl_control: ?GrpoKlControlTelemetry = null,
    benchmark: ?GrpoBenchmarkTelemetry = null,
    checkpoint_resume: ?PreferenceCheckpointResumeSummary = null,
    reward_pipeline: ?RewardPipelineTelemetry = null,
    metal_numerical_policy: ?GemmaMetalNumericalPolicy = null,
    evaluation_execution_policy: ?[]const u8 = null,
    baseline_evaluation: ?GrpoEvaluationSummary = null,
    baseline_relative: ?GrpoBaselineRelativeSummary = null,
    evaluation: ?GrpoEvaluationSummary = null,
    trained_adapter_dir: ?[]const u8 = null,
};

const PreferenceCheckpointResumeSummary = struct {
    enabled: bool,
    start_epoch: usize,
    checkpoint_path: ?[]const u8,
    checkpoint_state_path: ?[]const u8,
    checkpoint_state_sha256: ?[]const u8,
    checkpoint_epoch: ?usize,
    checkpoint_every_epochs: ?u32,
    run_fingerprint_sha256: []const u8,
    restored_micro_batch_steps: u64,
    restored_optimizer_steps: u64,
    restored_accumulation_micro_batches: u32,
};

const DpoCheckpointAggregates = struct {
    initial_adapter_digest: [std.crypto.hash.sha2.Sha256.digest_length]u8,
    examples_seen: usize,
    total_loss: f64,
    total_margin: f64,
    total_accuracy: f64,
    initial_logprob_parity: ?DpoInitialLogprobParity = null,
    initial_bucket_signature_parity: ?DpoInitialBucketSignatureParity = null,
};

const GrpoCheckpointAggregates = struct {
    initial_adapter_digest: [std.crypto.hash.sha2.Sha256.digest_length]u8,
    total_loss: f64,
    total_pg_loss: f64,
    total_kl_loss: f64,
    total_mean_kl: f64,
    total_clip_fraction: f64,
    total_groups: usize,
    total_completions: usize,
    total_tokens: usize,
    total_reward: f64,
    total_reward_squared: f64,
    saw_nonzero_reward_advantage: bool,
    saw_nonzero_policy_gradient: bool,
    initial_sampling_rescore_max_abs_error: f32,
    initial_policy_reference_max_abs_error: f32,
    initial_base_equivalent_policy: bool,
    captured_initial_logprob_parity: bool,
    policy_rescore_completions: usize,
    diagnostic_first_tokens: [8]i32,
    diagnostic_policy_first_token_logps: [8]f32,
    diagnostic_reference_first_token_logps: [8]f32,
    diagnostic_first_token_count: usize,
    kl_current_coef: f32,
    kl_admitted_groups: usize,
    kl_max_observed_mean: f32,
    kl_trace: []const u8,
    reward_call_index: usize,
    reward_external_calls: usize,
    reward_external_failures: usize,
    reward_trace: []const u8,
    /// Cumulative exact-sampler counters. Live KV pages are intentionally not
    /// serialized; checkpoints are admitted only after the sampler proves all
    /// logical and device sequence lifetimes have ended.
    incremental_kv: ?gemma4_real_autodiff.GrpoIncrementalKvTelemetry = null,
};

const PreferenceCheckpointState = struct {
    schema_version: []const u8 = "antfly_gemma4_preference_checkpoint_state/v1",
    task: []const u8,
    run_fingerprint_sha256: []const u8,
    epoch_index: usize,
    micro_batch_steps: u64,
    optimizer_steps: u64,
    accumulation_micro_batches: u32,
    dpo: ?DpoCheckpointAggregates = null,
    grpo: ?GrpoCheckpointAggregates = null,
};

const LoadedPreferenceCheckpointState = struct {
    parsed: std.json.Parsed(PreferenceCheckpointState),
    path: []u8,

    fn deinit(self: *LoadedPreferenceCheckpointState, allocator: std.mem.Allocator) void {
        self.parsed.deinit();
        allocator.free(self.path);
        self.* = undefined;
    }
};

const PreferenceCheckpointArtifactSummary = struct {
    state_path: []u8,
    state_sha256: []const u8,
    epoch: usize,

    fn deinit(self: *PreferenceCheckpointArtifactSummary, allocator: std.mem.Allocator) void {
        allocator.free(self.state_path);
        allocator.free(self.state_sha256);
        self.* = undefined;
    }
};

const RewardPipelineTelemetry = struct {
    aggregation: []const u8,
    failure_policy: []const u8,
    providers: usize,
    external_calls: usize,
    external_failures: usize,
    configuration_digest: []const u8,
    trace_path: ?[]const u8 = null,
    trace_digest: ?[]const u8 = null,
};

const GrpoEvaluationSummary = struct {
    report_path: []const u8,
    groups: usize,
    completions: usize,
    mean_reward: f32,
    top_rank_mean_reward: f32,
    positive_reward_group_rate: f32,
    reward_stddev: f32,
    kl_loss: f32,
    mean_kl: f32,
    sampling_seconds: ?f64 = null,
    reference_scoring_seconds: ?f64 = null,
    reward_loss_seconds: ?f64 = null,
    loop_seconds: ?f64 = null,
    passed: bool,
};

const GrpoBaselineRelativeSummary = struct {
    mean_reward_improvement: f32,
    top_rank_mean_reward_improvement: f32,
    positive_reward_group_rate_improvement: f32,
    passed: bool,
};

fn compareGrpoToBaseline(
    baseline: GrpoEvaluationSummary,
    evaluation: GrpoEvaluationSummary,
    minimums: GrpoEvalMinimums,
) GrpoBaselineRelativeSummary {
    const mean_reward_improvement = evaluation.mean_reward - baseline.mean_reward;
    const top_rank_mean_reward_improvement = evaluation.top_rank_mean_reward - baseline.top_rank_mean_reward;
    const positive_reward_group_rate_improvement = evaluation.positive_reward_group_rate - baseline.positive_reward_group_rate;
    return .{
        .mean_reward_improvement = mean_reward_improvement,
        .top_rank_mean_reward_improvement = top_rank_mean_reward_improvement,
        .positive_reward_group_rate_improvement = positive_reward_group_rate_improvement,
        .passed = mean_reward_improvement >= minimums.min_mean_reward_improvement.? and
            top_rank_mean_reward_improvement >= minimums.min_top_rank_mean_reward_improvement.? and
            positive_reward_group_rate_improvement >= minimums.min_positive_reward_group_rate_improvement.?,
    };
}

const GrpoEvaluationReport = struct {
    schema_version: []const u8 = "antfly_inference_finetune_grpo_evaluation/v3",
    status: []const u8,
    dataset_path: []const u8,
    dataset_fingerprint: PathFingerprint,
    policy_adapter_digest: []const u8,
    policy_backend: []const u8,
    execution_policy: []const u8 = canonical_preference_evaluation_policy,
    metal_numerical_policy: ?GemmaMetalNumericalPolicy = null,
    groups: usize,
    completions: usize,
    tokens: usize,
    prompt_overlap_count: usize,
    mean_reward: f32,
    top_rank_mean_reward: f32,
    positive_reward_group_rate: f32,
    reward_stddev: f32,
    loss: f32,
    pg_loss: f32,
    kl_loss: f32,
    mean_kl: f32,
    clip_fraction: f32,
    minimums: GrpoEvalMinimums,
    reference_mode: []const u8,
    execution_order: ?[]const u8 = null,
    sampling_seconds: ?f64 = null,
    incremental_kv: ?gemma4_real_autodiff.GrpoIncrementalKvTelemetry = null,
    reference_scoring_seconds: ?f64 = null,
    reward_loss_seconds: ?f64 = null,
    loop_seconds: ?f64 = null,
    reward_pipeline: ?RewardPipelineTelemetry = null,
};

const GrpoReferenceCacheTelemetry = struct {
    capacity: usize,
    entries: usize,
    hits: usize,
    misses: usize,
};

const GrpoInitialLogprobParity = struct {
    sampling_rescore_max_abs_error: f32,
    policy_reference_max_abs_error: f32,
    base_equivalent_policy: bool,
    completion_first_token_ids: []const i32,
    policy_first_token_logps: []const f32,
    reference_first_token_logps: []const f32,
};

const GrpoKlControlTelemetry = struct {
    mode: []const u8,
    train_max_kl: f32,
    target_kl: ?f32,
    kl_horizon: ?f32,
    initial_kl_coef: f32,
    final_kl_coef: f32,
    min_kl_coef: ?f32,
    max_kl_coef: ?f32,
    admitted_groups: usize,
    max_observed_mean_kl: f32,
    trace_path: []const u8,
    trace_digest: ?[]const u8,
};

const GrpoKlTraceRecord = struct {
    schema_version: []const u8 = "antfly_inference_grpo_kl_control_trace/v1",
    group_index: usize,
    epoch_index: usize,
    prompt_index: usize,
    optimizer_steps_before: u64,
    status: []const u8,
    mean_kl: f32,
    weighted_kl_loss: f32,
    train_max_kl: f32,
    target_kl: ?f32,
    kl_coef_before: f32,
    kl_coef_after: f32,
};

const GrpoKlControl = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    resolved: ResolvedGrpoKlControl,
    initial_kl_coef: f32,
    current_kl_coef: f32,
    controller: ?grpo.AdaptiveKLController,
    trace_path: []const u8,
    trace: std.ArrayList(u8) = .empty,
    trace_digest: ?[]const u8 = null,
    admitted_groups: usize = 0,
    max_observed_mean_kl: f32 = 0.0,
    finished: bool = false,

    fn init(allocator: std.mem.Allocator, io: std.Io, recipe: Recipe) !GrpoKlControl {
        const resolved = try resolveGrpoKlControl(recipe.grpo);
        const initial_kl_coef = recipe.grpo.kl_coef orelse 0.04;
        const trace_path = try grpoKlTracePath(allocator, recipe);
        errdefer allocator.free(trace_path);
        const controller = if (resolved.adaptive)
            try grpo.AdaptiveKLController.init(initial_kl_coef, .{
                .target = resolved.target_kl.?,
                .horizon = resolved.kl_horizon.?,
                .min_coef = resolved.min_kl_coef.?,
                .max_coef = resolved.max_kl_coef.?,
            })
        else
            null;
        return .{
            .allocator = allocator,
            .io = io,
            .resolved = resolved,
            .initial_kl_coef = initial_kl_coef,
            .current_kl_coef = initial_kl_coef,
            .controller = controller,
            .trace_path = trace_path,
        };
    }

    fn deinit(self: *GrpoKlControl) void {
        self.trace.deinit(self.allocator);
        self.allocator.free(self.trace_path);
        if (self.trace_digest) |digest| self.allocator.free(digest);
        self.* = undefined;
    }

    fn restoreCheckpoint(
        self: *GrpoKlControl,
        current_kl_coef: f32,
        admitted_groups: usize,
        max_observed_mean_kl: f32,
        trace: []const u8,
    ) !void {
        if (self.finished or self.trace.items.len != 0 or self.admitted_groups != 0 or
            !std.math.isFinite(current_kl_coef) or current_kl_coef < 0.0 or
            !std.math.isFinite(max_observed_mean_kl) or max_observed_mean_kl < 0.0 or
            trace.len > 16 * 1024 * 1024)
        {
            return error.InvalidPreferenceCheckpointState;
        }
        if (self.resolved.adaptive) {
            if (current_kl_coef < self.resolved.min_kl_coef.? or
                current_kl_coef > self.resolved.max_kl_coef.?)
            {
                return error.InvalidPreferenceCheckpointState;
            }
            self.controller.?.value = current_kl_coef;
        } else if (@as(u32, @bitCast(current_kl_coef)) != @as(u32, @bitCast(self.initial_kl_coef))) {
            return error.InvalidPreferenceCheckpointState;
        }
        try self.trace.appendSlice(self.allocator, trace);
        self.current_kl_coef = current_kl_coef;
        self.admitted_groups = admitted_groups;
        self.max_observed_mean_kl = max_observed_mean_kl;
    }

    /// Admits or rejects one group before optimizer mutation and returns the
    /// coefficient to use for the next group.
    fn observe(
        self: *GrpoKlControl,
        group_index: usize,
        epoch_index: usize,
        prompt_index: usize,
        optimizer_steps_before: u64,
        mean_kl: f32,
        weighted_kl_loss: f32,
    ) !f32 {
        const coefficient_before = self.current_kl_coef;
        const admitted = std.math.isFinite(mean_kl) and mean_kl >= 0.0 and
            mean_kl <= self.resolved.train_max_kl;
        var coefficient_after = coefficient_before;
        if (admitted) {
            self.max_observed_mean_kl = @max(self.max_observed_mean_kl, mean_kl);
            if (self.controller) |*controller| {
                coefficient_after = try controller.update(mean_kl, 1);
            }
        }
        try self.append(GrpoKlTraceRecord{
            .group_index = group_index,
            .epoch_index = epoch_index,
            .prompt_index = prompt_index,
            .optimizer_steps_before = optimizer_steps_before,
            .status = if (admitted) "admitted" else "budget-exceeded",
            .mean_kl = mean_kl,
            .weighted_kl_loss = weighted_kl_loss,
            .train_max_kl = self.resolved.train_max_kl,
            .target_kl = self.resolved.target_kl,
            .kl_coef_before = coefficient_before,
            .kl_coef_after = coefficient_after,
        });
        if (!admitted) {
            print(
                "grpo rejected optimizer group {d}: raw mean KL {d:.8} exceeds train_max_kl {d:.8}; no optimizer mutation was admitted\n",
                .{ group_index, mean_kl, self.resolved.train_max_kl },
            );
            try self.finish();
            return error.GrpoTrainKlBudgetExceeded;
        }
        self.admitted_groups += 1;
        self.current_kl_coef = coefficient_after;
        return coefficient_after;
    }

    fn append(self: *GrpoKlControl, record: GrpoKlTraceRecord) !void {
        const rendered = try std.json.Stringify.valueAlloc(self.allocator, record, .{});
        defer self.allocator.free(rendered);
        const record_size = std.math.add(usize, rendered.len, 1) catch return error.GrpoKlTraceLimitExceeded;
        const next_size = std.math.add(usize, self.trace.items.len, record_size) catch
            return error.GrpoKlTraceLimitExceeded;
        if (next_size > 16 * 1024 * 1024) return error.GrpoKlTraceLimitExceeded;
        try self.trace.ensureTotalCapacity(self.allocator, next_size);
        try self.trace.appendSlice(self.allocator, rendered);
        try self.trace.append(self.allocator, '\n');
    }

    fn finish(self: *GrpoKlControl) !void {
        if (self.finished) return;
        try artifact_publication.writeFileAtomicReplace(self.allocator, self.io, self.trace_path, self.trace.items);
        self.trace_digest = try sha256FileAlloc(self.allocator, self.io, self.trace_path);
        self.finished = true;
    }

    fn telemetry(self: *const GrpoKlControl) GrpoKlControlTelemetry {
        return .{
            .mode = if (self.resolved.adaptive) "adaptive" else "fixed-with-hard-budget",
            .train_max_kl = self.resolved.train_max_kl,
            .target_kl = self.resolved.target_kl,
            .kl_horizon = self.resolved.kl_horizon,
            .initial_kl_coef = self.initial_kl_coef,
            .final_kl_coef = self.current_kl_coef,
            .min_kl_coef = self.resolved.min_kl_coef,
            .max_kl_coef = self.resolved.max_kl_coef,
            .admitted_groups = self.admitted_groups,
            .max_observed_mean_kl = self.max_observed_mean_kl,
            .trace_path = self.trace_path,
            .trace_digest = self.trace_digest,
        };
    }
};

const GrpoBenchmarkUpdate = struct {
    seconds: f64,
    loss: f32,
    pg_loss: f32,
    kl_loss: f32,
    mean_reward: f32,
    reward_stddev: f32,
    completion_tokens: usize,
    policy_reference_max_abs_error: f32,
};

const GrpoBenchmarkProtocol = struct {
    cold: usize,
    first: usize,
    warmup: usize,
    measured: usize,
};

const GrpoBenchmarkTelemetry = struct {
    protocol: GrpoBenchmarkProtocol,
    cold: GrpoBenchmarkUpdate,
    first: GrpoBenchmarkUpdate,
    warmup: []const GrpoBenchmarkUpdate,
    measured: []const GrpoBenchmarkUpdate,
    median_seconds: f64,
    mean_seconds: f64,
};

const FastSmokeMode = enum {
    dry_run,
    execute,
    subprocess_execute,
};

const FastSmokeSetup = enum {
    none,
    synthetic_qwen2_dpo_execute,
    synthetic_qwen2_grpo_execute,
    synthetic_gemma_dpo_execute,
    synthetic_gemma_grpo_execute,
};

const FastSmokeCase = struct {
    name: []const u8,
    recipe_path: []const u8,
    mode: FastSmokeMode,
    setup: FastSmokeSetup = .none,
};

const FastSmokeCaseResult = struct {
    name: []const u8,
    recipe_path: []const u8,
    mode: FastSmokeMode,
    status: RunStatus,
    manifest_path: ?[]const u8 = null,
    training_report_path: ?[]const u8 = null,
};

const FastSmokeSummary = struct {
    schema_version: []const u8 = "antfly_inference_finetune_fast_smoke/v1",
    status: RunStatus,
    output_root: []const u8,
    cases: []const FastSmokeCaseResult,
};

const FastSmokeRecipeOverrides = struct {
    model_path: ?[]const u8 = null,
    reference_path: ?[]const u8 = null,
    dataset_path: ?[]const u8 = null,
    dataset_format: ?[]const u8 = null,
    train_path: ?[]const u8 = null,
    eval_path: ?[]const u8 = null,
    labels: ?[]const u8 = null,
    backend: ?[]const u8 = null,
    max_examples: ?usize = null,
    eval_max_examples: ?usize = null,
    max_seq_len: ?usize = null,
};

const SyntheticQwen2Assets = struct {
    model_dir: []const u8,
    dpo_path: []const u8,
    grpo_path: []const u8,
};

const SyntheticGemmaAssets = struct {
    model_dir: []const u8,
    dpo_path: []const u8,
    dpo_eval_path: []const u8,
    grpo_path: []const u8,
    grpo_eval_path: []const u8,
};

const WriteTensorF32 = struct {
    name: []const u8,
    shape: []const usize,
    data: []const f32,
};

pub fn main(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    if (args.len >= 1 and std.mem.eql(u8, args[0], "smoke-fast")) {
        return runFastSmoke(allocator, io, args[1..]);
    }
    if (args.len < 2 or !std.mem.eql(u8, args[0], "run")) {
        usage();
        return;
    }

    const recipe_path = args[1];
    var dry_run = false;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--dry-run") or std.mem.eql(u8, args[i], "--plan")) {
            dry_run = true;
        } else {
            return usageError();
        }
    }

    var parsed = try loadRecipe(allocator, io, recipe_path);
    defer parsed.deinit();

    const recipe = parsed.value;
    var plan_arena = std.heap.ArenaAllocator.init(allocator);
    defer plan_arena.deinit();
    const plan = try buildPlan(plan_arena.allocator(), recipe);

    try printPlan(io, recipe, plan);
    if (dry_run) return;

    const exe_dir = try std.process.executableDirPathAlloc(io, allocator);
    defer allocator.free(exe_dir);
    const manifest_path = try manifestPath(allocator, recipe);
    defer allocator.free(manifest_path);
    const training_config_path = try defaultArtifactPath(allocator, recipe, "training_config.json");
    defer allocator.free(training_config_path);
    const training_report_path = try defaultArtifactPath(allocator, recipe, "training_report.json");
    defer allocator.free(training_report_path);
    print("manifest: {s}\n", .{manifest_path});
    print("training config: {s}\n", .{training_config_path});
    print("training report: {s}\n", .{training_report_path});
    try runPlan(allocator, io, exe_dir, recipe, plan, manifest_path, training_config_path, training_report_path);
}

pub fn loadRecipe(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Parsed(Recipe) {
    const raw = try readFileMax(allocator, io, path, 32 * 1024 * 1024);
    defer allocator.free(raw);
    return std.json.parseFromSlice(Recipe, allocator, raw, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = false,
    });
}

fn runFastSmoke(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    var out_root: []const u8 = "/tmp/antfly-inference-finetune-smoke-fast";
    var selected_case: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--out-root")) {
            i += 1;
            if (i >= args.len) return usageError();
            out_root = args[i];
        } else if (std.mem.eql(u8, args[i], "--case")) {
            i += 1;
            if (i >= args.len or selected_case != null) return usageError();
            selected_case = args[i];
        } else {
            return usageError();
        }
    }

    try std.Io.Dir.cwd().createDirPath(io, out_root);
    const summary_path = try std.fs.path.join(allocator, &.{ out_root, "fast_smoke_summary.json" });
    defer allocator.free(summary_path);

    const cases = [_]FastSmokeCase{
        .{ .name = "gemma4_dry_run", .recipe_path = "pkg/inference/testdata/recipe_gemma4_lora.json", .mode = .dry_run },
        .{ .name = "layoutlmv3_dry_run", .recipe_path = "pkg/inference/testdata/recipe_layoutlmv3_lora_token.json", .mode = .dry_run },
        .{ .name = "reranker_head_dry_run", .recipe_path = "pkg/inference/testdata/recipe_reranker_head.json", .mode = .dry_run },
        .{ .name = "reranker_lora_dry_run", .recipe_path = "pkg/inference/testdata/recipe_reranker_lora.json", .mode = .dry_run },
        .{ .name = "colqwen2_dry_run", .recipe_path = "pkg/inference/testdata/recipe_colqwen2_vlm_retrieval.json", .mode = .dry_run },
        .{ .name = "dpo_text_dry_run", .recipe_path = "pkg/inference/testdata/recipe_dpo_text_preference_native_fast.json", .mode = .dry_run },
        .{ .name = "dpo_text_gemma_dry_run", .recipe_path = "pkg/inference/testdata/recipe_dpo_text_preference_gemma_fast.json", .mode = .dry_run },
        .{ .name = "dpo_rendered_text_gemma_dry_run", .recipe_path = "pkg/inference/testdata/recipe_dpo_rendered_text_preference_gemma_fast.json", .mode = .dry_run },
        .{ .name = "dpo_text_qwen2_dry_run", .recipe_path = "pkg/inference/testdata/recipe_dpo_text_preference_qwen2_fast.json", .mode = .dry_run },
        .{ .name = "grpo_text_dry_run", .recipe_path = "pkg/inference/testdata/recipe_grpo_text_native_fast.json", .mode = .dry_run },
        .{ .name = "grpo_text_gemma_dry_run", .recipe_path = "pkg/inference/testdata/recipe_grpo_text_gemma_fast.json", .mode = .dry_run },
        .{ .name = "grpo_rendered_text_gemma_dry_run", .recipe_path = "pkg/inference/testdata/recipe_grpo_rendered_text_gemma_fast.json", .mode = .dry_run },
        .{ .name = "grpo_text_qwen2_dry_run", .recipe_path = "pkg/inference/testdata/recipe_grpo_text_qwen2_fast.json", .mode = .dry_run },
        .{ .name = "grpo_text_colqwen2_dry_run", .recipe_path = "pkg/inference/testdata/recipe_grpo_text_colqwen2_fast.json", .mode = .dry_run },
        .{ .name = "grpo_ci_text_dry_run", .recipe_path = "pkg/inference/testdata/recipe_grpo_text_ci_native_fast.json", .mode = .dry_run },
        .{ .name = "grpo_prefix_text_dry_run", .recipe_path = "pkg/inference/testdata/recipe_grpo_text_prefix_native_fast.json", .mode = .dry_run },
        .{ .name = "qwen2_dpo_execute", .recipe_path = "pkg/inference/testdata/recipe_dpo_text_preference_qwen2_fast.json", .mode = .subprocess_execute, .setup = .synthetic_qwen2_dpo_execute },
        .{ .name = "qwen2_grpo_execute", .recipe_path = "pkg/inference/testdata/recipe_grpo_text_qwen2_fast.json", .mode = .execute, .setup = .synthetic_qwen2_grpo_execute },
        .{ .name = "gemma4_dpo_execute", .recipe_path = "pkg/inference/testdata/recipe_dpo_text_preference_gemma_fast.json", .mode = .execute, .setup = .synthetic_gemma_dpo_execute },
        .{ .name = "gemma4_grpo_execute", .recipe_path = "pkg/inference/testdata/recipe_grpo_text_gemma_fast.json", .mode = .execute, .setup = .synthetic_gemma_grpo_execute },
        .{ .name = "dpo_scalar_execute", .recipe_path = "pkg/inference/testdata/recipe_dpo_scalar.json", .mode = .execute },
        .{ .name = "grpo_scalar_execute", .recipe_path = "pkg/inference/testdata/recipe_grpo_scalar.json", .mode = .execute },
    };

    var selected_count: usize = 0;
    for (cases) |case| {
        if (selected_case == null or std.mem.eql(u8, selected_case.?, case.name)) selected_count += 1;
    }
    if (selected_count == 0) return error.UnknownFastSmokeCase;

    var results = try allocator.alloc(FastSmokeCaseResult, selected_count);
    var initialized_results: usize = 0;
    defer freeFastSmokeResults(allocator, results, initialized_results);
    for (cases) |case| {
        if (selected_case) |name| {
            if (!std.mem.eql(u8, name, case.name)) continue;
        }
        results[initialized_results] = try runFastSmokeCase(allocator, io, out_root, case);
        initialized_results += 1;
    }

    const overall = blk: {
        for (results) |result| {
            if (result.status != .succeeded) break :blk RunStatus.failed;
        }
        break :blk RunStatus.succeeded;
    };
    try writeJsonFile(allocator, io, summary_path, FastSmokeSummary{
        .status = overall,
        .output_root = out_root,
        .cases = results[0..initialized_results],
    });
    print("fast smoke summary: {s}\n", .{summary_path});
    if (overall != .succeeded) return error.FinetuneStepFailed;
}

fn runFastSmokeCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_root: []const u8,
    case: FastSmokeCase,
) !FastSmokeCaseResult {
    print("fast-smoke {s}: {s}\n", .{ case.name, case.recipe_path });
    var path_arena = std.heap.ArenaAllocator.init(allocator);
    defer path_arena.deinit();
    const path_allocator = path_arena.allocator();
    const case_root = try std.fs.path.join(allocator, &.{ out_root, case.name });
    defer allocator.free(case_root);
    const overrides = try setupFastSmokeCase(allocator, io, case_root, case.setup);
    defer freeFastSmokeRecipeOverrides(allocator, overrides);
    const recipe_path = try resolveCwdPath(path_allocator, io, case.recipe_path);
    var parsed = try loadRecipe(allocator, io, recipe_path);
    defer parsed.deinit();
    var recipe = parsed.value;
    try normalizeFastSmokeRecipePaths(path_allocator, io, &recipe);
    applyFastSmokeRecipeOverrides(&recipe, overrides);

    if (case.mode == .dry_run) {
        var plan_arena = std.heap.ArenaAllocator.init(allocator);
        defer plan_arena.deinit();
        const plan = try buildPlan(plan_arena.allocator(), recipe);
        try printPlan(io, recipe, plan);
        return .{
            .name = case.name,
            .recipe_path = case.recipe_path,
            .mode = case.mode,
            .status = .succeeded,
        };
    }

    recipe.artifacts.root = case_root;
    recipe.artifacts.manifest_path = null;
    recipe.artifacts.report_path = null;
    recipe.artifacts.prepared_path = null;
    recipe.artifacts.adapter_dir = null;
    recipe.artifacts.trained_adapter_dir = null;
    recipe.artifacts.materialized_dir = null;

    var plan_arena = std.heap.ArenaAllocator.init(allocator);
    defer plan_arena.deinit();
    const plan = try buildPlan(plan_arena.allocator(), recipe);
    const exe_dir = try std.process.executableDirPathAlloc(io, allocator);
    defer allocator.free(exe_dir);
    const manifest_path = try manifestPath(allocator, recipe);
    defer allocator.free(manifest_path);
    const training_config_path = try defaultArtifactPath(allocator, recipe, "training_config.json");
    defer allocator.free(training_config_path);
    const training_report_path = try defaultArtifactPath(allocator, recipe, "training_report.json");
    defer allocator.free(training_report_path);
    if (case.mode == .subprocess_execute) {
        const smoke_recipe_path = try std.fs.path.join(allocator, &.{ case_root, "smoke_recipe.json" });
        defer allocator.free(smoke_recipe_path);
        try writeJsonFile(allocator, io, smoke_recipe_path, recipe);
        const antfly_path = try std.fs.path.join(allocator, &.{ exe_dir, "antfly" });
        defer allocator.free(antfly_path);
        const result = try std.process.run(allocator, io, .{
            .argv = &.{ antfly_path, "inference", "finetune", "run", smoke_recipe_path },
            .stdout_limit = .limited(16 * 1024 * 1024),
            .stderr_limit = .limited(16 * 1024 * 1024),
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        if (result.stdout.len > 0) print("{s}", .{result.stdout});
        if (result.stderr.len > 0) print("{s}", .{result.stderr});
        switch (result.term) {
            .exited => |code| if (code != 0) return error.FinetuneStepFailed,
            else => return error.FinetuneStepFailed,
        }
        const dpo_path = try dpoReportPath(allocator, recipe);
        defer allocator.free(dpo_path);
        const trained_dir = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
        defer if (recipe.artifacts.trained_adapter_dir == null and recipe.artifacts.adapter_dir == null) allocator.free(trained_dir);
        try expectPathExists(io, dpo_path);
        try expectPathExists(io, trained_dir);
        try expectRunStatusFile(allocator, io, manifest_path, "succeeded");
        try expectRunStatusFile(allocator, io, training_report_path, "succeeded");
        return .{
            .name = case.name,
            .recipe_path = case.recipe_path,
            .mode = case.mode,
            .status = .succeeded,
            .manifest_path = try allocator.dupe(u8, manifest_path),
            .training_report_path = try allocator.dupe(u8, training_report_path),
        };
    }
    try runPlan(allocator, io, exe_dir, recipe, plan, manifest_path, training_config_path, training_report_path);
    try expectRunStatusFile(allocator, io, manifest_path, "succeeded");
    try expectRunStatusFile(allocator, io, training_report_path, "succeeded");
    return .{
        .name = case.name,
        .recipe_path = case.recipe_path,
        .mode = case.mode,
        .status = .succeeded,
        .manifest_path = try allocator.dupe(u8, manifest_path),
        .training_report_path = try allocator.dupe(u8, training_report_path),
    };
}

fn freeFastSmokeResults(allocator: std.mem.Allocator, results: []FastSmokeCaseResult, initialized_results: usize) void {
    std.debug.assert(initialized_results <= results.len);
    for (results[0..initialized_results]) |result| {
        if (result.manifest_path) |path| allocator.free(path);
        if (result.training_report_path) |path| allocator.free(path);
    }
    allocator.free(results);
}

fn freeFastSmokeRecipeOverrides(allocator: std.mem.Allocator, overrides: FastSmokeRecipeOverrides) void {
    var freed: [5][]const u8 = undefined;
    var freed_len: usize = 0;
    freeUniqueOptionalPath(allocator, &freed, &freed_len, overrides.model_path);
    freeUniqueOptionalPath(allocator, &freed, &freed_len, overrides.reference_path);
    freeUniqueOptionalPath(allocator, &freed, &freed_len, overrides.dataset_path);
    freeUniqueOptionalPath(allocator, &freed, &freed_len, overrides.train_path);
    freeUniqueOptionalPath(allocator, &freed, &freed_len, overrides.eval_path);
}

fn freeUniqueOptionalPath(
    allocator: std.mem.Allocator,
    freed: *[5][]const u8,
    freed_len: *usize,
    maybe_path: ?[]const u8,
) void {
    const path = maybe_path orelse return;
    for (freed[0..freed_len.*]) |existing| {
        if (existing.ptr == path.ptr and existing.len == path.len) return;
    }
    allocator.free(path);
    freed[freed_len.*] = path;
    freed_len.* += 1;
}

fn normalizeFastSmokeRecipePaths(allocator: std.mem.Allocator, io: std.Io, recipe: *Recipe) !void {
    recipe.dataset.path = try resolveOptionalCwdPath(allocator, io, recipe.dataset.path);
    recipe.dataset.train_path = try resolveOptionalCwdPath(allocator, io, recipe.dataset.train_path);
    recipe.dataset.eval_path = try resolveOptionalCwdPath(allocator, io, recipe.dataset.eval_path);
    recipe.dataset.prepared_path = try resolveOptionalCwdPath(allocator, io, recipe.dataset.prepared_path);
    recipe.dataset.cache_path = try resolveOptionalCwdPath(allocator, io, recipe.dataset.cache_path);
    recipe.dataset.train_cache_path = try resolveOptionalCwdPath(allocator, io, recipe.dataset.train_cache_path);
    recipe.dataset.eval_cache_path = try resolveOptionalCwdPath(allocator, io, recipe.dataset.eval_cache_path);
    if (recipe.eval) |*eval| eval.path = try resolveOptionalCwdPath(allocator, io, eval.path);
}

fn resolveOptionalCwdPath(allocator: std.mem.Allocator, io: std.Io, maybe_path: ?[]const u8) !?[]const u8 {
    const path = maybe_path orelse return null;
    return try resolveCwdPath(allocator, io, path);
}

fn resolveCwdPath(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    return resolvePathFromDir(allocator, io, std.Io.Dir.cwd(), path);
}

fn resolvePathFromDir(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) ![]const u8 {
    if (path.len == 0 or std.fs.path.isAbsolute(path)) return try allocator.dupe(u8, path);
    if (dirPathExists(dir, io, path)) return try allocator.dupe(u8, path);

    const package_prefix = "pkg/inference/";
    if (std.mem.startsWith(u8, path, package_prefix)) {
        const package_relative = path[package_prefix.len..];
        if (dirPathExists(dir, io, package_relative)) return try allocator.dupe(u8, package_relative);

        const repo_relative = try std.fs.path.join(allocator, &.{ "zig", path });
        if (dirPathExists(dir, io, repo_relative)) return repo_relative;
        allocator.free(repo_relative);
    } else {
        const package_relative = try std.fs.path.join(allocator, &.{ package_prefix[0 .. package_prefix.len - 1], path });
        if (dirPathExists(dir, io, package_relative)) return package_relative;
        allocator.free(package_relative);

        const repo_relative = try std.fs.path.join(allocator, &.{ "zig", package_prefix[0 .. package_prefix.len - 1], path });
        if (dirPathExists(dir, io, repo_relative)) return repo_relative;
        allocator.free(repo_relative);
    }
    return try allocator.dupe(u8, path);
}

fn cwdPathExists(io: std.Io, path: []const u8) bool {
    return dirPathExists(std.Io.Dir.cwd(), io, path);
}

fn dirPathExists(dir: std.Io.Dir, io: std.Io, path: []const u8) bool {
    dir.access(io, path, .{}) catch return false;
    return true;
}

fn setupFastSmokeCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    case_root: []const u8,
    setup: FastSmokeSetup,
) !FastSmokeRecipeOverrides {
    return switch (setup) {
        .none => .{},
        .synthetic_qwen2_dpo_execute => blk: {
            const assets = try writeSyntheticQwen2SmokeAssets(allocator, io, case_root);
            allocator.free(assets.grpo_path);
            break :blk .{
                .model_path = assets.model_dir,
                .reference_path = assets.model_dir,
                .dataset_path = assets.dpo_path,
                .dataset_format = "rendered-text-preference",
                .backend = "auto",
                .max_examples = 1,
                .max_seq_len = 32,
            };
        },
        .synthetic_qwen2_grpo_execute => blk: {
            const assets = try writeSyntheticQwen2SmokeAssets(allocator, io, case_root);
            allocator.free(assets.dpo_path);
            break :blk .{
                .model_path = assets.model_dir,
                .reference_path = assets.model_dir,
                .dataset_path = assets.grpo_path,
                .dataset_format = "rendered-text-grpo",
                .backend = "auto",
                .max_examples = 1,
                .max_seq_len = 32,
            };
        },
        .synthetic_gemma_dpo_execute => blk: {
            const assets = try writeSyntheticGemmaSmokeAssets(allocator, io, case_root);
            allocator.free(assets.grpo_path);
            allocator.free(assets.grpo_eval_path);
            break :blk .{
                .model_path = assets.model_dir,
                .reference_path = assets.model_dir,
                .dataset_path = assets.dpo_path,
                .eval_path = assets.dpo_eval_path,
                .dataset_format = "rendered-text-preference",
                .backend = "auto",
                .max_examples = 1,
                .eval_max_examples = 1,
                .max_seq_len = 32,
            };
        },
        .synthetic_gemma_grpo_execute => blk: {
            const assets = try writeSyntheticGemmaSmokeAssets(allocator, io, case_root);
            allocator.free(assets.dpo_path);
            allocator.free(assets.dpo_eval_path);
            break :blk .{
                .model_path = assets.model_dir,
                .reference_path = assets.model_dir,
                .dataset_path = assets.grpo_path,
                .eval_path = assets.grpo_eval_path,
                .dataset_format = "rendered-text-grpo",
                .backend = "auto",
                .max_examples = 1,
                .eval_max_examples = 1,
                .max_seq_len = 32,
            };
        },
    };
}

fn applyFastSmokeRecipeOverrides(recipe: *Recipe, overrides: FastSmokeRecipeOverrides) void {
    if (overrides.model_path) |value| recipe.model.path = value;
    if (overrides.reference_path) |value| recipe.model.reference_path = value;
    if (overrides.dataset_path) |value| recipe.dataset.path = value;
    if (overrides.dataset_format) |value| recipe.dataset.format = value;
    if (overrides.train_path) |value| recipe.dataset.train_path = value;
    if (overrides.eval_path) |value| {
        if (recipe.eval) |*eval| {
            eval.path = value;
            recipe.dataset.eval_path = null;
        } else {
            recipe.dataset.eval_path = value;
        }
    }
    if (overrides.labels) |value| recipe.dataset.labels = value;
    if (overrides.backend) |value| recipe.backend = value;
    if (overrides.max_examples) |value| recipe.dataset.max_examples = value;
    if (overrides.eval_max_examples) |value| recipe.dataset.eval_max_examples = value;
    if (overrides.max_seq_len) |value| recipe.dataset.max_seq_len = value;
}

fn writeSyntheticQwen2SmokeAssets(allocator: std.mem.Allocator, io: std.Io, case_root: []const u8) !SyntheticQwen2Assets {
    const assets_root = try std.fs.path.join(allocator, &.{ case_root, "synthetic_qwen2" });
    defer allocator.free(assets_root);
    try std.Io.Dir.cwd().createDirPath(io, assets_root);

    const model_dir = try std.fs.path.join(allocator, &.{ assets_root, "model" });
    errdefer allocator.free(model_dir);
    try std.Io.Dir.cwd().createDirPath(io, model_dir);

    try writeOwnedTextFile(allocator, io, try std.fs.path.join(allocator, &.{ model_dir, "config.json" }),
        \\{"model_type":"qwen2","vocab_size":200000,"hidden_size":32,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":64,"max_position_embeddings":32,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}
    );

    try copySmokeArtifactFromQwenTokenizerBundle(allocator, io, model_dir, "tokenizer.json");
    try copySmokeArtifactFromQwenTokenizerBundle(allocator, io, model_dir, "tokenizer_config.json");
    try writeOwnedTextFile(allocator, io, try std.fs.path.join(allocator, &.{ model_dir, "special_tokens_map.json" }),
        \\{"bos_token":"<|endoftext|>","eos_token":"<|im_end|>","pad_token":"<|endoftext|>"}
    );

    const checkpoint_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(checkpoint_path);
    try writeSyntheticQwen2Checkpoint(allocator, checkpoint_path);

    const dpo_path = try std.fs.path.join(allocator, &.{ assets_root, "dpo.jsonl" });
    errdefer allocator.free(dpo_path);
    const grpo_path = try std.fs.path.join(allocator, &.{ assets_root, "grpo.jsonl" });
    errdefer allocator.free(grpo_path);
    try writeTextFile(io, dpo_path,
        \\{"prompt":"Answer with one word: yes or no?\nAnswer:","chosen":" yes","rejected":" no"}
    );
    try writeTextFile(io, grpo_path,
        \\{"prompt":"Answer with one word: yes\nAnswer:","target":"yes"}
    );

    return .{
        .model_dir = model_dir,
        .dpo_path = dpo_path,
        .grpo_path = grpo_path,
    };
}

fn writeSyntheticGemmaSmokeAssets(allocator: std.mem.Allocator, io: std.Io, case_root: []const u8) !SyntheticGemmaAssets {
    const assets_root = try std.fs.path.join(allocator, &.{ case_root, "synthetic_gemma4" });
    defer allocator.free(assets_root);
    try std.Io.Dir.cwd().createDirPath(io, assets_root);

    const model_dir = try std.fs.path.join(allocator, &.{ assets_root, "model" });
    errdefer allocator.free(model_dir);
    try std.Io.Dir.cwd().createDirPath(io, model_dir);

    try writeOwnedTextFile(allocator, io, try std.fs.path.join(allocator, &.{ model_dir, "config.json" }),
        \\{"model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"attention_head_dim":8,"intermediate_size":64,"max_position_embeddings":32,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":false,"vocab_size":13}
    );

    try writeOwnedSyntheticPreferenceTokenizerArtifact(allocator, io, model_dir, "tokenizer.json");
    try writeOwnedSyntheticPreferenceTokenizerArtifact(allocator, io, model_dir, "tokenizer_config.json");
    try writeOwnedTextFile(allocator, io, try std.fs.path.join(allocator, &.{ model_dir, "special_tokens_map.json" }),
        \\{"bos_token":"<bos>","eos_token":"<eos>","pad_token":"<pad>"}
    );

    const checkpoint_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(checkpoint_path);
    try writeSyntheticGemmaCheckpoint(allocator, checkpoint_path);

    const dpo_path = try std.fs.path.join(allocator, &.{ assets_root, "dpo.jsonl" });
    errdefer allocator.free(dpo_path);
    const dpo_eval_path = try std.fs.path.join(allocator, &.{ assets_root, "dpo_eval.jsonl" });
    errdefer allocator.free(dpo_eval_path);
    const grpo_path = try std.fs.path.join(allocator, &.{ assets_root, "grpo.jsonl" });
    errdefer allocator.free(grpo_path);
    const grpo_eval_path = try std.fs.path.join(allocator, &.{ assets_root, "grpo_eval.jsonl" });
    errdefer allocator.free(grpo_eval_path);
    try writeTextFile(io, dpo_path,
        \\{"prompt":"Answer with one word: yes or no?\nAnswer:","chosen":" yes","rejected":" no"}
    );
    try writeTextFile(io, grpo_path,
        \\{"prompt":"Answer with one word: yes\nAnswer:","target":"yes"}
    );
    try writeTextFile(io, dpo_eval_path,
        \\{"prompt":"Answer yes\nAnswer:","chosen":" yes","rejected":" no"}
    );
    try writeTextFile(io, grpo_eval_path,
        \\{"prompt":"Answer yes\nAnswer:","target":"yes"}
    );

    return .{
        .model_dir = model_dir,
        .dpo_path = dpo_path,
        .dpo_eval_path = dpo_eval_path,
        .grpo_path = grpo_path,
        .grpo_eval_path = grpo_eval_path,
    };
}

fn copySmokeArtifactFromQwenTokenizerBundle(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir: []const u8,
    file_name: []const u8,
) !void {
    const src_root = "/tmp/antfly-inference-models/Qwen/Qwen2.5-0.5B-Instruct-GGUF";
    const src_path = try std.fs.path.join(allocator, &.{ src_root, file_name });
    defer allocator.free(src_path);
    const dst_path = try std.fs.path.join(allocator, &.{ out_dir, file_name });
    defer allocator.free(dst_path);
    if (c_file.fileExists(allocator, src_path)) {
        const contents = try c_file.readFile(allocator, src_path);
        defer allocator.free(contents);
        try compat.cwd().writeFile(io, .{ .sub_path = dst_path, .data = contents });
        return;
    }

    try writeSyntheticPreferenceTokenizerArtifact(io, dst_path, file_name);
}

fn writeSyntheticPreferenceTokenizerArtifact(io: std.Io, dst_path: []const u8, file_name: []const u8) !void {
    if (std.mem.eql(u8, file_name, "tokenizer.json")) {
        try writeTextFile(io, dst_path,
            \\{
            \\  "version":"1.0",
            \\  "normalizer":{"type":"BertNormalizer","lowercase":true},
            \\  "pre_tokenizer":{"type":"WhitespaceSplit"},
            \\  "added_tokens":[
            \\    {"id":0,"content":"<pad>"},
            \\    {"id":1,"content":"<unk>"},
            \\    {"id":2,"content":"<bos>"},
            \\    {"id":3,"content":"<eos>"}
            \\  ],
            \\  "model":{
            \\    "type":"Unigram",
            \\    "unk_id":1,
            \\    "vocab":[
            \\      ["<pad>",0.0],["<unk>",0.0],["<bos>",0.0],["<eos>",0.0],
            \\      ["answer",0.0],["with",0.0],["one",0.0],["word:",0.0],["yes",0.0],["or",0.0],["no?",0.0],["answer:",0.0],["no",0.0]
            \\    ]
            \\  }
            \\}
        );
        return;
    }
    if (std.mem.eql(u8, file_name, "tokenizer_config.json")) {
        try writeTextFile(io, dst_path,
            \\{"model_max_length":32,"unk_token":"<unk>","pad_token":"<pad>","bos_token":"<bos>","eos_token":"<eos>"}
        );
        return;
    }
    return error.FileNotFound;
}

fn writeSyntheticQwen2Checkpoint(allocator: std.mem.Allocator, path: []const u8) !void {
    const hidden: usize = 32;
    const intermediate: usize = 64;
    const vocab: usize = 200000;
    const num_layers: usize = 1;
    const q_dim: usize = 32;
    const kv_dim: usize = 16;

    var tensors: std.ArrayList(WriteTensorF32) = .empty;
    defer tensors.deinit(allocator);
    var owned_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (owned_names.items) |name| allocator.free(name);
        owned_names.deinit(allocator);
    }

    try tensors.append(allocator, .{ .name = "model.embed_tokens.weight", .shape = &.{ vocab, hidden }, .data = try makeRampF32(allocator, vocab * hidden, 0.00001) });
    try tensors.append(allocator, .{ .name = "model.norm.weight", .shape = &.{hidden}, .data = try makeFilledF32(allocator, hidden, 1.0) });

    var layer: usize = 0;
    while (layer < num_layers) : (layer += 1) {
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.input_layernorm.weight", layer, &.{hidden}, try makeFilledF32(allocator, hidden, 1.0));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.q_proj.weight", layer, &.{ q_dim, hidden }, try makeRampF32(allocator, q_dim * hidden, 0.0002));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.q_proj.bias", layer, &.{q_dim}, try makeFilledF32(allocator, q_dim, 0.0));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.k_proj.weight", layer, &.{ kv_dim, hidden }, try makeRampF32(allocator, kv_dim * hidden, 0.00025));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.k_proj.bias", layer, &.{kv_dim}, try makeFilledF32(allocator, kv_dim, 0.0));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.v_proj.weight", layer, &.{ kv_dim, hidden }, try makeRampF32(allocator, kv_dim * hidden, 0.0003));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.v_proj.bias", layer, &.{kv_dim}, try makeFilledF32(allocator, kv_dim, 0.0));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.o_proj.weight", layer, &.{ hidden, q_dim }, try makeRampF32(allocator, hidden * q_dim, 0.00035));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.post_attention_layernorm.weight", layer, &.{hidden}, try makeFilledF32(allocator, hidden, 1.0));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.mlp.gate_proj.weight", layer, &.{ intermediate, hidden }, try makeRampF32(allocator, intermediate * hidden, 0.0004));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.mlp.up_proj.weight", layer, &.{ intermediate, hidden }, try makeRampF32(allocator, intermediate * hidden, 0.00045));
        try appendOwnedQwenTensor(allocator, &tensors, &owned_names, "model.layers.{d}.mlp.down_proj.weight", layer, &.{ hidden, intermediate }, try makeRampF32(allocator, hidden * intermediate, 0.0005));
    }

    try writeHeaderAndTensorsF32(allocator, path, tensors.items);
}

fn writeSyntheticGemmaCheckpoint(allocator: std.mem.Allocator, path: []const u8) !void {
    const hidden: usize = 32;
    const intermediate: usize = 64;
    const vocab: usize = 13;
    const num_layers: usize = 1;
    const q_dim: usize = 32;
    const kv_dim: usize = 16;
    const head_dim: usize = 8;

    var tensors: std.ArrayList(WriteTensorF32) = .empty;
    defer tensors.deinit(allocator);
    var owned_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (owned_names.items) |name| allocator.free(name);
        owned_names.deinit(allocator);
    }

    try tensors.append(allocator, .{ .name = "model.embed_tokens.weight", .shape = &.{ vocab, hidden }, .data = try makeRampF32(allocator, vocab * hidden, 0.00001) });
    try tensors.append(allocator, .{ .name = "lm_head.weight", .shape = &.{ vocab, hidden }, .data = try makeSyntheticGemmaPreferenceLmHead(allocator, vocab, hidden) });
    try tensors.append(allocator, .{ .name = "model.norm.weight", .shape = &.{hidden}, .data = try makeFilledF32(allocator, hidden, 1.0) });

    var layer: usize = 0;
    while (layer < num_layers) : (layer += 1) {
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.input_layernorm.weight", layer, &.{hidden}, try makeFilledF32(allocator, hidden, 1.0));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.q_proj.weight", layer, &.{ q_dim, hidden }, try makeRampF32(allocator, q_dim * hidden, 0.0002));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.k_proj.weight", layer, &.{ kv_dim, hidden }, try makeRampF32(allocator, kv_dim * hidden, 0.00025));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.v_proj.weight", layer, &.{ kv_dim, hidden }, try makeRampF32(allocator, kv_dim * hidden, 0.0003));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.o_proj.weight", layer, &.{ hidden, q_dim }, try makeRampF32(allocator, hidden * q_dim, 0.00035));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.q_norm.weight", layer, &.{head_dim}, try makeFilledF32(allocator, head_dim, 1.0));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.self_attn.k_norm.weight", layer, &.{head_dim}, try makeFilledF32(allocator, head_dim, 1.0));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.post_attention_layernorm.weight", layer, &.{hidden}, try makeFilledF32(allocator, hidden, 1.0));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.pre_feedforward_layernorm.weight", layer, &.{hidden}, try makeFilledF32(allocator, hidden, 1.0));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.mlp.gate_proj.weight", layer, &.{ intermediate, hidden }, try makeRampF32(allocator, intermediate * hidden, 0.0004));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.mlp.up_proj.weight", layer, &.{ intermediate, hidden }, try makeRampF32(allocator, intermediate * hidden, 0.00045));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.mlp.down_proj.weight", layer, &.{ hidden, intermediate }, try makeRampF32(allocator, hidden * intermediate, 0.0005));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.post_feedforward_layernorm.weight", layer, &.{hidden}, try makeFilledF32(allocator, hidden, 1.0));
        try appendOwnedGemmaTensor(allocator, &tensors, &owned_names, "model.layers.{d}.layer_scalar", layer, &.{1}, try makeFilledF32(allocator, 1, 1.0));
    }

    try writeHeaderAndTensorsF32(allocator, path, tensors.items);
}

fn makeSyntheticGemmaPreferenceLmHead(allocator: std.mem.Allocator, vocab: usize, hidden: usize) ![]f32 {
    const yes_token_id: usize = 8;
    const no_token_id: usize = 12;
    if (yes_token_id >= vocab or no_token_id >= vocab) return error.InvalidSyntheticTokenizer;
    const data = try makeFilledF32(allocator, vocab * hidden, 0.0);
    @memset(data[yes_token_id * hidden ..][0..hidden], 1.0);
    @memset(data[no_token_id * hidden ..][0..hidden], 0.5);
    return data;
}

fn appendOwnedQwenTensor(
    allocator: std.mem.Allocator,
    tensors: *std.ArrayList(WriteTensorF32),
    owned_names: *std.ArrayList([]const u8),
    comptime fmt: []const u8,
    layer: usize,
    shape: []const usize,
    data: []f32,
) !void {
    const name = try std.fmt.allocPrint(allocator, fmt, .{layer});
    try owned_names.append(allocator, name);
    try tensors.append(allocator, .{
        .name = name,
        .shape = shape,
        .data = data,
    });
}

fn appendOwnedGemmaTensor(
    allocator: std.mem.Allocator,
    tensors: *std.ArrayList(WriteTensorF32),
    owned_names: *std.ArrayList([]const u8),
    comptime fmt: []const u8,
    layer: usize,
    shape: []const usize,
    data: []f32,
) !void {
    const name = try std.fmt.allocPrint(allocator, fmt, .{layer});
    try owned_names.append(allocator, name);
    try tensors.append(allocator, .{
        .name = name,
        .shape = shape,
        .data = data,
    });
}

fn writeOwnedSyntheticPreferenceTokenizerArtifact(
    allocator: std.mem.Allocator,
    io: std.Io,
    out_dir: []const u8,
    file_name: []const u8,
) !void {
    const path = try std.fs.path.join(allocator, &.{ out_dir, file_name });
    defer allocator.free(path);
    try writeSyntheticPreferenceTokenizerArtifact(io, path, file_name);
}

fn writeOwnedTextFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, contents: []const u8) !void {
    defer allocator.free(path);
    try writeTextFile(io, path, contents);
}

fn writeTextFile(io: std.Io, path: []const u8, contents: []const u8) !void {
    if (std.fs.path.dirname(path)) |dir_name| try compat.cwd().createDirPath(io, dir_name);
    try compat.cwd().writeFile(io, .{ .sub_path = path, .data = contents });
}

fn writeHeaderAndTensorsF32(allocator: std.mem.Allocator, path: []const u8, tensors: []const WriteTensorF32) !void {
    defer {
        for (tensors) |tensor| allocator.free(tensor.data);
    }

    var header_buf: std.Io.Writer.Allocating = .init(allocator);
    defer header_buf.deinit();
    const writer = &header_buf.writer;
    try writer.writeByte('{');
    var offset: u64 = 0;
    for (tensors, 0..) |tensor, idx| {
        if (idx != 0) try writer.writeByte(',');
        const byte_len = tensor.data.len * @sizeOf(f32);
        try writer.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{tensor.name});
        for (tensor.shape, 0..) |dim, dim_idx| {
            if (dim_idx != 0) try writer.writeByte(',');
            try writer.print("{}", .{dim});
        }
        try writer.print("],\"data_offsets\":[{},{}]}}", .{ offset, offset + byte_len });
        offset += byte_len;
    }
    try writer.writeByte('}');

    var file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    defer file.close(compat.io());

    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header_buf.written().len, .little);
    try file.writeStreamingAll(compat.io(), &len_buf);
    try file.writeStreamingAll(compat.io(), header_buf.written());
    for (tensors) |tensor| try file.writeStreamingAll(compat.io(), std.mem.sliceAsBytes(tensor.data));
}

fn makeFilledF32(allocator: std.mem.Allocator, len: usize, value: f32) ![]f32 {
    const data = try allocator.alloc(f32, len);
    @memset(data, value);
    return data;
}

fn makeRampF32(allocator: std.mem.Allocator, len: usize, scale: f32) ![]f32 {
    const data = try allocator.alloc(f32, len);
    for (data, 0..) |*item, idx| item.* = @as(f32, @floatFromInt(idx + 1)) * scale;
    return data;
}

pub fn buildPlan(allocator: std.mem.Allocator, recipe: Recipe) !Plan {
    const kind = try parseKind(recipe.recipe orelse recipe.kind orelse return error.MissingRecipeKind);
    const family = recipe.model.family orelse try inferFamily(recipe);

    if (recipe.optimizer.seed != null and
        (!(kind == .dpo or kind == .grpo) or !eqlAny(family, &.{ "gemma4", "gemma" })))
    {
        return error.UnsupportedOptimizerSeed;
    }

    if (eqlAny(family, &.{ "gemma4", "gemma" })) {
        switch (kind) {
            .sft => return error.Gemma4FullSftNotYetSupported,
            .qlora_sft => return error.Gemma4QLoRANotYetSupported,
            else => {},
        }
    }

    return switch (kind) {
        .sft, .lora_sft, .qlora_sft => try buildLoraSftPlan(allocator, recipe, family),
        .dpo => try buildDpoPlan(allocator, recipe),
        .grpo => try buildGrpoPlan(allocator, recipe),
        .reranker => try buildRerankerPlan(allocator, recipe, family),
        .vlm_retrieval => try buildVlmRetrievalPlan(allocator, recipe, family),
    };
}

fn buildLoraSftPlan(allocator: std.mem.Allocator, recipe: Recipe, family: []const u8) !Plan {
    if (isQwen35Family(family)) return buildQwen35TextSftPlan(allocator, recipe);
    if (eqlAny(family, &.{ "gemma4", "gemma" })) {
        return buildGemma4LoraPlan(allocator, recipe);
    }
    if (eqlAny(family, &.{"gliner2"})) {
        return buildGliner2LoraPlan(allocator, recipe);
    }
    if (eqlAny(family, &.{"layoutlmv3"})) {
        return buildLayoutLmv3LoraPlan(allocator, recipe);
    }
    if (eqlAny(family, &.{ "reranker", "text-reranker", "deberta", "modernbert" })) {
        return buildRerankerLoraPlan(allocator, recipe);
    }
    return error.UnsupportedRecipeFamily;
}

fn buildQwen35TextSftPlan(allocator: std.mem.Allocator, recipe: Recipe) !Plan {
    _ = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    return .{ .steps = try allocator.dupe(Step, &.{
        .{
            .kind = .direct_sft,
            .name = "train-eval",
            .argv = try argv(allocator, &.{"antfly-inference-internal-sft"}),
        },
    }) };
}

fn validateGemma4LoraRecipeContract(recipe: Recipe, adapter: AdapterConfig) !void {
    if (recipe.model.reference_path != null) return error.UnsupportedGemma4ModelOption;
    if (recipe.model.projector_path != null) return error.Gemma4MultimodalFinetuningNotSupported;
    if (recipe.model.allow_direct_gguf_training != null) return error.UnsupportedGemma4ModelOption;

    if (recipe.dataset.cache_path != null or
        recipe.dataset.train_cache_path != null or
        recipe.dataset.format != null or
        recipe.dataset.labels != null)
    {
        return error.UnsupportedGemma4DatasetOption;
    }
    if (evalDatasetPath(recipe) == null) return error.MissingEvaluationDataset;
    _ = try gemma4.validateTrainingSequenceLength(recipe.dataset.max_seq_len orelse 512, std.math.maxInt(u32));

    const rank = adapterRank(adapter, .lora_sft);
    if (rank == 0 or rank > std.math.maxInt(u32)) return error.InvalidLoRARank;
    const alpha = adapterAlpha(adapter);
    if (!std.math.isFinite(alpha) or alpha <= 0) return error.InvalidLoRAAlpha;
    if (adapter.dropout != null or adapter.layer_name != null or adapter.quantization != null) {
        return error.UnsupportedGemma4AdapterOption;
    }
    if (adapter.use_dora orelse false) return error.DoRAAutodiffNotYetSupported;
    try gemma4.validateLoRAInitializerBaseCompatibility(adapter.init_lora_weights);
    if (adapter.init_lora_weights) |initializer| {
        if (eqlName(initializer, "eva") or
            eqlName(initializer, "lora-ga") or
            eqlName(initializer, "loraga") or
            eqlName(initializer, "lora_ga"))
        {
            return error.Gemma4RecipeInitializerStatsNotYetSupported;
        }
    }

    if (recipe.optimizer.weight_decay != null or
        recipe.optimizer.lr_scheduler != null or
        recipe.optimizer.warmup_ratio != null or
        recipe.optimizer.warmup_steps != null or
        recipe.optimizer.num_cycles != null or
        recipe.optimizer.max_steps != null or
        recipe.optimizer.micro_batch_size != null or
        recipe.optimizer.llrd_decay != null or
        (recipe.optimizer.schedule_free orelse false))
    {
        return error.UnsupportedGemma4OptimizerOption;
    }

    if (recipe.eval) |eval| {
        if (eval.every_epochs != null or
            eval.batch_size != null or
            eval.early_stopping_patience != null or
            eval.improvement_threshold != null or
            eval.backend != null or
            eval.entity_minimums != null or
            eval.full_task_minimums != null or
            eval.dpo_minimums != null or
            eval.grpo_minimums != null)
        {
            return error.UnsupportedGemma4EvalOption;
        }
    }

    if (recipe.checkpoint) |checkpoint| {
        // Gemma4 production v1 publishes one atomic mutable trainer-state
        // checkpoint. Retention/generation policies require a future indexed
        // checkpoint store rather than silently pretending `keep_last` works.
        if (checkpoint.keep_last != null) return error.UnsupportedGemma4CheckpointOption;
        if (checkpoint.every_epochs) |every| {
            if (every == 0 or @as(usize, every) > (recipe.optimizer.epochs orelse 1)) {
                return error.InvalidGemma4CheckpointInterval;
            }
        }
        if (checkpoint.resume_path) |path| {
            if (std.mem.trim(u8, path, " \t\r\n").len == 0) return error.InvalidGemma4CheckpointPath;
        }
    }
    if (recipe.runtime) |runtime| {
        if (runtime.compiled_required != null or
            runtime.grpo_incremental_kv != null or
            runtime.grpo_incremental_kv_batch_active != null or
            runtime.grpo_incremental_kv_clone_prompt_tail != null or
            runtime.grpo_incremental_kv_shadow_exact != null)
        {
            return error.UnsupportedGemma4RuntimeOption;
        }
        if (runtime.sequence_length_bucket_quantum) |quantum| {
            if (quantum == 0) return error.InvalidGemma4SequenceLengthBucket;
            if (runtime.sequence_length_bucket_min) |minimum| {
                if (minimum == 0) return error.InvalidGemma4SequenceLengthBucket;
            }
            if (runtime.graph_cache_capacity) |capacity| {
                if (capacity == 0 or capacity > 8) return error.InvalidGemma4GraphCacheCapacity;
            }
        } else if (runtime.sequence_length_bucket_min != null or runtime.graph_cache_capacity != null) {
            return error.Gemma4SequenceLengthBucketQuantumRequired;
        }
    }
    if (recipe.trainer) |trainer| {
        if (!std.mem.eql(u8, trainer, "autodiff") and !std.mem.eql(u8, trainer, "auto")) {
            return error.UnsupportedGemma4Trainer;
        }
    }

    if (recipe.preference.beta != null or
        recipe.preference.simpo_gamma != null or
        recipe.preference.sft_lambda != null or
        recipe.preference.ipo_tau != null or
        recipe.grpo.group_size != null or
        recipe.grpo.clip_epsilon != null or
        recipe.grpo.kl_coef != null or
        recipe.grpo.train_max_kl != null or
        recipe.grpo.adaptive_kl != null or
        recipe.grpo.target_kl != null or
        recipe.grpo.kl_horizon != null or
        recipe.grpo.min_kl_coef != null or
        recipe.grpo.max_kl_coef != null or
        recipe.grpo.advantage_eps != null or
        recipe.grpo.normalize_advantage != null or
        recipe.grpo.max_completion_tokens != null or
        recipe.grpo.reward_mode != null or
        recipe.reward != null)
    {
        return error.UnsupportedGemma4AlgorithmOption;
    }

    if (recipe.artifacts.materialized_dir != null or
        recipe.artifacts.validation_report_path != null or
        recipe.artifacts.evaluation_report_path != null or
        recipe.artifacts.reload_report_path != null or
        recipe.artifacts.report_path != null)
    {
        return error.UnsupportedGemma4ArtifactOption;
    }
}

fn validateGemma4ArtifactDirectories(
    allocator: std.mem.Allocator,
    recipe: Recipe,
    prepared_path: []const u8,
    eval_prepared_path: []const u8,
    bootstrap_dir: []const u8,
    trained_dir: []const u8,
) !void {
    const planned_files = [_]?[]const u8{
        prepared_path,
        eval_prepared_path,
        recipe.artifacts.manifest_path,
    };
    try validateGemma4AdapterOutputDirectories(
        allocator,
        recipe,
        bootstrap_dir,
        trained_dir,
        &planned_files,
    );
}

fn validateGemma4AdapterOutputDirectories(
    allocator: std.mem.Allocator,
    recipe: Recipe,
    bootstrap_dir: []const u8,
    trained_dir: []const u8,
    planned_files: []const ?[]const u8,
) !void {
    const resolved_bootstrap = try path_isolation.resolveRequestedPath(allocator, compat.io(), bootstrap_dir);
    defer allocator.free(resolved_bootstrap);
    const resolved_trained = try path_isolation.resolveRequestedPath(allocator, compat.io(), trained_dir);
    defer allocator.free(resolved_trained);
    if (path_isolation.pathsOverlap(resolved_bootstrap, resolved_trained)) {
        return error.Gemma4BootstrapAndTrainingOutputConflict;
    }

    const artifact_root = recipe.artifacts.root orelse "antfly-inference-finetune-out";
    const resolved_root = try path_isolation.resolveRequestedPath(allocator, compat.io(), artifact_root);
    defer allocator.free(resolved_root);
    if (path_isolation.sameOrWithin(resolved_bootstrap, resolved_root) or
        path_isolation.sameOrWithin(resolved_trained, resolved_root))
    {
        return error.Gemma4OutputConflictsWithArtifactRoot;
    }

    for (planned_files) |maybe_path| {
        const path = maybe_path orelse continue;
        const resolved = try path_isolation.resolveRequestedPath(allocator, compat.io(), path);
        defer allocator.free(resolved);
        if (path_isolation.pathsOverlap(resolved_bootstrap, resolved) or
            path_isolation.pathsOverlap(resolved_trained, resolved))
        {
            return error.Gemma4OutputContainsPlannedArtifact;
        }
    }
}

fn validateGemma4PreferenceTrainingRecipeContract(
    allocator: std.mem.Allocator,
    recipe: Recipe,
    task: PreferenceTask,
) !void {
    const base_model_dir = recipe.model.path orelse return error.MissingModelPath;
    if (recipe.model.reference_path) |reference_path| {
        if (!std.mem.eql(u8, reference_path, base_model_dir)) return error.UnsupportedReferencePath;
    }
    if (task == .dpo and recipe.model.projector_path != null) {
        return error.Gemma4DpoMultimodalFinetuningNotSupported;
    }
    if (task == .grpo and recipe.model.projector_path != null and recipe.checkpoint != null) {
        return error.Gemma4MultimodalPreferenceCheckpointResumeNotSupported;
    }
    if ((recipe.model.allow_direct_gguf_training orelse false) and
        !eqlName(recipe.backend orelse "", "metal"))
    {
        return error.GgufAutodiffRequiresMetal;
    }
    // Direct Q4_0 GGUF preference training is qualified through the canonical
    // full-prefix scorer. Its paged one-token decode graph can select a
    // different lower-ranked continuation than that canonical graph, even
    // with prompt-tail cloning disabled. Keep this composition fail-closed
    // until quantized paged decode has its own exact trajectory proof.
    if (task == .grpo and
        (recipe.model.allow_direct_gguf_training orelse false) and
        gemmaGrpoIncrementalKvEnabled(recipe))
    {
        return error.DirectGgufGrpoIncrementalKvNotQualified;
    }

    if (recipe.dataset.cache_path != null or
        recipe.dataset.train_cache_path != null or
        recipe.dataset.eval_cache_path != null or
        recipe.dataset.prepared_path != null or
        recipe.dataset.labels != null)
    {
        return error.UnsupportedGemma4DatasetOption;
    }
    try validateGemma4PreferenceEvaluationContract(allocator, recipe, task);
    if (recipe.dataset.max_examples) |max_examples| {
        if (max_examples == 0) return error.InvalidMaxExamples;
    }
    _ = try gemma4.validateTrainingSequenceLength(
        recipe.dataset.max_seq_len orelse if (task == .dpo) 512 else 128,
        std.math.maxInt(u32),
    );

    const adapter = recipe.adapter orelse AdapterConfig{};
    const rank = adapterRank(adapter, if (task == .dpo) .dpo else .grpo);
    if (rank == 0 or rank > std.math.maxInt(u32)) return error.InvalidLoRARank;
    const alpha = adapterAlpha(adapter);
    if (!std.math.isFinite(alpha) or alpha <= 0) return error.InvalidLoRAAlpha;
    if (adapter.dropout != null or adapter.layer_name != null or adapter.quantization != null) {
        return error.UnsupportedGemma4AdapterOption;
    }
    if (adapter.use_dora orelse false) return error.DoRAAutodiffNotYetSupported;
    try gemma4.validateLoRAInitializerBaseCompatibility(adapter.init_lora_weights);
    if (adapter.init_lora_weights) |initializer| {
        if (eqlName(initializer, "eva") or
            eqlName(initializer, "lora-ga") or
            eqlName(initializer, "loraga") or
            eqlName(initializer, "lora_ga"))
        {
            return error.Gemma4RecipeInitializerStatsNotYetSupported;
        }
    }
    try validateGemmaAdapterOptions(adapter);

    if (recipe.optimizer.weight_decay != null or
        recipe.optimizer.lr_scheduler != null or
        recipe.optimizer.warmup_ratio != null or
        recipe.optimizer.warmup_steps != null or
        recipe.optimizer.num_cycles != null or
        recipe.optimizer.max_steps != null or
        recipe.optimizer.micro_batch_size != null or
        recipe.optimizer.llrd_decay != null or
        recipe.optimizer.schedule_free != null)
    {
        return error.UnsupportedGemma4OptimizerOption;
    }
    const learning_rate = recipe.optimizer.learning_rate orelse 0.0001;
    if (!std.math.isFinite(learning_rate) or learning_rate <= 0) return error.InvalidLearningRate;
    if ((recipe.optimizer.epochs orelse 1) == 0) return error.InvalidEpochCount;
    if ((recipe.optimizer.gradient_accumulation_steps orelse 1) == 0) {
        return error.InvalidGradientAccumulationSteps;
    }
    const max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0;
    if (!std.math.isFinite(max_grad_norm) or max_grad_norm <= 0) return error.InvalidMaxGradNorm;

    if (recipe.checkpoint) |checkpoint| {
        // Preference training publishes one mutable, atomically replaced
        // trainer checkpoint plus a content-addressed aggregate-state sidecar.
        // Retention is deliberately not emulated by leaving ambiguous files.
        if (checkpoint.keep_last != null) return error.UnsupportedGemma4CheckpointOption;
        if (checkpoint.every_epochs) |every| {
            if (every == 0 or @as(usize, every) > (recipe.optimizer.epochs orelse 1)) {
                return error.InvalidGemma4CheckpointInterval;
            }
        }
        if (checkpoint.resume_path) |path| {
            if (std.mem.trim(u8, path, " \t\r\n").len == 0) return error.InvalidGemma4CheckpointPath;
        } else if (checkpoint.every_epochs == null) {
            return error.CheckpointIntervalRequired;
        }
        if (task == .grpo) {
            if (recipe.reward) |reward| {
                // Resume publishes into a fresh artifact root. Custom trace or
                // exchange paths would otherwise alias the interrupted run's
                // mutable outputs, so keep that combination fail-closed until
                // the recipe owns a relocatable artifact-path contract.
                if (reward.trace_path != null or
                    reward.evaluation_trace_path != null or
                    reward.exchange_dir != null)
                {
                    return error.Gemma4GrpoCheckpointCustomRewardArtifactsNotSupported;
                }
            }
        }
    }
    if (recipe.runtime) |runtime| {
        if (runtime.compiled_required != null) return error.UnsupportedGemma4RuntimeOption;
        switch (task) {
            .dpo => {
                if (runtime.grpo_incremental_kv != null or
                    runtime.grpo_incremental_kv_batch_active != null or
                    runtime.grpo_incremental_kv_clone_prompt_tail != null or
                    runtime.grpo_incremental_kv_shadow_exact != null)
                {
                    return error.UnsupportedGemma4RuntimeOption;
                }
                if (runtime.sequence_length_bucket_quantum) |quantum| {
                    if (quantum == 0) return error.InvalidGemma4SequenceLengthBucket;
                    if (runtime.sequence_length_bucket_min) |minimum| {
                        if (minimum == 0) return error.InvalidGemma4SequenceLengthBucket;
                    }
                    if (runtime.graph_cache_capacity) |capacity| {
                        if (capacity == 0 or capacity > 8) return error.InvalidGemma4GraphCacheCapacity;
                    }
                } else if (runtime.sequence_length_bucket_min != null or runtime.graph_cache_capacity != null) {
                    return error.Gemma4SequenceLengthBucketQuantumRequired;
                }
            },
            .grpo => {
                if (runtime.sequence_length_bucket_quantum != null or
                    runtime.sequence_length_bucket_min != null or
                    runtime.graph_cache_capacity != null)
                {
                    return error.UnsupportedGemma4RuntimeOption;
                }
                const incremental_enabled = runtime.grpo_incremental_kv orelse false;
                if (!incremental_enabled and
                    (runtime.grpo_incremental_kv_batch_active != null or
                        runtime.grpo_incremental_kv_clone_prompt_tail != null or
                        runtime.grpo_incremental_kv_shadow_exact != null))
                {
                    return error.Gemma4GrpoIncrementalKvRequired;
                }
                if (incremental_enabled and !eqlName(recipe.backend orelse "", "metal")) {
                    return error.Gemma4GrpoIncrementalKvRequiresMetal;
                }
            },
        }
    }
    if (recipe.trainer) |trainer| {
        if (!eqlName(trainer, "autodiff") and !eqlName(trainer, "auto")) {
            return error.UnsupportedGemma4Trainer;
        }
    }

    switch (task) {
        .dpo => {
            const beta = recipe.preference.beta orelse 0.1;
            if (!std.math.isFinite(beta) or beta <= 0) return error.InvalidDpoBeta;
            if (recipe.preference.simpo_gamma != null or
                recipe.preference.sft_lambda != null or
                recipe.preference.ipo_tau != null or
                recipe.grpo.group_size != null or
                recipe.grpo.clip_epsilon != null or
                recipe.grpo.kl_coef != null or
                recipe.grpo.train_max_kl != null or
                recipe.grpo.adaptive_kl != null or
                recipe.grpo.target_kl != null or
                recipe.grpo.kl_horizon != null or
                recipe.grpo.min_kl_coef != null or
                recipe.grpo.max_kl_coef != null or
                recipe.grpo.advantage_eps != null or
                recipe.grpo.normalize_advantage != null or
                recipe.grpo.max_completion_tokens != null or
                recipe.grpo.reward_mode != null or
                recipe.reward != null)
            {
                return error.UnsupportedGemma4AlgorithmOption;
            }
        },
        .grpo => {
            if (recipe.preference.beta != null or
                recipe.preference.simpo_gamma != null or
                recipe.preference.sft_lambda != null or
                recipe.preference.ipo_tau != null)
            {
                return error.UnsupportedGemma4AlgorithmOption;
            }
            const group_size = recipe.grpo.group_size orelse 2;
            if (group_size < 2) return error.InvalidGrpoGroupSize;
            const max_completion_tokens = recipe.grpo.max_completion_tokens orelse 4;
            if (max_completion_tokens == 0) return error.InvalidMaxCompletionTokens;
            const clip_epsilon = recipe.grpo.clip_epsilon orelse 0.2;
            if (!std.math.isFinite(clip_epsilon) or clip_epsilon <= 0 or clip_epsilon > 1) {
                return error.InvalidGrpoClipEpsilon;
            }
            const kl_coef = recipe.grpo.kl_coef orelse 0.04;
            if (!std.math.isFinite(kl_coef) or kl_coef < 0) return error.InvalidGrpoKlCoefficient;
            _ = try resolveGrpoKlControl(recipe.grpo);
            const advantage_eps = recipe.grpo.advantage_eps orelse 1e-8;
            if (!std.math.isFinite(advantage_eps) or advantage_eps <= 0) return error.InvalidGrpoAdvantageEpsilon;
            try validateRewardPipelineConfig(recipe);
        },
    }

    if (recipe.artifacts.prepared_path != null or
        recipe.artifacts.materialized_dir != null or
        recipe.artifacts.validation_report_path != null or
        recipe.artifacts.reload_report_path != null)
    {
        return error.UnsupportedGemma4ArtifactOption;
    }

    const execution = try resolveGemmaPreferenceExecution(recipe.backend);
    // Validate the process environment during planning so the public `run`
    // command fails before runPlan publishes a manifest or status report.
    // The execution path validates it again immediately before model loading.
    try validateGemmaPreferenceEnvironmentContract(execution.backend_kind);

    const bootstrap_configured = adapter.path != null;
    const bootstrap_dir = adapter.path orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    defer if (!bootstrap_configured) allocator.free(bootstrap_dir);
    const trained_configured = recipe.artifacts.trained_adapter_dir != null or recipe.artifacts.adapter_dir != null;
    const trained_dir = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    defer if (!trained_configured) allocator.free(trained_dir);
    const manifest_path = try manifestPath(allocator, recipe);
    defer allocator.free(manifest_path);
    const training_config_path = try defaultArtifactPath(allocator, recipe, "training_config.json");
    defer allocator.free(training_config_path);
    const training_report_path = try defaultArtifactPath(allocator, recipe, "training_report.json");
    defer allocator.free(training_report_path);
    const algorithm_report_path = switch (task) {
        .dpo => try dpoReportPath(allocator, recipe),
        .grpo => try grpoReportPath(allocator, recipe),
    };
    defer allocator.free(algorithm_report_path);
    const evaluation_report_path = try preferenceEvaluationReportPath(allocator, recipe, task);
    defer allocator.free(evaluation_report_path);
    const checkpoint_path = try preferenceCheckpointPath(allocator, recipe, task);
    defer if (checkpoint_path) |path| allocator.free(path);
    var reward_trace_path: ?[]const u8 = null;
    defer if (reward_trace_path) |path| allocator.free(path);
    var evaluation_reward_trace_path: ?[]const u8 = null;
    defer if (evaluation_reward_trace_path) |path| allocator.free(path);
    var reward_exchange_dir: ?[]const u8 = null;
    defer if (reward_exchange_dir) |path| allocator.free(path);
    if (task == .grpo) {
        reward_trace_path = try grpoRewardTracePath(allocator, recipe, false);
        evaluation_reward_trace_path = try grpoRewardTracePath(allocator, recipe, true);
        if (rewardPipelineHasExternalProvider(recipe)) {
            reward_exchange_dir = try grpoRewardExchangeDir(allocator, recipe);
        }
    }
    const planned_files = [_]?[]const u8{
        manifest_path,
        training_config_path,
        training_report_path,
        algorithm_report_path,
        evaluation_report_path,
        reward_trace_path,
        evaluation_reward_trace_path,
        reward_exchange_dir,
        checkpoint_path,
    };
    try validateDistinctPreferenceOutputPaths(allocator, &planned_files);
    try validateGemma4AdapterOutputDirectories(
        allocator,
        recipe,
        bootstrap_dir,
        trained_dir,
        &planned_files,
    );
    try validatePreferenceOutputInputConflicts(
        allocator,
        recipe,
        bootstrap_dir,
        trained_dir,
        &planned_files,
    );
}

fn validateDistinctPreferenceOutputPaths(
    allocator: std.mem.Allocator,
    paths: []const ?[]const u8,
) !void {
    for (paths, 0..) |maybe_lhs, lhs_idx| {
        const lhs = maybe_lhs orelse continue;
        const resolved_lhs = try path_isolation.resolveRequestedPath(allocator, compat.io(), lhs);
        defer allocator.free(resolved_lhs);
        for (paths[lhs_idx + 1 ..]) |maybe_rhs| {
            const rhs = maybe_rhs orelse continue;
            const resolved_rhs = try path_isolation.resolveRequestedPath(allocator, compat.io(), rhs);
            defer allocator.free(resolved_rhs);
            if (path_isolation.pathsOverlap(resolved_lhs, resolved_rhs)) {
                return error.PreferenceArtifactPathConflict;
            }
        }
    }
}

fn validatePreferenceOutputInputConflicts(
    allocator: std.mem.Allocator,
    recipe: Recipe,
    bootstrap_dir: []const u8,
    trained_dir: []const u8,
    planned_files: []const ?[]const u8,
) !void {
    const inputs = [_]?[]const u8{
        recipe.model.path,
        recipe.model.reference_path,
        recipe.model.projector_path,
        trainDatasetPath(recipe),
        evalDatasetPath(recipe),
    };
    var outputs: [16]?[]const u8 = @splat(null);
    outputs[0] = bootstrap_dir;
    outputs[1] = trained_dir;
    for (planned_files, 0..) |path, idx| outputs[idx + 2] = path;
    for (inputs) |maybe_input| {
        const input = maybe_input orelse continue;
        const resolved_input = try path_isolation.resolveRequestedPath(allocator, compat.io(), input);
        defer allocator.free(resolved_input);
        for (outputs) |maybe_output| {
            const output = maybe_output orelse continue;
            const resolved_output = try path_isolation.resolveRequestedPath(allocator, compat.io(), output);
            defer allocator.free(resolved_output);
            if (path_isolation.pathsOverlap(resolved_input, resolved_output)) {
                return error.PreferenceArtifactInputConflict;
            }
        }
    }
    if (recipe.reward) |reward| if (reward.providers) |providers| {
        for (providers) |provider| {
            const provider_inputs = [_]?[]const u8{
                provider.executable_path,
                provider.model_path,
                provider.tokenizer_path,
                provider.chat_template_path,
                provider.calibration_dataset_path,
            };
            for (provider_inputs) |maybe_provider_input| {
                const provider_input = maybe_provider_input orelse continue;
                const resolved_input = try path_isolation.resolveRequestedPath(allocator, compat.io(), provider_input);
                defer allocator.free(resolved_input);
                for (outputs) |maybe_output| {
                    const output = maybe_output orelse continue;
                    const resolved_output = try path_isolation.resolveRequestedPath(allocator, compat.io(), output);
                    defer allocator.free(resolved_output);
                    if (path_isolation.pathsOverlap(resolved_input, resolved_output)) {
                        return error.PreferenceArtifactInputConflict;
                    }
                }
            }
        }
    };
}

fn validateGemma4PreferenceEvaluationContract(
    allocator: std.mem.Allocator,
    recipe: Recipe,
    task: PreferenceTask,
) !void {
    const train_path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const eval_path = evalDatasetPath(recipe) orelse return error.MissingPreferenceEvaluationDataset;
    if (std.mem.trim(u8, eval_path, " \t\r\n").len == 0) return error.MissingPreferenceEvaluationDataset;
    if (recipe.eval == null) return error.MissingPreferenceEvaluationConfig;

    if (recipe.dataset.path != null and recipe.dataset.train_path != null and
        !try resolvedPathsEqual(allocator, recipe.dataset.path.?, recipe.dataset.train_path.?))
    {
        return error.ConflictingPreferenceTrainingDataset;
    }
    if (recipe.dataset.eval_path != null and recipe.eval.?.path != null and
        !try resolvedPathsEqual(allocator, recipe.dataset.eval_path.?, recipe.eval.?.path.?))
    {
        return error.ConflictingPreferenceEvaluationDataset;
    }
    if (recipe.dataset.eval_max_examples != null and recipe.eval.?.max_examples != null and
        recipe.dataset.eval_max_examples.? != recipe.eval.?.max_examples.?)
    {
        return error.ConflictingPreferenceEvaluationLimit;
    }
    if (recipe.dataset.train_split) |split| {
        if (!std.mem.eql(u8, split, "train")) return error.UnsupportedPreferenceTrainingSplit;
    }
    if (recipe.dataset.eval_split != null or recipe.eval.?.split != null) {
        return error.UnsupportedPreferenceEvaluationSplit;
    }
    if (recipe.eval.?.every_epochs != null or
        recipe.eval.?.batch_size != null or
        recipe.eval.?.early_stopping_patience != null or
        recipe.eval.?.improvement_threshold != null or
        recipe.eval.?.backend != null or
        recipe.eval.?.entity_minimums != null or
        recipe.eval.?.full_task_minimums != null)
    {
        return error.UnsupportedGemma4PreferenceEvalOption;
    }
    if (evalMaxExamples(recipe)) |max_examples| {
        if (max_examples == 0) return error.InvalidEvaluationMaxExamples;
    }

    const resolved_train = try path_isolation.resolveRequestedPath(allocator, compat.io(), train_path);
    defer allocator.free(resolved_train);
    const resolved_eval = try path_isolation.resolveRequestedPath(allocator, compat.io(), eval_path);
    defer allocator.free(resolved_eval);
    if (std.mem.eql(u8, resolved_train, resolved_eval)) return error.PreferenceTrainEvalDatasetConflict;

    switch (task) {
        .dpo => {
            if (recipe.eval.?.grpo_minimums != null) return error.UnsupportedGemma4PreferenceEvalOption;
            const minimums = recipe.eval.?.dpo_minimums orelse return error.MissingDpoEvaluationMinimums;
            if (!std.math.isFinite(minimums.accuracy) or minimums.accuracy < 0.0 or minimums.accuracy > 1.0) {
                return error.InvalidDpoEvaluationMinimums;
            }
            if (!std.math.isFinite(minimums.max_loss) or minimums.max_loss < 0.0) {
                return error.InvalidDpoEvaluationMinimums;
            }
            const relative_count = @intFromBool(minimums.min_accuracy_improvement != null) +
                @intFromBool(minimums.min_reward_margin_improvement != null) +
                @intFromBool(minimums.min_loss_improvement != null);
            if (relative_count != 0 and relative_count != 3) return error.IncompleteDpoBaselineRelativeMinimums;
            if (minimums.min_accuracy_improvement) |value| {
                if (!std.math.isFinite(value) or value < 0.0 or
                    !std.math.isFinite(minimums.min_reward_margin_improvement.?) or minimums.min_reward_margin_improvement.? < 0.0 or
                    !std.math.isFinite(minimums.min_loss_improvement.?) or minimums.min_loss_improvement.? < 0.0)
                {
                    return error.InvalidDpoEvaluationMinimums;
                }
            }
        },
        .grpo => {
            if (recipe.model.projector_path != null) return error.Gemma4MultimodalPreferenceEvaluationNotYetSupported;
            if (recipe.eval.?.dpo_minimums != null) return error.UnsupportedGemma4PreferenceEvalOption;
            const minimums = recipe.eval.?.grpo_minimums orelse return error.MissingGrpoEvaluationMinimums;
            if (!std.math.isFinite(minimums.mean_reward) or
                !std.math.isFinite(minimums.top_rank_mean_reward) or
                !std.math.isFinite(minimums.positive_reward_group_rate) or
                minimums.positive_reward_group_rate < 0.0 or minimums.positive_reward_group_rate > 1.0 or
                !std.math.isFinite(minimums.max_kl_loss) or minimums.max_kl_loss < 0.0)
            {
                return error.InvalidGrpoEvaluationMinimums;
            }
            const relative_count = @intFromBool(minimums.min_mean_reward_improvement != null) +
                @intFromBool(minimums.min_top_rank_mean_reward_improvement != null) +
                @intFromBool(minimums.min_positive_reward_group_rate_improvement != null);
            if (relative_count != 0 and relative_count != 3) return error.IncompleteGrpoBaselineRelativeMinimums;
            if (minimums.min_mean_reward_improvement) |value| {
                if (!std.math.isFinite(value) or value < 0.0 or
                    !std.math.isFinite(minimums.min_top_rank_mean_reward_improvement.?) or minimums.min_top_rank_mean_reward_improvement.? < 0.0 or
                    !std.math.isFinite(minimums.min_positive_reward_group_rate_improvement.?) or minimums.min_positive_reward_group_rate_improvement.? < 0.0)
                {
                    return error.InvalidGrpoEvaluationMinimums;
                }
            }
        },
    }
}

fn resolvedPathsEqual(allocator: std.mem.Allocator, lhs: []const u8, rhs: []const u8) !bool {
    const resolved_lhs = try path_isolation.resolveRequestedPath(allocator, compat.io(), lhs);
    defer allocator.free(resolved_lhs);
    const resolved_rhs = try path_isolation.resolveRequestedPath(allocator, compat.io(), rhs);
    defer allocator.free(resolved_rhs);
    return std.mem.eql(u8, resolved_lhs, resolved_rhs);
}

fn validateRewardPipelineConfig(recipe: Recipe) !void {
    const reward = recipe.reward orelse {
        _ = try parseTextRewardMode(recipe.grpo.reward_mode orelse "exact-match");
        return;
    };
    if (recipe.grpo.reward_mode != null) return error.ConflictingRewardConfiguration;
    const aggregation = reward.aggregation orelse "weighted-mean";
    if (!std.mem.eql(u8, aggregation, "weighted-mean") and
        !std.mem.eql(u8, aggregation, "weighted-sum"))
    {
        return error.UnsupportedRewardAggregation;
    }
    if (!std.mem.eql(u8, reward.failure_policy orelse "fail", "fail")) {
        return error.UnsupportedRewardFailurePolicy;
    }
    const providers = reward.providers orelse return error.MissingRewardProviders;
    if (providers.len == 0) return error.MissingRewardProviders;
    if (providers.len > 32) return error.TooManyRewardProviders;
    const max_trace_bytes = reward.max_trace_bytes orelse 64 * 1024 * 1024;
    if (max_trace_bytes < 1024 or max_trace_bytes > 1024 * 1024 * 1024) return error.InvalidRewardTraceLimit;

    var positive_weight = false;
    for (providers, 0..) |provider, idx| {
        if (!validRewardProviderName(provider.name)) return error.InvalidRewardProviderName;
        for (providers[0..idx]) |prior| {
            if (std.mem.eql(u8, prior.name, provider.name)) return error.DuplicateRewardProviderName;
        }
        if (!std.math.isFinite(provider.weight) or provider.weight < 0.0) return error.InvalidRewardProviderWeight;
        positive_weight = positive_weight or provider.weight > 0.0;
        if (provider.min_reward) |minimum| {
            if (!std.math.isFinite(minimum)) return error.InvalidRewardBounds;
        }
        if (provider.max_reward) |maximum| {
            if (!std.math.isFinite(maximum)) return error.InvalidRewardBounds;
        }
        if (provider.min_reward != null and provider.max_reward != null and
            provider.min_reward.? > provider.max_reward.?)
        {
            return error.InvalidRewardBounds;
        }

        if (std.mem.eql(u8, provider.kind, "builtin")) {
            _ = try parseTextRewardMode(provider.mode orelse return error.MissingBuiltinRewardMode);
            if (provider.executable_path != null or provider.executable_sha256 != null or
                provider.args != null or provider.timeout_ms != null or providerHasModelConfiguration(provider))
            {
                return error.InvalidBuiltinRewardProvider;
            }
        } else if (std.mem.eql(u8, provider.kind, "external-command")) {
            if (providerHasModelConfiguration(provider)) return error.InvalidExternalRewardProvider;
            try validateCommandRewardProvider(provider);
        } else if (std.mem.eql(u8, provider.kind, "model-command")) {
            try validateCommandRewardProvider(provider);
            if (provider.min_reward == null or provider.max_reward == null) return error.MissingModelRewardBounds;
            try validatePinnedRewardArtifact(
                provider.model_path,
                provider.model_sha256,
                error.MissingRewardModel,
                error.MissingRewardModelDigest,
                error.InvalidRewardModelDigest,
            );
            try validatePinnedRewardArtifact(
                provider.tokenizer_path,
                provider.tokenizer_sha256,
                error.MissingRewardTokenizer,
                error.MissingRewardTokenizerDigest,
                error.InvalidRewardTokenizerDigest,
            );
            try validatePinnedRewardArtifact(
                provider.chat_template_path,
                provider.chat_template_sha256,
                error.MissingRewardChatTemplate,
                error.MissingRewardChatTemplateDigest,
                error.InvalidRewardChatTemplateDigest,
            );
            try validatePinnedRewardArtifact(
                provider.calibration_dataset_path,
                provider.calibration_dataset_sha256,
                error.MissingRewardCalibrationDataset,
                error.MissingRewardCalibrationDatasetDigest,
                error.InvalidRewardCalibrationDatasetDigest,
            );
            const max_input_tokens = provider.max_input_tokens orelse return error.MissingModelRewardTokenLimit;
            if (max_input_tokens == 0 or max_input_tokens > 1024 * 1024) return error.InvalidModelRewardTokenLimit;
            // The current exchange protocol invokes one completion at a time.
            // Refuse aspirational batching settings until batching is real.
            if ((provider.max_batch_size orelse return error.MissingModelRewardBatchLimit) != 1) {
                return error.UnsupportedModelRewardBatchSize;
            }
        } else return error.UnsupportedRewardProviderKind;
    }
    if (!positive_weight) return error.InvalidRewardProviderWeight;
}

fn providerHasModelConfiguration(provider: RewardProviderConfig) bool {
    return provider.model_path != null or provider.model_sha256 != null or
        provider.tokenizer_path != null or provider.tokenizer_sha256 != null or
        provider.chat_template_path != null or provider.chat_template_sha256 != null or
        provider.calibration_dataset_path != null or provider.calibration_dataset_sha256 != null or
        provider.max_input_tokens != null or provider.max_batch_size != null;
}

fn validateCommandRewardProvider(provider: RewardProviderConfig) !void {
    if (provider.mode != null) return error.InvalidExternalRewardProvider;
    const executable_path = provider.executable_path orelse return error.MissingRewardExecutable;
    if (!std.fs.path.isAbsolute(executable_path)) return error.RewardExecutableMustBeAbsolute;
    const expected_digest = provider.executable_sha256 orelse return error.MissingRewardExecutableDigest;
    if (!validSha256DigestString(expected_digest)) return error.InvalidRewardExecutableDigest;
    const timeout_ms = provider.timeout_ms orelse 10_000;
    if (timeout_ms < 10 or timeout_ms > 60_000) return error.InvalidRewardTimeout;
    if (provider.args) |args| {
        if (args.len > 16) return error.TooManyRewardArguments;
        for (args) |arg| {
            if (arg.len > 4096 or std.mem.indexOfScalar(u8, arg, 0) != null) return error.InvalidRewardArgument;
        }
    }
}

fn validatePinnedRewardArtifact(
    maybe_path: ?[]const u8,
    maybe_digest: ?[]const u8,
    missing_path_error: anyerror,
    missing_digest_error: anyerror,
    invalid_digest_error: anyerror,
) !void {
    const path = maybe_path orelse return missing_path_error;
    if (!std.fs.path.isAbsolute(path)) return error.RewardModelArtifactMustBeAbsolute;
    const digest = maybe_digest orelse return missing_digest_error;
    if (!validSha256DigestString(digest)) return invalid_digest_error;
}

fn validSha256DigestString(value: []const u8) bool {
    const prefix = "sha256:";
    if (!std.mem.startsWith(u8, value, prefix) or value.len != prefix.len + 64) return false;
    for (value[prefix.len..]) |byte| {
        if (!std.ascii.isHex(byte)) return false;
    }
    return true;
}

fn validRewardProviderName(value: []const u8) bool {
    if (value.len == 0 or value.len > 64) return false;
    for (value) |byte| {
        if (!std.ascii.isAlphanumeric(byte) and byte != '-' and byte != '_' and byte != '.') return false;
    }
    return true;
}

fn buildGemma4LoraPlan(allocator: std.mem.Allocator, recipe: Recipe) !Plan {
    const model_path = recipe.model.path orelse return error.MissingModelPath;
    const dataset_path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const adapter = recipe.adapter orelse AdapterConfig{};
    const prepared_path = recipe.artifacts.prepared_path orelse recipe.dataset.prepared_path orelse try defaultArtifactPath(allocator, recipe, "prepared_inputs.json");
    const eval_prepared_path = recipe.dataset.eval_cache_path orelse try defaultArtifactPath(allocator, recipe, "prepared_eval_inputs.json");
    const bootstrap_dir = adapter.path orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    const trained_dir = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    const checkpoint = recipe.checkpoint orelse CheckpointConfig{};
    const runtime = recipe.runtime orelse RuntimeConfig{};
    const checkpoint_path: ?[]const u8 = if (checkpoint.resume_path) |path|
        path
    else if (checkpoint.every_epochs != null)
        try defaultArtifactPath(allocator, recipe, "gemma4_trainer_state.safetensors")
    else
        null;
    try validateGemma4LoraRecipeContract(recipe, adapter);
    try validateGemma4ArtifactDirectories(allocator, recipe, prepared_path, eval_prepared_path, bootstrap_dir, trained_dir);

    var steps: std.ArrayList(Step) = .empty;
    errdefer freeSteps(allocator, steps.items);

    var prepare_argv: std.ArrayList([]const u8) = .empty;
    try appendMany(allocator, &prepare_argv, &.{
        "prepare-gemma4-lora-inputs",
        model_path,
        dataset_path,
        recipe.dataset.train_split orelse "-",
        prepared_path,
        "--max-examples",
        try fmtInt(allocator, recipe.dataset.max_examples orelse 0),
        "--max-seq-len",
        try fmtInt(allocator, recipe.dataset.max_seq_len orelse 512),
    });
    if (recipe.model.projector_path) |path| {
        try appendMany(allocator, &prepare_argv, &.{ "--gguf-projector", path });
    }
    try steps.append(allocator, .{ .name = "prepare", .argv = try prepare_argv.toOwnedSlice(allocator) });

    var prepare_eval_argv: std.ArrayList([]const u8) = .empty;
    try appendMany(allocator, &prepare_eval_argv, &.{
        "prepare-gemma4-lora-inputs",
        model_path,
        evalDatasetPath(recipe).?,
        if (recipe.eval) |eval| eval.split orelse recipe.dataset.eval_split orelse "-" else recipe.dataset.eval_split orelse "-",
        eval_prepared_path,
        "--max-examples",
        try fmtInt(allocator, evalMaxExamples(recipe) orelse 0),
        "--max-seq-len",
        try fmtInt(allocator, recipe.dataset.max_seq_len orelse 512),
    });
    if (recipe.model.projector_path) |path| {
        try appendMany(allocator, &prepare_eval_argv, &.{ "--gguf-projector", path });
    }
    try steps.append(allocator, .{ .name = "prepare-eval", .argv = try prepare_eval_argv.toOwnedSlice(allocator) });

    var bootstrap_argv: std.ArrayList([]const u8) = .empty;
    try appendMany(allocator, &bootstrap_argv, &.{
        "bootstrap-gemma4-lora",
        model_path,
        bootstrap_dir,
    });
    try appendGemmaBootstrapAdapterArgs(allocator, &bootstrap_argv, adapter, .lora_sft);
    try steps.append(allocator, .{ .name = "bootstrap-adapter", .argv = try bootstrap_argv.toOwnedSlice(allocator) });

    const backend = recipe.backend orelse return error.MissingBackend;
    if (!std.mem.eql(u8, backend, "native") and !std.mem.eql(u8, backend, "metal")) {
        return error.UnsupportedBackend;
    }
    var train_argv: std.ArrayList([]const u8) = .empty;
    try appendMany(allocator, &train_argv, &.{
        "train-eval-gemma4-lora-bundle",
        model_path,
        bootstrap_dir,
        prepared_path,
        trained_dir,
        "--lr",
        try fmtFloat(allocator, recipe.optimizer.learning_rate orelse 0.001),
        "--max-examples",
        try fmtInt(allocator, recipe.dataset.max_examples orelse 32),
        "--eval-prepared",
        eval_prepared_path,
        "--epochs",
        try fmtInt(allocator, recipe.optimizer.epochs orelse 1),
    });
    if (evalMaxExamples(recipe)) |max| try appendMany(allocator, &train_argv, &.{ "--eval-max-examples", try fmtInt(allocator, max) });
    if (recipe.optimizer.gradient_accumulation_steps) |steps_count| try appendMany(allocator, &train_argv, &.{ "--grad-accum", try fmtInt(allocator, steps_count) });
    if (recipe.optimizer.max_grad_norm) |norm| try appendMany(allocator, &train_argv, &.{ "--max-grad-norm", try fmtFloat(allocator, norm) });
    if (runtime.sequence_length_bucket_quantum) |quantum| {
        try appendMany(allocator, &train_argv, &.{ "--sequence-length-bucket-quantum", try fmtInt(allocator, quantum) });
    }
    if (runtime.sequence_length_bucket_min) |minimum| {
        try appendMany(allocator, &train_argv, &.{ "--sequence-length-bucket-min", try fmtInt(allocator, minimum) });
    }
    if (runtime.graph_cache_capacity) |capacity| {
        try appendMany(allocator, &train_argv, &.{ "--graph-cache-capacity", try fmtInt(allocator, capacity) });
    }
    if (checkpoint_path) |path| try appendMany(allocator, &train_argv, &.{ "--checkpoint-path", path });
    if (checkpoint.every_epochs) |every| try appendMany(allocator, &train_argv, &.{ "--checkpoint-every-epochs", try fmtInt(allocator, every) });
    if (checkpoint.resume_path != null) try train_argv.append(allocator, "--resume");
    try appendMany(allocator, &train_argv, &.{ "--backend", backend });
    if (recipe.trainer) |trainer| try appendMany(allocator, &train_argv, &.{ "--trainer", trainer });
    if (recipe.model.projector_path) |path| try appendMany(allocator, &train_argv, &.{ "--gguf-projector", path });
    try steps.append(allocator, .{ .name = "train-eval", .argv = try train_argv.toOwnedSlice(allocator) });

    return .{ .steps = try steps.toOwnedSlice(allocator) };
}

fn buildGliner2LoraPlan(allocator: std.mem.Allocator, recipe: Recipe) !Plan {
    const model_path = recipe.model.path orelse return error.MissingModelPath;
    const train_path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const eval_path = evalDatasetPath(recipe);
    const out_dir = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    const validation_report_path = recipe.artifacts.validation_report_path orelse try defaultArtifactPath(allocator, recipe, "run_validation.json");
    const adapter = recipe.adapter orelse AdapterConfig{};
    const checkpoint = recipe.checkpoint orelse CheckpointConfig{};
    const runtime = recipe.runtime orelse RuntimeConfig{};
    const eval = recipe.eval orelse EvalConfig{};
    if ((eval.early_stopping_patience orelse 0) > 0 and eval_path == null) return error.EarlyStoppingRequiresEvalPath;
    if (eval_path != null and eval.full_task_minimums == null) return error.MissingFullTaskQualityThreshold;
    if (eval_path != null and eval.entity_minimums == null) return error.MissingEntityQualityThreshold;
    const eval_backend = eval.backend orelse "native";
    if (eval_path != null and !std.ascii.eqlIgnoreCase(eval_backend, "native")) return error.FullTaskEvaluationRequiresNativeBackend;

    var steps: std.ArrayList(Step) = .empty;
    errdefer freeSteps(allocator, steps.items);
    var train_argv: std.ArrayList([]const u8) = .empty;
    try appendMany(allocator, &train_argv, &.{
        "train-gliner2-autodiff",
        "--model-dir",
        model_path,
        "--train-data",
        train_path,
        "--out-dir",
        out_dir,
        "--objective",
        "gliner2-total-loss",
        "--lora-only-trainables",
    });
    if (recipe.optimizer.epochs) |epochs| try appendMany(allocator, &train_argv, &.{ "--epochs", try fmtInt(allocator, epochs) });
    if (recipe.optimizer.learning_rate) |lr| try appendMany(allocator, &train_argv, &.{ "--learning-rate", try fmtFloat(allocator, lr) });
    if (recipe.optimizer.weight_decay) |decay| try appendMany(allocator, &train_argv, &.{ "--weight-decay", try fmtFloat(allocator, decay) });
    if (recipe.optimizer.lr_scheduler) |scheduler| try appendMany(allocator, &train_argv, &.{ "--lr-scheduler", scheduler });
    if (recipe.optimizer.warmup_ratio) |ratio| try appendMany(allocator, &train_argv, &.{ "--warmup-ratio", try fmtFloat(allocator, ratio) });
    if (recipe.optimizer.warmup_steps) |warmup| try appendMany(allocator, &train_argv, &.{ "--warmup-steps", try fmtInt(allocator, warmup) });
    if (recipe.optimizer.num_cycles) |cycles| try appendMany(allocator, &train_argv, &.{ "--num-cycles", try fmtFloat(allocator, cycles) });
    if (recipe.optimizer.max_steps) |max_steps| try appendMany(allocator, &train_argv, &.{ "--max-steps", try fmtInt(allocator, max_steps) });
    if (recipe.optimizer.micro_batch_size) |batch| try appendMany(allocator, &train_argv, &.{ "--batch-size", try fmtInt(allocator, batch) });
    if (recipe.optimizer.gradient_accumulation_steps) |steps_count| try appendMany(allocator, &train_argv, &.{ "--grad-accum", try fmtInt(allocator, steps_count) });
    if (recipe.optimizer.max_grad_norm) |norm| try appendMany(allocator, &train_argv, &.{ "--max-grad-norm", try fmtFloat(allocator, norm) });
    if (recipe.dataset.max_seq_len) |seq_len| try appendMany(allocator, &train_argv, &.{ "--seq-len", try fmtInt(allocator, seq_len) });
    if (adapter.rank) |rank| try appendMany(allocator, &train_argv, &.{ "--lora-rank", try fmtInt(allocator, rank) });
    if (adapter.alpha) |alpha| try appendMany(allocator, &train_argv, &.{ "--lora-alpha", try fmtFloat(allocator, alpha) });
    if (adapter.dropout) |dropout| try appendMany(allocator, &train_argv, &.{ "--lora-dropout", try fmtFloat(allocator, dropout) });
    if (adapter.target_modules) |modules| try appendMany(allocator, &train_argv, &.{ "--lora-targets", try std.mem.join(allocator, ",", modules) });
    if (recipe.dataset.max_examples) |max| try appendMany(allocator, &train_argv, &.{ "--max-examples", try fmtInt(allocator, max) });
    if (recipe.dataset.labels) |labels| try appendMany(allocator, &train_argv, &.{ "--entity-types", labels });
    if (eval_path) |path| try appendMany(allocator, &train_argv, &.{ "--eval-data", path });
    if (eval.every_epochs) |every| try appendMany(allocator, &train_argv, &.{ "--eval-every-epochs", try fmtInt(allocator, every) });
    if (eval.batch_size) |batch| try appendMany(allocator, &train_argv, &.{ "--eval-batch-size", try fmtInt(allocator, batch) });
    if (eval.early_stopping_patience) |patience| try appendMany(allocator, &train_argv, &.{ "--early-stopping-patience", try fmtInt(allocator, patience) });
    if (eval.improvement_threshold) |threshold| try appendMany(allocator, &train_argv, &.{ "--early-stopping-threshold", try fmtFloat(allocator, threshold) });
    if (checkpoint.every_epochs) |every| try appendMany(allocator, &train_argv, &.{ "--checkpoint-every-epochs", try fmtInt(allocator, every) });
    if (checkpoint.keep_last) |keep| try appendMany(allocator, &train_argv, &.{ "--checkpoint-keep-last", try fmtInt(allocator, keep) });
    if (checkpoint.resume_path) |path| try appendMany(allocator, &train_argv, &.{ "--resume-checkpoint", path });
    if (runtime.compiled_required orelse false) try train_argv.append(allocator, "--compiled-required");
    if (runtime.graph_cache_capacity) |capacity| try appendMany(allocator, &train_argv, &.{ "--graph-cache-capacity", try fmtInt(allocator, capacity) });
    if (recipe.backend) |backend| try appendMany(allocator, &train_argv, &.{ "--backend", backend });
    try steps.append(allocator, .{ .name = "train", .argv = try train_argv.toOwnedSlice(allocator) });

    try steps.append(allocator, .{ .name = "validate", .argv = try argv(allocator, &.{
        "validate-gliner2-autodiff-run", out_dir, "--out", validation_report_path,
    }) });

    if (eval_path) |path| {
        const evaluation_report_path = recipe.artifacts.evaluation_report_path orelse try defaultArtifactPath(allocator, recipe, "heldout_eval.json");
        var eval_argv: std.ArrayList([]const u8) = .empty;
        try appendMany(allocator, &eval_argv, &.{
            "eval-gliner2-autodiff-adapter-dataset",
            model_path,
            out_dir,
            path,
            recipe.dataset.labels orelse "-",
            "--objective",
            "gliner2-total-loss",
            "--out",
            evaluation_report_path,
        });
        if (evalMaxExamples(recipe)) |max| try appendMany(allocator, &eval_argv, &.{ "--max-examples", try fmtInt(allocator, max) });
        if (recipe.dataset.max_seq_len) |seq_len| try appendMany(allocator, &eval_argv, &.{ "--seq-len", try fmtInt(allocator, seq_len) });
        try appendMany(allocator, &eval_argv, &.{ "--backend", eval_backend });
        if (eval.entity_minimums) |minimums| {
            if (minimums.precision) |threshold| try appendEntityMinimum(allocator, &eval_argv, "--min-precision", threshold);
            if (minimums.recall) |threshold| try appendEntityMinimum(allocator, &eval_argv, "--min-recall", threshold);
            try appendEntityMinimum(allocator, &eval_argv, "--min-f1", minimums.f1);
            try appendEntityMinimum(allocator, &eval_argv, "--min-exact-match", minimums.exact_match);
        }
        const full_task_minimums = eval.full_task_minimums.?;
        try appendFullTaskMinimum(allocator, &eval_argv, "classifications.micro_f1", full_task_minimums.classifications_micro_f1);
        try appendFullTaskMinimum(allocator, &eval_argv, "classifications.exact_match", full_task_minimums.classifications_exact_match);
        try appendFullTaskMinimum(allocator, &eval_argv, "json_structures.micro_f1", full_task_minimums.json_structures_micro_f1);
        try appendFullTaskMinimum(allocator, &eval_argv, "json_structures.exact_match", full_task_minimums.json_structures_exact_match);
        try appendFullTaskMinimum(allocator, &eval_argv, "relations.micro_f1", full_task_minimums.relations_micro_f1);
        try appendFullTaskMinimum(allocator, &eval_argv, "relations.exact_match", full_task_minimums.relations_exact_match);
        try appendFullTaskMinimum(allocator, &eval_argv, "count.accuracy", full_task_minimums.count_accuracy);
        try steps.append(allocator, .{ .name = "evaluate", .argv = try eval_argv.toOwnedSlice(allocator) });
    }

    if (recipe.artifacts.materialized_dir) |materialized_dir| {
        try steps.append(allocator, .{ .name = "materialize", .argv = try argv(allocator, &.{
            "materialize-gliner2-lora", model_path, out_dir, materialized_dir,
        }) });
        const reload_report_path = recipe.artifacts.reload_report_path orelse try defaultArtifactPath(allocator, recipe, "materialized_reload.json");
        try steps.append(allocator, .{ .name = "reload-validate", .argv = try argv(allocator, &.{
            "inspect-gliner2-checkpoint", materialized_dir, "--out", reload_report_path,
        }) });
    }

    return .{ .steps = try steps.toOwnedSlice(allocator) };
}

fn buildLayoutLmv3LoraPlan(allocator: std.mem.Allocator, recipe: Recipe) !Plan {
    const model_path = recipe.model.path orelse return error.MissingModelPath;
    const train_path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const eval_path = evalDatasetPath(recipe) orelse train_path;
    const bootstrap_dir = adapterBootstrapDir(recipe) orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    const trained_dir = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    const task = recipe.dataset.format orelse "sequence";
    const train_cmd = if (std.mem.eql(u8, task, "token")) "train-eval-layoutlmv3-lora-token" else "train-eval-layoutlmv3-lora-sequence";
    const adapter = recipe.adapter orelse AdapterConfig{};

    var steps: std.ArrayList(Step) = .empty;
    errdefer freeSteps(allocator, steps.items);
    var bootstrap_argv: std.ArrayList([]const u8) = .empty;
    try appendMany(allocator, &bootstrap_argv, &.{ "bootstrap-layoutlmv3-lora", model_path, bootstrap_dir });
    try appendGenericBootstrapAdapterArgs(allocator, &bootstrap_argv, adapter, .lora_sft);
    try steps.append(allocator, .{ .name = "bootstrap-adapter", .argv = try bootstrap_argv.toOwnedSlice(allocator) });
    try steps.append(allocator, .{ .name = "train-eval", .argv = try argv(allocator, &.{
        train_cmd,                                                    model_path,                                                           bootstrap_dir,                                            train_path,                                              eval_path, trained_dir,
        try fmtInt(allocator, recipe.dataset.max_examples orelse 32), try fmtFloat(allocator, recipe.optimizer.learning_rate orelse 0.001), try fmtInt(allocator, evalMaxExamples(recipe) orelse 32), try fmtInt(allocator, recipe.optimizer.epochs orelse 1),
    }) });
    if (recipe.artifacts.materialized_dir) |out_dir| {
        try steps.append(allocator, .{ .name = "materialize", .argv = try argv(allocator, &.{
            "materialize-layoutlmv3-checkpoint", model_path, trained_dir, task, out_dir,
        }) });
    }
    return .{ .steps = try steps.toOwnedSlice(allocator) };
}

fn buildRerankerPlan(allocator: std.mem.Allocator, recipe: Recipe, family: []const u8) !Plan {
    if (eqlAny(family, &.{ "reranker", "text-reranker", "deberta", "modernbert" })) {
        const model_path = recipe.model.path orelse return error.MissingModelPath;
        const train_path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
        const eval_path = evalDatasetPath(recipe);
        const train_cache_path = trainCachePath(recipe) orelse recipe.artifacts.prepared_path orelse try defaultArtifactPath(allocator, recipe, "reranker_train_pooled_cache.json");
        const eval_cache_path = evalCachePath(recipe) orelse if (eval_path != null) try defaultArtifactPath(allocator, recipe, "reranker_eval_pooled_cache.json") else train_cache_path;
        const out_dir = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir orelse try defaultArtifactPath(allocator, recipe, "reranker-head");
        var steps: std.ArrayList(Step) = .empty;
        errdefer freeSteps(allocator, steps.items);
        try steps.append(allocator, .{ .name = "prepare", .argv = try argv(allocator, &.{
            "prepare-reranker-pooled-cache", model_path, train_path, train_cache_path, recipe.dataset.train_split orelse "train", "--backend", recipe.backend orelse "auto", "--max-examples", try fmtInt(allocator, recipe.dataset.max_examples orelse 256),
        }) });
        if (eval_path) |path| {
            try steps.append(allocator, .{ .name = "prepare-eval", .argv = try argv(allocator, &.{
                "prepare-reranker-pooled-cache", model_path, path, eval_cache_path, recipe.dataset.eval_split orelse "eval", "--backend", recipe.backend orelse "auto", "--max-examples", try fmtInt(allocator, evalMaxExamples(recipe) orelse 256),
            }) });
        }
        try steps.append(allocator, .{ .name = "train-eval", .argv = try argv(allocator, &.{
            "train-eval-reranker-head-cached",                             model_path,                                                           train_cache_path,             eval_cache_path,                                         out_dir,
            "--learning-rate",                                             try fmtFloat(allocator, recipe.optimizer.learning_rate orelse 0.001), "--epochs",                   try fmtInt(allocator, recipe.optimizer.epochs orelse 1), "--max-examples",
            try fmtInt(allocator, recipe.dataset.max_examples orelse 256), "--backend",                                                          recipe.backend orelse "auto",
        }) });
        if (recipe.artifacts.materialized_dir) |materialized_dir| {
            try steps.append(allocator, .{ .name = "materialize", .argv = try argv(allocator, &.{
                "materialize-reranker-head", model_path, out_dir, materialized_dir,
            }) });
        }
        return .{ .steps = try steps.toOwnedSlice(allocator) };
    }
    return error.UnsupportedRecipeFamily;
}

fn buildRerankerLoraPlan(allocator: std.mem.Allocator, recipe: Recipe) !Plan {
    const model_path = recipe.model.path orelse return error.MissingModelPath;
    const train_path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const eval_path = evalDatasetPath(recipe);
    const train_cache_path = trainCachePath(recipe) orelse recipe.artifacts.prepared_path orelse try defaultArtifactPath(allocator, recipe, "reranker_train_top_layer_cache.json");
    const eval_cache_path = evalCachePath(recipe) orelse if (eval_path != null) try defaultArtifactPath(allocator, recipe, "reranker_eval_top_layer_cache.json") else train_cache_path;
    const bootstrap_dir = adapterBootstrapDir(recipe) orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    const trained_dir = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    const adapter = recipe.adapter orelse AdapterConfig{};
    var steps: std.ArrayList(Step) = .empty;
    errdefer freeSteps(allocator, steps.items);
    var bootstrap_argv: std.ArrayList([]const u8) = .empty;
    try appendMany(allocator, &bootstrap_argv, &.{ "bootstrap-reranker-lora", model_path, bootstrap_dir });
    try appendGenericBootstrapAdapterArgs(allocator, &bootstrap_argv, adapter, .lora_sft);
    try steps.append(allocator, .{ .name = "bootstrap-adapter", .argv = try bootstrap_argv.toOwnedSlice(allocator) });
    try steps.append(allocator, .{ .name = "prepare", .argv = try argv(allocator, &.{
        "prepare-reranker-top-layer-cache", model_path, train_path, train_cache_path, recipe.dataset.train_split orelse "train", "--backend", recipe.backend orelse "auto", "--max-examples", try fmtInt(allocator, recipe.dataset.max_examples orelse 128),
    }) });
    if (eval_path) |path| {
        try steps.append(allocator, .{ .name = "prepare-eval", .argv = try argv(allocator, &.{
            "prepare-reranker-top-layer-cache", model_path, path, eval_cache_path, recipe.dataset.eval_split orelse "eval", "--backend", recipe.backend orelse "auto", "--max-examples", try fmtInt(allocator, evalMaxExamples(recipe) orelse 128),
        }) });
    }
    const head_input = recipe.artifacts.report_path orelse return error.MissingRerankerHeadInput;
    try steps.append(allocator, .{ .name = "train-eval", .argv = try argv(allocator, &.{
        "train-eval-reranker-lora-top-layer-cached-surrogate", model_path,                                                           bootstrap_dir, head_input,                                              train_cache_path, eval_cache_path,                                               trained_dir,
        "--learning-rate",                                     try fmtFloat(allocator, recipe.optimizer.learning_rate orelse 0.001), "--epochs",    try fmtInt(allocator, recipe.optimizer.epochs orelse 1), "--max-examples", try fmtInt(allocator, recipe.dataset.max_examples orelse 128), "--backend",
        recipe.backend orelse "auto",
    }) });
    if (recipe.artifacts.materialized_dir) |out_dir| {
        try steps.append(allocator, .{ .name = "materialize", .argv = try argv(allocator, &.{
            "materialize-reranker-lora", model_path, trained_dir, out_dir,
        }) });
    }
    return .{ .steps = try steps.toOwnedSlice(allocator) };
}

fn buildVlmRetrievalPlan(allocator: std.mem.Allocator, recipe: Recipe, family: []const u8) !Plan {
    if (!eqlAny(family, &.{ "colqwen2", "colqwen", "qwen2vl" })) return error.UnsupportedRecipeFamily;
    const model_path = recipe.model.path orelse return error.MissingModelPath;
    const dataset_path = recipe.dataset.path orelse return error.MissingDatasetRoot;
    const examples_jsonl = trainDatasetPath(recipe) orelse recipe.dataset.prepared_path orelse return error.MissingExamplesJsonl;
    const prepared_path = recipe.artifacts.prepared_path orelse try defaultArtifactPath(allocator, recipe, "colqwen2_inputs.json");
    const bootstrap_dir = adapterBootstrapDir(recipe) orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    const trained_dir = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    const adapter = recipe.adapter orelse AdapterConfig{};

    var steps: std.ArrayList(Step) = .empty;
    errdefer freeSteps(allocator, steps.items);
    try steps.append(allocator, .{ .name = "prepare", .argv = try argv(allocator, &.{
        "prepare-colqwen2-inputs", model_path, dataset_path, examples_jsonl, prepared_path, try fmtInt(allocator, recipe.dataset.max_examples orelse 32),
    }) });
    var bootstrap_argv: std.ArrayList([]const u8) = .empty;
    try appendMany(allocator, &bootstrap_argv, &.{ "bootstrap-colqwen2-lora", model_path, bootstrap_dir });
    try appendGenericBootstrapAdapterArgs(allocator, &bootstrap_argv, adapter, .vlm_retrieval);
    try steps.append(allocator, .{ .name = "bootstrap-adapter", .argv = try bootstrap_argv.toOwnedSlice(allocator) });
    try steps.append(allocator, .{ .name = "train-eval", .argv = try argv(allocator, &.{
        "train-eval-colqwen2-lora-bundle",                       model_path,                                                           bootstrap_dir,    prepared_path,                                                trained_dir,
        "--lr",                                                  try fmtFloat(allocator, recipe.optimizer.learning_rate orelse 0.001), "--max-examples", try fmtInt(allocator, recipe.dataset.max_examples orelse 32), "--epochs",
        try fmtInt(allocator, recipe.optimizer.epochs orelse 1),
    }) });
    return .{ .steps = try steps.toOwnedSlice(allocator) };
}

fn parsePreferenceExecutionMode(recipe: Recipe) !PreferenceExecutionMode {
    const raw = recipe.execution.mode orelse return error.MissingPreferenceExecutionMode;
    if (eqlName(raw, "train")) return .train;
    if (eqlName(raw, "score") or eqlName(raw, "metrics") or eqlName(raw, "analyze")) return .score;
    return error.UnsupportedPreferenceExecutionMode;
}

fn isDpoModelFormat(format: []const u8) bool {
    return std.mem.eql(u8, format, "text-preference") or
        std.mem.eql(u8, format, "rendered-text-preference");
}

fn isGrpoModelFormat(format: []const u8) bool {
    return std.mem.eql(u8, format, "text-grpo") or
        std.mem.eql(u8, format, "rendered-text-grpo");
}

fn optimizerConfigHasAnyValue(config: OptimizerConfig) bool {
    return config.seed != null or
        config.learning_rate != null or
        config.weight_decay != null or
        config.lr_scheduler != null or
        config.warmup_ratio != null or
        config.warmup_steps != null or
        config.num_cycles != null or
        config.max_steps != null or
        config.epochs != null or
        config.micro_batch_size != null or
        config.gradient_accumulation_steps != null or
        config.max_grad_norm != null or
        config.schedule_free != null or
        config.llrd_decay != null;
}

fn preferenceTrainingFamilySupported(family: []const u8) bool {
    return eqlAny(family, &.{ "gemma4", "gemma", "qwen2", "qwen", "colqwen2", "colqwen", "qwen2vl" }) or
        isQwen35Family(family);
}

fn validatePreferenceExecutionContract(recipe: Recipe, task: PreferenceTask, mode: PreferenceExecutionMode, format: []const u8) !void {
    const fixture_format = switch (task) {
        .dpo => std.mem.eql(u8, format, "scalar-logprobs"),
        .grpo => std.mem.eql(u8, format, "token-logprobs"),
    };
    const model_format = switch (task) {
        .dpo => isDpoModelFormat(format),
        .grpo => isGrpoModelFormat(format),
    };
    if (!fixture_format and !model_format) {
        return switch (task) {
            .dpo => error.UnsupportedDpoFormat,
            .grpo => error.UnsupportedGrpoFormat,
        };
    }

    switch (mode) {
        .train => {
            if (!model_format) return error.PreferenceTrainingRequiresModelDataset;
            if (!requestsAdapterTraining(recipe)) return error.MissingAdapterTrainingIntent;
            _ = recipe.model.path orelse return error.MissingModelPath;
            const family = recipe.model.family orelse try inferFamily(recipe);
            if (!preferenceTrainingFamilySupported(family)) return error.UnsupportedPreferenceTrainingFamily;
        },
        .score => {
            if (requestsAdapterTraining(recipe)) return error.AdapterTrainingRequiresTrainMode;
            if (optimizerConfigHasAnyValue(recipe.optimizer)) return error.OptimizerRequiresTrainMode;
            if (recipe.checkpoint != null or recipe.runtime != null or recipe.trainer != null) {
                return error.TrainingOptionRequiresTrainMode;
            }
            if (recipe.grpo.train_max_kl != null or recipe.grpo.adaptive_kl != null or
                recipe.grpo.target_kl != null or recipe.grpo.kl_horizon != null or
                recipe.grpo.min_kl_coef != null or recipe.grpo.max_kl_coef != null)
            {
                return error.TrainingOptionRequiresTrainMode;
            }
            if (model_format) {
                _ = recipe.model.path orelse return error.MissingModelPath;
            } else if (recipe.model.path != null or
                recipe.model.reference_path != null or
                recipe.model.projector_path != null or
                recipe.backend != null)
            {
                return error.ModelOptionNotUsedByFixtureScoring;
            }
        },
    }
}

fn buildDpoPlan(allocator: std.mem.Allocator, recipe: Recipe) !Plan {
    _ = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const format = recipe.dataset.format orelse return error.MissingDatasetFormat;
    const mode = try parsePreferenceExecutionMode(recipe);
    try validatePreferenceExecutionContract(recipe, .dpo, mode, format);
    if (recipe.reward != null) return error.RewardConfigurationOnlySupportedForGrpo;
    if (mode == .train) {
        const family = recipe.model.family orelse try inferFamily(recipe);
        if (eqlAny(family, &.{ "gemma4", "gemma" })) {
            try validateGemma4PreferenceTrainingRecipeContract(allocator, recipe, .dpo);
        }
    }
    return .{ .steps = try allocator.dupe(Step, &.{
        .{
            .kind = .direct_dpo,
            .name = if (mode == .train) "train" else "score",
            .argv = try argv(allocator, &.{"antfly-inference-internal-dpo"}),
        },
    }) };
}

fn buildGrpoPlan(allocator: std.mem.Allocator, recipe: Recipe) !Plan {
    _ = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const format = recipe.dataset.format orelse return error.MissingDatasetFormat;
    const mode = try parsePreferenceExecutionMode(recipe);
    try validatePreferenceExecutionContract(recipe, .grpo, mode, format);
    if (recipe.reward != null) {
        if (mode != .train) return error.TypedRewardPipelineRequiresGemma4Training;
        const family = recipe.model.family orelse try inferFamily(recipe);
        if (!eqlAny(family, &.{ "gemma4", "gemma" })) {
            return error.TypedRewardPipelineRequiresGemma4Training;
        }
        try validateRewardPipelineConfig(recipe);
    }
    if (mode == .train) {
        const family = recipe.model.family orelse try inferFamily(recipe);
        if (eqlAny(family, &.{ "gemma4", "gemma" })) {
            try validateGemma4PreferenceTrainingRecipeContract(allocator, recipe, .grpo);
        }
    }
    return .{ .steps = try allocator.dupe(Step, &.{
        .{
            .kind = .direct_grpo,
            .name = if (mode == .train) "train" else "score",
            .argv = try argv(allocator, &.{"antfly-inference-internal-grpo"}),
        },
    }) };
}

fn runPlan(
    allocator: std.mem.Allocator,
    io: std.Io,
    exe_dir: []const u8,
    recipe: Recipe,
    plan: Plan,
    manifest_path: []const u8,
    training_config_path: []const u8,
    training_report_path: []const u8,
) !void {
    var step_manifests = try initStepManifests(allocator, plan);
    defer allocator.free(step_manifests);
    var static_meta_arena = std.heap.ArenaAllocator.init(allocator);
    defer static_meta_arena.deinit();
    const static_metadata = try collectStaticMetadata(static_meta_arena.allocator(), io, recipe);
    try writeTrainingConfig(allocator, io, training_config_path, .{
        .recipe = recipe,
        .steps = step_manifests,
        .metadata = static_metadata,
    });
    try writeRunManifest(allocator, io, manifest_path, .{
        .status = .planned,
        .recipe = recipe,
        .artifact_root = recipe.artifacts.root,
        .steps = step_manifests,
    });
    try writeTrainingReport(allocator, io, training_report_path, .{
        .status = .planned,
        .recipe = recipe,
        .artifact_root = recipe.artifacts.root,
        .steps = step_manifests,
        .metadata = .{
            .dataset_fingerprints = static_metadata.dataset_fingerprints,
            .backend = static_metadata.backend,
            .optimizer = static_metadata.optimizer,
        },
    });

    for (plan.steps, 0..) |step, idx| {
        step_manifests[idx].status = .running;
        try writeRunManifest(allocator, io, manifest_path, .{
            .status = .running,
            .recipe = recipe,
            .artifact_root = recipe.artifacts.root,
            .steps = step_manifests,
        });
        try writeTrainingReport(allocator, io, training_report_path, .{
            .status = .running,
            .recipe = recipe,
            .artifact_root = recipe.artifacts.root,
            .steps = step_manifests,
            .metadata = .{
                .dataset_fingerprints = static_metadata.dataset_fingerprints,
                .backend = static_metadata.backend,
                .optimizer = static_metadata.optimizer,
            },
        });

        print("finetune[{d}/{d}] {s}: ", .{ idx + 1, plan.steps.len, step.name });
        switch (step.kind) {
            .direct_sft => {
                print("{s}\n", .{step.argv[0]});
                const report_path = try sftReportPath(allocator, recipe);
                defer allocator.free(report_path);
                runDirectSft(allocator, io, recipe, report_path) catch |err| {
                    step_manifests[idx].status = .failed;
                    try writeRunManifest(allocator, io, manifest_path, .{
                        .status = .failed,
                        .recipe = recipe,
                        .artifact_root = recipe.artifacts.root,
                        .steps = step_manifests,
                    });
                    try writeTrainingReport(allocator, io, training_report_path, .{
                        .status = .failed,
                        .recipe = recipe,
                        .artifact_root = recipe.artifacts.root,
                        .steps = step_manifests,
                        .metadata = try collectReportMetadata(allocator, io, recipe, plan, manifest_path, training_config_path, training_report_path, static_metadata),
                    });
                    return err;
                };
                step_manifests[idx].stdout_bytes = 0;
                step_manifests[idx].stderr_bytes = 0;
                step_manifests[idx].exit_code = 0;
                step_manifests[idx].status = .succeeded;
                if (idx + 1 == plan.steps.len) {
                    try writeSucceededRunStatus(allocator, io, recipe, step_manifests, manifest_path, training_report_path, static_metadata);
                }
                continue;
            },
            .direct_dpo => {
                print("{s}\n", .{step.argv[0]});
                const report_path = try dpoReportPath(allocator, recipe);
                defer allocator.free(report_path);
                runDirectDpo(allocator, io, recipe, report_path) catch |err| {
                    step_manifests[idx].status = .failed;
                    try writeRunManifest(allocator, io, manifest_path, .{
                        .status = .failed,
                        .recipe = recipe,
                        .artifact_root = recipe.artifacts.root,
                        .steps = step_manifests,
                    });
                    try writeTrainingReport(allocator, io, training_report_path, .{
                        .status = .failed,
                        .recipe = recipe,
                        .artifact_root = recipe.artifacts.root,
                        .steps = step_manifests,
                        .metadata = try collectReportMetadata(allocator, io, recipe, plan, manifest_path, training_config_path, training_report_path, static_metadata),
                    });
                    return err;
                };
                step_manifests[idx].stdout_bytes = 0;
                step_manifests[idx].stderr_bytes = 0;
                step_manifests[idx].exit_code = 0;
                step_manifests[idx].status = .succeeded;
                if (idx + 1 == plan.steps.len) {
                    try writeSucceededRunStatus(allocator, io, recipe, step_manifests, manifest_path, training_report_path, static_metadata);
                }
                continue;
            },
            .direct_grpo => {
                print("{s}\n", .{step.argv[0]});
                const report_path = try grpoReportPath(allocator, recipe);
                defer allocator.free(report_path);
                runDirectGrpo(allocator, io, recipe, report_path) catch |err| {
                    step_manifests[idx].status = .failed;
                    try writeRunManifest(allocator, io, manifest_path, .{
                        .status = .failed,
                        .recipe = recipe,
                        .artifact_root = recipe.artifacts.root,
                        .steps = step_manifests,
                    });
                    try writeTrainingReport(allocator, io, training_report_path, .{
                        .status = .failed,
                        .recipe = recipe,
                        .artifact_root = recipe.artifacts.root,
                        .steps = step_manifests,
                        .metadata = try collectReportMetadata(allocator, io, recipe, plan, manifest_path, training_config_path, training_report_path, static_metadata),
                    });
                    return err;
                };
                step_manifests[idx].stdout_bytes = 0;
                step_manifests[idx].stderr_bytes = 0;
                step_manifests[idx].exit_code = 0;
                step_manifests[idx].status = .succeeded;
                if (idx + 1 == plan.steps.len) {
                    try writeSucceededRunStatus(allocator, io, recipe, step_manifests, manifest_path, training_report_path, static_metadata);
                }
                continue;
            },
            .command => {},
        }

        const direct_adapter_ran = runDirectCommandAdapter(allocator, io, recipe, step) catch |err| {
            step_manifests[idx].status = .failed;
            try writeFailedRunStatus(allocator, io, recipe, plan, step_manifests, manifest_path, training_config_path, training_report_path, static_metadata);
            return err;
        };
        if (direct_adapter_ran) {
            step_manifests[idx].stdout_bytes = 0;
            step_manifests[idx].stderr_bytes = 0;
            step_manifests[idx].exit_code = 0;
            step_manifests[idx].status = .succeeded;
            continue;
        }

        const peer_path = try std.fs.path.join(allocator, &.{ exe_dir, step.argv[0] });
        defer allocator.free(peer_path);

        var full_argv: std.ArrayList([]const u8) = .empty;
        defer full_argv.deinit(allocator);
        var cwd: std.process.Child.Cwd = .inherit;
        if (std.Io.Dir.cwd().access(io, peer_path, .{})) |_| {
            try full_argv.append(allocator, peer_path);
            try full_argv.appendSlice(allocator, step.argv[1..]);
        } else |_| {
            const pkg_root = try installedPackageRoot(allocator, exe_dir);
            defer allocator.free(pkg_root);
            try full_argv.appendSlice(allocator, &.{ "zig", "build", step.argv[0], "--" });
            try full_argv.appendSlice(allocator, step.argv[1..]);
            cwd = .{ .path = pkg_root };
        }

        for (full_argv.items, 0..) |part, part_idx| {
            if (part_idx != 0) print(" ", .{});
            print("{s}", .{part});
        }
        print("\n", .{});
        const result = std.process.run(allocator, io, .{
            .argv = full_argv.items,
            .cwd = cwd,
            .stdout_limit = .limited(16 * 1024 * 1024),
            .stderr_limit = .limited(16 * 1024 * 1024),
        }) catch |err| {
            step_manifests[idx].status = .failed;
            try writeRunManifest(allocator, io, manifest_path, .{
                .status = .failed,
                .recipe = recipe,
                .artifact_root = recipe.artifacts.root,
                .steps = step_manifests,
            });
            try writeTrainingReport(allocator, io, training_report_path, .{
                .status = .failed,
                .recipe = recipe,
                .artifact_root = recipe.artifacts.root,
                .steps = step_manifests,
                .metadata = try collectReportMetadata(allocator, io, recipe, plan, manifest_path, training_config_path, training_report_path, static_metadata),
            });
            return err;
        };
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        if (result.stdout.len > 0) print("{s}", .{result.stdout});
        if (result.stderr.len > 0) print("{s}", .{result.stderr});
        step_manifests[idx].stdout_bytes = result.stdout.len;
        step_manifests[idx].stderr_bytes = result.stderr.len;
        switch (result.term) {
            .exited => |code| {
                step_manifests[idx].exit_code = code;
                step_manifests[idx].status = if (code == 0) .succeeded else .failed;
                if (code != 0) {
                    try writeRunManifest(allocator, io, manifest_path, .{
                        .status = .failed,
                        .recipe = recipe,
                        .artifact_root = recipe.artifacts.root,
                        .steps = step_manifests,
                    });
                    try writeTrainingReport(allocator, io, training_report_path, .{
                        .status = .failed,
                        .recipe = recipe,
                        .artifact_root = recipe.artifacts.root,
                        .steps = step_manifests,
                        .metadata = try collectReportMetadata(allocator, io, recipe, plan, manifest_path, training_config_path, training_report_path, static_metadata),
                    });
                    return error.FinetuneStepFailed;
                }
            },
            else => {
                step_manifests[idx].status = .failed;
                try writeRunManifest(allocator, io, manifest_path, .{
                    .status = .failed,
                    .recipe = recipe,
                    .artifact_root = recipe.artifacts.root,
                    .steps = step_manifests,
                });
                try writeTrainingReport(allocator, io, training_report_path, .{
                    .status = .failed,
                    .recipe = recipe,
                    .artifact_root = recipe.artifacts.root,
                    .steps = step_manifests,
                    .metadata = try collectReportMetadata(allocator, io, recipe, plan, manifest_path, training_config_path, training_report_path, static_metadata),
                });
                return error.FinetuneStepFailed;
            },
        }
    }
    try finalizeSucceededRun(allocator, io, recipe, plan, step_manifests, manifest_path, training_config_path, training_report_path, static_metadata);
}

fn writeSucceededRunStatus(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    step_manifests: []const StepManifest,
    manifest_path: []const u8,
    training_report_path: []const u8,
    static_metadata: StaticMetadata,
) !void {
    try writeRunManifest(allocator, io, manifest_path, .{
        .status = .succeeded,
        .recipe = recipe,
        .artifact_root = recipe.artifacts.root,
        .steps = step_manifests,
    });
    try writeTrainingReport(allocator, io, training_report_path, .{
        .status = .succeeded,
        .recipe = recipe,
        .artifact_root = recipe.artifacts.root,
        .steps = step_manifests,
        .metadata = .{
            .dataset_fingerprints = static_metadata.dataset_fingerprints,
            .backend = static_metadata.backend,
            .optimizer = static_metadata.optimizer,
        },
    });
}

fn writeFailedRunStatus(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    plan: Plan,
    step_manifests: []const StepManifest,
    manifest_path: []const u8,
    training_config_path: []const u8,
    training_report_path: []const u8,
    static_metadata: StaticMetadata,
) !void {
    var report_arena = std.heap.ArenaAllocator.init(allocator);
    defer report_arena.deinit();
    try writeRunManifest(allocator, io, manifest_path, .{
        .status = .failed,
        .recipe = recipe,
        .artifact_root = recipe.artifacts.root,
        .steps = step_manifests,
    });
    try writeTrainingReport(allocator, io, training_report_path, .{
        .status = .failed,
        .recipe = recipe,
        .artifact_root = recipe.artifacts.root,
        .steps = step_manifests,
        .metadata = try collectReportMetadata(report_arena.allocator(), io, recipe, plan, manifest_path, training_config_path, training_report_path, static_metadata),
    });
}

fn finalizeSucceededRun(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    plan: Plan,
    step_manifests: []const StepManifest,
    manifest_path: []const u8,
    training_config_path: []const u8,
    training_report_path: []const u8,
    static_metadata: StaticMetadata,
) !void {
    try writeSucceededRunStatus(allocator, io, recipe, step_manifests, manifest_path, training_report_path, static_metadata);
    var report_arena = std.heap.ArenaAllocator.init(allocator);
    defer report_arena.deinit();
    try writeTrainingReport(allocator, io, training_report_path, .{
        .status = .succeeded,
        .recipe = recipe,
        .artifact_root = recipe.artifacts.root,
        .steps = step_manifests,
        .metadata = try collectReportMetadata(report_arena.allocator(), io, recipe, plan, manifest_path, training_config_path, training_report_path, static_metadata),
    });
}

fn runDirectCommandAdapter(allocator: std.mem.Allocator, io: std.Io, recipe: Recipe, step: Step) !bool {
    _ = recipe;
    const command = step.argv[0];
    if (!isDirectCommandAdapter(command)) return false;
    if (std.mem.eql(u8, command, "prepare-gemma4-lora-inputs")) {
        try runDirectPrepareGemma4LoraInputs(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "bootstrap-gemma4-lora")) {
        try runDirectBootstrapGemma4Lora(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "train-eval-gemma4-lora-bundle")) {
        try runDirectTrainEvalGemma4LoraBundle(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "bootstrap-layoutlmv3-lora")) {
        try runDirectBootstrapLayoutlmv3Lora(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "train-eval-layoutlmv3-lora-sequence")) {
        try runDirectTrainEvalLayoutlmv3LoraSequence(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "train-eval-layoutlmv3-lora-token")) {
        try runDirectTrainEvalLayoutlmv3LoraToken(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "materialize-layoutlmv3-checkpoint")) {
        try runDirectMaterializeLayoutlmv3Checkpoint(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "bootstrap-reranker-lora")) {
        try runDirectBootstrapRerankerLora(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "prepare-reranker-top-layer-cache")) {
        try runDirectPrepareRerankerTopLayerCache(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "train-eval-reranker-lora-top-layer-cached-surrogate")) {
        try runDirectTrainEvalRerankerLoraTopLayerCachedSurrogate(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "materialize-reranker-lora")) {
        try runDirectMaterializeRerankerLora(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "prepare-colqwen2-inputs")) {
        try runDirectPrepareColqwen2Inputs(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "bootstrap-colqwen2-lora")) {
        try runDirectBootstrapColqwen2Lora(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "train-eval-colqwen2-lora-bundle")) {
        try runDirectTrainEvalColqwen2LoraBundle(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "prepare-reranker-pooled-cache")) {
        try runDirectPrepareRerankerPooledCache(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "train-eval-reranker-head-cached")) {
        try runDirectTrainEvalRerankerHeadCached(allocator, io, step.argv);
        return true;
    }
    if (std.mem.eql(u8, command, "materialize-reranker-head")) {
        try runDirectMaterializeRerankerHead(allocator, io, step.argv);
        return true;
    }
    return false;
}

fn isDirectCommandAdapter(command: []const u8) bool {
    return command_registry.isDirectCommandAdapter(command);
}

fn printPlan(io: std.Io, recipe: Recipe, plan: Plan) !void {
    _ = io;
    print("recipe: {s}\n", .{recipe.recipe orelse recipe.kind orelse "unknown"});
    print("steps: {d}\n", .{plan.steps.len});
    for (plan.steps, 0..) |step, idx| {
        print("  {d}. {s}: ", .{ idx + 1, step.name });
        for (step.argv, 0..) |part, part_idx| {
            if (part_idx != 0) print(" ", .{});
            print("{s}", .{part});
        }
        print("\n", .{});
    }
}

fn parseKind(value: []const u8) !RecipeKind {
    if (eqlName(value, "sft")) return .sft;
    if (eqlName(value, "lora-sft") or eqlName(value, "lora_sft")) return .lora_sft;
    if (eqlName(value, "qlora-sft") or eqlName(value, "qlora_sft")) return .qlora_sft;
    if (eqlName(value, "dpo")) return .dpo;
    if (eqlName(value, "grpo")) return .grpo;
    if (eqlName(value, "reranker")) return .reranker;
    if (eqlName(value, "vlm-retrieval") or eqlName(value, "vlm_retrieval")) return .vlm_retrieval;
    return error.UnsupportedRecipeKind;
}

fn inferFamily(recipe: Recipe) ![]const u8 {
    if (recipe.model.path) |path| {
        if (inferFamilyFromModelPath(path)) |family| return family;
    }
    return error.MissingModelFamily;
}

fn inferFamilyFromModelPath(path: []const u8) ?[]const u8 {
    if (containsIgnoreCase(path, "gemma")) return "gemma4";
    if (containsQwen35Signal(path)) return "qwen3_5";
    if (containsIgnoreCase(path, "colqwen")) return "colqwen2";
    if (containsIgnoreCase(path, "qwen")) return "qwen2";
    if (containsIgnoreCase(path, "gliner")) return "gliner2";
    if (containsIgnoreCase(path, "layoutlmv3")) return "layoutlmv3";
    if (containsIgnoreCase(path, "reranker") or containsIgnoreCase(path, "deberta") or containsIgnoreCase(path, "modernbert")) return "reranker";
    return null;
}

fn requestsAdapterTraining(recipe: Recipe) bool {
    return recipe.artifacts.trained_adapter_dir != null or recipe.artifacts.adapter_dir != null or recipe.adapter != null;
}

fn rejectUnsupportedQwen35AdapterTraining(recipe: Recipe) !void {
    if (!requestsAdapterTraining(recipe)) return;
    const family = recipe.model.family orelse blk: {
        if (recipe.model.path) |path| break :blk inferFamilyFromModelPath(path) orelse return;
        return;
    };
    if (isQwen35Family(family)) return error.UnsupportedQwen35FinetuneGraph;
}

fn qwenLoraTargetModulesForFamily(family: []const u8) []const []const u8 {
    if (isQwen35Family(family)) return qwen2_real_autodiff.qwen35_lora_target_modules[0..];
    return qwen2_real_autodiff.default_lora_target_modules[0..];
}

fn defaultLoraRankForKind(kind: RecipeKind) usize {
    return switch (kind) {
        .grpo => default_policy_lora_rank,
        else => default_lora_rank,
    };
}

fn adapterRank(adapter: AdapterConfig, kind: RecipeKind) usize {
    return adapter.rank orelse defaultLoraRankForKind(kind);
}

fn adapterAlpha(adapter: AdapterConfig) f32 {
    return adapter.alpha orelse default_lora_alpha;
}

fn validateAdapterScaling(adapter: AdapterConfig) !void {
    const scaling = adapter.scaling orelse return;
    if (eqlName(scaling, "standard") or
        eqlName(scaling, "alpha/r") or
        eqlName(scaling, "alpha-over-r"))
    {
        return;
    }
    return error.UnsupportedLoRAScaling;
}

fn validateAdapterTargetSelection(adapter: AdapterConfig) !void {
    if (adapter.target_modules != null and adapter.target_preset != null) return error.ConflictingLoRATargetSelection;
}

fn validateGemmaAdapterOptions(adapter: AdapterConfig) !void {
    try validateAdapterScaling(adapter);
    try validateAdapterTargetSelection(adapter);
    if (adapter.target_modules == null) {
        const name = adapter.target_preset orelse default_gemma4_lora_target_preset;
        if (gemma4.parseGemma4LoRATargetPreset(name) == null) return error.UnsupportedLoRATargetPreset;
    }
}

fn validateNonGemmaAdapterOptions(adapter: AdapterConfig) !void {
    try validateAdapterScaling(adapter);
    try validateAdapterTargetSelection(adapter);
    if (adapter.use_dora orelse false) return error.UnsupportedLoRAOption;
    if (adapter.init_lora_weights != null) return error.UnsupportedLoRAOption;
}

fn validateGenericBootstrapAdapterOptions(adapter: AdapterConfig) !void {
    try validateNonGemmaAdapterOptions(adapter);
    if (adapter.target_preset != null) return error.UnsupportedLoRATargetPreset;
}

fn parseAdapterTargetPreset(name: []const u8) !peft.TargetPreset {
    return peft.parseTargetPreset(name) orelse error.UnsupportedLoRATargetPreset;
}

fn gemma4TargetPreset(adapter: AdapterConfig) ?gemma4.Gemma4LoRATargetPreset {
    if (adapter.target_modules != null) return null;
    return gemma4.parseGemma4LoRATargetPreset(adapter.target_preset orelse default_gemma4_lora_target_preset);
}

fn gemmaLegacyTargetPreset(adapter: AdapterConfig) !?peft.TargetPreset {
    if (adapter.target_modules != null or gemma4TargetPreset(adapter) != null) return null;
    return try parseAdapterTargetPreset(adapter.target_preset orelse default_gemma4_lora_target_preset);
}

fn adapterTargetModulesForQwen(adapter: AdapterConfig, default_target_modules: []const []const u8) ![]const []const u8 {
    try validateAdapterTargetSelection(adapter);
    if (adapter.target_modules) |modules| return modules;
    const preset_name = adapter.target_preset orelse return default_target_modules;
    const preset = try parseAdapterTargetPreset(preset_name);
    return switch (preset) {
        .all_linear => default_target_modules,
        .attention_only => qwen_attention_lora_target_modules[0..],
        .mlp_only => qwen_mlp_lora_target_modules[0..],
        .moe_experts => error.UnsupportedLoRATargetPreset,
    };
}

fn appendTargetModulesCsv(allocator: std.mem.Allocator, list: *std.ArrayList([]const u8), modules: []const []const u8) !void {
    try appendMany(allocator, list, &.{ "--target-modules", try joinCsv(allocator, modules) });
}

fn appendGemmaBootstrapAdapterArgs(
    allocator: std.mem.Allocator,
    list: *std.ArrayList([]const u8),
    adapter: AdapterConfig,
    kind: RecipeKind,
) !void {
    try validateGemmaAdapterOptions(adapter);
    try appendMany(allocator, list, &.{
        try fmtInt(allocator, adapterRank(adapter, kind)),
        try fmtFloat(allocator, adapterAlpha(adapter)),
    });
    if (adapter.base_model_name_or_path) |base_name| try list.append(allocator, base_name);
    if (adapter.target_modules) |modules| {
        try appendTargetModulesCsv(allocator, list, modules);
    } else {
        const preset = adapter.target_preset orelse default_gemma4_lora_target_preset;
        if (gemma4.parseGemma4LoRATargetPreset(preset) == null) return error.UnsupportedLoRATargetPreset;
        try appendMany(allocator, list, &.{ "--target-preset", preset });
    }
    if (adapter.layer_name) |layer| try appendMany(allocator, list, &.{ "--layer-name", layer });
    if (adapter.use_dora orelse false) try list.append(allocator, "--use-dora");
    if (adapter.init_lora_weights) |init| try appendMany(allocator, list, &.{ "--init-lora-weights", init });
}

fn appendGenericBootstrapAdapterArgs(
    allocator: std.mem.Allocator,
    list: *std.ArrayList([]const u8),
    adapter: AdapterConfig,
    kind: RecipeKind,
) !void {
    try validateGenericBootstrapAdapterOptions(adapter);
    try appendMany(allocator, list, &.{
        try fmtInt(allocator, adapterRank(adapter, kind)),
        try fmtFloat(allocator, adapterAlpha(adapter)),
    });
    if (adapter.base_model_name_or_path) |base_name| try list.append(allocator, base_name);
    if (adapter.target_modules) |modules| try appendTargetModulesCsv(allocator, list, modules);
}

fn adapterBootstrapDir(recipe: Recipe) ?[]const u8 {
    if (recipe.adapter) |adapter| if (adapter.path) |path| return path;
    return recipe.artifacts.adapter_dir;
}

fn trainDatasetPath(recipe: Recipe) ?[]const u8 {
    return recipe.dataset.train_path orelse recipe.dataset.path;
}

fn evalDatasetPath(recipe: Recipe) ?[]const u8 {
    if (recipe.eval) |eval| if (eval.path) |path| return path;
    return recipe.dataset.eval_path;
}

fn trainCachePath(recipe: Recipe) ?[]const u8 {
    return recipe.dataset.train_cache_path orelse recipe.dataset.cache_path;
}

fn evalCachePath(recipe: Recipe) ?[]const u8 {
    return recipe.dataset.eval_cache_path;
}

fn evalMaxExamples(recipe: Recipe) ?usize {
    if (recipe.eval) |eval| if (eval.max_examples) |max| return max;
    return recipe.dataset.eval_max_examples;
}

fn manifestPath(allocator: std.mem.Allocator, recipe: Recipe) ![]const u8 {
    if (recipe.artifacts.manifest_path) |path| return allocator.dupe(u8, path);
    return defaultArtifactPath(allocator, recipe, "recipe_run_manifest.json");
}

fn defaultArtifactPath(allocator: std.mem.Allocator, recipe: Recipe, leaf: []const u8) ![]const u8 {
    const root = recipe.artifacts.root orelse "antfly-inference-finetune-out";
    return std.fs.path.join(allocator, &.{ root, leaf });
}

fn initStepManifests(allocator: std.mem.Allocator, plan: Plan) ![]StepManifest {
    const steps = try allocator.alloc(StepManifest, plan.steps.len);
    for (plan.steps, 0..) |step, idx| {
        steps[idx] = .{
            .index = idx,
            .name = step.name,
            .argv = step.argv,
        };
    }
    return steps;
}

fn writeRunManifest(allocator: std.mem.Allocator, io: std.Io, path: []const u8, manifest: RunManifest) !void {
    try writeJsonFile(allocator, io, path, manifest);
}

fn writeTrainingConfig(allocator: std.mem.Allocator, io: std.Io, path: []const u8, config: TrainingConfigFile) !void {
    try writeJsonFile(allocator, io, path, config);
}

fn writeTrainingReport(allocator: std.mem.Allocator, io: std.Io, path: []const u8, report: TrainingReportFile) !void {
    try writeJsonFile(allocator, io, path, report);
}

fn collectStaticMetadata(allocator: std.mem.Allocator, io: std.Io, recipe: Recipe) !StaticMetadata {
    return .{
        .dataset_fingerprints = try collectDatasetFingerprints(allocator, io, recipe),
        .backend = .{
            .requested = recipe.backend,
            .build = .{
                .inference_version = build_options.inference_version,
                .enable_native = build_options.enable_native,
                .enable_onnx = build_options.enable_onnx,
                .enable_mlx = false,
                .enable_pjrt = build_options.enable_pjrt,
                .skip_openapi = build_options.skip_openapi,
            },
        },
        .optimizer = .{
            .seed = recipe.optimizer.seed,
            .learning_rate = recipe.optimizer.learning_rate,
            .weight_decay = recipe.optimizer.weight_decay,
            .lr_scheduler = recipe.optimizer.lr_scheduler,
            .warmup_ratio = recipe.optimizer.warmup_ratio,
            .warmup_steps = recipe.optimizer.warmup_steps,
            .num_cycles = recipe.optimizer.num_cycles,
            .max_steps = recipe.optimizer.max_steps,
            .epochs = recipe.optimizer.epochs,
            .micro_batch_size = recipe.optimizer.micro_batch_size,
            .gradient_accumulation_steps = recipe.optimizer.gradient_accumulation_steps,
            .max_grad_norm = recipe.optimizer.max_grad_norm,
            .schedule_free = recipe.optimizer.schedule_free,
            .llrd_decay = recipe.optimizer.llrd_decay,
        },
    };
}

fn collectReportMetadata(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    plan: Plan,
    manifest_path: []const u8,
    training_config_path: []const u8,
    training_report_path: []const u8,
    static_metadata: StaticMetadata,
) !ReportMetadata {
    return .{
        .dataset_fingerprints = static_metadata.dataset_fingerprints,
        .backend = static_metadata.backend,
        .optimizer = static_metadata.optimizer,
        .artifact_checksums = try collectArtifactChecksums(allocator, io, recipe, plan, manifest_path, training_config_path, training_report_path),
    };
}

fn collectDatasetFingerprints(allocator: std.mem.Allocator, io: std.Io, recipe: Recipe) ![]const PathFingerprint {
    var planned: std.ArrayListUnmanaged(PlannedPath) = .empty;
    errdefer planned.deinit(allocator);
    try appendUniquePlannedPath(allocator, &planned, "dataset", recipe.dataset.path);
    try appendUniquePlannedPath(allocator, &planned, "train_dataset", recipe.dataset.train_path);
    try appendUniquePlannedPath(allocator, &planned, "eval_dataset", recipe.dataset.eval_path);
    if (recipe.eval) |eval| try appendUniquePlannedPath(allocator, &planned, "eval_dataset", eval.path);
    try appendUniquePlannedPath(allocator, &planned, "dataset_cache", recipe.dataset.cache_path);
    try appendUniquePlannedPath(allocator, &planned, "train_cache", recipe.dataset.train_cache_path);
    try appendUniquePlannedPath(allocator, &planned, "eval_cache", recipe.dataset.eval_cache_path);
    try appendUniquePlannedPath(allocator, &planned, "prepared_dataset", recipe.dataset.prepared_path);
    return try fingerprintPlannedPaths(allocator, io, planned.items);
}

fn collectArtifactChecksums(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    plan: Plan,
    manifest_path: []const u8,
    training_config_path: []const u8,
    training_report_path: []const u8,
) ![]const PathFingerprint {
    var planned: std.ArrayListUnmanaged(PlannedPath) = .empty;
    errdefer planned.deinit(allocator);
    try appendUniquePlannedPath(allocator, &planned, "manifest", manifest_path);
    try appendUniquePlannedPath(allocator, &planned, "training_config", training_config_path);
    try appendArtifactPathsFromPlan(allocator, &planned, recipe, plan);
    removePlannedPath(&planned, training_report_path);
    return try fingerprintPlannedPaths(allocator, io, planned.items);
}

fn appendArtifactPathsFromPlan(allocator: std.mem.Allocator, planned: *std.ArrayListUnmanaged(PlannedPath), recipe: Recipe, plan: Plan) !void {
    for (plan.steps) |step| {
        switch (step.kind) {
            .direct_sft => {
                const report_path = try sftReportPath(allocator, recipe);
                defer allocator.free(report_path);
                try appendUniquePlannedPathOwned(allocator, planned, "sft_report", report_path);
            },
            .direct_dpo => {
                const report_path = try dpoReportPath(allocator, recipe);
                defer allocator.free(report_path);
                try appendUniquePlannedPathOwned(allocator, planned, "dpo_report", report_path);
                if ((try parsePreferenceExecutionMode(recipe)) == .train) {
                    try appendDirectPreferenceTrainingArtifacts(allocator, planned, recipe);
                    if (try preferenceCheckpointPath(allocator, recipe, .dpo)) |checkpoint_path| {
                        defer allocator.free(checkpoint_path);
                        try appendUniquePlannedPathOwned(allocator, planned, "dpo_training_checkpoint", checkpoint_path);
                    }
                    const evaluation_path = try preferenceEvaluationReportPath(allocator, recipe, .dpo);
                    defer allocator.free(evaluation_path);
                    try appendUniquePlannedPathOwned(allocator, planned, "dpo_evaluation", evaluation_path);
                }
            },
            .direct_grpo => {
                const report_path = try grpoReportPath(allocator, recipe);
                defer allocator.free(report_path);
                try appendUniquePlannedPathOwned(allocator, planned, "grpo_report", report_path);
                if ((try parsePreferenceExecutionMode(recipe)) == .train) {
                    try appendDirectPreferenceTrainingArtifacts(allocator, planned, recipe);
                    if (try preferenceCheckpointPath(allocator, recipe, .grpo)) |checkpoint_path| {
                        defer allocator.free(checkpoint_path);
                        try appendUniquePlannedPathOwned(allocator, planned, "grpo_training_checkpoint", checkpoint_path);
                    }
                    const evaluation_path = try preferenceEvaluationReportPath(allocator, recipe, .grpo);
                    defer allocator.free(evaluation_path);
                    try appendUniquePlannedPathOwned(allocator, planned, "grpo_evaluation", evaluation_path);
                    const reward_trace_path = try grpoRewardTracePath(allocator, recipe, false);
                    defer allocator.free(reward_trace_path);
                    try appendUniquePlannedPathOwned(allocator, planned, "grpo_reward_trace", reward_trace_path);
                    const eval_reward_trace_path = try grpoRewardTracePath(allocator, recipe, true);
                    defer allocator.free(eval_reward_trace_path);
                    try appendUniquePlannedPathOwned(allocator, planned, "grpo_evaluation_reward_trace", eval_reward_trace_path);
                    const kl_trace_path = try grpoKlTracePath(allocator, recipe);
                    defer allocator.free(kl_trace_path);
                    try appendUniquePlannedPathOwned(allocator, planned, "grpo_kl_control_trace", kl_trace_path);
                    if (rewardPipelineHasExternalProvider(recipe)) {
                        const exchange_dir = try grpoRewardExchangeDir(allocator, recipe);
                        defer allocator.free(exchange_dir);
                        try appendUniquePlannedPathOwned(allocator, planned, "reward_verifier_exchanges", exchange_dir);
                    }
                }
            },
            .command => {
                const command = step.argv[0];
                if (std.mem.eql(u8, command, "prepare-gemma4-lora-inputs")) {
                    try appendUniquePlannedPath(allocator, planned, "prepared_inputs", step.argv[4]);
                } else if (std.mem.eql(u8, command, "bootstrap-gemma4-lora")) {
                    try appendUniquePlannedPath(allocator, planned, "adapter_bootstrap", step.argv[2]);
                } else if (std.mem.eql(u8, command, "train-eval-gemma4-lora-bundle")) {
                    try appendUniquePlannedPath(allocator, planned, "trained_adapter", step.argv[4]);
                } else if (std.mem.eql(u8, command, "bootstrap-layoutlmv3-lora")) {
                    try appendUniquePlannedPath(allocator, planned, "adapter_bootstrap", step.argv[2]);
                } else if (std.mem.eql(u8, command, "train-eval-layoutlmv3-lora-sequence") or std.mem.eql(u8, command, "train-eval-layoutlmv3-lora-token")) {
                    try appendUniquePlannedPath(allocator, planned, "trained_adapter", step.argv[5]);
                } else if (std.mem.eql(u8, command, "materialize-layoutlmv3-checkpoint")) {
                    try appendUniquePlannedPath(allocator, planned, "materialized_model", step.argv[4]);
                } else if (std.mem.eql(u8, command, "prepare-reranker-pooled-cache")) {
                    const label = if (std.mem.eql(u8, step.name, "prepare-eval")) "eval_cache" else "train_cache";
                    try appendUniquePlannedPath(allocator, planned, label, step.argv[3]);
                } else if (std.mem.eql(u8, command, "train-eval-reranker-head-cached")) {
                    try appendUniquePlannedPath(allocator, planned, "trained_head", step.argv[4]);
                } else if (std.mem.eql(u8, command, "materialize-reranker-head")) {
                    try appendUniquePlannedPath(allocator, planned, "materialized_model", step.argv[3]);
                } else if (std.mem.eql(u8, command, "bootstrap-reranker-lora")) {
                    try appendUniquePlannedPath(allocator, planned, "adapter_bootstrap", step.argv[2]);
                } else if (std.mem.eql(u8, command, "prepare-reranker-top-layer-cache")) {
                    const label = if (std.mem.eql(u8, step.name, "prepare-eval")) "eval_cache" else "train_cache";
                    try appendUniquePlannedPath(allocator, planned, label, step.argv[3]);
                } else if (std.mem.eql(u8, command, "train-eval-reranker-lora-top-layer-cached-surrogate")) {
                    try appendUniquePlannedPath(allocator, planned, "trained_adapter", step.argv[6]);
                } else if (std.mem.eql(u8, command, "materialize-reranker-lora")) {
                    try appendUniquePlannedPath(allocator, planned, "materialized_model", step.argv[3]);
                } else if (std.mem.eql(u8, command, "prepare-colqwen2-inputs")) {
                    try appendUniquePlannedPath(allocator, planned, "prepared_inputs", step.argv[4]);
                } else if (std.mem.eql(u8, command, "bootstrap-colqwen2-lora")) {
                    try appendUniquePlannedPath(allocator, planned, "adapter_bootstrap", step.argv[2]);
                } else if (std.mem.eql(u8, command, "train-eval-colqwen2-lora-bundle")) {
                    try appendUniquePlannedPath(allocator, planned, "trained_adapter", step.argv[4]);
                } else if (std.mem.eql(u8, command, "train-gliner2-autodiff")) {
                    try appendUniquePlannedPath(allocator, planned, "trained_adapter", step.argv[6]);
                } else if (std.mem.eql(u8, command, "validate-gliner2-autodiff-run")) {
                    try appendUniquePlannedPath(allocator, planned, "run_validation", step.argv[3]);
                } else if (std.mem.eql(u8, command, "eval-gliner2-autodiff-adapter-dataset")) {
                    try appendUniquePlannedPath(allocator, planned, "heldout_eval", step.argv[8]);
                } else if (std.mem.eql(u8, command, "materialize-gliner2-lora")) {
                    try appendUniquePlannedPath(allocator, planned, "materialized_model", step.argv[3]);
                } else if (std.mem.eql(u8, command, "inspect-gliner2-checkpoint") and std.mem.eql(u8, step.name, "reload-validate")) {
                    try appendUniquePlannedPath(allocator, planned, "materialized_reload", step.argv[3]);
                }
            },
        }
    }
}

fn appendDirectPreferenceTrainingArtifacts(
    allocator: std.mem.Allocator,
    planned: *std.ArrayListUnmanaged(PlannedPath),
    recipe: Recipe,
) !void {
    const adapter = recipe.adapter orelse AdapterConfig{};
    if (adapter.path) |bootstrap_dir| {
        try appendUniquePlannedPath(allocator, planned, "adapter_bootstrap", bootstrap_dir);
    } else {
        const bootstrap_dir = try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
        defer allocator.free(bootstrap_dir);
        try appendUniquePlannedPathOwned(allocator, planned, "adapter_bootstrap", bootstrap_dir);
    }

    if (recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir) |trained_dir| {
        try appendUniquePlannedPath(allocator, planned, "trained_adapter", trained_dir);
    } else {
        const trained_dir = try defaultArtifactPath(allocator, recipe, "adapter-trained");
        defer allocator.free(trained_dir);
        try appendUniquePlannedPathOwned(allocator, planned, "trained_adapter", trained_dir);
    }
}

fn appendUniquePlannedPath(allocator: std.mem.Allocator, planned: *std.ArrayListUnmanaged(PlannedPath), label: []const u8, maybe_path: ?[]const u8) !void {
    const path = maybe_path orelse return;
    for (planned.items) |item| {
        if (std.mem.eql(u8, item.path, path)) return;
    }
    try planned.append(allocator, .{
        .label = label,
        .path = try allocator.dupe(u8, path),
    });
}

fn appendUniquePlannedPathOwned(allocator: std.mem.Allocator, planned: *std.ArrayListUnmanaged(PlannedPath), label: []const u8, path: []const u8) !void {
    for (planned.items) |item| {
        if (std.mem.eql(u8, item.path, path)) return;
    }
    try planned.append(allocator, .{
        .label = label,
        .path = try allocator.dupe(u8, path),
    });
}

fn removePlannedPath(planned: *std.ArrayListUnmanaged(PlannedPath), path: []const u8) void {
    var i: usize = 0;
    while (i < planned.items.len) {
        if (std.mem.eql(u8, planned.items[i].path, path)) {
            _ = planned.swapRemove(i);
            continue;
        }
        i += 1;
    }
}

fn fingerprintPlannedPaths(allocator: std.mem.Allocator, io: std.Io, planned: []const PlannedPath) ![]const PathFingerprint {
    const out = try allocator.alloc(PathFingerprint, planned.len);
    for (planned, 0..) |item, idx| {
        out[idx] = try fingerprintPath(allocator, io, item.label, item.path);
    }
    return out;
}

fn fingerprintPath(allocator: std.mem.Allocator, io: std.Io, label: []const u8, path: []const u8) !PathFingerprint {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return .{
            .label = label,
            .path = path,
            .exists = false,
        },
        else => return err,
    };
    switch (stat.kind) {
        .file => {
            const digest = try sha256FileAlloc(allocator, io, path);
            return .{
                .label = label,
                .path = path,
                .exists = true,
                .kind = "file",
                .size_bytes = stat.size,
                .digest = digest,
            };
        },
        .directory => {
            const summary = try digestDirectoryAlloc(allocator, io, path);
            return .{
                .label = label,
                .path = path,
                .exists = true,
                .kind = "directory",
                .size_bytes = summary.size_bytes,
                .entries = summary.entries,
                .digest = summary.digest,
            };
        },
        else => {
            return .{
                .label = label,
                .path = path,
                .exists = true,
                .kind = @tagName(stat.kind),
            };
        },
    }
}

fn sha256FileAlloc(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [64 * 1024]u8 = undefined;
    while (true) {
        const n = file.readStreaming(io, &.{buf[0..]}) catch |err| switch (err) {
            error.EndOfStream => 0,
            else => return err,
        };
        if (n == 0) break;
        hasher.update(buf[0..n]);
    }

    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.allocPrint(allocator, "sha256:{s}", .{std.fmt.bytesToHex(digest, .lower)});
}

const PreferenceFingerprintPolicy = struct {
    seed: u64 = 42,
    max_examples: usize,
    max_seq_len: usize,
    epochs: usize,
    learning_rate: f32,
    max_grad_norm: f32,
    requested_gradient_accumulation_steps: u32,
    physical_micro_batches_per_unit: usize,
    graph_cache_capacity: u8,
    sequence_length_bucket_quantum: ?u32 = null,
    sequence_length_bucket_min: ?u32 = null,
    direct_gguf_base: bool,
    fused_linear_cross_entropy: bool,
    execution_flags: u64,
    metal_numerical_policy_flags: u64 = 0,
    metal_sparse_loss_chunk_rows: ?u32 = null,
    metal_linear_cce_tile_vocab: ?usize = null,
    dpo_beta: ?f32 = null,
    dpo_activation_checkpoint_layer_interval: ?u32 = null,
    dpo_activation_checkpoint_recursive: ?bool = null,
    grpo_group_size: ?usize = null,
    grpo_backward_batch_size: ?usize = null,
    grpo_max_completion_tokens: ?usize = null,
    grpo_clip_epsilon: ?f32 = null,
    grpo_kl_coef: ?f32 = null,
    grpo_train_max_kl: ?f32 = null,
    grpo_adaptive_kl: ?bool = null,
    grpo_target_kl: ?f32 = null,
    grpo_kl_horizon: ?f32 = null,
    grpo_min_kl_coef: ?f32 = null,
    grpo_max_kl_coef: ?f32 = null,
    grpo_advantage_eps: ?f32 = null,
    grpo_normalize_advantage: ?bool = null,
    reward_configuration_digest: ?[]const u8 = null,
};

fn preferenceHashField(hasher: *std.crypto.hash.sha2.Sha256, value: []const u8) void {
    preferenceHashU64(hasher, value.len);
    hasher.update(value);
}

fn preferenceHashOptionalField(hasher: *std.crypto.hash.sha2.Sha256, value: ?[]const u8) void {
    preferenceHashU64(hasher, @intFromBool(value != null));
    if (value) |present| preferenceHashField(hasher, present);
}

fn preferenceHashU64(hasher: *std.crypto.hash.sha2.Sha256, value: anytype) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, @intCast(value), .little);
    hasher.update(&bytes);
}

fn preferenceHashOptionalU64(hasher: *std.crypto.hash.sha2.Sha256, value: anytype) void {
    preferenceHashU64(hasher, @intFromBool(value != null));
    if (value) |present| preferenceHashU64(hasher, present);
}

fn preferenceHashF32(hasher: *std.crypto.hash.sha2.Sha256, value: f32) void {
    preferenceHashU64(hasher, @as(u32, @bitCast(value)));
}

fn preferenceHashF64(hasher: *std.crypto.hash.sha2.Sha256, value: f64) void {
    preferenceHashU64(hasher, @as(u64, @bitCast(value)));
}

fn preferenceHashOptionalF32(hasher: *std.crypto.hash.sha2.Sha256, value: ?f32) void {
    preferenceHashU64(hasher, @intFromBool(value != null));
    if (value) |present| preferenceHashF32(hasher, present);
}

fn preferenceHashOptionalBool(hasher: *std.crypto.hash.sha2.Sha256, value: ?bool) void {
    preferenceHashU64(hasher, @intFromBool(value != null));
    if (value) |present| preferenceHashU64(hasher, @intFromBool(present));
}

fn gemmaPreferenceRunFingerprint(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    task: PreferenceTask,
    base_model_dir: []const u8,
    bootstrap_dir: []const u8,
    target_modules: []const []const u8,
    lora_rank: u32,
    lora_alpha: f32,
    recursive_lora: bool,
    backend_kind: gemma4_real_autodiff.BackendKind,
    policy: PreferenceFingerprintPolicy,
) ![std.crypto.hash.sha2.Sha256.digest_length]u8 {
    var provenance = try gemma4.fingerprintGemma4Model(allocator, base_model_dir);
    defer provenance.deinit(allocator);
    const train_path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const eval_path = evalDatasetPath(recipe) orelse return error.MissingPreferenceEvaluationDataset;
    const train_digest = try sha256FileAlloc(allocator, io, train_path);
    defer allocator.free(train_digest);
    const eval_digest = try sha256FileAlloc(allocator, io, eval_path);
    defer allocator.free(eval_digest);
    const adapter_payload_path = try std.fs.path.join(allocator, &.{ bootstrap_dir, gemma4.adapter_checkpoint_file_name });
    defer allocator.free(adapter_payload_path);
    const adapter_digest = try sha256FileAlloc(allocator, io, adapter_payload_path);
    defer allocator.free(adapter_digest);
    const projector_digest = if (recipe.model.projector_path) |path|
        try sha256FileAlloc(allocator, io, path)
    else
        null;
    defer if (projector_digest) |digest| allocator.free(digest);

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    preferenceHashField(&hasher, "antfly.gemma4.preference.run/v5");
    preferenceHashField(&hasher, @tagName(task));
    preferenceHashField(&hasher, provenance.base_model_sha256);
    preferenceHashField(&hasher, provenance.tokenizer_sha256);
    preferenceHashField(&hasher, provenance.chat_template_sha256);
    preferenceHashField(&hasher, adapter_digest);
    preferenceHashField(&hasher, train_digest);
    preferenceHashField(&hasher, eval_digest);
    preferenceHashField(&hasher, recipe.dataset.format orelse "");
    preferenceHashOptionalU64(&hasher, evalMaxExamples(recipe));
    switch (task) {
        .dpo => {
            const minimums = recipe.eval.?.dpo_minimums.?;
            preferenceHashF64(&hasher, minimums.accuracy);
            preferenceHashF64(&hasher, minimums.max_loss);
        },
        .grpo => {
            const minimums = recipe.eval.?.grpo_minimums.?;
            preferenceHashF64(&hasher, minimums.mean_reward);
            preferenceHashF64(&hasher, minimums.top_rank_mean_reward);
            preferenceHashF64(&hasher, minimums.positive_reward_group_rate);
            preferenceHashF64(&hasher, minimums.max_kl_loss);
        },
    }
    preferenceHashOptionalField(&hasher, projector_digest);
    preferenceHashU64(&hasher, target_modules.len);
    for (target_modules) |module| preferenceHashField(&hasher, module);
    preferenceHashU64(&hasher, lora_rank);
    preferenceHashF32(&hasher, lora_alpha);
    preferenceHashU64(&hasher, @intFromBool(recursive_lora));
    preferenceHashField(&hasher, @tagName(backend_kind));
    preferenceHashU64(&hasher, policy.max_examples);
    preferenceHashU64(&hasher, policy.max_seq_len);
    preferenceHashU64(&hasher, policy.epochs);
    preferenceHashF32(&hasher, policy.learning_rate);
    preferenceHashF32(&hasher, policy.max_grad_norm);
    preferenceHashU64(&hasher, policy.requested_gradient_accumulation_steps);
    preferenceHashU64(&hasher, policy.physical_micro_batches_per_unit);
    preferenceHashU64(&hasher, policy.graph_cache_capacity);
    preferenceHashOptionalU64(&hasher, policy.sequence_length_bucket_quantum);
    preferenceHashOptionalU64(&hasher, policy.sequence_length_bucket_min);
    preferenceHashU64(&hasher, @intFromBool(policy.direct_gguf_base));
    preferenceHashU64(&hasher, @intFromBool(policy.fused_linear_cross_entropy));
    preferenceHashU64(&hasher, policy.execution_flags);
    preferenceHashU64(&hasher, policy.metal_numerical_policy_flags);
    preferenceHashOptionalU64(&hasher, policy.metal_sparse_loss_chunk_rows);
    preferenceHashOptionalU64(&hasher, policy.metal_linear_cce_tile_vocab);
    preferenceHashOptionalF32(&hasher, policy.dpo_beta);
    preferenceHashOptionalU64(&hasher, policy.dpo_activation_checkpoint_layer_interval);
    preferenceHashOptionalBool(&hasher, policy.dpo_activation_checkpoint_recursive);
    preferenceHashOptionalU64(&hasher, policy.grpo_group_size);
    preferenceHashOptionalU64(&hasher, policy.grpo_backward_batch_size);
    preferenceHashOptionalU64(&hasher, policy.grpo_max_completion_tokens);
    preferenceHashOptionalF32(&hasher, policy.grpo_clip_epsilon);
    preferenceHashOptionalF32(&hasher, policy.grpo_kl_coef);
    preferenceHashOptionalF32(&hasher, policy.grpo_train_max_kl);
    preferenceHashU64(&hasher, @intFromBool(policy.grpo_adaptive_kl orelse false));
    preferenceHashOptionalF32(&hasher, policy.grpo_target_kl);
    preferenceHashOptionalF32(&hasher, policy.grpo_kl_horizon);
    preferenceHashOptionalF32(&hasher, policy.grpo_min_kl_coef);
    preferenceHashOptionalF32(&hasher, policy.grpo_max_kl_coef);
    preferenceHashOptionalF32(&hasher, policy.grpo_advantage_eps);
    preferenceHashU64(&hasher, @intFromBool(policy.grpo_normalize_advantage orelse false));
    preferenceHashOptionalField(&hasher, policy.reward_configuration_digest);
    // Preserve the established default-seed v5 identity so in-flight seed-42
    // checkpoints remain resumable. Non-default seeds extend the domain and
    // cannot be confused with either default or one another.
    if (policy.seed != 42) {
        preferenceHashField(&hasher, "typed-training-seed/v1");
        preferenceHashU64(&hasher, policy.seed);
    }
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return digest;
}

fn preferenceCheckpointMagic(task: PreferenceTask) u64 {
    return switch (task) {
        .dpo => 0x44504f2d43504b31,
        .grpo => 0x4752504f43504b31,
    };
}

fn preferenceDigestWords(digest: [std.crypto.hash.sha2.Sha256.digest_length]u8) [4]u64 {
    var words: [4]u64 = undefined;
    for (&words, 0..) |*word, idx| {
        word.* = std.mem.readInt(u64, digest[idx * 8 ..][0..8], .little);
    }
    return words;
}

fn preferenceDigestFromProgress(progress: real_autodiff.TrainingProgress) [std.crypto.hash.sha2.Sha256.digest_length]u8 {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    for (progress.rng_state, 0..) |word, idx| {
        std.mem.writeInt(u64, digest[idx * 8 ..][0..8], word, .little);
    }
    return digest;
}

fn preferenceCheckpointStatePath(
    allocator: std.mem.Allocator,
    checkpoint_path: []const u8,
    digest: [std.crypto.hash.sha2.Sha256.digest_length]u8,
) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s}.preference-state-{s}.json",
        .{ checkpoint_path, std.fmt.bytesToHex(digest, .lower) },
    );
}

fn savePreferenceCheckpoint(
    allocator: std.mem.Allocator,
    io: std.Io,
    trainer: *real_autodiff.RealAutodiffTrainer,
    checkpoint_path: []const u8,
    task: PreferenceTask,
    epoch_index: usize,
    examples_seen: usize,
    run_fingerprint: *const [std.crypto.hash.sha2.Sha256.digest_length]u8,
    state: PreferenceCheckpointState,
) !void {
    if (state.epoch_index != epoch_index or
        state.micro_batch_steps != trainer.microBatchSteps() or
        state.optimizer_steps != trainer.optimizerSteps() or
        state.accumulation_micro_batches != trainer.accumulatedMicroBatches())
    {
        return error.InvalidPreferenceCheckpointState;
    }
    const previous_progress = trainer.trainingProgress();
    const previous_state_path = if (previous_progress.order_seed == preferenceCheckpointMagic(task))
        try preferenceCheckpointStatePath(
            allocator,
            checkpoint_path,
            preferenceDigestFromProgress(previous_progress),
        )
    else
        null;
    defer if (previous_state_path) |path| allocator.free(path);
    const rendered = try std.json.Stringify.valueAlloc(allocator, state, .{});
    defer allocator.free(rendered);
    var state_digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(rendered, &state_digest, .{});
    const state_path = try preferenceCheckpointStatePath(allocator, checkpoint_path, state_digest);
    defer allocator.free(state_path);
    try artifact_publication.writeFileAtomicReplace(allocator, io, state_path, rendered);
    const persisted_digest = try sha256FileAlloc(allocator, io, state_path);
    defer allocator.free(persisted_digest);
    const expected_digest = try formatSha256DigestAlloc(allocator, state_digest);
    defer allocator.free(expected_digest);
    if (!std.ascii.eqlIgnoreCase(persisted_digest, expected_digest)) {
        return error.PreferenceCheckpointStateDigestMismatch;
    }
    trainer.setTrainingProgress(.{
        .epoch_index = @intCast(epoch_index),
        .examples_seen = @intCast(examples_seen),
        .order_seed = preferenceCheckpointMagic(task),
        .rng_state = preferenceDigestWords(state_digest),
    });
    try trainer.saveTrainingCheckpoint(checkpoint_path, run_fingerprint, null);
    // The newly synced trainer checkpoint is now the sole authority for the
    // sidecar generation. Remove the previously referenced sidecar only after
    // that atomic publication; a crash before this point leaves the old pair
    // valid, while a cleanup failure merely leaves a harmless orphan.
    if (previous_state_path) |previous_path| {
        if (!std.mem.eql(u8, previous_path, state_path)) {
            std.Io.Dir.cwd().deleteFile(io, previous_path) catch {};
        }
    }
}

fn loadPreferenceCheckpointState(
    allocator: std.mem.Allocator,
    io: std.Io,
    checkpoint_path: []const u8,
    task: PreferenceTask,
    run_fingerprint: *const [std.crypto.hash.sha2.Sha256.digest_length]u8,
    restored: real_autodiff.RestoredTrainingCheckpoint,
) !LoadedPreferenceCheckpointState {
    if (restored.progress.next_example_index != 0 or
        restored.progress.order_cursor != 0 or
        restored.progress.order_seed != preferenceCheckpointMagic(task))
    {
        return error.InvalidPreferenceCheckpointProgress;
    }
    const state_digest = preferenceDigestFromProgress(restored.progress);
    const state_path = try preferenceCheckpointStatePath(allocator, checkpoint_path, state_digest);
    errdefer allocator.free(state_path);
    const bytes = try readFileMax(allocator, io, state_path, 192 * 1024 * 1024);
    defer allocator.free(bytes);
    var actual_digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual_digest, .{});
    if (!std.mem.eql(u8, &actual_digest, &state_digest)) {
        return error.PreferenceCheckpointStateDigestMismatch;
    }
    var parsed = try std.json.parseFromSlice(
        PreferenceCheckpointState,
        allocator,
        bytes,
        .{ .ignore_unknown_fields = false, .allocate = .alloc_always },
    );
    errdefer parsed.deinit();
    const expected_fingerprint = try formatSha256DigestAlloc(allocator, run_fingerprint.*);
    defer allocator.free(expected_fingerprint);
    const state = parsed.value;
    if (!std.mem.eql(u8, state.schema_version, "antfly_gemma4_preference_checkpoint_state/v1") or
        !std.mem.eql(u8, state.task, @tagName(task)) or
        !std.ascii.eqlIgnoreCase(state.run_fingerprint_sha256, expected_fingerprint) or
        state.epoch_index != restored.progress.epoch_index or
        state.micro_batch_steps != restored.micro_batch_steps or
        state.optimizer_steps != restored.optimizer_steps or
        state.accumulation_micro_batches != restored.accumulation_micro_batches or
        state.epoch_index > std.math.maxInt(usize) or
        restored.progress.examples_seen > std.math.maxInt(usize))
    {
        return error.InvalidPreferenceCheckpointState;
    }
    if ((task == .dpo) != (state.dpo != null) or (task == .grpo) != (state.grpo != null)) {
        return error.InvalidPreferenceCheckpointState;
    }
    const aggregate_examples = switch (task) {
        .dpo => state.dpo.?.examples_seen,
        .grpo => state.grpo.?.total_groups,
    };
    if (aggregate_examples != restored.progress.examples_seen) {
        return error.InvalidPreferenceCheckpointState;
    }
    if (state.grpo) |aggregate| {
        if (aggregate.diagnostic_first_token_count > aggregate.diagnostic_first_tokens.len or
            aggregate.kl_admitted_groups != aggregate.total_groups or
            aggregate.policy_rescore_completions > aggregate.total_completions or
            aggregate.reward_call_index != aggregate.total_completions or
            aggregate.reward_external_calls > aggregate.reward_call_index or
            aggregate.reward_external_failures > aggregate.reward_external_calls or
            aggregate.total_tokens < aggregate.total_completions)
        {
            return error.InvalidPreferenceCheckpointState;
        }
        if (aggregate.incremental_kv) |telemetry| {
            if (telemetry.groups != aggregate.total_groups or
                telemetry.prompt_prefill_forwards != aggregate.total_groups or
                telemetry.cache_page_tokens == 0 or
                !std.mem.eql(u8, telemetry.cache_dtype, "f32"))
            {
                return error.InvalidPreferenceCheckpointState;
            }
        }
    }
    return .{ .parsed = parsed, .path = state_path };
}

fn preferenceCheckpointArtifactSummary(
    allocator: std.mem.Allocator,
    checkpoint_path: ?[]const u8,
    task: PreferenceTask,
    progress: real_autodiff.TrainingProgress,
) !?PreferenceCheckpointArtifactSummary {
    const path = checkpoint_path orelse return null;
    if (progress.order_seed != preferenceCheckpointMagic(task) or
        progress.next_example_index != 0 or
        progress.order_cursor != 0)
    {
        return error.InvalidPreferenceCheckpointProgress;
    }
    const epoch = std.math.cast(usize, progress.epoch_index) orelse
        return error.InvalidPreferenceCheckpointProgress;
    const digest = preferenceDigestFromProgress(progress);
    const state_path = try preferenceCheckpointStatePath(allocator, path, digest);
    errdefer allocator.free(state_path);
    const state_sha256 = try formatSha256DigestAlloc(allocator, digest);
    return .{
        .state_path = state_path,
        .state_sha256 = state_sha256,
        .epoch = epoch,
    };
}

fn digestDirectoryAlloc(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !DirectoryDigest {
    var entries: std.ArrayListUnmanaged(DirectoryDigestEntry) = .empty;
    defer {
        for (entries.items) |entry| {
            allocator.free(entry.relative_path);
            allocator.free(entry.digest);
        }
        entries.deinit(allocator);
    }
    try appendDirectoryDigestEntries(allocator, io, dir_path, "", &entries);
    std.sort.heap(DirectoryDigestEntry, entries.items, {}, lessThanDirectoryDigestEntry);

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var total_size: u64 = 0;
    for (entries.items) |entry| {
        hasher.update(entry.relative_path);
        hasher.update(&.{0});
        hasher.update(entry.digest);
        hasher.update(&.{0});
        const size_text = try std.fmt.allocPrint(allocator, "{d}", .{entry.size_bytes});
        defer allocator.free(size_text);
        hasher.update(size_text);
        hasher.update(&.{'\n'});
        total_size += entry.size_bytes;
    }

    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return .{
        .digest = try std.fmt.allocPrint(allocator, "sha256:{s}", .{std.fmt.bytesToHex(digest, .lower)}),
        .size_bytes = total_size,
        .entries = entries.items.len,
    };
}

fn appendDirectoryDigestEntries(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir_path: []const u8,
    rel_prefix: []const u8,
    entries: *std.ArrayListUnmanaged(DirectoryDigestEntry),
) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);

    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .directory) continue;
        const child_path = try std.fs.path.join(allocator, &.{ dir_path, entry.name });
        defer allocator.free(child_path);
        const rel_path = if (rel_prefix.len == 0)
            try allocator.dupe(u8, entry.name)
        else
            try std.fmt.allocPrint(allocator, "{s}/{s}", .{ rel_prefix, entry.name });
        errdefer allocator.free(rel_path);

        if (entry.kind == .directory) {
            try appendDirectoryDigestEntries(allocator, io, child_path, rel_path, entries);
            allocator.free(rel_path);
            continue;
        }

        const stat = try std.Io.Dir.cwd().statFile(io, child_path, .{});
        const digest = try sha256FileAlloc(allocator, io, child_path);
        errdefer allocator.free(digest);
        try entries.append(allocator, .{
            .relative_path = rel_path,
            .size_bytes = stat.size,
            .digest = digest,
        });
    }
}

fn validatePublishedAdapterChanged(
    allocator: std.mem.Allocator,
    io: std.Io,
    bootstrap_dir: []const u8,
    trained_dir: []const u8,
) !void {
    const trained = try digestDirectoryAlloc(allocator, io, trained_dir);
    defer allocator.free(trained.digest);
    if (trained.entries == 0) return error.EmptyTrainedAdapter;

    // Directory metadata can change even when serialization accidentally
    // republishes the original adapter weights. Compare the actual PEFT
    // payloads so the post-publication gate proves the trained tensor bytes
    // differ from bootstrap, complementing the pre-publication host digest.
    const bootstrap_payload_path = try std.fs.path.join(allocator, &.{ bootstrap_dir, "adapter_model.safetensors" });
    defer allocator.free(bootstrap_payload_path);
    const trained_payload_path = try std.fs.path.join(allocator, &.{ trained_dir, "adapter_model.safetensors" });
    defer allocator.free(trained_payload_path);
    const bootstrap_payload_digest = try sha256FileAlloc(allocator, io, bootstrap_payload_path);
    defer allocator.free(bootstrap_payload_digest);
    const trained_payload_digest = try sha256FileAlloc(allocator, io, trained_payload_path);
    defer allocator.free(trained_payload_digest);
    if (std.mem.eql(u8, bootstrap_payload_digest, trained_payload_digest)) return error.UnchangedTrainedAdapter;
}

fn trainerLoRAParameterDigest(trainer: *const real_autodiff.RealAutodiffTrainer) [std.crypto.hash.sha2.Sha256.digest_length]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    for (trainer.lora_params.items) |slot| {
        hasher.update(slot.name);
        hasher.update(&.{0});
        hasher.update(std.mem.sliceAsBytes(slot.weights));
        hasher.update(&.{0});
    }
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return digest;
}

fn formatSha256DigestAlloc(
    allocator: std.mem.Allocator,
    digest: [std.crypto.hash.sha2.Sha256.digest_length]u8,
) ![]const u8 {
    return std.fmt.allocPrint(allocator, "sha256:{s}", .{std.fmt.bytesToHex(digest, .lower)});
}

fn validateTrainerAdapterChanged(
    trainer: *real_autodiff.RealAutodiffTrainer,
    initial_digest: [std.crypto.hash.sha2.Sha256.digest_length]u8,
) !void {
    try trainer.syncDeviceTrainablesToHost();
    if (trainer.lora_params.items.len == 0) return error.MissingAdapterParameters;
    for (trainer.lora_params.items) |slot| {
        for (slot.weights) |value| {
            if (!std.math.isFinite(value)) return error.NonFiniteTrainedAdapter;
        }
    }
    const final_digest = trainerLoRAParameterDigest(trainer);
    if (std.mem.eql(u8, &initial_digest, &final_digest)) return error.UnchangedTrainedAdapter;
}

fn requireMissingPreferencePublicationTarget(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().access(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    return error.PreferenceTrainedAdapterAlreadyExists;
}

fn lessThanDirectoryDigestEntry(_: void, lhs: DirectoryDigestEntry, rhs: DirectoryDigestEntry) bool {
    return std.mem.order(u8, lhs.relative_path, rhs.relative_path) == .lt;
}

fn writeJsonFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, value: anytype) !void {
    const rendered = try std.json.Stringify.valueAlloc(allocator, value, .{ .whitespace = .indent_2 });
    defer allocator.free(rendered);
    try artifact_publication.writeFileAtomicReplace(allocator, io, path, rendered);
}

fn sftReportPath(allocator: std.mem.Allocator, recipe: Recipe) ![]const u8 {
    if (recipe.artifacts.report_path) |path| return allocator.dupe(u8, path);
    return defaultArtifactPath(allocator, recipe, "sft_report.json");
}

fn dpoReportPath(allocator: std.mem.Allocator, recipe: Recipe) ![]const u8 {
    if (recipe.artifacts.report_path) |path| return allocator.dupe(u8, path);
    return defaultArtifactPath(allocator, recipe, "dpo_report.json");
}

fn grpoReportPath(allocator: std.mem.Allocator, recipe: Recipe) ![]const u8 {
    if (recipe.artifacts.report_path) |path| return allocator.dupe(u8, path);
    return defaultArtifactPath(allocator, recipe, "grpo_report.json");
}

fn preferenceCheckpointPath(
    allocator: std.mem.Allocator,
    recipe: Recipe,
    task: PreferenceTask,
) !?[]const u8 {
    const checkpoint = recipe.checkpoint orelse return null;
    if (checkpoint.resume_path) |path| return try allocator.dupe(u8, path);
    if (checkpoint.every_epochs == null) return error.CheckpointIntervalRequired;
    return try defaultArtifactPath(
        allocator,
        recipe,
        switch (task) {
            .dpo => "gemma4_dpo_trainer_state.safetensors",
            .grpo => "gemma4_grpo_trainer_state.safetensors",
        },
    );
}

fn preferenceEvaluationReportPath(
    allocator: std.mem.Allocator,
    recipe: Recipe,
    task: PreferenceTask,
) ![]const u8 {
    if (recipe.artifacts.evaluation_report_path) |path| return allocator.dupe(u8, path);
    return defaultArtifactPath(allocator, recipe, switch (task) {
        .dpo => "dpo_evaluation_report.json",
        .grpo => "grpo_evaluation_report.json",
    });
}

fn preferenceBaselineEvaluationReportPath(
    allocator: std.mem.Allocator,
    recipe: Recipe,
    task: PreferenceTask,
) ![]const u8 {
    return defaultArtifactPath(allocator, recipe, switch (task) {
        .dpo => "dpo_baseline_evaluation_report.json",
        .grpo => "grpo_baseline_evaluation_report.json",
    });
}

fn grpoRewardTracePath(allocator: std.mem.Allocator, recipe: Recipe, evaluation: bool) ![]const u8 {
    if (recipe.reward) |reward| {
        if (evaluation) {
            if (reward.evaluation_trace_path) |path| return allocator.dupe(u8, path);
        } else if (reward.trace_path) |path| return allocator.dupe(u8, path);
    }
    return defaultArtifactPath(
        allocator,
        recipe,
        if (evaluation) "grpo_evaluation_reward_trace.jsonl" else "grpo_reward_trace.jsonl",
    );
}

fn grpoKlTracePath(allocator: std.mem.Allocator, recipe: Recipe) ![]const u8 {
    return defaultArtifactPath(allocator, recipe, "grpo_kl_control_trace.jsonl");
}

fn grpoRewardExchangeDir(allocator: std.mem.Allocator, recipe: Recipe) ![]const u8 {
    if (recipe.reward) |reward| if (reward.exchange_dir) |path| return allocator.dupe(u8, path);
    return defaultArtifactPath(allocator, recipe, "reward-verifier-exchanges");
}

fn expectRunStatusFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expected_status: []const u8) !void {
    const raw = try readFileMax(allocator, io, path, 16 * 1024 * 1024);
    defer allocator.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSmokeArtifact;
    const status_val = parsed.value.object.get("status") orelse return error.InvalidSmokeArtifact;
    if (status_val != .string) return error.InvalidSmokeArtifact;
    if (!std.mem.eql(u8, status_val.string, expected_status)) return error.InvalidSmokeArtifact;
}

fn expectPathExists(io: std.Io, path: []const u8) !void {
    _ = try std.Io.Dir.cwd().statFile(io, path, .{});
}

fn runDirectPrepareGemma4LoraInputs(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len < 5) return error.InvalidArguments;
    const model_dir = argv_in[1];
    const dataset_path = argv_in[2];
    const split_arg = argv_in[3];
    const out_path = argv_in[4];
    const split = if (std.mem.eql(u8, split_arg, "-")) null else split_arg;

    var max_examples: usize = 0;
    var max_seq_len: usize = 512;
    var gguf_projector_path: ?[]const u8 = null;
    var i: usize = 5;
    while (i < argv_in.len) : (i += 1) {
        const arg = argv_in[i];
        if (std.mem.eql(u8, arg, "--max-examples")) {
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            max_examples = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
        } else if (std.mem.eql(u8, arg, "--max-seq-len")) {
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            max_seq_len = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
        } else if (std.mem.eql(u8, arg, "--gguf-projector")) {
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            gguf_projector_path = argv_in[i];
        } else {
            return error.InvalidArguments;
        }
    }

    var loaded = try gemma_chat_data.loadExamples(allocator, dataset_path, split);
    defer loaded.deinit();
    const has_multimodal = gemmaMessagesHaveMedia(loaded.examples);
    if (has_multimodal and gguf_projector_path == null) return error.MissingGgufProjector;
    if (has_multimodal) return error.Gemma4MultimodalFinetuningNotSupported;
    var summary = try gemma4.prepareInputsFromChatDataWithSource(
        allocator,
        model_dir,
        loaded.examples,
        max_examples,
        max_seq_len,
        .{ .dataset_path = dataset_path, .split = split },
    );
    defer gemma4.freePreparedInputsSummary(allocator, &summary);
    try gemma4.savePreparedInputsSummary(allocator, out_path, summary);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectBootstrapGemma4Lora(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len < 3) return error.InvalidArguments;
    const model_dir = argv_in[1];
    const out_dir = argv_in[2];
    var rank = default_lora_rank;
    var alpha = default_lora_alpha;
    var rank_set = false;
    var alpha_set = false;
    var rank_alpha_flag_seen = false;

    var base_model_name_or_path: ?[]const u8 = null;
    var layer_name: ?[]const u8 = null;
    var target_preset: ?peft.TargetPreset = null;
    var gemma4_target_preset: ?gemma4.Gemma4LoRATargetPreset = null;
    var target_modules: ?[]const []const u8 = null;
    defer if (target_modules) |modules| allocator.free(modules);
    var use_dora = false;
    var init_lora_weights: ?[]const u8 = null;
    var i: usize = 3;
    while (i < argv_in.len) : (i += 1) {
        const arg = argv_in[i];
        if (std.mem.eql(u8, arg, "--rank")) {
            if (rank_set) return error.InvalidArguments;
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            rank = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
            rank_set = true;
            rank_alpha_flag_seen = true;
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            if (alpha_set) return error.InvalidArguments;
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            alpha = try std.fmt.parseFloat(f32, argv_in[i]);
            alpha_set = true;
            rank_alpha_flag_seen = true;
        } else if (std.mem.eql(u8, arg, "--layer-name") or std.mem.eql(u8, arg, "--layer")) {
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            layer_name = argv_in[i];
        } else if (std.mem.eql(u8, arg, "--target-preset")) {
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            if (gemma4.parseGemma4LoRATargetPreset(argv_in[i])) |preset| {
                gemma4_target_preset = preset;
            } else {
                target_preset = peft.parseTargetPreset(argv_in[i]) orelse return error.InvalidArguments;
            }
        } else if (std.mem.eql(u8, arg, "--target-modules")) {
            if (target_modules != null) return error.InvalidArguments;
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            target_modules = try parseCsvBorrowed(allocator, argv_in[i]);
        } else if (std.mem.eql(u8, arg, "--use-dora")) {
            use_dora = true;
        } else if (std.mem.eql(u8, arg, "--init-lora-weights")) {
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            init_lora_weights = argv_in[i];
        } else if (!rank_alpha_flag_seen and !rank_set) {
            rank = try std.fmt.parseUnsigned(usize, arg, 10);
            rank_set = true;
        } else if (!rank_alpha_flag_seen and !alpha_set) {
            alpha = try std.fmt.parseFloat(f32, arg);
            alpha_set = true;
        } else if (base_model_name_or_path == null) {
            base_model_name_or_path = arg;
        } else {
            return error.InvalidArguments;
        }
    }
    const selection_count = @intFromBool(target_modules != null) +
        @intFromBool(target_preset != null) +
        @intFromBool(gemma4_target_preset != null);
    if (selection_count > 1) return error.InvalidArguments;

    var summary = try gemma4.bootstrapLoRABundle(allocator, model_dir, out_dir, .{
        .rank = rank,
        .alpha = alpha,
        .base_model_name_or_path = base_model_name_or_path,
        .layer_name = layer_name,
        .target_modules = target_modules,
        .target_preset = target_preset,
        .gemma4_target_preset = gemma4_target_preset,
        .use_dora = use_dora,
        .init_lora_weights = init_lora_weights,
    });
    defer gemma4.freeBootstrapSummary(allocator, &summary);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectTrainEvalGemma4LoraBundle(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    if (argv_in.len < 5) return error.InvalidArguments;
    try train_eval_gemma4_lora_bundle.runFromArgs(allocator, io, argv_in[1..]);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn gemmaMessagesHaveMedia(examples: []const gemma_chat_data.Example) bool {
    for (examples) |example| {
        if (example.image_paths.len > 0 or example.audio_paths.len > 0) return true;
    }
    return false;
}

fn parseCsvBorrowed(allocator: std.mem.Allocator, value: []const u8) ![]const []const u8 {
    var out: std.ArrayListUnmanaged([]const u8) = .empty;
    errdefer out.deinit(allocator);
    var iter = std.mem.splitScalar(u8, value, ',');
    while (iter.next()) |raw| {
        const item = std.mem.trim(u8, raw, " \t\r\n");
        if (item.len == 0) continue;
        try out.append(allocator, item);
    }
    if (out.items.len == 0) return error.InvalidArguments;
    return try out.toOwnedSlice(allocator);
}

fn runDirectBootstrapLayoutlmv3Lora(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len < 3) return error.InvalidArguments;
    const model_dir = argv_in[1];
    const out_dir = argv_in[2];
    var rank = default_lora_rank;
    var alpha = default_lora_alpha;
    var base_model_name_or_path: ?[]const u8 = null;
    var target_modules: ?[]const []const u8 = null;
    defer if (target_modules) |modules| allocator.free(modules);
    var i: usize = 3;
    if (i < argv_in.len and !std.mem.startsWith(u8, argv_in[i], "--")) {
        rank = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
        i += 1;
    }
    if (i < argv_in.len and !std.mem.startsWith(u8, argv_in[i], "--")) {
        alpha = try std.fmt.parseFloat(f32, argv_in[i]);
        i += 1;
    }
    while (i < argv_in.len) : (i += 1) {
        const arg = argv_in[i];
        if (std.mem.eql(u8, arg, "--target-modules")) {
            if (target_modules != null) return error.InvalidArguments;
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            target_modules = try parseCsvBorrowed(allocator, argv_in[i]);
        } else if (base_model_name_or_path == null) {
            base_model_name_or_path = arg;
        } else {
            return error.InvalidArguments;
        }
    }

    var summary = try layoutlmv3.bootstrapLoRABundle(allocator, model_dir, out_dir, .{
        .rank = rank,
        .alpha = alpha,
        .base_model_name_or_path = base_model_name_or_path,
        .target_modules = target_modules,
    });
    defer layoutlmv3.freeBootstrapSummary(allocator, &summary);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectTrainEvalLayoutlmv3LoraSequence(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    if (argv_in.len < 6) return error.InvalidArguments;
    try train_eval_layoutlmv3_lora_sequence.runFromArgs(allocator, io, argv_in[1..]);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectTrainEvalLayoutlmv3LoraToken(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    if (argv_in.len < 6) return error.InvalidArguments;
    try train_eval_layoutlmv3_lora_token.runFromArgs(allocator, io, argv_in[1..]);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectMaterializeLayoutlmv3Checkpoint(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    if (argv_in.len < 5) return error.InvalidArguments;
    var summary = try layoutlmv3.materializeMergedModel(allocator, argv_in[1], argv_in[2], argv_in[3], argv_in[4]);
    defer layoutlmv3.freeMaterializeSummary(allocator, &summary);
    if (argv_in.len >= 6) {
        try writeJsonFile(allocator, io, argv_in[5], summary);
    }
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectBootstrapRerankerLora(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len < 3) return error.InvalidArguments;
    const model_dir = argv_in[1];
    const out_dir = argv_in[2];
    var rank = default_lora_rank;
    var alpha = default_lora_alpha;
    var top_layer_count: usize = 1;
    var base_model_name_or_path: ?[]const u8 = null;
    var target_modules: ?[]const []const u8 = null;
    defer if (target_modules) |modules| allocator.free(modules);
    var i: usize = 3;
    if (i < argv_in.len and !std.mem.startsWith(u8, argv_in[i], "--")) {
        rank = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
        i += 1;
    }
    if (i < argv_in.len and !std.mem.startsWith(u8, argv_in[i], "--")) {
        alpha = try std.fmt.parseFloat(f32, argv_in[i]);
        i += 1;
    }
    if (i < argv_in.len and !std.mem.startsWith(u8, argv_in[i], "--")) {
        top_layer_count = std.fmt.parseUnsigned(usize, argv_in[i], 10) catch |err| switch (err) {
            error.InvalidCharacter => blk: {
                base_model_name_or_path = argv_in[i];
                break :blk top_layer_count;
            },
            else => return err,
        };
        i += 1;
    }
    while (i < argv_in.len) : (i += 1) {
        const arg = argv_in[i];
        if (std.mem.eql(u8, arg, "--target-modules")) {
            if (target_modules != null) return error.InvalidArguments;
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            target_modules = try parseCsvBorrowed(allocator, argv_in[i]);
        } else if (std.mem.eql(u8, arg, "--base-model-name-or-path") or std.mem.eql(u8, arg, "--base-model")) {
            if (base_model_name_or_path != null) return error.InvalidArguments;
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            base_model_name_or_path = argv_in[i];
        } else {
            return error.InvalidArguments;
        }
    }

    var summary = try reranker_lora.bootstrapLoRABundle(allocator, model_dir, out_dir, .{
        .rank = rank,
        .alpha = alpha,
        .top_layer_count = top_layer_count,
        .base_model_name_or_path = base_model_name_or_path,
        .target_modules = target_modules,
    });
    defer reranker_lora.freeBootstrapSummary(allocator, &summary);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectPrepareRerankerTopLayerCache(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len < 4) return error.InvalidArguments;
    const model_dir = argv_in[1];
    const input_path = argv_in[2];
    const out_path = argv_in[3];
    const split = if (argv_in.len >= 5 and !std.mem.startsWith(u8, argv_in[4], "--")) argv_in[4] else null;

    var backend: reranker_head.BackendChoice = .auto;
    var max_examples: usize = 128;
    var top_layer_count: usize = 1;
    var i: usize = if (split == null) 4 else 5;
    while (i < argv_in.len) : (i += 1) {
        const arg = argv_in[i];
        if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i >= argv_in.len) return error.MissingBackendValue;
            backend = parseRerankerBackendChoice(argv_in[i]) orelse return error.InvalidBackend;
        } else if (std.mem.eql(u8, arg, "--max-examples")) {
            i += 1;
            if (i >= argv_in.len) return error.MissingMaxExamples;
            max_examples = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
        } else if (std.mem.eql(u8, arg, "--top-layer-count")) {
            i += 1;
            if (i >= argv_in.len) return error.MissingTopLayerCount;
            top_layer_count = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
        } else {
            return error.InvalidArguments;
        }
    }

    var loaded = try reranker_data.loadExamples(allocator, input_path, split);
    defer loaded.deinit();
    var summary = try reranker_head.prepareCachedTopLayerSummary(
        allocator,
        model_dir,
        input_path,
        split,
        loaded.examples,
        backend,
        max_examples,
        top_layer_count,
    );
    defer reranker_head.freeCachedTopLayerSummary(allocator, &summary);
    try reranker_head.saveCachedTopLayerSummary(allocator, out_path, summary);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectTrainEvalRerankerLoraTopLayerCachedSurrogate(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    if (argv_in.len < 7) return error.InvalidArguments;
    try train_eval_reranker_lora_top_layer_cached_surrogate.runFromArgs(allocator, io, argv_in[1..]);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectMaterializeRerankerLora(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len != 4) return error.InvalidArguments;
    var summary = try reranker_lora.materializeMergedModel(allocator, argv_in[1], argv_in[2], argv_in[3]);
    defer reranker_lora.freeMaterializeSummary(allocator, &summary);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectPrepareColqwen2Inputs(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len < 5) return error.InvalidArguments;
    const model_input = argv_in[1];
    const dataset_root = argv_in[2];
    const examples_jsonl = argv_in[3];
    const out_path = argv_in[4];
    const max_examples = if (argv_in.len >= 6) try std.fmt.parseUnsigned(usize, argv_in[5], 10) else 32;

    const examples = try colqwen2.loadExamples(allocator, examples_jsonl);
    defer colqwen2.freeExamples(allocator, examples);
    var summary = try colqwen2.prepareInputsAgainstExamples(allocator, model_input, dataset_root, examples, max_examples);
    defer colqwen2.freePreparedInputsSummary(allocator, &summary);
    try colqwen2.savePreparedInputsSummary(allocator, out_path, summary);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectBootstrapColqwen2Lora(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len < 3) return error.InvalidArguments;
    const model_dir = argv_in[1];
    const out_dir = argv_in[2];
    var rank = default_lora_rank;
    var alpha = default_lora_alpha;
    var base_model_name_or_path: ?[]const u8 = null;
    var target_modules: ?[]const []const u8 = null;
    defer if (target_modules) |modules| allocator.free(modules);
    var i: usize = 3;
    if (i < argv_in.len and !std.mem.startsWith(u8, argv_in[i], "--")) {
        rank = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
        i += 1;
    }
    if (i < argv_in.len and !std.mem.startsWith(u8, argv_in[i], "--")) {
        alpha = try std.fmt.parseFloat(f32, argv_in[i]);
        i += 1;
    }
    while (i < argv_in.len) : (i += 1) {
        const arg = argv_in[i];
        if (std.mem.eql(u8, arg, "--target-modules")) {
            if (target_modules != null) return error.InvalidArguments;
            i += 1;
            if (i >= argv_in.len) return error.InvalidArguments;
            target_modules = try parseCsvBorrowed(allocator, argv_in[i]);
        } else if (base_model_name_or_path == null) {
            base_model_name_or_path = arg;
        } else {
            return error.InvalidArguments;
        }
    }

    var summary = try colqwen2.bootstrapLoRABundle(allocator, model_dir, out_dir, .{
        .rank = rank,
        .alpha = alpha,
        .base_model_name_or_path = base_model_name_or_path,
        .target_modules = target_modules,
    });
    defer colqwen2.freeBootstrapSummary(allocator, &summary);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectTrainEvalColqwen2LoraBundle(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    if (argv_in.len < 5) return error.InvalidArguments;
    try train_eval_colqwen2_lora_bundle.runFromArgs(allocator, io, argv_in[1..]);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectPrepareRerankerPooledCache(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len < 4) return error.InvalidArguments;
    const model_dir = argv_in[1];
    const input_path = argv_in[2];
    const out_path = argv_in[3];
    const split = if (argv_in.len >= 5 and !std.mem.startsWith(u8, argv_in[4], "--")) argv_in[4] else null;

    var backend: reranker_head.BackendChoice = .auto;
    var max_examples: usize = 256;
    var i: usize = if (split == null) 4 else 5;
    while (i < argv_in.len) : (i += 1) {
        const arg = argv_in[i];
        if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i >= argv_in.len) return error.MissingBackendValue;
            backend = parseRerankerBackendChoice(argv_in[i]) orelse return error.InvalidBackend;
        } else if (std.mem.eql(u8, arg, "--max-examples")) {
            i += 1;
            if (i >= argv_in.len) return error.MissingMaxExamples;
            max_examples = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
        } else {
            return error.InvalidArguments;
        }
    }

    var loaded = try reranker_data.loadExamples(allocator, input_path, split);
    defer loaded.deinit();
    var summary = try reranker_head.prepareCachedPooledSummary(
        allocator,
        model_dir,
        input_path,
        split,
        loaded.examples,
        backend,
        max_examples,
    );
    defer reranker_head.freeCachedPooledSummary(allocator, &summary);
    try reranker_head.saveCachedPooledSummary(allocator, out_path, summary);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectTrainEvalRerankerHeadCached(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len < 5) return error.InvalidArguments;
    const model_dir = argv_in[1];
    const train_cache_path = argv_in[2];
    const eval_cache_path = argv_in[3];
    const out_dir = argv_in[4];

    var backend: reranker_head.BackendChoice = .auto;
    var max_examples: usize = 256;
    var epochs: usize = 1;
    var learning_rate: f32 = 0.001;
    var i: usize = 5;
    while (i < argv_in.len) : (i += 1) {
        const arg = argv_in[i];
        if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i >= argv_in.len) return error.MissingBackendValue;
            backend = parseRerankerBackendChoice(argv_in[i]) orelse return error.InvalidBackend;
        } else if (std.mem.eql(u8, arg, "--max-examples")) {
            i += 1;
            if (i >= argv_in.len) return error.MissingMaxExamples;
            max_examples = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
        } else if (std.mem.eql(u8, arg, "--epochs")) {
            i += 1;
            if (i >= argv_in.len) return error.MissingEpochs;
            epochs = try std.fmt.parseUnsigned(usize, argv_in[i], 10);
        } else if (std.mem.eql(u8, arg, "--learning-rate")) {
            i += 1;
            if (i >= argv_in.len) return error.MissingLearningRate;
            learning_rate = try std.fmt.parseFloat(f32, argv_in[i]);
        } else {
            return error.InvalidArguments;
        }
    }

    var train_summary = try reranker_head.loadCachedPooledSummary(allocator, train_cache_path);
    defer reranker_head.freeCachedPooledSummary(allocator, &train_summary);
    var eval_summary = try reranker_head.loadCachedPooledSummary(allocator, eval_cache_path);
    defer reranker_head.freeCachedPooledSummary(allocator, &eval_summary);
    const summary = try reranker_head.trainEvalHeadCachedSummary(
        allocator,
        model_dir,
        &train_summary,
        &eval_summary,
        out_dir,
        backend,
        epochs,
        .{
            .learning_rate = learning_rate,
            .max_examples = max_examples,
        },
    );
    defer allocator.free(summary.output_dir);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn runDirectMaterializeRerankerHead(allocator: std.mem.Allocator, io: std.Io, argv_in: []const []const u8) !void {
    _ = io;
    if (argv_in.len != 4) return error.InvalidArguments;
    try reranker_head.materializeHeadFromDir(allocator, argv_in[1], argv_in[2], argv_in[3]);
    print("direct adapter: {s}\n", .{argv_in[0]});
}

fn parseRerankerBackendChoice(value: []const u8) ?reranker_head.BackendChoice {
    if (std.mem.eql(u8, value, "auto")) return .auto;
    // Accept both the current "blas" spelling and the legacy "native" alias so
    // recipes that forward "backend": "native" verbatim keep working.
    if (std.mem.eql(u8, value, "blas") or std.mem.eql(u8, value, "native")) return .native;
    if (std.mem.eql(u8, value, "metal")) return .metal;
    return null;
}

const DpoScalarRow = struct {
    policy_chosen_logp: f32,
    policy_rejected_logp: f32,
    ref_chosen_logp: f32,
    ref_rejected_logp: f32,
    chosen_length: ?u32 = null,
    rejected_length: ?u32 = null,
    sft_chosen_loss: ?f32 = null,
};

const DpoTextRow = struct {
    prompt: []const u8,
    chosen: []const u8,
    rejected: []const u8,
    sft_chosen_loss: ?f32 = null,
};

const SftTextRow = struct {
    prompt: []const u8,
    response: ?[]const u8 = null,
    completion: ?[]const u8 = null,
    chosen: ?[]const u8 = null,
};

const DpoBatchOwned = struct {
    policy_chosen_logps: []f32,
    policy_rejected_logps: []f32,
    ref_chosen_logps: []f32,
    ref_rejected_logps: []f32,
    chosen_lengths: []u32,
    rejected_lengths: []u32,
    sft_chosen_loss: []f32,

    fn deinit(self: DpoBatchOwned, allocator: std.mem.Allocator) void {
        allocator.free(self.policy_chosen_logps);
        allocator.free(self.policy_rejected_logps);
        allocator.free(self.ref_chosen_logps);
        allocator.free(self.ref_rejected_logps);
        allocator.free(self.chosen_lengths);
        allocator.free(self.rejected_lengths);
        allocator.free(self.sft_chosen_loss);
    }

    fn batch(self: DpoBatchOwned) preference_loss.PairedBatch {
        return .{
            .policy_chosen_logps = self.policy_chosen_logps,
            .policy_rejected_logps = self.policy_rejected_logps,
            .ref_chosen_logps = self.ref_chosen_logps,
            .ref_rejected_logps = self.ref_rejected_logps,
            .chosen_lengths = self.chosen_lengths,
            .rejected_lengths = self.rejected_lengths,
            .sft_chosen_loss = self.sft_chosen_loss,
        };
    }
};

const DpoPreferenceSamplesOwned = struct {
    arena: std.heap.ArenaAllocator,
    samples: []const preference_harness.PreferenceSample,

    fn deinit(self: *DpoPreferenceSamplesOwned) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

const SftPreparedExamplesOwned = struct {
    examples: []gemma4.PreparedExampleInput,

    fn deinit(self: *SftPreparedExamplesOwned, allocator: std.mem.Allocator) void {
        for (self.examples) |*example| freeGemmaPreparedExample(allocator, example);
        allocator.free(self.examples);
        self.* = undefined;
    }
};

const GrpoTextRow = struct {
    prompt: []const u8,
    target: []const u8,
    image_paths: ?[]const []const u8 = null,
    audio_paths: ?[]const []const u8 = null,
};

const GrpoPromptBatchOwned = struct {
    arena: std.heap.ArenaAllocator,
    prompts: []const []const i32,
    prompt_texts: []const []const u8,
    targets: []const []const u8,

    fn deinit(self: *GrpoPromptBatchOwned) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

const GemmaPreparedPromptBatchOwned = struct {
    allocator: std.mem.Allocator,
    prompts: []const *const gemma4.PreparedExampleInput,
    summaries: []gemma4.PreparedInputsSummary,
    targets: []const []const u8,

    fn deinit(self: *GemmaPreparedPromptBatchOwned) void {
        for (self.summaries) |*summary| gemma4.freePreparedInputsSummary(self.allocator, summary);
        self.allocator.free(self.prompts);
        self.allocator.free(self.summaries);
        for (self.targets) |target| self.allocator.free(target);
        self.allocator.free(self.targets);
        self.* = undefined;
    }
};

const DecoderLogprobScorer = struct {
    allocator: std.mem.Allocator,
    model: *model_manager_mod.LoadedModel,
    max_seq_len: usize,

    fn modelForward(
        ctx: *anyopaque,
        prompts: []const []const i32,
        completion_tokens: []const []const i32,
        out_logps: []f32,
    ) !void {
        const self: *DecoderLogprobScorer = @ptrCast(@alignCast(ctx));
        if (prompts.len != completion_tokens.len or prompts.len != out_logps.len) return error.BatchSizeMismatch;

        const gpt_config = session_factory.getGptConfig(self.model.session) orelse return error.InvalidModelForGeneration;
        var cb = try session_factory.getComputeBackend(self.model.session, self.allocator);
        defer cb.deinit();

        for (prompts, completion_tokens, out_logps) |prompt_ids, completion_ids, *out_logp| {
            if (prompt_ids.len == 0) return error.EmptyPrompt;
            if (completion_ids.len == 0) return error.EmptyCompletion;
            const total_len = prompt_ids.len + completion_ids.len;
            if (total_len > self.max_seq_len) return error.SequenceTooLong;

            const input_ids = try self.allocator.alloc(i64, total_len);
            defer self.allocator.free(input_ids);
            for (prompt_ids, 0..) |token_id, idx| input_ids[idx] = token_id;
            for (completion_ids, 0..) |token_id, idx| input_ids[prompt_ids.len + idx] = token_id;

            const logits = try gpt_arch.forward(&cb, self.allocator, gpt_config, input_ids, 1, total_len, null);
            defer self.allocator.free(logits);

            const vocab_size: usize = @intCast(gpt_config.vocab_size);
            var sum_logp: f32 = 0.0;
            for (completion_ids, 0..) |token_id, comp_idx| {
                const row_idx = prompt_ids.len + comp_idx - 1;
                const row = logits[row_idx * vocab_size ..][0..vocab_size];
                sum_logp += logProbAtToken(row, token_id);
            }
            out_logp.* = sum_logp;
        }
    }

    fn tokenLogprobs(
        ctx: *anyopaque,
        prompt: []const i32,
        completion: []const i32,
        out_per_token_logp: []f32,
    ) !void {
        const self: *DecoderLogprobScorer = @ptrCast(@alignCast(ctx));
        if (completion.len != out_per_token_logp.len) return error.LogpLenMismatch;
        if (prompt.len == 0) return error.EmptyPrompt;
        if (completion.len == 0) return error.EmptyCompletion;

        const gpt_config = session_factory.getGptConfig(self.model.session) orelse return error.InvalidModelForGeneration;
        const total_len = prompt.len + completion.len;
        if (total_len > self.max_seq_len) return error.SequenceTooLong;

        var cb = try session_factory.getComputeBackend(self.model.session, self.allocator);
        defer cb.deinit();

        const input_ids = try self.allocator.alloc(i64, total_len);
        defer self.allocator.free(input_ids);
        for (prompt, 0..) |token_id, idx| input_ids[idx] = token_id;
        for (completion, 0..) |token_id, idx| input_ids[prompt.len + idx] = token_id;

        const logits = try gpt_arch.forward(&cb, self.allocator, gpt_config, input_ids, 1, total_len, null);
        defer self.allocator.free(logits);

        const vocab_size: usize = @intCast(gpt_config.vocab_size);
        for (completion, 0..) |token_id, comp_idx| {
            const row_idx = prompt.len + comp_idx - 1;
            const row = logits[row_idx * vocab_size ..][0..vocab_size];
            out_per_token_logp[comp_idx] = logProbAtToken(row, token_id);
        }
    }
};

const DecoderGrpoSampler = struct {
    allocator: std.mem.Allocator,
    model: *model_manager_mod.LoadedModel,
    max_seq_len: usize,
    max_completion_tokens: usize,

    fn sample(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        prompt: []const i32,
        num_samples: usize,
        out_tokens: *std.ArrayList([]i32),
        out_old_logps: *std.ArrayList([]f32),
    ) !void {
        const self: *DecoderGrpoSampler = @ptrCast(@alignCast(ctx));
        if (prompt.len == 0) return error.EmptyPrompt;
        if (num_samples == 0) return;

        const gpt_config = session_factory.getGptConfig(self.model.session) orelse return error.InvalidModelForGeneration;
        var cb = try session_factory.getComputeBackend(self.model.session, self.allocator);
        defer cb.deinit();

        const top_rank_cap: usize = @min(num_samples, 8);
        const eos_id = self.model.getTokenizer().specialTokens().sep_id;

        for (0..num_samples) |sample_idx| {
            var seq = std.ArrayListUnmanaged(i64).empty;
            defer seq.deinit(allocator);
            try seq.ensureTotalCapacity(allocator, prompt.len + self.max_completion_tokens);
            for (prompt) |token_id| try seq.append(allocator, token_id);

            var completion = std.ArrayListUnmanaged(i32).empty;
            defer completion.deinit(allocator);
            var old_logps = std.ArrayListUnmanaged(f32).empty;
            defer old_logps.deinit(allocator);

            var step: usize = 0;
            while (step < self.max_completion_tokens and seq.items.len < self.max_seq_len) : (step += 1) {
                const logits = try gpt_arch.forward(&cb, self.allocator, gpt_config, seq.items, 1, seq.items.len, null);
                defer self.allocator.free(logits);
                const vocab_size: usize = @intCast(gpt_config.vocab_size);
                const row = logits[(seq.items.len - 1) * vocab_size ..][0..vocab_size];
                const token_id = try selectRankedTokenFromLogits(allocator, row, sample_idx % top_rank_cap);
                const token_logp = logProbAtToken(row, token_id);
                try completion.append(allocator, token_id);
                try old_logps.append(allocator, token_logp);
                try seq.append(allocator, token_id);
                if (eos_id >= 0 and token_id == eos_id) break;
            }

            if (completion.items.len == 0) return error.EmptyCompletion;
            try out_tokens.append(allocator, try completion.toOwnedSlice(allocator));
            try out_old_logps.append(allocator, try old_logps.toOwnedSlice(allocator));
        }
    }
};

const TextRewardMode = enum {
    exact_match,
    exact_match_ci,
    prefix_match,
    token_exact_match,
    token_prefix_match,
};

const TextRewardCtx = struct {
    allocator: std.mem.Allocator,
    tokenizer: tokenizer_mod.Tokenizer,
    targets: []const []const u8,
    mode: TextRewardMode,

    fn score(
        ctx: *anyopaque,
        prompt_idx: usize,
        completion_tokens: []const i32,
    ) !f32 {
        const self: *TextRewardCtx = @ptrCast(@alignCast(ctx));
        if (prompt_idx >= self.targets.len) return error.InvalidPromptIndex;
        const target_trimmed = std.mem.trim(u8, self.targets[prompt_idx], " \t\r\n");
        if (target_trimmed.len == 0) return error.EmptyRewardTarget;

        if (self.mode == .token_exact_match or self.mode == .token_prefix_match) {
            const target_tokens = try self.tokenizer.encode(self.allocator, target_trimmed);
            defer self.allocator.free(target_tokens);
            if (target_tokens.len == 0) return error.EmptyRewardTarget;
            return scoreTokenReward(self.mode, completion_tokens, target_tokens);
        }

        const decoded = try self.tokenizer.decode(self.allocator, completion_tokens);
        defer self.allocator.free(decoded);
        const completion_trimmed = std.mem.trim(u8, decoded, " \t\r\n");
        return scoreTextReward(self.mode, completion_trimmed, target_trimmed);
    }
};

const RewardPhase = enum { train, evaluation };

const RewardProviderTrace = struct {
    name: []const u8,
    kind: []const u8,
    weight: f32,
    reward: f32,
    mode: ?[]const u8 = null,
    executable_path: ?[]const u8 = null,
    executable_digest: ?[]const u8 = null,
    evidence: ?[]const u8 = null,
    request_path: ?[]const u8 = null,
    request_digest: ?[]const u8 = null,
    response_path: ?[]const u8 = null,
    response_digest: ?[]const u8 = null,
    model_path: ?[]const u8 = null,
    model_digest: ?[]const u8 = null,
    tokenizer_path: ?[]const u8 = null,
    tokenizer_digest: ?[]const u8 = null,
    chat_template_path: ?[]const u8 = null,
    chat_template_digest: ?[]const u8 = null,
    calibration_dataset_path: ?[]const u8 = null,
    calibration_dataset_digest: ?[]const u8 = null,
    max_input_tokens: ?usize = null,
    max_batch_size: ?usize = null,
    input_tokens: ?usize = null,
};

const RewardTraceRecord = struct {
    schema_version: []const u8 = "antfly_inference_grpo_reward_trace/v1",
    phase: []const u8,
    call_index: usize,
    prompt_index: usize,
    completion_tokens: []const i32,
    aggregate_reward: f32,
    aggregation: []const u8,
    failure_policy: []const u8,
    pipeline_configuration_digest: []const u8,
    providers: []const RewardProviderTrace,
};

const ExternalRewardFailureTraceRecord = struct {
    schema_version: []const u8 = "antfly_inference_grpo_reward_failure/v1",
    status: []const u8 = "failed",
    phase: []const u8,
    call_index: usize,
    prompt_index: usize,
    completion_tokens: []const i32,
    provider_name: []const u8,
    provider_kind: []const u8,
    executable_path: []const u8,
    executable_digest: []const u8,
    model_path: ?[]const u8 = null,
    model_digest: ?[]const u8 = null,
    tokenizer_path: ?[]const u8 = null,
    tokenizer_digest: ?[]const u8 = null,
    chat_template_path: ?[]const u8 = null,
    chat_template_digest: ?[]const u8 = null,
    calibration_dataset_path: ?[]const u8 = null,
    calibration_dataset_digest: ?[]const u8 = null,
    error_name: []const u8,
    failure_policy: []const u8,
    pipeline_configuration_digest: []const u8,
    request_path: []const u8,
    request_digest: ?[]const u8 = null,
    response_path: []const u8,
    response_digest: ?[]const u8 = null,
    stderr_path: []const u8,
    stderr_digest: ?[]const u8 = null,
};

const ExternalRewardRequest = struct {
    schema_version: []const u8,
    phase: []const u8,
    call_index: usize,
    prompt_index: usize,
    prompt: []const u8,
    target: []const u8,
    completion: []const u8,
    completion_tokens: []const i32,
    model: ?ModelRewardRequestIdentity = null,
};

fn rewardRequestSchemaVersion(provider_kind: []const u8) []const u8 {
    return if (std.mem.eql(u8, provider_kind, "model-command"))
        "antfly_inference_grpo_reward_request/v2"
    else
        "antfly_inference_grpo_reward_request/v1";
}

const ModelRewardRequestIdentity = struct {
    model_path: []const u8,
    model_sha256: []const u8,
    tokenizer_path: []const u8,
    tokenizer_sha256: []const u8,
    chat_template_path: []const u8,
    chat_template_sha256: []const u8,
    calibration_dataset_path: []const u8,
    calibration_dataset_sha256: []const u8,
    max_input_tokens: usize,
    max_batch_size: usize,
};

const ExternalRewardResponse = struct {
    reward: f32,
    evidence: ?[]const u8 = null,
    input_tokens: ?usize = null,
    model_sha256: ?[]const u8 = null,
    tokenizer_sha256: ?[]const u8 = null,
    chat_template_sha256: ?[]const u8 = null,
    calibration_dataset_sha256: ?[]const u8 = null,
};

const ExternalRewardResult = struct {
    reward: f32,
    evidence: ?[]const u8,
    request_path: []const u8,
    request_digest: []const u8,
    response_path: []const u8,
    response_digest: []const u8,
    input_tokens: ?usize,
};

const RewardPipeline = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    tokenizer: tokenizer_mod.Tokenizer,
    prompt_texts: []const []const u8,
    targets: []const []const u8,
    providers: ?[]const RewardProviderConfig,
    legacy_mode: ?TextRewardMode,
    aggregation: []const u8,
    failure_policy: []const u8,
    phase: RewardPhase,
    trace_path: []const u8,
    exchange_dir: []const u8,
    configuration_digest: []const u8,
    max_trace_bytes: usize,
    trace: std.ArrayList(u8) = .empty,
    trace_digest: ?[]const u8 = null,
    call_index: usize = 0,
    external_calls: usize = 0,
    external_failures: usize = 0,
    finished: bool = false,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        recipe: Recipe,
        tokenizer: tokenizer_mod.Tokenizer,
        prompt_texts: []const []const u8,
        targets: []const []const u8,
        phase: RewardPhase,
    ) !RewardPipeline {
        if (prompt_texts.len != targets.len) return error.RewardPromptTargetAlignmentMismatch;
        try validateRewardPipelineConfig(recipe);
        const trace_path = try grpoRewardTracePath(allocator, recipe, phase == .evaluation);
        errdefer allocator.free(trace_path);
        const exchange_dir = try grpoRewardExchangeDir(allocator, recipe);
        errdefer allocator.free(exchange_dir);
        const configuration_digest = try rewardPipelineConfigurationDigestAlloc(allocator, recipe);
        errdefer allocator.free(configuration_digest);
        const reward = recipe.reward;
        const providers = if (reward) |config| config.providers else null;
        const legacy_mode = if (reward == null)
            try parseTextRewardMode(recipe.grpo.reward_mode orelse "exact-match")
        else
            null;

        if (rewardPipelineHasExternalProvider(recipe)) {
            try preflightRewardExecutables(allocator, io, recipe);
            try std.Io.Dir.cwd().createDirPath(io, exchange_dir);
        }
        return .{
            .allocator = allocator,
            .io = io,
            .tokenizer = tokenizer,
            .prompt_texts = prompt_texts,
            .targets = targets,
            .providers = providers,
            .legacy_mode = legacy_mode,
            .aggregation = if (reward) |config| config.aggregation orelse "weighted-mean" else "weighted-mean",
            .failure_policy = if (reward) |config| config.failure_policy orelse "fail" else "fail",
            .phase = phase,
            .trace_path = trace_path,
            .exchange_dir = exchange_dir,
            .configuration_digest = configuration_digest,
            .max_trace_bytes = if (reward) |config| config.max_trace_bytes orelse 64 * 1024 * 1024 else 64 * 1024 * 1024,
        };
    }

    fn deinit(self: *RewardPipeline) void {
        self.trace.deinit(self.allocator);
        self.allocator.free(self.trace_path);
        self.allocator.free(self.exchange_dir);
        self.allocator.free(self.configuration_digest);
        if (self.trace_digest) |digest| self.allocator.free(digest);
        self.* = undefined;
    }

    fn restoreCheckpoint(
        self: *RewardPipeline,
        call_index: usize,
        external_calls: usize,
        external_failures: usize,
        trace: []const u8,
    ) !void {
        if (self.finished or self.trace.items.len != 0 or self.call_index != 0 or
            self.external_calls != 0 or self.external_failures != 0 or
            trace.len > self.max_trace_bytes or external_calls > call_index)
        {
            return error.InvalidPreferenceCheckpointState;
        }
        try self.trace.appendSlice(self.allocator, trace);
        self.call_index = call_index;
        self.external_calls = external_calls;
        self.external_failures = external_failures;
    }

    fn score(ctx: *anyopaque, prompt_idx: usize, completion_tokens: []const i32) !f32 {
        const self: *RewardPipeline = @ptrCast(@alignCast(ctx));
        if (prompt_idx >= self.targets.len or prompt_idx >= self.prompt_texts.len) return error.InvalidPromptIndex;
        const target = std.mem.trim(u8, self.targets[prompt_idx], " \t\r\n");
        if (target.len == 0) return error.EmptyRewardTarget;
        const decoded_owned = try self.tokenizer.decode(self.allocator, completion_tokens);
        defer self.allocator.free(decoded_owned);
        const completion = std.mem.trim(u8, decoded_owned, " \t\r\n");

        var temp = std.heap.ArenaAllocator.init(self.allocator);
        defer temp.deinit();
        const aa = temp.allocator();
        const provider_count = if (self.providers) |providers| providers.len else 1;
        const traces = try aa.alloc(RewardProviderTrace, provider_count);
        var weighted_sum: f64 = 0.0;
        var weight_sum: f64 = 0.0;

        if (self.providers) |providers| {
            for (providers, 0..) |provider, idx| {
                var trace = RewardProviderTrace{
                    .name = provider.name,
                    .kind = provider.kind,
                    .weight = provider.weight,
                    .reward = 0.0,
                    .mode = provider.mode,
                    .executable_path = provider.executable_path,
                    .executable_digest = provider.executable_sha256,
                    .model_path = provider.model_path,
                    .model_digest = provider.model_sha256,
                    .tokenizer_path = provider.tokenizer_path,
                    .tokenizer_digest = provider.tokenizer_sha256,
                    .chat_template_path = provider.chat_template_path,
                    .chat_template_digest = provider.chat_template_sha256,
                    .calibration_dataset_path = provider.calibration_dataset_path,
                    .calibration_dataset_digest = provider.calibration_dataset_sha256,
                    .max_input_tokens = provider.max_input_tokens,
                    .max_batch_size = provider.max_batch_size,
                };
                if (std.mem.eql(u8, provider.kind, "builtin")) {
                    const mode = try parseTextRewardMode(provider.mode.?);
                    trace.reward = try scoreBuiltinReward(
                        self.allocator,
                        self.tokenizer,
                        mode,
                        completion_tokens,
                        completion,
                        target,
                    );
                } else {
                    const external = try self.scoreExternalProvider(
                        aa,
                        provider,
                        prompt_idx,
                        completion_tokens,
                        completion,
                        target,
                    );
                    trace.reward = external.reward;
                    trace.evidence = external.evidence;
                    trace.request_path = external.request_path;
                    trace.request_digest = external.request_digest;
                    trace.response_path = external.response_path;
                    trace.response_digest = external.response_digest;
                    trace.input_tokens = external.input_tokens;
                }
                try validateProviderReward(provider, trace.reward);
                weighted_sum += @as(f64, trace.reward) * @as(f64, provider.weight);
                weight_sum += provider.weight;
                traces[idx] = trace;
            }
        } else {
            const reward = try scoreBuiltinReward(
                self.allocator,
                self.tokenizer,
                self.legacy_mode.?,
                completion_tokens,
                completion,
                target,
            );
            traces[0] = .{
                .name = "legacy-builtin",
                .kind = "builtin",
                .weight = 1.0,
                .reward = reward,
                .mode = @tagName(self.legacy_mode.?),
            };
            weighted_sum = reward;
            weight_sum = 1.0;
        }
        if (!(weight_sum > 0.0) or !std.math.isFinite(weighted_sum)) return error.InvalidAggregateReward;
        const aggregate: f32 = @floatCast(if (std.mem.eql(u8, self.aggregation, "weighted-sum"))
            weighted_sum
        else
            weighted_sum / weight_sum);
        if (!std.math.isFinite(aggregate)) return error.InvalidAggregateReward;

        try self.appendTraceRecord(aa, RewardTraceRecord{
            .phase = @tagName(self.phase),
            .call_index = self.call_index,
            .prompt_index = prompt_idx,
            .completion_tokens = completion_tokens,
            .aggregate_reward = aggregate,
            .aggregation = self.aggregation,
            .failure_policy = self.failure_policy,
            .pipeline_configuration_digest = self.configuration_digest,
            .providers = traces,
        });
        self.call_index += 1;
        return aggregate;
    }

    fn appendTraceRecord(self: *RewardPipeline, allocator: std.mem.Allocator, record: anytype) !void {
        const rendered = try std.json.Stringify.valueAlloc(allocator, record, .{});
        const record_size = std.math.add(usize, rendered.len, 1) catch return error.RewardTraceLimitExceeded;
        const next_trace_size = std.math.add(usize, self.trace.items.len, record_size) catch
            return error.RewardTraceLimitExceeded;
        if (next_trace_size > self.max_trace_bytes) return error.RewardTraceLimitExceeded;
        try self.trace.ensureTotalCapacity(self.allocator, next_trace_size);
        try self.trace.appendSlice(self.allocator, rendered);
        try self.trace.append(self.allocator, '\n');
    }

    fn appendExternalFailureTrace(
        self: *RewardPipeline,
        allocator: std.mem.Allocator,
        provider: RewardProviderConfig,
        prompt_idx: usize,
        completion_tokens: []const i32,
        request_path: []const u8,
        response_path: []const u8,
        stderr_path: []const u8,
        failure: anyerror,
    ) !void {
        const request_digest = sha256FileAlloc(self.allocator, self.io, request_path) catch null;
        defer if (request_digest) |digest| self.allocator.free(digest);
        const response_digest = sha256FileAlloc(self.allocator, self.io, response_path) catch null;
        defer if (response_digest) |digest| self.allocator.free(digest);
        const stderr_digest = sha256FileAlloc(self.allocator, self.io, stderr_path) catch null;
        defer if (stderr_digest) |digest| self.allocator.free(digest);
        self.external_failures +|= 1;
        try self.appendTraceRecord(allocator, ExternalRewardFailureTraceRecord{
            .phase = @tagName(self.phase),
            .call_index = self.call_index,
            .prompt_index = prompt_idx,
            .completion_tokens = completion_tokens,
            .provider_name = provider.name,
            .provider_kind = provider.kind,
            .executable_path = provider.executable_path.?,
            .executable_digest = provider.executable_sha256.?,
            .model_path = provider.model_path,
            .model_digest = provider.model_sha256,
            .tokenizer_path = provider.tokenizer_path,
            .tokenizer_digest = provider.tokenizer_sha256,
            .chat_template_path = provider.chat_template_path,
            .chat_template_digest = provider.chat_template_sha256,
            .calibration_dataset_path = provider.calibration_dataset_path,
            .calibration_dataset_digest = provider.calibration_dataset_sha256,
            .error_name = @errorName(failure),
            .failure_policy = self.failure_policy,
            .pipeline_configuration_digest = self.configuration_digest,
            .request_path = request_path,
            .request_digest = request_digest,
            .response_path = response_path,
            .response_digest = response_digest,
            .stderr_path = stderr_path,
            .stderr_digest = stderr_digest,
        });
    }

    fn scoreExternalProvider(
        self: *RewardPipeline,
        allocator: std.mem.Allocator,
        provider: RewardProviderConfig,
        prompt_idx: usize,
        completion_tokens: []const i32,
        completion: []const u8,
        target: []const u8,
    ) !ExternalRewardResult {
        const stem = try std.fmt.allocPrint(
            allocator,
            "{s}-{d:0>8}-{s}",
            .{ @tagName(self.phase), self.call_index, provider.name },
        );
        const request_name = try std.fmt.allocPrint(allocator, "{s}.request.json", .{stem});
        const response_name = try std.fmt.allocPrint(allocator, "{s}.response.json", .{stem});
        const stderr_name = try std.fmt.allocPrint(allocator, "{s}.stderr.txt", .{stem});
        const request_path = try std.fs.path.join(allocator, &.{ self.exchange_dir, request_name });
        const response_path = try std.fs.path.join(allocator, &.{ self.exchange_dir, response_name });
        const stderr_path = try std.fs.path.join(allocator, &.{ self.exchange_dir, stderr_name });
        errdefer |failure| self.appendExternalFailureTrace(
            allocator,
            provider,
            prompt_idx,
            completion_tokens,
            request_path,
            response_path,
            stderr_path,
            failure,
        ) catch {};

        try verifyRewardProviderArtifacts(self.allocator, self.io, provider);
        if (self.prompt_texts[prompt_idx].len > 1024 * 1024 or
            target.len > 64 * 1024 or completion.len > 1024 * 1024)
        {
            return error.RewardProviderRequestTooLarge;
        }
        const request_argument = try std.fs.path.resolve(allocator, &.{request_path});
        try writeJsonFile(self.allocator, self.io, request_path, ExternalRewardRequest{
            .schema_version = rewardRequestSchemaVersion(provider.kind),
            .phase = @tagName(self.phase),
            .call_index = self.call_index,
            .prompt_index = prompt_idx,
            .prompt = self.prompt_texts[prompt_idx],
            .target = target,
            .completion = completion,
            .completion_tokens = completion_tokens,
            .model = if (std.mem.eql(u8, provider.kind, "model-command")) .{
                .model_path = provider.model_path.?,
                .model_sha256 = provider.model_sha256.?,
                .tokenizer_path = provider.tokenizer_path.?,
                .tokenizer_sha256 = provider.tokenizer_sha256.?,
                .chat_template_path = provider.chat_template_path.?,
                .chat_template_sha256 = provider.chat_template_sha256.?,
                .calibration_dataset_path = provider.calibration_dataset_path.?,
                .calibration_dataset_sha256 = provider.calibration_dataset_sha256.?,
                .max_input_tokens = provider.max_input_tokens.?,
                .max_batch_size = provider.max_batch_size.?,
            } else null,
        });

        var argv_list: std.ArrayList([]const u8) = .empty;
        defer argv_list.deinit(self.allocator);
        try argv_list.append(self.allocator, provider.executable_path.?);
        if (provider.args) |args| try argv_list.appendSlice(self.allocator, args);
        try argv_list.append(self.allocator, request_argument);
        var empty_environment = std.process.Environ.Map.init(self.allocator);
        defer empty_environment.deinit();
        const result = std.process.run(self.allocator, self.io, .{
            .argv = argv_list.items,
            .cwd = .{ .path = "/" },
            .environ_map = &empty_environment,
            .stdout_limit = .limited(64 * 1024),
            .stderr_limit = .limited(64 * 1024),
            .timeout = .{ .duration = .{ .raw = .fromMilliseconds(provider.timeout_ms orelse 10_000), .clock = .awake } },
        }) catch |err| {
            const marker = std.fmt.allocPrint(allocator, "provider invocation failed: {s}\n", .{@errorName(err)}) catch "provider invocation failed\n";
            artifact_publication.writeFileAtomicReplace(self.allocator, self.io, stderr_path, marker) catch {};
            if (err == error.Timeout) return error.RewardProviderTimeout;
            return error.RewardProviderInvocationFailed;
        };
        defer self.allocator.free(result.stdout);
        defer self.allocator.free(result.stderr);
        try artifact_publication.writeFileAtomicReplace(self.allocator, self.io, response_path, result.stdout);
        try artifact_publication.writeFileAtomicReplace(self.allocator, self.io, stderr_path, result.stderr);
        try verifyRewardProviderArtifacts(self.allocator, self.io, provider);
        switch (result.term) {
            .exited => |code| if (code != 0) return error.RewardProviderFailed,
            else => return error.RewardProviderFailed,
        }
        const response = std.json.parseFromSliceLeaky(
            ExternalRewardResponse,
            allocator,
            std.mem.trim(u8, result.stdout, " \t\r\n"),
            .{ .ignore_unknown_fields = false, .allocate = .alloc_always },
        ) catch return error.InvalidRewardProviderResponse;
        if (!std.math.isFinite(response.reward)) return error.NonFiniteRewardProviderResponse;
        if (std.mem.eql(u8, provider.kind, "model-command")) {
            try validateModelRewardResponse(provider, response);
        } else if (response.input_tokens != null or response.model_sha256 != null or
            response.tokenizer_sha256 != null or response.chat_template_sha256 != null or
            response.calibration_dataset_sha256 != null)
        {
            return error.UnexpectedModelRewardAttestation;
        }
        // Validate external bounds while the invocation-scoped errdefer still
        // owns the exchange paths, so an out-of-contract score is represented
        // by a structured failure trace rather than by orphaned files alone.
        try validateProviderReward(provider, response.reward);
        const request_digest_owned = try sha256FileAlloc(self.allocator, self.io, request_path);
        defer self.allocator.free(request_digest_owned);
        const response_digest_owned = try sha256FileAlloc(self.allocator, self.io, response_path);
        defer self.allocator.free(response_digest_owned);
        self.external_calls += 1;
        return .{
            .reward = response.reward,
            .evidence = if (response.evidence) |evidence| try allocator.dupe(u8, evidence) else null,
            .request_path = request_path,
            .request_digest = try allocator.dupe(u8, request_digest_owned),
            .response_path = response_path,
            .response_digest = try allocator.dupe(u8, response_digest_owned),
            .input_tokens = response.input_tokens,
        };
    }

    fn finish(self: *RewardPipeline) !void {
        if (self.finished) return;
        try artifact_publication.writeFileAtomicReplace(self.allocator, self.io, self.trace_path, self.trace.items);
        self.trace_digest = try sha256FileAlloc(self.allocator, self.io, self.trace_path);
        self.finished = true;
    }

    fn telemetry(self: *const RewardPipeline) RewardPipelineTelemetry {
        return .{
            .aggregation = self.aggregation,
            .failure_policy = self.failure_policy,
            .providers = if (self.providers) |providers| providers.len else 1,
            .external_calls = self.external_calls,
            .external_failures = self.external_failures,
            .configuration_digest = self.configuration_digest,
            .trace_path = self.trace_path,
            .trace_digest = self.trace_digest,
        };
    }
};

fn rewardPipelineHasExternalProvider(recipe: Recipe) bool {
    const reward = recipe.reward orelse return false;
    const providers = reward.providers orelse return false;
    for (providers) |provider| {
        if (std.mem.eql(u8, provider.kind, "external-command") or
            std.mem.eql(u8, provider.kind, "model-command")) return true;
    }
    return false;
}

fn preflightRewardExecutables(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
) !void {
    const reward = recipe.reward orelse return;
    const providers = reward.providers orelse return;
    for (providers) |provider| {
        if (!std.mem.eql(u8, provider.kind, "external-command") and
            !std.mem.eql(u8, provider.kind, "model-command")) continue;
        try verifyRewardProviderArtifacts(allocator, io, provider);
    }
}

fn rewardPipelineConfigurationDigestAlloc(
    allocator: std.mem.Allocator,
    recipe: Recipe,
) ![]const u8 {
    const rendered = try std.json.Stringify.valueAlloc(allocator, .{
        .legacy_reward_mode = recipe.grpo.reward_mode,
        .reward = recipe.reward,
    }, .{});
    defer allocator.free(rendered);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(rendered);
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.allocPrint(allocator, "sha256:{s}", .{std.fmt.bytesToHex(digest, .lower)});
}

fn verifyRewardExecutableDigest(
    allocator: std.mem.Allocator,
    io: std.Io,
    provider: RewardProviderConfig,
) !void {
    const stat = try std.Io.Dir.cwd().statFile(io, provider.executable_path.?, .{});
    if (stat.kind != .file) return error.InvalidRewardExecutable;
    const actual = try sha256FileAlloc(allocator, io, provider.executable_path.?);
    defer allocator.free(actual);
    if (!std.ascii.eqlIgnoreCase(actual, provider.executable_sha256.?)) {
        return error.RewardExecutableDigestMismatch;
    }
}

fn verifyRewardProviderArtifacts(
    allocator: std.mem.Allocator,
    io: std.Io,
    provider: RewardProviderConfig,
) !void {
    try verifyRewardExecutableDigest(allocator, io, provider);
    if (!std.mem.eql(u8, provider.kind, "model-command")) return;
    try verifyRewardArtifactDigest(allocator, io, provider.model_path.?, provider.model_sha256.?);
    try verifyRewardArtifactDigest(allocator, io, provider.tokenizer_path.?, provider.tokenizer_sha256.?);
    try verifyRewardArtifactDigest(allocator, io, provider.chat_template_path.?, provider.chat_template_sha256.?);
    try verifyRewardArtifactDigest(allocator, io, provider.calibration_dataset_path.?, provider.calibration_dataset_sha256.?);
}

fn verifyRewardArtifactDigest(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    expected_digest: []const u8,
) !void {
    const fingerprint = try fingerprintPath(allocator, io, "reward_model_artifact", path);
    defer if (fingerprint.digest) |digest| allocator.free(digest);
    if (!fingerprint.exists or fingerprint.digest == null or
        !std.ascii.eqlIgnoreCase(fingerprint.digest.?, expected_digest))
    {
        return error.RewardModelArtifactDigestMismatch;
    }
}

fn validateModelRewardResponse(provider: RewardProviderConfig, response: ExternalRewardResponse) !void {
    const input_tokens = response.input_tokens orelse return error.MissingModelRewardAttestation;
    if (input_tokens == 0 or input_tokens > provider.max_input_tokens.?) return error.ModelRewardTokenLimitExceeded;
    if (!std.ascii.eqlIgnoreCase(response.model_sha256 orelse return error.MissingModelRewardAttestation, provider.model_sha256.?) or
        !std.ascii.eqlIgnoreCase(response.tokenizer_sha256 orelse return error.MissingModelRewardAttestation, provider.tokenizer_sha256.?) or
        !std.ascii.eqlIgnoreCase(response.chat_template_sha256 orelse return error.MissingModelRewardAttestation, provider.chat_template_sha256.?) or
        !std.ascii.eqlIgnoreCase(response.calibration_dataset_sha256 orelse return error.MissingModelRewardAttestation, provider.calibration_dataset_sha256.?))
    {
        return error.ModelRewardIdentityMismatch;
    }
}

fn validateProviderReward(provider: RewardProviderConfig, reward: f32) !void {
    if (!std.math.isFinite(reward)) return error.NonFiniteRewardProviderResponse;
    if (provider.min_reward) |minimum| if (reward < minimum) return error.RewardProviderOutOfBounds;
    if (provider.max_reward) |maximum| if (reward > maximum) return error.RewardProviderOutOfBounds;
}

fn scoreBuiltinReward(
    allocator: std.mem.Allocator,
    tokenizer: tokenizer_mod.Tokenizer,
    mode: TextRewardMode,
    completion_tokens: []const i32,
    completion_trimmed: []const u8,
    target_trimmed: []const u8,
) !f32 {
    if (mode == .token_exact_match or mode == .token_prefix_match) {
        const target_tokens = try tokenizer.encode(allocator, target_trimmed);
        defer allocator.free(target_tokens);
        if (target_tokens.len == 0) return error.EmptyRewardTarget;
        return scoreTokenReward(mode, completion_tokens, target_tokens);
    }
    return scoreTextReward(mode, completion_trimmed, target_trimmed);
}

fn scoreTextReward(mode: TextRewardMode, completion_trimmed: []const u8, target_trimmed: []const u8) f32 {
    return switch (mode) {
        .exact_match => if (std.mem.eql(u8, completion_trimmed, target_trimmed)) 1.0 else 0.0,
        .exact_match_ci => if (std.ascii.eqlIgnoreCase(completion_trimmed, target_trimmed)) 1.0 else 0.0,
        .prefix_match => if (std.mem.startsWith(u8, completion_trimmed, target_trimmed)) 1.0 else 0.0,
        .token_exact_match, .token_prefix_match => unreachable,
    };
}

fn scoreTokenReward(mode: TextRewardMode, completion_tokens: []const i32, target_tokens: []const i32) f32 {
    return switch (mode) {
        .token_exact_match => if (std.mem.eql(i32, completion_tokens, target_tokens)) 1.0 else 0.0,
        .token_prefix_match => if (std.mem.startsWith(i32, completion_tokens, target_tokens)) 1.0 else 0.0,
        .exact_match, .exact_match_ci, .prefix_match => unreachable,
    };
}

fn runDirectSft(allocator: std.mem.Allocator, io: std.Io, recipe: Recipe, report_path: []const u8) !void {
    const path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const format = recipe.dataset.format orelse "text-sft";
    if (!std.mem.eql(u8, format, "text-sft") and !std.mem.eql(u8, format, "rendered-text-sft")) {
        return error.UnsupportedSftFormat;
    }
    const family = recipe.model.family orelse try inferFamily(recipe);
    if (!isQwen35Family(family)) return error.UnsupportedRecipeFamily;
    try runOptimizerBackedQwen2Sft(allocator, io, recipe, path, report_path);
}

fn shouldRunOptimizerBackedQwen35Sft(recipe: Recipe, format: []const u8) !bool {
    if (!std.mem.eql(u8, format, "text-sft") and !std.mem.eql(u8, format, "rendered-text-sft")) return false;
    const family = recipe.model.family orelse try inferFamily(recipe);
    return isQwen35Family(family) and requestsAdapterTraining(recipe);
}

fn runDirectDpo(allocator: std.mem.Allocator, io: std.Io, recipe: Recipe, report_path: []const u8) !void {
    const path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const format = recipe.dataset.format orelse return error.MissingDatasetFormat;
    const mode = try parsePreferenceExecutionMode(recipe);
    try validatePreferenceExecutionContract(recipe, .dpo, mode, format);
    if (std.mem.eql(u8, format, "scalar-logprobs")) {
        const batch = try loadDpoScalarJsonl(allocator, io, path);
        defer batch.deinit(allocator);
        var result = try preference_loss.pairedPreferenceLoss(allocator, batch.batch(), .{
            .kind = .dpo,
            .beta = recipe.preference.beta orelse 0.1,
            .simpo_gamma = recipe.preference.simpo_gamma orelse 0.5,
            .sft_lambda = recipe.preference.sft_lambda orelse 1.0,
            .ipo_tau = recipe.preference.ipo_tau orelse 0.1,
        });
        defer result.deinit();
        try writeJsonFile(allocator, io, report_path, DpoReport{
            .execution_mode = "score",
            .dataset_format = format,
            .examples = batch.policy_chosen_logps.len,
            .loss = result.loss,
            .mean_reward_margin = result.mean_reward_margin,
            .accuracy = result.accuracy,
            .beta = recipe.preference.beta orelse 0.1,
        });
        print("dpo report: {s}\n", .{report_path});
        return;
    }
    if (mode == .train) {
        if (try shouldRunOptimizerBackedQwen2Dpo(recipe, format)) {
            try runOptimizerBackedQwen2Dpo(allocator, io, recipe, path, report_path);
            return;
        }
        if (try shouldRunOptimizerBackedGemmaDpo(recipe, format)) {
            try runOptimizerBackedGemmaDpo(allocator, io, recipe, path, report_path);
            return;
        }
        return error.UnsupportedPreferenceTrainingFamily;
    }

    const policy_path = recipe.model.path orelse return error.MissingModelPath;
    const reference_path = recipe.model.reference_path orelse policy_path;
    const backend_choice = try parseRecipeBackendChoice(recipe.backend);

    var session_manager = backends.SessionManager.init(allocator);
    native_backend_choice.configureSessionPreference(&session_manager, backend_choice);
    var model_manager = model_manager_mod.ModelManager.init(allocator, session_manager);
    defer model_manager.deinit();

    const policy_model = try model_manager.loadFromDir(policy_path);
    const reference_model = if (std.mem.eql(u8, reference_path, policy_path))
        policy_model
    else
        try model_manager.loadFromDir(reference_path);

    var samples = try loadDpoTextPreferenceSamples(allocator, io, path, recipe, policy_model);
    defer samples.deinit();

    var policy_scorer = DecoderLogprobScorer{
        .allocator = allocator,
        .model = policy_model,
        .max_seq_len = recipe.dataset.max_seq_len orelse 2048,
    };
    var ref_scorer = DecoderLogprobScorer{
        .allocator = allocator,
        .model = reference_model,
        .max_seq_len = recipe.dataset.max_seq_len orelse 2048,
    };
    var result = try preference_harness.pairedStep(allocator, .{
        .ctx = &policy_scorer,
        .call = DecoderLogprobScorer.modelForward,
    }, .{
        .ctx = &ref_scorer,
        .call = DecoderLogprobScorer.modelForward,
    }, samples.samples, .{
        .pref = .{
            .kind = .dpo,
            .beta = recipe.preference.beta orelse 0.1,
            .simpo_gamma = recipe.preference.simpo_gamma orelse 0.5,
            .sft_lambda = recipe.preference.sft_lambda orelse 1.0,
            .ipo_tau = recipe.preference.ipo_tau orelse 0.1,
        },
        .reference_from_disabled_adapter = false,
    });
    defer result.deinit();
    try writeJsonFile(allocator, io, report_path, DpoReport{
        .execution_mode = "score",
        .dataset_format = format,
        .examples = samples.samples.len,
        .loss = result.loss,
        .mean_reward_margin = result.mean_reward_margin,
        .accuracy = result.accuracy,
        .beta = recipe.preference.beta orelse 0.1,
    });
    print("dpo report: {s}\n", .{report_path});
}

fn shouldRunOptimizerBackedGemmaDpo(recipe: Recipe, format: []const u8) !bool {
    if (!std.mem.eql(u8, format, "text-preference") and !std.mem.eql(u8, format, "rendered-text-preference")) return false;
    const family = recipe.model.family orelse try inferFamily(recipe);
    if (!eqlAny(family, &.{ "gemma4", "gemma" })) return false;
    return recipe.artifacts.trained_adapter_dir != null or recipe.artifacts.adapter_dir != null or recipe.adapter != null;
}

fn shouldRunOptimizerBackedQwen2Dpo(recipe: Recipe, format: []const u8) !bool {
    if (!std.mem.eql(u8, format, "text-preference") and !std.mem.eql(u8, format, "rendered-text-preference")) return false;
    const family = recipe.model.family orelse try inferFamily(recipe);
    if (!isQwen35Family(family) and !eqlAny(family, &.{ "qwen2", "qwen", "colqwen2", "colqwen", "qwen2vl" })) return false;
    return recipe.artifacts.trained_adapter_dir != null or recipe.artifacts.adapter_dir != null or recipe.adapter != null;
}

fn runOptimizerBackedQwen2Sft(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    dataset_path: []const u8,
    report_path: []const u8,
) !void {
    const base_model_dir = recipe.model.path orelse return error.MissingModelPath;
    const adapter = recipe.adapter orelse AdapterConfig{};
    const bootstrap_dir_config = adapter.path orelse adapterBootstrapDir(recipe);
    const bootstrap_dir = bootstrap_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    defer if (bootstrap_dir_config == null) allocator.free(bootstrap_dir);
    const trained_dir_config = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir;
    const trained_dir = trained_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    defer if (trained_dir_config == null) allocator.free(trained_dir);
    const backend_kind: qwen2_real_autodiff.BackendKind = .native;
    const max_examples = recipe.dataset.max_examples orelse 32;
    const max_seq_len = recipe.dataset.max_seq_len orelse 512;
    const family = recipe.model.family orelse try inferFamily(recipe);
    const default_target_modules = qwenLoraTargetModulesForFamily(family);
    try validateNonGemmaAdapterOptions(adapter);
    const bootstrap_target_modules = try adapterTargetModulesForQwen(adapter, default_target_modules);

    compat.cwd().access(compat.io(), bootstrap_dir, .{}) catch {
        var bootstrap = try colqwen2.bootstrapLoRABundle(allocator, base_model_dir, bootstrap_dir, .{
            .rank = adapterRank(adapter, .lora_sft),
            .alpha = adapterAlpha(adapter),
            .base_model_name_or_path = adapter.base_model_name_or_path,
            .target_modules = bootstrap_target_modules,
        });
        defer colqwen2.freeBootstrapSummary(allocator, &bootstrap);
    };

    var session_manager = backends.SessionManager.init(allocator);
    native_backend_choice.configureSessionPreference(&session_manager, try parseRecipeBackendChoice(recipe.backend));
    var model_manager = model_manager_mod.ModelManager.init(allocator, session_manager);
    defer model_manager.deinit();
    const tokenizer_model = try model_manager.loadFromDir(base_model_dir);

    var prepared = try loadSftPreparedExamples(allocator, io, dataset_path, recipe, tokenizer_model, max_examples, max_seq_len);
    defer prepared.deinit(allocator);

    const graph_config = try qwen2_real_autodiff.loadGraphConfig(allocator, base_model_dir);
    var backend = try qwen2_real_autodiff.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();

    var adapter_inspect = try colqwen2.inspectCheckpoint(allocator, bootstrap_dir);
    defer colqwen2.freeInspectionSummary(allocator, &adapter_inspect);
    const lora_rank = adapter_inspect.lora_rank orelse return error.MissingAdapterConfig;
    const lora_alpha = @as(f32, @floatCast(adapter_inspect.lora_alpha orelse return error.MissingAdapterConfig));
    const target_modules = adapter_inspect.target_modules orelse bootstrap_target_modules;
    const lora_config = ml.graph.lora.LoRAConfig{
        .rank = @intCast(lora_rank),
        .alpha = lora_alpha,
        .target_patterns = target_modules,
    };

    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
        .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
        .grad_accum_steps = recipe.optimizer.gradient_accumulation_steps orelse 1,
        .hidden_size_hint = graph_config.arch.hidden_size,
        .num_layers_hint = graph_config.arch.num_hidden_layers,
    });
    defer trainer.deinit();

    var ctx = qwen2_real_autodiff.Qwen2AutodiffCtx.init(graph_config);
    const bootstrap_example = qwen2_real_autodiff.findFirstSupervisedExample(prepared.examples) orelse return error.NoTrainingData;
    try qwen2_real_autodiff.initializeTrainerFromAdapterDir(allocator, &trainer, &ctx, bootstrap_dir, bootstrap_example, @intCast(max_seq_len));

    const epochs = recipe.optimizer.epochs orelse 1;
    var total_loss: f64 = 0.0;
    var examples_seen: usize = 0;
    var supervised_tokens: usize = 0;

    var epoch_idx: usize = 0;
    while (epoch_idx < epochs) : (epoch_idx += 1) {
        for (prepared.examples) |*example| {
            if (example.num_supervised_tokens == 0) continue;
            var input = try qwen2_real_autodiff.makeTrainerInputForExample(allocator, &ctx, example, @intCast(max_seq_len));
            errdefer input.deinit(allocator);
            const step = try trainer.step(input.trainer_input);
            input.deinit(allocator);
            total_loss += step.loss;
            examples_seen += 1;
            supervised_tokens += example.num_supervised_tokens;
        }
    }
    if (examples_seen == 0) return error.NoTrainingData;

    try qwen2_real_autodiff.saveTrainerAsQwenAdapterDir(allocator, &trainer, base_model_dir, bootstrap_dir, trained_dir);

    const denom = @as(f64, @floatFromInt(examples_seen));
    try writeJsonFile(allocator, io, report_path, SftReport{
        .examples = examples_seen,
        .supervised_tokens = supervised_tokens,
        .loss = @floatCast(total_loss / denom),
        .epochs = epochs,
        .trained_adapter_dir = trained_dir,
    });
    print("sft report: {s}\ntrained adapter: {s}\n", .{ report_path, trained_dir });
}

const DpoTextRowsOwned = struct {
    arena: std.heap.ArenaAllocator,
    rows: []DpoTextRow,

    fn deinit(self: *DpoTextRowsOwned) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

fn loadDpoTextRows(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    max_examples: usize,
) !DpoTextRowsOwned {
    const raw = try readFileMax(allocator, io, path, 256 * 1024 * 1024);
    defer allocator.free(raw);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const aa = arena.allocator();

    var rows: std.ArrayListUnmanaged(DpoTextRow) = .empty;
    errdefer rows.deinit(aa);
    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;
        if (rows.items.len >= max_examples) break;
        try rows.append(aa, try std.json.parseFromSliceLeaky(DpoTextRow, aa, line, .{ .ignore_unknown_fields = true }));
    }
    if (rows.items.len == 0) return error.EmptyBatch;
    return .{
        .arena = arena,
        .rows = try rows.toOwnedSlice(aa),
    };
}

fn loadSftPreparedExamples(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    recipe: Recipe,
    model: *model_manager_mod.LoadedModel,
    max_examples: usize,
    max_seq_len: usize,
) !SftPreparedExamplesOwned {
    const raw = try readFileMax(allocator, io, path, 256 * 1024 * 1024);
    defer allocator.free(raw);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var examples: std.ArrayList(gemma4.PreparedExampleInput) = .empty;
    errdefer {
        for (examples.items) |*example| freeGemmaPreparedExample(allocator, example);
        examples.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;
        if (examples.items.len >= max_examples) break;
        const row = try std.json.parseFromSliceLeaky(SftTextRow, aa, line, .{ .ignore_unknown_fields = true });
        const completion = row.response orelse row.completion orelse row.chosen orelse return error.EmptyCompletion;
        var prepared = try tokenizeSftTextRow(allocator, model, recipe, row.prompt, completion, max_seq_len);
        errdefer freeGemmaPreparedExample(allocator, &prepared);
        try examples.append(allocator, prepared);
    }
    if (examples.items.len == 0) return error.EmptyBatch;
    return .{ .examples = try examples.toOwnedSlice(allocator) };
}

fn tokenizeSftTextRow(
    allocator: std.mem.Allocator,
    model: *model_manager_mod.LoadedModel,
    recipe: Recipe,
    prompt: []const u8,
    completion: []const u8,
    max_seq_len: usize,
) !gemma4.PreparedExampleInput {
    const tokenizer = model.getTokenizer();
    const render_prompt = !std.mem.eql(u8, recipe.dataset.format orelse "text-sft", "rendered-text-sft");
    const prompt_text = if (render_prompt)
        try renderDpoPrompt(allocator, model, prompt)
    else
        try allocator.dupe(u8, prompt);
    defer allocator.free(prompt_text);

    var prompt_encoded = try generation.encodePromptForGeneration(
        tokenizer,
        allocator,
        prompt_text,
        max_seq_len,
        model.manifest.add_bos_token,
        model.manifest.bos_token,
    );
    defer prompt_encoded.deinit();

    const prompt_len = countAttentionMask(prompt_encoded.attention_mask);
    if (prompt_len == 0) return error.EmptyPrompt;
    const remaining_budget = max_seq_len - prompt_len;
    if (remaining_budget == 0) return error.NoCompletionBudget;

    const completion_tokens = try tokenizeCompletion(allocator, tokenizer, completion, remaining_budget);
    defer allocator.free(completion_tokens);
    if (completion_tokens.len == 0) return error.EmptyCompletion;

    const prompt_tokens = try allocator.alloc(i32, prompt_len);
    defer allocator.free(prompt_tokens);
    for (0..prompt_len) |idx| prompt_tokens[idx] = prompt_encoded.ids[idx];
    return buildGemmaPreparedExampleFromTokens(allocator, prompt_tokens, completion_tokens, max_seq_len);
}

const GemmaDpoReferenceCache = struct {
    allocator: std.mem.Allocator,
    chosen_logps: []f32,
    rejected_logps: []f32,
    precompute_seconds: f64,
    base_equivalent_policy: bool,

    fn deinit(self: *GemmaDpoReferenceCache) void {
        self.allocator.free(self.chosen_logps);
        self.allocator.free(self.rejected_logps);
        self.* = undefined;
    }
};

const GemmaGrpoReferenceCache = struct {
    const Entry = struct {
        prompt_idx: usize,
        completion_tokens: []i32,
        reference_logps: []f32,
    };

    allocator: std.mem.Allocator,
    capacity: usize,
    entries: std.ArrayList(Entry) = .empty,
    next_evict: usize = 0,
    hits: usize = 0,
    misses: usize = 0,

    fn init(allocator: std.mem.Allocator, capacity: usize) GemmaGrpoReferenceCache {
        return .{
            .allocator = allocator,
            .capacity = capacity,
        };
    }

    fn deinit(self: *GemmaGrpoReferenceCache) void {
        for (self.entries.items) |entry| {
            self.allocator.free(entry.completion_tokens);
            self.allocator.free(entry.reference_logps);
        }
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    fn lookup(
        self: *GemmaGrpoReferenceCache,
        prompt_idx: usize,
        completion_tokens: []const i32,
        out_reference_logps: []f32,
    ) !bool {
        if (completion_tokens.len != out_reference_logps.len) return error.LogpLenMismatch;
        for (self.entries.items) |entry| {
            if (entry.prompt_idx != prompt_idx) continue;
            if (!std.mem.eql(i32, entry.completion_tokens, completion_tokens)) continue;
            if (entry.reference_logps.len != out_reference_logps.len) return error.CorruptGrpoReferenceCache;
            @memcpy(out_reference_logps, entry.reference_logps);
            self.hits += 1;
            return true;
        }
        self.misses += 1;
        return false;
    }

    fn contains(
        self: *const GemmaGrpoReferenceCache,
        prompt_idx: usize,
        completion_tokens: []const i32,
    ) bool {
        for (self.entries.items) |entry| {
            if (entry.prompt_idx != prompt_idx) continue;
            if (std.mem.eql(i32, entry.completion_tokens, completion_tokens)) return true;
        }
        return false;
    }

    fn insert(
        self: *GemmaGrpoReferenceCache,
        prompt_idx: usize,
        completion_tokens: []const i32,
        reference_logps: []const f32,
    ) !void {
        if (completion_tokens.len != reference_logps.len) return error.LogpLenMismatch;
        if (self.capacity == 0) return;
        const owned_tokens = try self.allocator.dupe(i32, completion_tokens);
        errdefer self.allocator.free(owned_tokens);
        const owned_logps = try self.allocator.dupe(f32, reference_logps);
        errdefer self.allocator.free(owned_logps);
        const entry = Entry{
            .prompt_idx = prompt_idx,
            .completion_tokens = owned_tokens,
            .reference_logps = owned_logps,
        };
        if (self.entries.items.len < self.capacity) {
            try self.entries.append(self.allocator, entry);
            return;
        }

        const victim = &self.entries.items[self.next_evict];
        self.allocator.free(victim.completion_tokens);
        self.allocator.free(victim.reference_logps);
        victim.* = entry;
        self.next_evict = (self.next_evict + 1) % self.capacity;
    }

    fn telemetry(self: *const GemmaGrpoReferenceCache) GrpoReferenceCacheTelemetry {
        return .{
            .capacity = self.capacity,
            .entries = self.entries.items.len,
            .hits = self.hits,
            .misses = self.misses,
        };
    }
};

/// Scores every one-token completion for a shared prompt with one frozen-base
/// sparse projection. Cache accounting remains completion-granular: existing
/// entries are hits, the first uncached occurrence is a miss and insertion,
/// and a duplicate later in the same group observes that insertion as a hit.
fn cachedSingleTokenGroupReferenceLogps(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_real_autodiff.GemmaAutodiffCtx,
    prompt: []const i32,
    prompt_idx: usize,
    sampled_tokens: []const std.ArrayList(i32),
    seq_len: u32,
    frozen_lora: *const gemma4_real_autodiff.FrozenBaseLoraBindings,
    reference_cache: *GemmaGrpoReferenceCache,
    out_logps: []f32,
) !void {
    if (sampled_tokens.len == 0 or sampled_tokens.len != out_logps.len) {
        return error.InvalidCompletionGroup;
    }
    const candidate_token_ids = try allocator.alloc(i32, sampled_tokens.len);
    defer allocator.free(candidate_token_ids);

    var any_miss = false;
    for (sampled_tokens, candidate_token_ids) |tokens, *token_id| {
        if (tokens.items.len != 1) return error.ExpectedSingleTokenCompletion;
        token_id.* = tokens.items[0];
        any_miss = any_miss or !reference_cache.contains(prompt_idx, tokens.items);
    }

    if (any_miss) {
        try gemma4_real_autodiff.singleTokenCandidateLogprobsForPromptFrozenBase(
            allocator,
            trainer,
            ctx,
            prompt,
            candidate_token_ids,
            seq_len,
            out_logps,
            frozen_lora,
        );
    }

    for (sampled_tokens, 0..) |tokens, completion_idx| {
        var cached = [_]f32{0.0};
        if (try reference_cache.lookup(prompt_idx, tokens.items, &cached)) {
            out_logps[completion_idx] = cached[0];
        } else {
            if (!any_miss) return error.CorruptGrpoReferenceCache;
            try reference_cache.insert(
                prompt_idx,
                tokens.items,
                out_logps[completion_idx .. completion_idx + 1],
            );
        }
    }
}

const CompletionGroupLogps = struct {
    allocator: std.mem.Allocator,
    flat: []f32,
    rows: [][]f32,

    fn init(allocator: std.mem.Allocator, sampled_tokens: []const std.ArrayList(i32)) !CompletionGroupLogps {
        var total_tokens: usize = 0;
        for (sampled_tokens) |tokens| {
            if (tokens.items.len == 0) return error.EmptyCompletion;
            total_tokens = std.math.add(usize, total_tokens, tokens.items.len) catch
                return error.InvalidCompletionGroup;
        }
        const flat = try allocator.alloc(f32, total_tokens);
        errdefer allocator.free(flat);
        const rows = try allocator.alloc([]f32, sampled_tokens.len);
        errdefer allocator.free(rows);
        var offset: usize = 0;
        for (sampled_tokens, rows) |tokens, *row| {
            row.* = flat[offset .. offset + tokens.items.len];
            offset += tokens.items.len;
        }
        return .{ .allocator = allocator, .flat = flat, .rows = rows };
    }

    fn deinit(self: *CompletionGroupLogps) void {
        self.allocator.free(self.rows);
        self.allocator.free(self.flat);
        self.* = undefined;
    }
};

fn completionTokenSlices(
    allocator: std.mem.Allocator,
    sampled_tokens: []const std.ArrayList(i32),
) ![][]const i32 {
    const completions = try allocator.alloc([]const i32, sampled_tokens.len);
    for (sampled_tokens, completions) |tokens, *completion| completion.* = tokens.items;
    return completions;
}

fn gemmaGrpoMultiTokenBatchEnabled(group_size: usize) bool {
    if (!platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_BATCH_MULTI_TOKEN_SCORING", false)) return false;
    const max_batch = platform.env.getenvUsize("ANTFLY_GEMMA4_GRPO_MULTI_TOKEN_MAX_BATCH") orelse 4;
    return max_batch != 0 and group_size <= max_batch;
}

/// Research lane for one contamination-free multi-token GRPO backward per
/// completion group. Independent batch rows preserve causal/RoPE isolation,
/// while one sparse weighted objective sums every completion contribution
/// before the optimizer boundary. Keep default-off until E2B and E4B pass the
/// exact trajectory and peak-memory gates against the serial rollback.
fn gemmaGrpoMultiTokenBackwardBatchSize(group_size: usize) usize {
    if (!platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_BATCH_MULTI_TOKEN_BACKWARD", false)) return 1;
    const max_batch = platform.env.getenvUsize("ANTFLY_GEMMA4_GRPO_MULTI_TOKEN_BACKWARD_MAX_BATCH") orelse 4;
    if (max_batch < 2) return 1;
    return @min(group_size, max_batch);
}

/// Qualified on real E2B and E4B Metal GRPO with two- and four-token
/// completions. Sampling and both policy/reference rescoring must select the
/// same projection geometry or the on-policy sampling/rescore contract is not
/// numerically stable. Keep this as one switch rather than independently
/// configurable phases; setting it to false is the production rollback.
fn gemmaGrpoSparseMultiTokenEnabled() bool {
    return platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_SPARSE_MULTI_TOKEN", true);
}

/// The exact E2B/E4B replay campaign passes, but the first serial paged-decode
/// implementation is slower than the qualified full-prefix sampler. Keep it
/// default-off until prompt/prefix-aware active-candidate batching beats the
/// rollback on both model sizes. The shadow gate runs the legacy full-prefix
/// sampler and the incremental sampler against the same live adapter, then
/// rejects any token or f32-logprob bit drift.
fn gemmaGrpoIncrementalKvEnabled(recipe: Recipe) bool {
    if (recipe.runtime) |runtime| if (runtime.grpo_incremental_kv) |enabled| return enabled;
    return platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_INCREMENTAL_KV", false);
}

fn gemmaGrpoIncrementalKvShadowExactEnabled(recipe: Recipe) bool {
    if (recipe.runtime) |runtime| if (runtime.grpo_incremental_kv_shadow_exact) |enabled| return enabled;
    return platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_INCREMENTAL_KV_SHADOW_EXACT", false);
}

/// Incremental KV is itself default-off. Within that research lane, batch
/// active candidates that have the same decode position so Q/V LoRA and the
/// vocabulary head execute once for the group. Set false for the serial
/// incremental rollback used by the performance gate.
fn gemmaGrpoIncrementalKvBatchActiveEnabled(recipe: Recipe) bool {
    if (recipe.runtime) |runtime| if (runtime.grpo_incremental_kv_batch_active) |enabled| return enabled;
    return platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_INCREMENTAL_KV_BATCH_ACTIVE", true);
}

/// Research-only device fan-out of a single qualified segmented prompt-tail
/// replay. This avoids replaying the same prompt tail for every GRPO candidate
/// while keeping each subsequent decode page independently writable.
fn gemmaGrpoIncrementalKvClonePromptTailEnabled(recipe: Recipe) bool {
    if (recipe.runtime) |runtime| if (runtime.grpo_incremental_kv_clone_prompt_tail) |enabled| return enabled;
    return platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_INCREMENTAL_KV_CLONE_PROMPT_TAIL", false);
}

fn sampleGemmaGrpoCompletionGroup(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_real_autodiff.GemmaAutodiffCtx,
    incremental_sampler: ?*gemma4_real_autodiff.GrpoIncrementalKvSampler,
    incremental_shadow_exact: bool,
    prompt: []const i32,
    seq_len: u32,
    max_completion_tokens: usize,
    rank_cap: usize,
    eos_token_id: ?i32,
    sparse_multi_token_projection: bool,
    out_tokens: []std.ArrayList(i32),
    out_logps: []std.ArrayList(f32),
) !void {
    const sampler = incremental_sampler orelse return gemma4_real_autodiff.sampleCompletionGroupRanked(
        allocator,
        trainer,
        ctx,
        prompt,
        seq_len,
        max_completion_tokens,
        rank_cap,
        eos_token_id,
        sparse_multi_token_projection,
        out_tokens,
        out_logps,
    );

    const run_shadow = incremental_shadow_exact and sampler.telemetry.groups == 0;
    var shadow_tokens: ?[]std.ArrayList(i32) = null;
    defer if (shadow_tokens) |lists| {
        for (lists) |*tokens| tokens.deinit(allocator);
        allocator.free(lists);
    };
    var shadow_logps: ?[]std.ArrayList(f32) = null;
    defer if (shadow_logps) |lists| {
        for (lists) |*logps| logps.deinit(allocator);
        allocator.free(lists);
    };
    if (run_shadow) {
        const baseline_tokens = try allocator.alloc(std.ArrayList(i32), out_tokens.len);
        shadow_tokens = baseline_tokens;
        const baseline_logps = try allocator.alloc(std.ArrayList(f32), out_logps.len);
        shadow_logps = baseline_logps;
        for (baseline_tokens, baseline_logps) |*tokens, *logps| {
            tokens.* = .empty;
            logps.* = .empty;
        }
        try gemma4_real_autodiff.sampleCompletionGroupRanked(
            allocator,
            trainer,
            ctx,
            prompt,
            seq_len,
            max_completion_tokens,
            rank_cap,
            eos_token_id,
            sparse_multi_token_projection,
            baseline_tokens,
            baseline_logps,
        );
    }

    try sampler.sampleCompletionGroupRanked(
        prompt,
        seq_len,
        max_completion_tokens,
        rank_cap,
        eos_token_id,
        out_tokens,
        out_logps,
    );

    // Paged one-token decode and the qualified fixed-shape graph can select
    // identical tokens while differing by a few f32 bits in their reduction
    // order. Canonicalize the sampled-policy log-probabilities with the same
    // sparse, sequence-wide graph used by the legacy sampler. This retains
    // incremental KV for token selection and replaces N per-token full-prefix
    // forwards with one exact full-sequence rescore per completion.
    for (out_tokens, out_logps) |tokens, *logps| {
        if (tokens.items.len != logps.items.len) return error.GrpoIncrementalKvLogprobParityFailed;
        try gemma4_real_autodiff.tokenLogprobsForPromptCompletionSparseRows(
            allocator,
            trainer,
            ctx,
            prompt,
            tokens.items,
            seq_len,
            logps.items,
        );
        sampler.telemetry.exact_logprob_rescore_forwards += 1;
    }

    if (shadow_tokens) |baseline_tokens| {
        const baseline_logps = shadow_logps.?;
        for (baseline_tokens, out_tokens, 0..) |want_tokens, got_tokens, candidate_index| {
            if (!std.mem.eql(i32, want_tokens.items, got_tokens.items)) {
                std.log.err(
                    "GRPO incremental-KV shadow token mismatch candidate={d} expected={any} actual={any}",
                    .{ candidate_index, want_tokens.items, got_tokens.items },
                );
                return error.GrpoIncrementalKvTokenParityFailed;
            }
        }
        // The legacy sampler records log-softmax values from its selection
        // forward, while the product incremental path deliberately replaces
        // placeholder decode logprobs with the canonical sparse sequence-wide
        // rescore above. Those two qualified graphs can differ by a few F32
        // ULPs, especially for direct Q4_0 GGUF weights. Canonicalize the
        // shadow through the same authoritative scorer before demanding bit
        // equality. Token selection remains an independent exact comparison,
        // so this cannot mask a paged-decode decision change.
        for (baseline_tokens, baseline_logps) |tokens, *logps| {
            if (tokens.items.len != logps.items.len) return error.GrpoIncrementalKvLogprobParityFailed;
            try gemma4_real_autodiff.tokenLogprobsForPromptCompletionSparseRows(
                allocator,
                trainer,
                ctx,
                prompt,
                tokens.items,
                seq_len,
                logps.items,
            );
        }
        for (baseline_logps, out_logps, 0..) |want_logps, got_logps, candidate_index| {
            if (want_logps.items.len != got_logps.items.len) return error.GrpoIncrementalKvLogprobParityFailed;
            for (want_logps.items, got_logps.items, 0..) |want, got, token_index| {
                if (@as(u32, @bitCast(want)) != @as(u32, @bitCast(got))) {
                    std.log.err(
                        "GRPO incremental-KV shadow logprob mismatch candidate={d} token={d} expected_bits=0x{x} actual_bits=0x{x}",
                        .{ candidate_index, token_index, @as(u32, @bitCast(want)), @as(u32, @bitCast(got)) },
                    );
                    return error.GrpoIncrementalKvLogprobParityFailed;
                }
            }
        }
    }
}

/// Scores an entire multi-token frozen-reference group in one sparse batch.
/// Cache lookup/insertion remains completion-granular and ordered, including
/// duplicate candidates. A false return leaves the cache telemetry untouched
/// and asks the caller to execute the legacy per-completion scorer.
fn cachedMultiTokenGroupReferenceLogps(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_real_autodiff.GemmaAutodiffCtx,
    prompt: []const i32,
    prompt_idx: usize,
    sampled_tokens: []const std.ArrayList(i32),
    seq_len: u32,
    frozen_lora: *const gemma4_real_autodiff.FrozenBaseLoraBindings,
    reference_cache: *GemmaGrpoReferenceCache,
    out_logps: *CompletionGroupLogps,
) !bool {
    if (sampled_tokens.len == 0 or sampled_tokens.len != out_logps.rows.len) {
        return error.InvalidCompletionGroup;
    }
    var any_miss = false;
    for (sampled_tokens) |tokens| {
        if (tokens.items.len == 0) return error.EmptyCompletion;
        any_miss = any_miss or !reference_cache.contains(prompt_idx, tokens.items);
    }

    if (any_miss) {
        const completions = try completionTokenSlices(allocator, sampled_tokens);
        defer allocator.free(completions);
        if (!try gemma4_real_autodiff.tokenLogprobsForPromptCompletionGroupSparseRowsFrozenBase(
            allocator,
            trainer,
            ctx,
            prompt,
            completions,
            seq_len,
            out_logps.rows,
            frozen_lora,
        )) return false;
    }

    for (sampled_tokens, out_logps.rows) |tokens, output| {
        if (try reference_cache.lookup(prompt_idx, tokens.items, output)) continue;
        if (!any_miss) return error.CorruptGrpoReferenceCache;
        try reference_cache.insert(prompt_idx, tokens.items, output);
    }
    return true;
}

fn batchedMultiTokenGroupReferenceLogps(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_real_autodiff.GemmaAutodiffCtx,
    prompt: []const i32,
    prompt_idx: usize,
    sampled_tokens: []const std.ArrayList(i32),
    seq_len: u32,
    frozen_lora: *const gemma4_real_autodiff.FrozenBaseLoraBindings,
    reference_cache: *GemmaGrpoReferenceCache,
) !?CompletionGroupLogps {
    var logps = try CompletionGroupLogps.init(allocator, sampled_tokens);
    errdefer logps.deinit();
    if (!try cachedMultiTokenGroupReferenceLogps(
        allocator,
        trainer,
        ctx,
        prompt,
        prompt_idx,
        sampled_tokens,
        seq_len,
        frozen_lora,
        reference_cache,
        &logps,
    )) {
        logps.deinit();
        return null;
    }
    return logps;
}

fn batchedMultiTokenGroupPolicyLogps(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_real_autodiff.GemmaAutodiffCtx,
    prompt: []const i32,
    sampled_tokens: []const std.ArrayList(i32),
    seq_len: u32,
) !?CompletionGroupLogps {
    var logps = try CompletionGroupLogps.init(allocator, sampled_tokens);
    errdefer logps.deinit();
    const completions = try completionTokenSlices(allocator, sampled_tokens);
    defer allocator.free(completions);
    if (!try gemma4_real_autodiff.tokenLogprobsForPromptCompletionGroupSparseRows(
        allocator,
        trainer,
        ctx,
        prompt,
        completions,
        seq_len,
        logps.rows,
    )) {
        logps.deinit();
        return null;
    }
    return logps;
}

const DpoBenchmarkRecorder = struct {
    const warmup_updates = 3;
    const measured_updates = 20;
    const total_updates = 2 + warmup_updates + measured_updates;

    allocator: std.mem.Allocator,
    updates_seen: usize = 0,
    cold_seconds: f64 = 0.0,
    cold_loss: f32 = 0.0,
    first_seconds: f64 = 0.0,
    first_loss: f32 = 0.0,
    warmup_seconds: []f64,
    warmup_losses: []f32,
    measured_seconds: []f64,
    measured_losses: []f32,

    fn init(allocator: std.mem.Allocator) !DpoBenchmarkRecorder {
        const warmup_seconds = try allocator.alloc(f64, warmup_updates);
        errdefer allocator.free(warmup_seconds);
        const warmup_losses = try allocator.alloc(f32, warmup_updates);
        errdefer allocator.free(warmup_losses);
        const measured_seconds = try allocator.alloc(f64, measured_updates);
        errdefer allocator.free(measured_seconds);
        const measured_losses = try allocator.alloc(f32, measured_updates);
        return .{
            .allocator = allocator,
            .warmup_seconds = warmup_seconds,
            .warmup_losses = warmup_losses,
            .measured_seconds = measured_seconds,
            .measured_losses = measured_losses,
        };
    }

    fn deinit(self: *DpoBenchmarkRecorder) void {
        self.allocator.free(self.warmup_seconds);
        self.allocator.free(self.warmup_losses);
        self.allocator.free(self.measured_seconds);
        self.allocator.free(self.measured_losses);
        self.* = undefined;
    }

    fn record(self: *DpoBenchmarkRecorder, elapsed_seconds: f64, loss: f32) !void {
        const idx = self.updates_seen;
        if (idx >= total_updates) return error.DpoBenchmarkUpdateCountMismatch;
        if (idx == 0) {
            self.cold_seconds = elapsed_seconds;
            self.cold_loss = loss;
        } else if (idx == 1) {
            self.first_seconds = elapsed_seconds;
            self.first_loss = loss;
        } else if (idx < 2 + warmup_updates) {
            const warmup_idx = idx - 2;
            self.warmup_seconds[warmup_idx] = elapsed_seconds;
            self.warmup_losses[warmup_idx] = loss;
        } else {
            const measured_idx = idx - 2 - warmup_updates;
            self.measured_seconds[measured_idx] = elapsed_seconds;
            self.measured_losses[measured_idx] = loss;
        }
        self.updates_seen += 1;
    }

    fn finish(self: *DpoBenchmarkRecorder) !DpoBenchmarkTelemetry {
        if (self.updates_seen != total_updates) return error.DpoBenchmarkUpdateCountMismatch;
        var saw_policy_movement = @abs(self.first_loss - self.cold_loss) > 1e-6;
        for (self.warmup_losses) |loss| saw_policy_movement = saw_policy_movement or @abs(loss - self.cold_loss) > 1e-6;
        for (self.measured_losses) |loss| saw_policy_movement = saw_policy_movement or @abs(loss - self.cold_loss) > 1e-6;
        if (!saw_policy_movement) return error.DpoBenchmarkNoPolicyMovement;
        const sorted = try self.allocator.dupe(f64, self.measured_seconds);
        defer self.allocator.free(sorted);
        std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
        const median = (sorted[measured_updates / 2 - 1] + sorted[measured_updates / 2]) / 2.0;
        var total: f64 = 0.0;
        for (self.measured_seconds) |seconds| total += seconds;
        return .{
            .protocol = .{
                .cold = 1,
                .first = 1,
                .warmup = warmup_updates,
                .measured = measured_updates,
            },
            .cold_seconds = self.cold_seconds,
            .cold_loss = self.cold_loss,
            .first_seconds = self.first_seconds,
            .first_loss = self.first_loss,
            .warmup_seconds = self.warmup_seconds,
            .warmup_losses = self.warmup_losses,
            .measured_seconds = self.measured_seconds,
            .measured_losses = self.measured_losses,
            .median_seconds = median,
            .mean_seconds = total / measured_updates,
        };
    }
};

const GrpoBenchmarkRecorder = struct {
    const warmup_updates = 3;
    const measured_updates = 20;
    const total_updates = 2 + warmup_updates + measured_updates;

    allocator: std.mem.Allocator,
    updates_seen: usize = 0,
    cold: GrpoBenchmarkUpdate = undefined,
    first: GrpoBenchmarkUpdate = undefined,
    warmup: []GrpoBenchmarkUpdate,
    measured: []GrpoBenchmarkUpdate,

    fn init(allocator: std.mem.Allocator) !GrpoBenchmarkRecorder {
        const warmup = try allocator.alloc(GrpoBenchmarkUpdate, warmup_updates);
        errdefer allocator.free(warmup);
        const measured = try allocator.alloc(GrpoBenchmarkUpdate, measured_updates);
        return .{
            .allocator = allocator,
            .warmup = warmup,
            .measured = measured,
        };
    }

    fn deinit(self: *GrpoBenchmarkRecorder) void {
        self.allocator.free(self.warmup);
        self.allocator.free(self.measured);
        self.* = undefined;
    }

    fn record(self: *GrpoBenchmarkRecorder, update: GrpoBenchmarkUpdate) !void {
        const idx = self.updates_seen;
        if (idx >= total_updates) return error.GrpoBenchmarkUpdateCountMismatch;
        if (idx == 0) {
            self.cold = update;
        } else if (idx == 1) {
            self.first = update;
        } else if (idx < 2 + warmup_updates) {
            self.warmup[idx - 2] = update;
        } else {
            self.measured[idx - 2 - warmup_updates] = update;
        }
        self.updates_seen += 1;
    }

    fn finish(self: *GrpoBenchmarkRecorder) !GrpoBenchmarkTelemetry {
        if (self.updates_seen != total_updates) return error.GrpoBenchmarkUpdateCountMismatch;
        var saw_policy_movement = self.first.policy_reference_max_abs_error > 1e-5;
        for (self.warmup) |update| saw_policy_movement = saw_policy_movement or update.policy_reference_max_abs_error > 1e-5;
        for (self.measured) |update| saw_policy_movement = saw_policy_movement or update.policy_reference_max_abs_error > 1e-5;
        if (!saw_policy_movement) return error.GrpoBenchmarkNoPolicyMovement;

        const sorted = try self.allocator.alloc(f64, measured_updates);
        defer self.allocator.free(sorted);
        var total: f64 = 0.0;
        for (self.measured, 0..) |update, idx| {
            sorted[idx] = update.seconds;
            total += update.seconds;
        }
        std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
        const median = (sorted[measured_updates / 2 - 1] + sorted[measured_updates / 2]) / 2.0;
        return .{
            .protocol = .{
                .cold = 1,
                .first = 1,
                .warmup = warmup_updates,
                .measured = measured_updates,
            },
            .cold = self.cold,
            .first = self.first,
            .warmup = self.warmup,
            .measured = self.measured,
            .median_seconds = median,
            .mean_seconds = total / measured_updates,
        };
    }
};

fn gemmaLoraAdapterIsBaseEquivalent(trainer: *const real_autodiff.RealAutodiffTrainer) bool {
    var saw_lora_b = false;
    for (trainer.lora_params.items) |slot| {
        if (!std.mem.endsWith(u8, slot.name, ".lora_B")) continue;
        saw_lora_b = true;
        for (slot.weights) |weight| {
            if (weight != 0.0) return false;
        }
    }
    return saw_lora_b;
}

const GemmaDpoSingleTokenPair = struct {
    prompt: []const i32,
    chosen_token: i32,
    rejected_token: i32,
};

/// Returns a shared-prompt view only when the prepared pair proves the exact
/// causal layout required by the one-row scorer and weighted backward path.
/// Any richer or structurally ambiguous preference pair keeps the general
/// two-sequence implementation.
fn gemmaDpoSingleTokenPair(
    chosen: *const gemma4.PreparedExampleInput,
    rejected: *const gemma4.PreparedExampleInput,
) ?GemmaDpoSingleTokenPair {
    const prompt_len = chosen.prompt_input_ids.len;
    if (prompt_len == 0 or !std.mem.eql(i32, chosen.prompt_input_ids, rejected.prompt_input_ids)) return null;
    if (chosen.num_prompt_tokens != prompt_len or rejected.num_prompt_tokens != prompt_len) return null;
    if (chosen.response_input_ids.len != 1 or rejected.response_input_ids.len != 1) return null;
    if (chosen.num_response_tokens != 1 or rejected.num_response_tokens != 1) return null;
    if (chosen.num_supervised_tokens != 1 or rejected.num_supervised_tokens != 1) return null;
    if (chosen.input_ids.len != prompt_len + 1 or rejected.input_ids.len != prompt_len + 1) return null;
    if (chosen.labels.len != chosen.input_ids.len or rejected.labels.len != rejected.input_ids.len) return null;
    if (chosen.num_input_tokens != chosen.input_ids.len or rejected.num_input_tokens != rejected.input_ids.len) return null;
    if (!std.mem.eql(i32, chosen.input_ids[0..prompt_len], chosen.prompt_input_ids) or
        !std.mem.eql(i32, rejected.input_ids[0..prompt_len], rejected.prompt_input_ids)) return null;
    if (chosen.input_ids[prompt_len] != chosen.response_input_ids[0] or
        rejected.input_ids[prompt_len] != rejected.response_input_ids[0]) return null;
    for (chosen.labels[0..prompt_len]) |label| if (label != -100) return null;
    for (rejected.labels[0..prompt_len]) |label| if (label != -100) return null;
    if (chosen.labels[prompt_len] != chosen.response_input_ids[0] or
        rejected.labels[prompt_len] != rejected.response_input_ids[0]) return null;
    if (chosen.image_paths.len != 0 or chosen.audio_paths.len != 0 or
        rejected.image_paths.len != 0 or rejected.audio_paths.len != 0) return null;
    if (chosen.teacher_top_k != 0 or rejected.teacher_top_k != 0 or
        chosen.teacher_top_k_token_ids.len != 0 or rejected.teacher_top_k_token_ids.len != 0 or
        chosen.teacher_top_k_probs.len != 0 or rejected.teacher_top_k_probs.len != 0) return null;

    return .{
        .prompt = chosen.prompt_input_ids,
        .chosen_token = chosen.response_input_ids[0],
        .rejected_token = rejected.response_input_ids[0],
    };
}

fn allGemmaDpoPairsAreSingleTokenSharedPrompt(
    chosen_examples: []const gemma4.PreparedExampleInput,
    rejected_examples: []const gemma4.PreparedExampleInput,
) bool {
    if (chosen_examples.len == 0 or chosen_examples.len != rejected_examples.len) return false;
    for (chosen_examples, rejected_examples) |*chosen, *rejected| {
        if (gemmaDpoSingleTokenPair(chosen, rejected) == null) return false;
    }
    return true;
}

/// Invert `loss = -log(sigmoid(reward_margin))` without another policy
/// scoring pass. `-expm1(-loss)` is stable when loss is close to zero.
fn rewardMarginFromDpoLoss(loss: f32) !f32 {
    if (!std.math.isFinite(loss) or loss <= 0.0) return error.InvalidDpoCompiledLoss;
    const one_minus_sigmoid = -std.math.expm1(-loss);
    if (!(one_minus_sigmoid > 0.0) or !std.math.isFinite(one_minus_sigmoid)) {
        return error.InvalidDpoCompiledLoss;
    }
    const margin = -loss - @log(one_minus_sigmoid);
    if (!std.math.isFinite(margin)) return error.InvalidDpoCompiledLoss;
    return margin;
}

const GemmaDpoPairSchedule = struct {
    sequence_length: u32,
    weighted_target_rows: usize,
};

fn gemmaDpoLengthBuckets(recipe: Recipe) ?gemma4_real_autodiff.SequenceLengthBuckets {
    const runtime = recipe.runtime orelse return null;
    const quantum = runtime.sequence_length_bucket_quantum orelse return null;
    return .{
        .quantum = quantum,
        .minimum = runtime.sequence_length_bucket_min orelse 0,
    };
}

fn gemmaDpoGraphCacheCapacity(recipe: Recipe, coalesce_single_token_pairs: bool) u8 {
    if (recipe.runtime) |runtime| {
        if (runtime.graph_cache_capacity) |capacity| return capacity;
        if (runtime.sequence_length_bucket_quantum != null) return 4;
    }
    return if (coalesce_single_token_pairs) 1 else 4;
}

fn gemmaDpoPairGraphSignature(schedule: GemmaDpoPairSchedule) u128 {
    return (@as(u128, schedule.sequence_length) << 64) |
        @as(u128, schedule.weighted_target_rows);
}

/// One DPO unit always schedules its chosen and rejected branches together.
/// Sequence length is the rounded maximum logical row; sparse weighted-target
/// rows use the maximum completion bucket. Therefore neither attention nor
/// loss metadata can acquire a branch-specific compiled signature.
fn gemmaDpoPairSchedule(
    chosen: *const gemma4.PreparedExampleInput,
    rejected: *const gemma4.PreparedExampleInput,
    max_seq_len: u32,
    buckets: ?gemma4_real_autodiff.SequenceLengthBuckets,
) !GemmaDpoPairSchedule {
    return .{
        .sequence_length = try gemma4_real_autodiff.sequenceLengthForExample(
            @max(chosen.num_input_tokens, rejected.num_input_tokens),
            max_seq_len,
            buckets,
        ),
        .weighted_target_rows = @max(
            try gemma4_real_autodiff.preferenceTargetRows(chosen.num_supervised_tokens),
            try gemma4_real_autodiff.preferenceTargetRows(rejected.num_supervised_tokens),
        ),
    };
}

fn summarizeGemmaDpoPairLengthPolicy(
    allocator: std.mem.Allocator,
    chosen_examples: []const gemma4.PreparedExampleInput,
    rejected_examples: []const gemma4.PreparedExampleInput,
    max_seq_len: u32,
    buckets: ?gemma4_real_autodiff.SequenceLengthBuckets,
    graph_cache_capacity: u8,
    scope: []const u8,
) !DpoPairLengthPolicyTelemetry {
    if (chosen_examples.len == 0 or chosen_examples.len != rejected_examples.len) {
        return error.DpoBatchAlignmentMismatch;
    }

    var logical_branch_rows: usize = 0;
    var scheduled_branch_rows: usize = 0;
    var minimum_pair_sequence_length: u32 = 0;
    var maximum_pair_sequence_length: u32 = 0;
    var unique_sequence_lengths = std.AutoHashMap(u32, void).init(allocator);
    defer unique_sequence_lengths.deinit();
    var unique_graph_signatures = std.AutoHashMap(u128, void).init(allocator);
    defer unique_graph_signatures.deinit();

    for (chosen_examples, rejected_examples) |*chosen, *rejected| {
        const schedule = try gemmaDpoPairSchedule(chosen, rejected, max_seq_len, buckets);
        logical_branch_rows = std.math.add(usize, logical_branch_rows, chosen.num_input_tokens) catch
            return error.SequenceTooLong;
        logical_branch_rows = std.math.add(usize, logical_branch_rows, rejected.num_input_tokens) catch
            return error.SequenceTooLong;
        scheduled_branch_rows = std.math.add(
            usize,
            scheduled_branch_rows,
            try std.math.mul(usize, @as(usize, schedule.sequence_length), 2),
        ) catch return error.SequenceTooLong;
        if (minimum_pair_sequence_length == 0) {
            minimum_pair_sequence_length = schedule.sequence_length;
        } else {
            minimum_pair_sequence_length = @min(minimum_pair_sequence_length, schedule.sequence_length);
        }
        maximum_pair_sequence_length = @max(maximum_pair_sequence_length, schedule.sequence_length);

        try unique_sequence_lengths.put(schedule.sequence_length, {});
        try unique_graph_signatures.put(gemmaDpoPairGraphSignature(schedule), {});
    }

    const fixed_shape_branch_rows = try std.math.mul(
        usize,
        try std.math.mul(usize, chosen_examples.len, @as(usize, max_seq_len)),
        2,
    );
    const padding_rows_avoided = fixed_shape_branch_rows - scheduled_branch_rows;
    return .{
        .mode = if (buckets == null) "fixed-pair-padding" else "pair-safe-length-buckets",
        .scope = scope,
        .maximum_sequence_length = max_seq_len,
        .bucket_quantum = if (buckets) |policy| policy.quantum else null,
        .bucket_minimum = if (buckets) |policy| if (policy.minimum == 0) policy.quantum else policy.minimum else null,
        .graph_cache_capacity = graph_cache_capacity,
        .pairs = chosen_examples.len,
        .logical_branch_rows = logical_branch_rows,
        .scheduled_branch_rows = scheduled_branch_rows,
        .fixed_shape_branch_rows = fixed_shape_branch_rows,
        .padding_rows_avoided = padding_rows_avoided,
        .padding_reduction_fraction = @as(f64, @floatFromInt(padding_rows_avoided)) /
            @as(f64, @floatFromInt(fixed_shape_branch_rows)),
        .minimum_pair_sequence_length = minimum_pair_sequence_length,
        .maximum_pair_sequence_length = maximum_pair_sequence_length,
        .unique_pair_sequence_lengths = unique_sequence_lengths.count(),
        .unique_pair_graph_signatures = if (buckets == null) null else unique_graph_signatures.count(),
        .weighted_target_row_policy = if (buckets == null) "branch-local" else "pair-shared-maximum-bucket",
    };
}

fn validateGemmaDpoInitialBucketSignatureParity(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_real_autodiff.GemmaAutodiffCtx,
    chosen_examples: []const gemma4.PreparedExampleInput,
    rejected_examples: []const gemma4.PreparedExampleInput,
    reference_cache: *const GemmaDpoReferenceCache,
    seq_len: u32,
    coalesce_single_token_pairs: bool,
    buckets: ?gemma4_real_autodiff.SequenceLengthBuckets,
) !?DpoInitialBucketSignatureParity {
    if (buckets == null) return null;
    if (chosen_examples.len == 0 or
        chosen_examples.len != rejected_examples.len or
        chosen_examples.len != reference_cache.chosen_logps.len or
        chosen_examples.len != reference_cache.rejected_logps.len)
    {
        return error.DpoBatchAlignmentMismatch;
    }

    var seen = std.AutoHashMap(u128, void).init(allocator);
    defer seen.deinit();
    var result: ?DpoInitialBucketSignatureParity = null;

    for (chosen_examples, rejected_examples, 0..) |*chosen, *rejected, pair_idx| {
        const schedule = try gemmaDpoPairSchedule(chosen, rejected, seq_len, buckets);
        const signature = gemmaDpoPairGraphSignature(schedule);
        if (seen.contains(signature)) continue;
        try seen.put(signature, {});

        var policy_logps: [2]f32 = undefined;
        if (coalesce_single_token_pairs) {
            const pair = gemmaDpoSingleTokenPair(chosen, rejected) orelse
                return error.DpoSingleTokenPairContractMismatch;
            const candidate_tokens = [_]i32{ pair.chosen_token, pair.rejected_token };
            try gemma4_real_autodiff.singleTokenCandidateLogprobsForPrompt(
                allocator,
                trainer,
                ctx,
                pair.prompt,
                &candidate_tokens,
                schedule.sequence_length,
                &policy_logps,
            );
        } else {
            policy_logps[0] = try gemma4_real_autodiff.sequenceLogprobForExampleScheduled(
                allocator,
                trainer,
                ctx,
                chosen,
                schedule.sequence_length,
                schedule.weighted_target_rows,
            );
            policy_logps[1] = try gemma4_real_autodiff.sequenceLogprobForExampleScheduled(
                allocator,
                trainer,
                ctx,
                rejected,
                schedule.sequence_length,
                schedule.weighted_target_rows,
            );
        }

        const error_for_pair = @max(
            @abs(policy_logps[0] - reference_cache.chosen_logps[pair_idx]),
            @abs(policy_logps[1] - reference_cache.rejected_logps[pair_idx]),
        );
        if (result == null or error_for_pair > result.?.max_abs_error) {
            result = .{
                .graph_signatures_checked = 0,
                .representative_pair_index = pair_idx,
                .policy_chosen_logp = policy_logps[0],
                .policy_rejected_logp = policy_logps[1],
                .reference_chosen_logp = reference_cache.chosen_logps[pair_idx],
                .reference_rejected_logp = reference_cache.rejected_logps[pair_idx],
                .max_abs_error = error_for_pair,
                .base_equivalent_policy = reference_cache.base_equivalent_policy,
            };
        }
    }

    var parity = result orelse return error.DpoBatchAlignmentMismatch;
    parity.graph_signatures_checked = seen.count();
    if (parity.base_equivalent_policy and parity.max_abs_error > 1e-4) {
        return error.GemmaDpoInitialReferenceParityMismatch;
    }
    return parity;
}

fn precomputeGemmaDpoBaseReferenceCache(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_real_autodiff.GemmaAutodiffCtx,
    chosen_examples: []const gemma4.PreparedExampleInput,
    rejected_examples: []const gemma4.PreparedExampleInput,
    seq_len: u32,
    coalesce_single_token_pairs: bool,
    buckets: ?gemma4_real_autodiff.SequenceLengthBuckets,
) !GemmaDpoReferenceCache {
    if (chosen_examples.len != rejected_examples.len) return error.DpoBatchAlignmentMismatch;

    const base_equivalent_policy = gemmaLoraAdapterIsBaseEquivalent(trainer);
    var frozen_lora = try gemma4_real_autodiff.FrozenBaseLoraBindings.init(allocator, trainer);
    defer frozen_lora.deinit();

    const chosen_logps = try allocator.alloc(f32, chosen_examples.len);
    errdefer allocator.free(chosen_logps);
    const rejected_logps = try allocator.alloc(f32, rejected_examples.len);
    errdefer allocator.free(rejected_logps);

    const started_ns = platform.time.monotonicNs();
    for (chosen_examples, rejected_examples, 0..) |*chosen, *rejected, idx| {
        const schedule = try gemmaDpoPairSchedule(chosen, rejected, seq_len, buckets);
        if (coalesce_single_token_pairs) {
            const pair = gemmaDpoSingleTokenPair(chosen, rejected) orelse return error.DpoSingleTokenPairContractMismatch;
            const candidate_tokens = [_]i32{ pair.chosen_token, pair.rejected_token };
            var pair_logps: [2]f32 = undefined;
            try gemma4_real_autodiff.singleTokenCandidateLogprobsForPromptFrozenBase(
                allocator,
                trainer,
                ctx,
                pair.prompt,
                &candidate_tokens,
                schedule.sequence_length,
                &pair_logps,
                &frozen_lora,
            );
            chosen_logps[idx] = pair_logps[0];
            rejected_logps[idx] = pair_logps[1];
        } else {
            chosen_logps[idx] = if (buckets == null)
                try gemma4_real_autodiff.sequenceLogprobForExampleFrozenBase(
                    allocator,
                    trainer,
                    ctx,
                    chosen,
                    seq_len,
                    &frozen_lora,
                )
            else
                try gemma4_real_autodiff.sequenceLogprobForExampleFrozenBaseScheduled(
                    allocator,
                    trainer,
                    ctx,
                    chosen,
                    schedule.sequence_length,
                    schedule.weighted_target_rows,
                    &frozen_lora,
                );
            if (platform.env.getenvBoolDefault("ANTFLY_GEMMA4_PREFERENCE_TRACE", false)) {
                print("gemma4 dpo frozen lora after chosen: max_abs_prefix={d:.9}\n", .{try frozen_lora.debugMaxAbsPrefix(8)});
            }
            rejected_logps[idx] = if (buckets == null)
                try gemma4_real_autodiff.sequenceLogprobForExampleFrozenBase(
                    allocator,
                    trainer,
                    ctx,
                    rejected,
                    seq_len,
                    &frozen_lora,
                )
            else
                try gemma4_real_autodiff.sequenceLogprobForExampleFrozenBaseScheduled(
                    allocator,
                    trainer,
                    ctx,
                    rejected,
                    schedule.sequence_length,
                    schedule.weighted_target_rows,
                    &frozen_lora,
                );
            if (platform.env.getenvBoolDefault("ANTFLY_GEMMA4_PREFERENCE_TRACE", false)) {
                print("gemma4 dpo frozen lora after rejected: max_abs_prefix={d:.9}\n", .{try frozen_lora.debugMaxAbsPrefix(8)});
            }
        }
    }
    const elapsed_ns = platform.time.monotonicNs() - started_ns;

    return .{
        .allocator = allocator,
        .chosen_logps = chosen_logps,
        .rejected_logps = rejected_logps,
        .precompute_seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s,
        .base_equivalent_policy = base_equivalent_policy,
    };
}

fn runOptimizerBackedGemmaDpo(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    dataset_path: []const u8,
    report_path: []const u8,
) !void {
    const base_model_dir = recipe.model.path orelse return error.MissingModelPath;
    const adapter = recipe.adapter orelse AdapterConfig{};
    const bootstrap_dir_config = adapter.path;
    const bootstrap_dir = bootstrap_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    defer if (bootstrap_dir_config == null) allocator.free(bootstrap_dir);
    const trained_dir_config = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir;
    const trained_dir = trained_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    defer if (trained_dir_config == null) allocator.free(trained_dir);
    try requireMissingPreferencePublicationTarget(io, trained_dir);
    const reference_path = recipe.model.reference_path orelse base_model_dir;
    if (!std.mem.eql(u8, reference_path, base_model_dir)) return error.UnsupportedReferencePath;
    const execution = try resolveGemmaPreferenceExecution(recipe.backend);
    const backend_kind = execution.backend_kind;
    try validateGemmaPreferenceEnvironmentContract(backend_kind);
    const execution_policy = train_eval_gemma4_lora_bundle.autodiffExecutionPolicy(backend_kind);
    const max_examples = recipe.dataset.max_examples orelse 32;
    const max_seq_len = recipe.dataset.max_seq_len orelse 512;
    try validateGemmaAdapterOptions(adapter);
    var graph_executor_scope = try train_eval_gemma4_lora_bundle.acquireMetalGraphExecutorScope(backend_kind);
    defer if (graph_executor_scope) |*scope| scope.deinit();
    try train_eval_gemma4_lora_bundle.validateAutodiffBaseArtifactForRecipe(
        allocator,
        base_model_dir,
        backend_kind,
        recipe.model.allow_direct_gguf_training orelse false,
    );
    const direct_gguf_base = try train_eval_gemma4_lora_bundle.autodiffBaseUsesGguf(allocator, base_model_dir);
    if ((recipe.model.allow_direct_gguf_training orelse false) != direct_gguf_base) {
        return error.DirectGgufTrainingAdmissionMismatch;
    }

    compat.cwd().access(compat.io(), bootstrap_dir, .{}) catch {
        var bootstrap = try gemma4.bootstrapLoRABundle(allocator, base_model_dir, bootstrap_dir, .{
            .rank = adapterRank(adapter, .dpo),
            .alpha = adapterAlpha(adapter),
            .base_model_name_or_path = adapter.base_model_name_or_path,
            .target_modules = adapter.target_modules,
            .gemma4_target_preset = gemma4TargetPreset(adapter),
            .target_preset = try gemmaLegacyTargetPreset(adapter),
            .use_dora = adapter.use_dora orelse false,
            .init_lora_weights = adapter.init_lora_weights,
            .initialization_seed = adapter.initialization_seed orelse 0,
        });
        defer gemma4.freeBootstrapSummary(allocator, &bootstrap);
    };

    var session_manager = backends.SessionManager.init(allocator);
    native_backend_choice.configureSessionPreference(&session_manager, execution.session_choice);
    var model_manager = model_manager_mod.ModelManager.init(allocator, session_manager);
    defer model_manager.deinit();
    const tokenizer_model = try model_manager.loadFromDir(base_model_dir);

    var samples = try loadDpoTextPreferenceSamples(allocator, io, dataset_path, recipe, tokenizer_model);
    defer samples.deinit();

    var chosen_prepared = try prepareGemmaDpoPreparedExamplesFromSamples(allocator, base_model_dir, samples.samples, max_examples, max_seq_len, .chosen);
    defer gemma4.freePreparedInputsSummary(allocator, &chosen_prepared);
    var rejected_prepared = try prepareGemmaDpoPreparedExamplesFromSamples(allocator, base_model_dir, samples.samples, max_examples, max_seq_len, .rejected);
    defer gemma4.freePreparedInputsSummary(allocator, &rejected_prepared);
    if (chosen_prepared.examples.len != rejected_prepared.examples.len or chosen_prepared.examples.len != samples.samples.len) {
        return error.DpoBatchAlignmentMismatch;
    }
    // The weighted sparse-row backward is currently qualified only through
    // the strict Metal executor. Native retains the general two-sequence path;
    // its rank-2 dot implementation does not yet support this sparse target
    // graph and must not receive it through a structural fast-path match.
    const coalesce_single_token_pairs = backend_kind == .metal and
        allGemmaDpoPairsAreSingleTokenSharedPrompt(
            chosen_prepared.examples,
            rejected_prepared.examples,
        );
    const length_buckets = gemmaDpoLengthBuckets(recipe);
    const graph_cache_capacity = gemmaDpoGraphCacheCapacity(recipe, coalesce_single_token_pairs);
    const sequence_length_policy = try summarizeGemmaDpoPairLengthPolicy(
        allocator,
        chosen_prepared.examples,
        rejected_prepared.examples,
        @intCast(max_seq_len),
        length_buckets,
        graph_cache_capacity,
        "train-dataset-one-pass",
    );
    // The simultaneous whole-objective graph remains an explicit research
    // lane until its activation lifetime is below the MLX memory gate. The
    // production fast path instead detaches adapter-sized branch gradients.
    const pair_objective_requested = backend_kind == .metal and
        !coalesce_single_token_pairs and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_PAIR_GRAPH", false);
    const batch2_pair_forward_requested = pair_objective_requested and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_BATCH2_FORWARD_GRAPH", false);
    const detached_pair_gradients_requested = backend_kind == .metal and
        !coalesce_single_token_pairs and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_DETACHED_GRADIENTS", true);
    const coalesced_snapshot_frame_requested = backend_kind == .metal and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_COALESCED_SNAPSHOT_FRAME", false);
    const ping_pong_gradients_requested = backend_kind == .metal and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_PING_PONG_GRADIENTS", false);
    const slot_bound_outputs = backend_kind == .metal and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_SLOT_BOUND_OUTPUTS", false);
    const completion_cache_enabled = backend_kind == .metal and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_COMPLETION_FENCED_CACHE", true);
    const checkpoint_interval_raw = platform.env.getenvUsize("ANTFLY_GEMMA4_DPO_CHECKPOINT_LAYER_INTERVAL") orelse 1;
    const checkpoint_interval: u32 = @intCast(@min(
        @max(checkpoint_interval_raw, 1),
        @as(usize, std.math.maxInt(u32)),
    ));
    const dpo_checkpoint_config: ?ml.graph.checkpoint.CheckpointConfig = if (backend_kind == .metal and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_ACTIVATION_CHECKPOINTING", false))
        .{
            .strategy = .every_n_layers,
            .layer_interval = checkpoint_interval,
            .recursive_recompute_dependencies = platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_RECURSIVE_CHECKPOINTING", false),
        }
    else
        null;

    const graph_config = try gemma4_real_autodiff.loadGraphConfig(allocator, base_model_dir);
    // The allocator fences aliases recycled inside a planned encoder and
    // quarantines releases made outside one. The exact E2B topology is
    // production-qualified; other shapes remain fail-closed unless explicitly
    // enabled for research. Setting the variable to 0 is the E2B kill switch.
    const in_frame_buffer_reuse_enabled = backend_kind == .metal and
        platform.env.getenvBoolDefault(
            "ANTFLY_GEMMA4_DPO_IN_FRAME_BUFFER_REUSE",
            gemma4_real_autodiff.qualifiedE2BTrainingTopology(graph_config),
        );
    var backend = try gemma4_real_autodiff.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();

    var adapter_inspect = try gemma4.inspectCheckpoint(allocator, bootstrap_dir);
    defer gemma4.freeInspectionSummary(allocator, &adapter_inspect);
    const lora_rank = adapter_inspect.lora_rank orelse return error.MissingAdapterConfig;
    const lora_alpha = @as(f32, @floatCast(adapter_inspect.lora_alpha orelse return error.MissingAdapterConfig));
    const target_modules = adapter_inspect.target_modules orelse (adapter.target_modules orelse gemma4.default_lora_target_modules[0..]);
    // Recursive adapters intentionally train distinct use-site parameters.
    // A dual branch would double those sites rather than share one policy, so
    // keep the proven two-microbatch path until a recursive-aware pair graph
    // has an explicit parameter-sharing contract.
    const requested_grad_accum_steps = recipe.optimizer.gradient_accumulation_steps orelse 1;
    const compile_pair_objective = pair_objective_requested and !adapter_inspect.recursive_lora_enabled;
    const dpo_pair_graph_mode: gemma4_real_autodiff.DpoPairGraphMode = if (batch2_pair_forward_requested)
        .batched_forward
    else
        .split_batch1;
    const detach_pair_gradients = detached_pair_gradients_requested and
        !compile_pair_objective and
        requested_grad_accum_steps == 1 and
        !adapter_inspect.recursive_lora_enabled;
    const physical_micro_batches_per_pair: usize = if (coalesce_single_token_pairs or compile_pair_objective) 1 else 2;
    const grad_accum_steps = try preferenceGradAccumSteps(
        requested_grad_accum_steps,
        physical_micro_batches_per_pair,
    );
    const lora_config = ml.graph.lora.LoRAConfig{
        .rank = @intCast(lora_rank),
        .alpha = lora_alpha,
        .target_patterns = target_modules,
        .strict_target_patterns = true,
        .sharing = if (adapter_inspect.recursive_lora_enabled) .by_use else .by_weight,
    };

    var trainer = try @import("real_autodiff_trainer.zig").RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
        .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
        .grad_accum_steps = grad_accum_steps,
        .seed = recipe.optimizer.seed orelse 42,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
        .checkpoint_config = dpo_checkpoint_config,
        .metal_slot_bound_outputs = slot_bound_outputs,
        // Fixed preference examples and the qualified bucketed policy retain
        // four graphs by default. Recipes may explicitly admit up to eight;
        // every run reports actual build/eviction behavior.
        .graph_cache_capacity = graph_cache_capacity,
    });
    defer trainer.deinit();

    var ctx = gemma4_real_autodiff.GemmaAutodiffCtx.init(graph_config);
    ctx.enable_fused_rms_norm_backward = backend_kind == .metal;
    ctx.enable_fused_gqa_attention_backward = backend_kind == .metal and gemma4_real_autodiff.fusedGqaAttentionExperimentEnabled(graph_config);
    ctx.enable_fused_linear_cross_entropy = backend_kind == .metal and
        !direct_gguf_base and
        !platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_LINEAR_CCE", false);
    const bootstrap_example = gemma4_real_autodiff.findFirstSupervisedExample(chosen_prepared.examples) orelse return error.NoTrainingData;
    try gemma4_real_autodiff.initializeTrainerFromAdapterDir(allocator, &trainer, &ctx, bootstrap_dir, bootstrap_example, @intCast(max_seq_len));

    const dpo_minimums = recipe.eval.?.dpo_minimums.?;
    const dpo_baseline_report_path = if (dpo_minimums.min_accuracy_improvement != null)
        try preferenceBaselineEvaluationReportPath(allocator, recipe, .dpo)
    else
        null;
    defer if (dpo_baseline_report_path) |path| allocator.free(path);
    const baseline_evaluation: ?DpoEvaluationSummary = if (dpo_baseline_report_path) |path|
        try evaluateGemmaDpoHeldout(
            allocator,
            io,
            recipe,
            tokenizer_model,
            samples.samples,
            &trainer,
            &ctx,
            base_model_dir,
            backend_kind,
            path,
            false,
        )
    else
        null;

    var dpo_execution_flags: u64 = 0;
    if (coalesce_single_token_pairs) dpo_execution_flags |= @as(u64, 1) << 0;
    if (compile_pair_objective) dpo_execution_flags |= @as(u64, 1) << 1;
    if (batch2_pair_forward_requested) dpo_execution_flags |= @as(u64, 1) << 2;
    if (detach_pair_gradients) dpo_execution_flags |= @as(u64, 1) << 3;
    if (coalesced_snapshot_frame_requested) dpo_execution_flags |= @as(u64, 1) << 4;
    if (ping_pong_gradients_requested) dpo_execution_flags |= @as(u64, 1) << 5;
    if (slot_bound_outputs) dpo_execution_flags |= @as(u64, 1) << 6;
    if (completion_cache_enabled) dpo_execution_flags |= @as(u64, 1) << 7;
    if (in_frame_buffer_reuse_enabled) dpo_execution_flags |= @as(u64, 1) << 8;
    if (dpo_checkpoint_config != null) dpo_execution_flags |= @as(u64, 1) << 9;
    if (ctx.enable_fused_gqa_attention_backward) dpo_execution_flags |= @as(u64, 1) << 10;
    const metal_numerical_policy = resolveGemmaMetalNumericalPolicy(backend_kind, &ctx);
    const dpo_run_fingerprint = try gemmaPreferenceRunFingerprint(
        allocator,
        io,
        recipe,
        .dpo,
        base_model_dir,
        bootstrap_dir,
        target_modules,
        @intCast(lora_rank),
        lora_alpha,
        adapter_inspect.recursive_lora_enabled,
        backend_kind,
        .{
            .seed = recipe.optimizer.seed orelse 42,
            .max_examples = max_examples,
            .max_seq_len = max_seq_len,
            .epochs = recipe.optimizer.epochs orelse 1,
            .learning_rate = recipe.optimizer.learning_rate orelse 0.0001,
            .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
            .requested_gradient_accumulation_steps = requested_grad_accum_steps,
            .physical_micro_batches_per_unit = physical_micro_batches_per_pair,
            .graph_cache_capacity = graph_cache_capacity,
            .sequence_length_bucket_quantum = if (length_buckets) |policy| policy.quantum else null,
            .sequence_length_bucket_min = if (length_buckets) |policy| policy.minimum else null,
            .direct_gguf_base = direct_gguf_base,
            .fused_linear_cross_entropy = ctx.enable_fused_linear_cross_entropy orelse false,
            .execution_flags = dpo_execution_flags,
            .metal_numerical_policy_flags = if (metal_numerical_policy) |policy| policy.fingerprint_flags else 0,
            .metal_sparse_loss_chunk_rows = if (metal_numerical_policy) |policy| policy.sparse_loss_chunk_rows else null,
            .metal_linear_cce_tile_vocab = if (metal_numerical_policy) |policy| policy.linear_cce_tile_vocab else null,
            .dpo_beta = recipe.preference.beta orelse 0.1,
            .dpo_activation_checkpoint_layer_interval = if (dpo_checkpoint_config) |cfg| cfg.layer_interval else null,
            .dpo_activation_checkpoint_recursive = if (dpo_checkpoint_config) |cfg| cfg.recursive_recompute_dependencies else null,
        },
    );
    const dpo_run_fingerprint_text = try formatSha256DigestAlloc(allocator, dpo_run_fingerprint);
    defer allocator.free(dpo_run_fingerprint_text);
    const dpo_checkpoint_path = try preferenceCheckpointPath(allocator, recipe, .dpo);
    defer if (dpo_checkpoint_path) |path| allocator.free(path);
    const dpo_resume_enabled = if (recipe.checkpoint) |checkpoint| checkpoint.resume_path != null else false;
    var dpo_restored = real_autodiff.RestoredTrainingCheckpoint{
        .micro_batch_steps = 0,
        .optimizer_steps = 0,
        .accumulation_micro_batches = 0,
        .configured_accumulation_steps = grad_accum_steps,
        .stochastic_steps = 0,
        .progress = .{},
    };
    var loaded_dpo_state: ?LoadedPreferenceCheckpointState = null;
    defer if (loaded_dpo_state) |*state| state.deinit(allocator);
    var start_epoch: usize = 0;
    var initial_adapter_digest = trainerLoRAParameterDigest(&trainer);
    if (dpo_resume_enabled) {
        const path = dpo_checkpoint_path orelse return error.CheckpointPathRequired;
        dpo_restored = try trainer.loadTrainingCheckpoint(path, &dpo_run_fingerprint);
        loaded_dpo_state = try loadPreferenceCheckpointState(
            allocator,
            io,
            path,
            .dpo,
            &dpo_run_fingerprint,
            dpo_restored,
        );
        start_epoch = std.math.cast(usize, dpo_restored.progress.epoch_index) orelse
            return error.InvalidPreferenceCheckpointState;
        if (start_epoch > (recipe.optimizer.epochs orelse 1)) {
            return error.CheckpointBeyondRequestedEpochCount;
        }
        initial_adapter_digest = loaded_dpo_state.?.parsed.value.dpo.?.initial_adapter_digest;
    }
    const graph_cache_after_initialization = trainer.graphCacheStats();

    // Scope both reuse tiers across the complete DPO workload. The default
    // remains fail-closed while the planned-encoder path is requalified; the
    // research admission is explicit and recorded in the DPO report.
    var dpo_buffer_reuse_scope = try gemma4_real_autodiff.configureMetalBufferReuseForPreferenceRun(
        &trainer,
        in_frame_buffer_reuse_enabled,
        completion_cache_enabled,
    );
    defer dpo_buffer_reuse_scope.deinit();

    const epochs = recipe.optimizer.epochs orelse 1;
    const benchmark_enabled = platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_BENCHMARK", false);
    if (benchmark_enabled and recipe.checkpoint != null) return error.PreferenceBenchmarkCheckpointingNotSupported;
    const planned_updates = std.math.mul(usize, chosen_prepared.examples.len, epochs) catch return error.DpoBenchmarkUpdateCountMismatch;
    if (benchmark_enabled and planned_updates != DpoBenchmarkRecorder.total_updates) {
        return error.DpoBenchmarkUpdateCountMismatch;
    }
    var benchmark: ?DpoBenchmarkRecorder = if (benchmark_enabled) try DpoBenchmarkRecorder.init(allocator) else null;
    defer if (benchmark) |*recorder| recorder.deinit();

    var reference_cache = try precomputeGemmaDpoBaseReferenceCache(
        allocator,
        &trainer,
        &ctx,
        chosen_prepared.examples,
        rejected_prepared.examples,
        @intCast(max_seq_len),
        coalesce_single_token_pairs,
        length_buckets,
    );
    defer reference_cache.deinit();
    const graph_cache_after_reference_precompute = trainer.graphCacheStats();
    const observed_bucket_signature_parity = try validateGemmaDpoInitialBucketSignatureParity(
        allocator,
        &trainer,
        &ctx,
        chosen_prepared.examples,
        rejected_prepared.examples,
        &reference_cache,
        @intCast(max_seq_len),
        coalesce_single_token_pairs,
        length_buckets,
    );
    const graph_cache_after_initial_bucket_signature_parity = trainer.graphCacheStats();

    const restored_dpo_aggregates = if (loaded_dpo_state) |*state| state.parsed.value.dpo else null;
    if (restored_dpo_aggregates) |state| {
        const expected_examples = std.math.mul(usize, start_epoch, chosen_prepared.examples.len) catch
            return error.InvalidPreferenceCheckpointState;
        if (state.examples_seen != expected_examples) return error.InvalidPreferenceCheckpointState;
    }
    var total_loss: f64 = if (restored_dpo_aggregates) |state| state.total_loss else 0.0;
    var total_margin: f64 = if (restored_dpo_aggregates) |state| state.total_margin else 0.0;
    var total_accuracy: f64 = if (restored_dpo_aggregates) |state| state.total_accuracy else 0.0;
    var examples_seen: usize = if (restored_dpo_aggregates) |state| state.examples_seen else 0;
    var initial_logprob_parity: ?DpoInitialLogprobParity = if (restored_dpo_aggregates) |state|
        state.initial_logprob_parity
    else
        null;
    const initial_bucket_signature_parity: ?DpoInitialBucketSignatureParity = if (restored_dpo_aggregates) |state|
        state.initial_bucket_signature_parity
    else
        observed_bucket_signature_parity;
    var single_pc = [_]f32{0};
    var single_pr = [_]f32{0};
    var single_rc = [_]f32{0};
    var single_rr = [_]f32{0};
    var single_cl = [_]u32{0};
    var single_rl = [_]u32{0};
    var single_sft = [_]f32{0};

    var epoch_idx: usize = start_epoch;
    while (epoch_idx < epochs) : (epoch_idx += 1) {
        for (chosen_prepared.examples, rejected_prepared.examples, samples.samples, 0..) |*chosen_ex, *rejected_ex, sample, sample_idx| {
            const update_started_ns = if (benchmark_enabled) platform.time.monotonicNs() else 0;
            const pair_schedule = try gemmaDpoPairSchedule(
                chosen_ex,
                rejected_ex,
                @intCast(max_seq_len),
                length_buckets,
            );
            var policy_chosen: f32 = 0.0;
            var policy_rejected: f32 = 0.0;
            var detached_device_gradients: ?real_autodiff.DetachedDeviceGradients = null;
            defer if (detached_device_gradients) |*gradients| gradients.deinit();
            // The pair graph owns policy scoring after the initial oracle
            // check. Keeping exactly one live-policy comparison preserves the
            // base-equivalence gate without putting two score-only forwards
            // back into every measured update.
            if (detach_pair_gradients) {
                // Execute each exact batch-1 branch once. A coefficient of one
                // makes the graph's scalar loss equal the raw summed sequence
                // log-probability and its gradient equal d(logp)/d(theta).
                // The chosen gradient is detached before rejected executes, so
                // neither branch updates weights or retains model activations.
                var chosen_raw_input = if (length_buckets == null)
                    try gemma4_real_autodiff.makeTrainerInputForLogprobCoeff(
                        allocator,
                        &ctx,
                        chosen_ex,
                        @intCast(max_seq_len),
                        1.0,
                    )
                else
                    try gemma4_real_autodiff.makeTrainerInputForLogprobCoeffScheduled(
                        allocator,
                        &ctx,
                        chosen_ex,
                        pair_schedule.sequence_length,
                        pair_schedule.weighted_target_rows,
                        1.0,
                    );
                defer chosen_raw_input.deinit(allocator);
                const chosen_raw_step = try trainer.step(chosen_raw_input.trainer_input);
                if (chosen_raw_step.optimizer_stepped) return error.DpoDetachedGradientSteppedEarly;
                policy_chosen = chosen_raw_step.loss;
                detached_device_gradients = try trainer.detachAccumulatedDeviceGradients();

                var rejected_raw_input = if (length_buckets == null)
                    try gemma4_real_autodiff.makeTrainerInputForLogprobCoeff(
                        allocator,
                        &ctx,
                        rejected_ex,
                        @intCast(max_seq_len),
                        1.0,
                    )
                else
                    try gemma4_real_autodiff.makeTrainerInputForLogprobCoeffScheduled(
                        allocator,
                        &ctx,
                        rejected_ex,
                        pair_schedule.sequence_length,
                        pair_schedule.weighted_target_rows,
                        1.0,
                    );
                defer rejected_raw_input.deinit(allocator);
                const rejected_raw_step = try trainer.step(rejected_raw_input.trainer_input);
                if (rejected_raw_step.optimizer_stepped) return error.DpoDetachedGradientSteppedEarly;
                policy_rejected = rejected_raw_step.loss;
            } else if (!compile_pair_objective or examples_seen == 0) {
                if (coalesce_single_token_pairs) {
                    const pair = gemmaDpoSingleTokenPair(chosen_ex, rejected_ex) orelse return error.DpoSingleTokenPairContractMismatch;
                    const candidate_tokens = [_]i32{ pair.chosen_token, pair.rejected_token };
                    var pair_logps: [2]f32 = undefined;
                    try gemma4_real_autodiff.singleTokenCandidateLogprobsForPrompt(
                        allocator,
                        &trainer,
                        &ctx,
                        pair.prompt,
                        &candidate_tokens,
                        pair_schedule.sequence_length,
                        &pair_logps,
                    );
                    policy_chosen = pair_logps[0];
                    policy_rejected = pair_logps[1];
                } else {
                    policy_chosen = if (length_buckets == null)
                        try gemma4_real_autodiff.sequenceLogprobForExample(
                            allocator,
                            &trainer,
                            &ctx,
                            chosen_ex,
                            @intCast(max_seq_len),
                        )
                    else
                        try gemma4_real_autodiff.sequenceLogprobForExampleScheduled(
                            allocator,
                            &trainer,
                            &ctx,
                            chosen_ex,
                            pair_schedule.sequence_length,
                            pair_schedule.weighted_target_rows,
                        );
                    policy_rejected = if (length_buckets == null)
                        try gemma4_real_autodiff.sequenceLogprobForExample(
                            allocator,
                            &trainer,
                            &ctx,
                            rejected_ex,
                            @intCast(max_seq_len),
                        )
                    else
                        try gemma4_real_autodiff.sequenceLogprobForExampleScheduled(
                            allocator,
                            &trainer,
                            &ctx,
                            rejected_ex,
                            pair_schedule.sequence_length,
                            pair_schedule.weighted_target_rows,
                        );
                }
            }

            single_rc[0] = reference_cache.chosen_logps[sample_idx];
            single_rr[0] = reference_cache.rejected_logps[sample_idx];

            if (examples_seen == 0) {
                const max_abs_error = @max(
                    @abs(policy_chosen - single_rc[0]),
                    @abs(policy_rejected - single_rr[0]),
                );
                initial_logprob_parity = .{
                    .policy_chosen_logp = policy_chosen,
                    .policy_rejected_logp = policy_rejected,
                    .reference_chosen_logp = single_rc[0],
                    .reference_rejected_logp = single_rr[0],
                    .max_abs_error = max_abs_error,
                    .base_equivalent_policy = reference_cache.base_equivalent_policy,
                };
                if (platform.env.getenvBoolDefault("ANTFLY_GEMMA4_PREFERENCE_TRACE", false)) {
                    print(
                        "gemma4 dpo initial logps: policy_chosen={d:.9} policy_rejected={d:.9} ref_chosen={d:.9} ref_rejected={d:.9} prompt_tokens={} chosen_tokens={} rejected_tokens={}\n",
                        .{
                            policy_chosen,
                            policy_rejected,
                            single_rc[0],
                            single_rr[0],
                            sample.prompt_tokens.len,
                            sample.chosen_tokens.len,
                            sample.rejected_tokens.len,
                        },
                    );
                }
                if (reference_cache.base_equivalent_policy and max_abs_error > 1e-4) {
                    return error.GemmaDpoInitialReferenceParityMismatch;
                }
            }

            var update_loss: f32 = undefined;
            var update_margin: f32 = undefined;
            var update_accuracy: f32 = undefined;
            if (compile_pair_objective) {
                var pair_input = try gemma4_real_autodiff.makeTrainerInputForDpoPair(
                    allocator,
                    &ctx,
                    chosen_ex,
                    rejected_ex,
                    pair_schedule.sequence_length,
                    dpo_pair_graph_mode,
                    single_rc[0],
                    single_rr[0],
                    recipe.preference.beta orelse 0.1,
                );
                defer pair_input.deinit(allocator);
                const pair_step = try trainer.step(pair_input.trainer_input);
                update_loss = pair_step.loss;
                update_margin = try rewardMarginFromDpoLoss(pair_step.loss);
                update_accuracy = if (update_margin > 0.0) 1.0 else 0.0;
            } else {
                single_pc[0] = policy_chosen;
                single_pr[0] = policy_rejected;
                single_cl[0] = @intCast(sample.chosen_tokens.len);
                single_rl[0] = @intCast(sample.rejected_tokens.len);
                single_sft[0] = sample.sft_chosen_loss orelse 0;

                var step_result = try preference_loss.pairedPreferenceLoss(allocator, .{
                    .policy_chosen_logps = single_pc[0..1],
                    .policy_rejected_logps = single_pr[0..1],
                    .ref_chosen_logps = single_rc[0..1],
                    .ref_rejected_logps = single_rr[0..1],
                    .chosen_lengths = single_cl[0..1],
                    .rejected_lengths = single_rl[0..1],
                    .sft_chosen_loss = single_sft[0..1],
                }, .{
                    .kind = .dpo,
                    .beta = recipe.preference.beta orelse 0.1,
                    .simpo_gamma = recipe.preference.simpo_gamma orelse 0.5,
                    .sft_lambda = recipe.preference.sft_lambda orelse 1.0,
                    .ipo_tau = recipe.preference.ipo_tau orelse 0.1,
                });
                defer step_result.deinit();
                if (!coalesce_single_token_pairs) {
                    try scalePreferenceUnitGradients(step_result.grad_chosen, 2);
                    try scalePreferenceUnitGradients(step_result.grad_rejected, 2);
                }
                update_loss = step_result.loss;
                update_margin = step_result.mean_reward_margin;
                update_accuracy = step_result.accuracy;

                if (detach_pair_gradients) {
                    if (detached_device_gradients) |*detached| {
                        try trainer.combineDetachedDeviceGradients(
                            detached,
                            step_result.grad_chosen[0],
                            step_result.grad_rejected[0],
                        );
                    } else return error.MissingDpoDetachedGradient;
                    detached_device_gradients = null;

                    const flush = (try trainer.flushAccumulatedGradients()) orelse
                        return error.MissingDpoDetachedGradientUpdate;
                    if (flush.micro_batches != 2 or
                        trainer.accumulatedMicroBatches() != 0 or
                        trainer.optimizerSteps() != @as(u64, @intCast(examples_seen + 1)))
                    {
                        return error.InvalidDpoDetachedGradientUpdate;
                    }
                } else if (coalesce_single_token_pairs) {
                    const pair = gemmaDpoSingleTokenPair(chosen_ex, rejected_ex) orelse return error.DpoSingleTokenPairContractMismatch;
                    const candidate_tokens = [_]i32{ pair.chosen_token, pair.rejected_token };
                    const logprob_grads = [_]f32{ step_result.grad_chosen[0], step_result.grad_rejected[0] };
                    var pair_input = try gemma4_real_autodiff.makeTrainerInputForSingleTokenCandidatesLogprobGrads(
                        allocator,
                        &ctx,
                        chosen_ex,
                        pair_schedule.sequence_length,
                        &candidate_tokens,
                        &logprob_grads,
                    );
                    defer pair_input.deinit(allocator);
                    _ = try trainer.step(pair_input.trainer_input);
                } else {
                    var chosen_input = if (length_buckets == null)
                        try gemma4_real_autodiff.makeTrainerInputForLogprobCoeff(
                            allocator,
                            &ctx,
                            chosen_ex,
                            @intCast(max_seq_len),
                            step_result.grad_chosen[0],
                        )
                    else
                        try gemma4_real_autodiff.makeTrainerInputForLogprobCoeffScheduled(
                            allocator,
                            &ctx,
                            chosen_ex,
                            pair_schedule.sequence_length,
                            pair_schedule.weighted_target_rows,
                            step_result.grad_chosen[0],
                        );
                    defer chosen_input.deinit(allocator);
                    _ = try trainer.step(chosen_input.trainer_input);

                    var rejected_input = if (length_buckets == null)
                        try gemma4_real_autodiff.makeTrainerInputForLogprobCoeff(
                            allocator,
                            &ctx,
                            rejected_ex,
                            @intCast(max_seq_len),
                            step_result.grad_rejected[0],
                        )
                    else
                        try gemma4_real_autodiff.makeTrainerInputForLogprobCoeffScheduled(
                            allocator,
                            &ctx,
                            rejected_ex,
                            pair_schedule.sequence_length,
                            pair_schedule.weighted_target_rows,
                            step_result.grad_rejected[0],
                        );
                    defer rejected_input.deinit(allocator);
                    _ = try trainer.step(rejected_input.trainer_input);
                }
            }

            total_loss += update_loss;
            total_margin += update_margin;
            total_accuracy += update_accuracy;
            examples_seen += 1;

            if (benchmark) |*recorder| {
                const elapsed_ns = platform.time.monotonicNs() - update_started_ns;
                try recorder.record(
                    @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s,
                    update_loss,
                );
            }
        }
        const completed_epochs = epoch_idx + 1;
        const checkpoint_every = if (recipe.checkpoint) |checkpoint| checkpoint.every_epochs else null;
        if (checkpoint_every) |every| {
            if (completed_epochs % @as(usize, every) == 0 or completed_epochs == epochs) {
                const path = dpo_checkpoint_path orelse return error.CheckpointPathRequired;
                try savePreferenceCheckpoint(
                    allocator,
                    io,
                    &trainer,
                    path,
                    .dpo,
                    completed_epochs,
                    examples_seen,
                    &dpo_run_fingerprint,
                    .{
                        .task = @tagName(PreferenceTask.dpo),
                        .run_fingerprint_sha256 = dpo_run_fingerprint_text,
                        .epoch_index = completed_epochs,
                        .micro_batch_steps = trainer.microBatchSteps(),
                        .optimizer_steps = trainer.optimizerSteps(),
                        .accumulation_micro_batches = trainer.accumulatedMicroBatches(),
                        .dpo = .{
                            .initial_adapter_digest = initial_adapter_digest,
                            .examples_seen = examples_seen,
                            .total_loss = total_loss,
                            .total_margin = total_margin,
                            .total_accuracy = total_accuracy,
                            .initial_logprob_parity = initial_logprob_parity,
                            .initial_bucket_signature_parity = initial_bucket_signature_parity,
                        },
                    },
                );
            }
        }
    }

    const benchmark_telemetry = if (benchmark) |*recorder| try recorder.finish() else null;

    _ = try trainer.flushAccumulatedGradients();
    if (recipe.checkpoint) |checkpoint| if (checkpoint.every_epochs != null) {
        const path = dpo_checkpoint_path orelse return error.CheckpointPathRequired;
        try savePreferenceCheckpoint(
            allocator,
            io,
            &trainer,
            path,
            .dpo,
            epochs,
            examples_seen,
            &dpo_run_fingerprint,
            .{
                .task = @tagName(PreferenceTask.dpo),
                .run_fingerprint_sha256 = dpo_run_fingerprint_text,
                .epoch_index = epochs,
                .micro_batch_steps = trainer.microBatchSteps(),
                .optimizer_steps = trainer.optimizerSteps(),
                .accumulation_micro_batches = trainer.accumulatedMicroBatches(),
                .dpo = .{
                    .initial_adapter_digest = initial_adapter_digest,
                    .examples_seen = examples_seen,
                    .total_loss = total_loss,
                    .total_margin = total_margin,
                    .total_accuracy = total_accuracy,
                    .initial_logprob_parity = initial_logprob_parity,
                    .initial_bucket_signature_parity = initial_bucket_signature_parity,
                },
            },
        );
    };
    if (trainer.optimizerSteps() == 0) return error.NoOptimizerSteps;
    try validateTrainerAdapterChanged(&trainer, initial_adapter_digest);
    const graph_cache_after_training = trainer.graphCacheStats();

    const evaluation_report_path = try preferenceEvaluationReportPath(allocator, recipe, .dpo);
    defer allocator.free(evaluation_report_path);
    // Quality admission must be a function of the published policy and held-
    // out data, not of how many allocator generations the training process
    // happened to retire before evaluation. In particular, uninterrupted and
    // resumed E4B runs can reach byte-identical adapter weights while the live
    // Metal runtime retains different allocation history. Snapshot the exact
    // final values to host, retire the training device state, and evaluate on
    // a separately initialized backend/trainer with both private-buffer reuse
    // tiers disabled.
    var graph_cache_after_evaluation = graph_cache_after_training;
    const evaluation = evaluation: {
        try trainer.prepareTerminalEvaluationFromHostSnapshot();

        var evaluation_backend = try gemma4_real_autodiff.loadBackendForModelDir(
            allocator,
            base_model_dir,
            backend_kind,
        );
        defer evaluation_backend.deinit();
        var evaluation_trainer = try real_autodiff.RealAutodiffTrainer.init(
            allocator,
            evaluation_backend.backendPtr(),
            .{
                .lora = lora_config,
                .optimizer = .{},
                .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
                .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
                .grad_accum_steps = grad_accum_steps,
                .seed = recipe.optimizer.seed orelse 42,
                .hidden_size_hint = graph_config.hidden_size,
                .num_layers_hint = graph_config.num_hidden_layers,
                .execution_engine = execution_policy.engine,
                .compiled_required = execution_policy.compiled_required,
                .strict_metal_execution = execution_policy.strict_metal_execution,
                .checkpoint_config = dpo_checkpoint_config,
                .metal_slot_bound_outputs = slot_bound_outputs,
                .graph_cache_capacity = graph_cache_capacity,
            },
        );
        defer evaluation_trainer.deinit();
        var evaluation_ctx = gemma4_real_autodiff.GemmaAutodiffCtx.init(graph_config);
        evaluation_ctx.enable_fused_rms_norm_backward = ctx.enable_fused_rms_norm_backward;
        evaluation_ctx.enable_fused_gqa_attention_backward = ctx.enable_fused_gqa_attention_backward;
        evaluation_ctx.enable_fused_linear_cross_entropy = ctx.enable_fused_linear_cross_entropy;
        var canonical_eval_reuse_scope = try gemma4_real_autodiff.configureMetalBufferReuseForPreferenceRun(
            &evaluation_trainer,
            false,
            false,
        );
        defer canonical_eval_reuse_scope.deinit();
        try gemma4_real_autodiff.initializeTrainerFromAdapterDir(
            allocator,
            &evaluation_trainer,
            &evaluation_ctx,
            bootstrap_dir,
            bootstrap_example,
            @intCast(max_seq_len),
        );
        try evaluation_trainer.initializeTerminalEvaluationFromHostSnapshot(&trainer);
        const summary = try evaluateGemmaDpoHeldout(
            allocator,
            io,
            recipe,
            tokenizer_model,
            samples.samples,
            &evaluation_trainer,
            &evaluation_ctx,
            base_model_dir,
            backend_kind,
            evaluation_report_path,
            true,
        );
        graph_cache_after_evaluation = evaluation_trainer.graphCacheStats();
        break :evaluation summary;
    };

    const baseline_relative: ?DpoBaselineRelativeSummary = if (baseline_evaluation) |baseline| relative: {
        const summary = compareDpoToBaseline(baseline, evaluation, dpo_minimums);
        if (!summary.passed) return error.DpoBaselineRelativeEvaluationGateFailed;
        break :relative summary;
    } else null;

    try gemma4_real_autodiff.saveTrainerAsGemmaBundle(allocator, &trainer, base_model_dir, bootstrap_dir, trained_dir);
    try validatePublishedAdapterChanged(allocator, io, bootstrap_dir, trained_dir);
    // Report only completion-published storage. This also guarantees the
    // override can drain/restore the cache at its defer boundary.
    try trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame();
    const completion_cache_stats = gemma4_real_autodiff.metalCompletionCacheStats(&trainer);
    const completion_cache_telemetry: ?DpoMetalCompletionCacheTelemetry = if (backend_kind == .metal)
        .{
            .enabled = completion_cache_stats.enabled,
            .max_bytes = completion_cache_stats.max_bytes,
            .available_bytes = completion_cache_stats.available_bytes,
            .available_slots = completion_cache_stats.available_slots,
            .peak_bytes = completion_cache_stats.peak_bytes,
            .peak_slots = completion_cache_stats.peak_slots,
            .requests = completion_cache_stats.requests,
            .hits = completion_cache_stats.hits,
            .misses = completion_cache_stats.misses,
            .retired = completion_cache_stats.retired,
            .evictions = completion_cache_stats.evictions,
            .completed_generation = completion_cache_stats.completed_generation,
        }
    else
        null;

    var dpo_checkpoint_artifact = try preferenceCheckpointArtifactSummary(
        allocator,
        dpo_checkpoint_path,
        .dpo,
        trainer.trainingProgress(),
    );
    defer if (dpo_checkpoint_artifact) |*artifact| artifact.deinit(allocator);

    const denom = @as(f64, @floatFromInt(@max(examples_seen, 1)));
    try writeJsonFile(allocator, io, report_path, DpoReport{
        .execution_mode = "train",
        .dataset_format = recipe.dataset.format.?,
        .examples = examples_seen,
        .loss = @floatCast(total_loss / denom),
        .mean_reward_margin = @floatCast(total_margin / denom),
        .accuracy = @floatCast(total_accuracy / denom),
        .beta = recipe.preference.beta orelse 0.1,
        .training_seed = recipe.optimizer.seed orelse 42,
        .policy_backend = @tagName(backend_kind),
        .optimizer_steps = trainer.optimizerSteps(),
        .micro_batch_steps = trainer.microBatchSteps(),
        .policy_scoring_mode = if (coalesce_single_token_pairs)
            "shared-prompt-single-row"
        else if (compile_pair_objective)
            "initial-parity-only-then-in-graph"
        else if (detach_pair_gradients)
            "backward-loss-reuse-device-detached"
        else
            "compiled-loss-only-device-reduced",
        .training_microbatch_mode = if (coalesce_single_token_pairs)
            "coalesced-single-token-pair-sparse-weighted-row"
        else if (compile_pair_objective)
            if (dpo_pair_graph_mode == .batched_forward)
                "compiled-single-forward-batch2-pair-in-graph-dpo"
            else
                "compiled-split-batch1-pair-in-graph-dpo"
        else if (detach_pair_gradients)
            "chosen-rejected-raw-gradients-device-combined"
        else
            "chosen-rejected-pair-fused-uniform-cce",
        .device_gradient_snapshot_mode = if (!detach_pair_gradients)
            "not-applicable"
        else if (ping_pong_gradients_requested)
            "persistent-ping-pong-accumulator-swap"
        else if (coalesced_snapshot_frame_requested)
            "single-frame-copy-then-clear"
        else
            "per-copy-wait-then-single-frame-clear",
        .activation_checkpointing_mode = if (dpo_checkpoint_config) |cfg|
            if (cfg.recursive_recompute_dependencies)
                "every-n-layers-recursive-recompute"
            else
                "every-n-layers-direct-recompute"
        else
            "disabled",
        .activation_checkpointing_layer_interval = if (dpo_checkpoint_config) |cfg| cfg.layer_interval else null,
        .metal_buffer_reuse_mode = if (backend_kind != .metal)
            "not-applicable"
        else if (in_frame_buffer_reuse_enabled and completion_cache_enabled)
            "planned-encoder-fenced-in-frame-reuse;completion-fenced-cross-frame-cache"
        else if (in_frame_buffer_reuse_enabled)
            "planned-encoder-fenced-in-frame-reuse;cross-frame-cache-disabled"
        else if (completion_cache_enabled)
            "completion-fenced-cross-frame-cache;in-frame-reuse-disabled"
        else if (slot_bound_outputs)
            "compiler-slot-workspace;in-frame-reuse-disabled"
        else
            "disabled-for-dpo-run",
        .metal_completion_cache = completion_cache_telemetry,
        .reference_mode = if (coalesce_single_token_pairs) "compiled-base-cache-shared-prompt-single-row" else "device-reduced-base-cache",
        .reference_precompute_seconds = reference_cache.precompute_seconds,
        .initial_logprob_parity = initial_logprob_parity,
        .initial_bucket_signature_parity = initial_bucket_signature_parity,
        .sequence_length_policy = sequence_length_policy,
        .graph_cache = .{
            .after_initialization = graph_cache_after_initialization,
            .after_reference_precompute = graph_cache_after_reference_precompute,
            .after_initial_bucket_signature_parity = graph_cache_after_initial_bucket_signature_parity,
            .after_training = graph_cache_after_training,
            .after_evaluation = graph_cache_after_evaluation,
        },
        .benchmark = benchmark_telemetry,
        .checkpoint_resume = .{
            .enabled = dpo_resume_enabled,
            .start_epoch = start_epoch,
            .checkpoint_path = dpo_checkpoint_path,
            .checkpoint_state_path = if (dpo_checkpoint_artifact) |artifact| artifact.state_path else null,
            .checkpoint_state_sha256 = if (dpo_checkpoint_artifact) |artifact| artifact.state_sha256 else null,
            .checkpoint_epoch = if (dpo_checkpoint_artifact) |artifact| artifact.epoch else null,
            .checkpoint_every_epochs = if (recipe.checkpoint) |checkpoint| checkpoint.every_epochs else null,
            .run_fingerprint_sha256 = dpo_run_fingerprint_text,
            .restored_micro_batch_steps = dpo_restored.micro_batch_steps,
            .restored_optimizer_steps = dpo_restored.optimizer_steps,
            .restored_accumulation_micro_batches = dpo_restored.accumulation_micro_batches,
        },
        .metal_numerical_policy = metal_numerical_policy,
        .evaluation_execution_policy = canonical_preference_evaluation_policy,
        .baseline_evaluation = baseline_evaluation,
        .baseline_relative = baseline_relative,
        .evaluation = evaluation,
        .trained_adapter_dir = trained_dir,
    });
    print("dpo report: {s}\ntrained adapter: {s}\n", .{ report_path, trained_dir });
}

fn evaluateGemmaDpoHeldout(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    tokenizer_model: *model_manager_mod.LoadedModel,
    train_samples: []const preference_harness.PreferenceSample,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_real_autodiff.GemmaAutodiffCtx,
    base_model_dir: []const u8,
    backend_kind: gemma4_real_autodiff.BackendKind,
    report_path: []const u8,
    enforce_minimums: bool,
) !DpoEvaluationSummary {
    const eval_path = evalDatasetPath(recipe) orelse return error.MissingPreferenceEvaluationDataset;
    const minimums = recipe.eval.?.dpo_minimums orelse return error.MissingDpoEvaluationMinimums;
    var eval_recipe = recipe;
    eval_recipe.dataset.max_examples = evalMaxExamples(recipe);
    var samples = try loadDpoTextPreferenceSamples(allocator, io, eval_path, eval_recipe, tokenizer_model);
    defer samples.deinit();
    const dataset_fingerprint = try fingerprintPath(allocator, io, "eval_dataset", eval_path);
    defer if (dataset_fingerprint.digest) |digest| allocator.free(digest);
    const policy_adapter_digest = try formatSha256DigestAlloc(allocator, trainerLoRAParameterDigest(trainer));
    defer allocator.free(policy_adapter_digest);

    const overlap_count = try countPreferencePromptOverlaps(allocator, train_samples, samples.samples);
    if (overlap_count != 0) {
        try writeJsonFile(allocator, io, report_path, DpoEvaluationReport{
            .status = "failed-prompt-overlap",
            .dataset_path = eval_path,
            .dataset_fingerprint = dataset_fingerprint,
            .policy_adapter_digest = policy_adapter_digest,
            .policy_backend = @tagName(backend_kind),
            .metal_numerical_policy = resolveGemmaMetalNumericalPolicy(backend_kind, ctx),
            .examples = samples.samples.len,
            .prompt_overlap_count = overlap_count,
            .loss = 0.0,
            .mean_reward_margin = 0.0,
            .accuracy = 0.0,
            .minimums = minimums,
            .reference_mode = "compiled-zero-lora",
        });
        return error.PreferenceTrainEvalPromptOverlap;
    }

    const max_examples = evalMaxExamples(recipe) orelse samples.samples.len;
    const max_seq_len = recipe.dataset.max_seq_len orelse 512;
    var chosen = try prepareGemmaDpoPreparedExamplesFromSamples(
        allocator,
        base_model_dir,
        samples.samples,
        max_examples,
        max_seq_len,
        .chosen,
    );
    defer gemma4.freePreparedInputsSummary(allocator, &chosen);
    var rejected = try prepareGemmaDpoPreparedExamplesFromSamples(
        allocator,
        base_model_dir,
        samples.samples,
        max_examples,
        max_seq_len,
        .rejected,
    );
    defer gemma4.freePreparedInputsSummary(allocator, &rejected);
    if (chosen.examples.len != rejected.examples.len or chosen.examples.len != samples.samples.len) {
        return error.DpoBatchAlignmentMismatch;
    }

    const coalesce = backend_kind == .metal and
        allGemmaDpoPairsAreSingleTokenSharedPrompt(chosen.examples, rejected.examples);
    const length_buckets = gemmaDpoLengthBuckets(recipe);
    const sequence_length_policy = try summarizeGemmaDpoPairLengthPolicy(
        allocator,
        chosen.examples,
        rejected.examples,
        @intCast(max_seq_len),
        length_buckets,
        @intCast(trainer.graphCacheStats().capacity),
        "heldout-dataset-one-pass",
    );
    var reference = try precomputeGemmaDpoBaseReferenceCache(
        allocator,
        trainer,
        ctx,
        chosen.examples,
        rejected.examples,
        @intCast(max_seq_len),
        coalesce,
        length_buckets,
    );
    defer reference.deinit();

    const policy_chosen = try allocator.alloc(f32, samples.samples.len);
    defer allocator.free(policy_chosen);
    const policy_rejected = try allocator.alloc(f32, samples.samples.len);
    defer allocator.free(policy_rejected);
    const chosen_lengths = try allocator.alloc(u32, samples.samples.len);
    defer allocator.free(chosen_lengths);
    const rejected_lengths = try allocator.alloc(u32, samples.samples.len);
    defer allocator.free(rejected_lengths);
    const sft_loss = try allocator.alloc(f32, samples.samples.len);
    defer allocator.free(sft_loss);

    for (chosen.examples, rejected.examples, samples.samples, 0..) |*chosen_example, *rejected_example, sample, idx| {
        const pair_schedule = try gemmaDpoPairSchedule(
            chosen_example,
            rejected_example,
            @intCast(max_seq_len),
            length_buckets,
        );
        if (coalesce) {
            const pair = gemmaDpoSingleTokenPair(chosen_example, rejected_example) orelse
                return error.DpoSingleTokenPairContractMismatch;
            const candidates = [_]i32{ pair.chosen_token, pair.rejected_token };
            var pair_logps: [2]f32 = undefined;
            try gemma4_real_autodiff.singleTokenCandidateLogprobsForPrompt(
                allocator,
                trainer,
                ctx,
                pair.prompt,
                &candidates,
                pair_schedule.sequence_length,
                &pair_logps,
            );
            policy_chosen[idx] = pair_logps[0];
            policy_rejected[idx] = pair_logps[1];
        } else {
            policy_chosen[idx] = if (length_buckets == null)
                try gemma4_real_autodiff.sequenceLogprobForExample(
                    allocator,
                    trainer,
                    ctx,
                    chosen_example,
                    @intCast(max_seq_len),
                )
            else
                try gemma4_real_autodiff.sequenceLogprobForExampleScheduled(
                    allocator,
                    trainer,
                    ctx,
                    chosen_example,
                    pair_schedule.sequence_length,
                    pair_schedule.weighted_target_rows,
                );
            policy_rejected[idx] = if (length_buckets == null)
                try gemma4_real_autodiff.sequenceLogprobForExample(
                    allocator,
                    trainer,
                    ctx,
                    rejected_example,
                    @intCast(max_seq_len),
                )
            else
                try gemma4_real_autodiff.sequenceLogprobForExampleScheduled(
                    allocator,
                    trainer,
                    ctx,
                    rejected_example,
                    pair_schedule.sequence_length,
                    pair_schedule.weighted_target_rows,
                );
        }
        chosen_lengths[idx] = @intCast(sample.chosen_tokens.len);
        rejected_lengths[idx] = @intCast(sample.rejected_tokens.len);
        sft_loss[idx] = sample.sft_chosen_loss orelse 0.0;
    }

    var result = try preference_loss.pairedPreferenceLoss(allocator, .{
        .policy_chosen_logps = policy_chosen,
        .policy_rejected_logps = policy_rejected,
        .ref_chosen_logps = reference.chosen_logps,
        .ref_rejected_logps = reference.rejected_logps,
        .chosen_lengths = chosen_lengths,
        .rejected_lengths = rejected_lengths,
        .sft_chosen_loss = sft_loss,
    }, .{
        .kind = .dpo,
        .beta = recipe.preference.beta orelse 0.1,
        .simpo_gamma = recipe.preference.simpo_gamma orelse 0.5,
        .sft_lambda = recipe.preference.sft_lambda orelse 1.0,
        .ipo_tau = recipe.preference.ipo_tau orelse 0.1,
    });
    defer result.deinit();

    const passed = @as(f64, result.accuracy) >= minimums.accuracy and
        @as(f64, result.loss) <= minimums.max_loss;
    try writeJsonFile(allocator, io, report_path, DpoEvaluationReport{
        .status = if (passed) "passed" else "failed-quality-gate",
        .dataset_path = eval_path,
        .dataset_fingerprint = dataset_fingerprint,
        .policy_adapter_digest = policy_adapter_digest,
        .policy_backend = @tagName(backend_kind),
        .metal_numerical_policy = resolveGemmaMetalNumericalPolicy(backend_kind, ctx),
        .examples = samples.samples.len,
        .prompt_overlap_count = 0,
        .loss = result.loss,
        .mean_reward_margin = result.mean_reward_margin,
        .accuracy = result.accuracy,
        .minimums = minimums,
        .reference_mode = if (coalesce) "compiled-zero-lora-shared-prompt-single-row" else "compiled-zero-lora",
        .sequence_length_policy = sequence_length_policy,
    });
    if (!passed and enforce_minimums) return error.DpoEvaluationGateFailed;
    return .{
        .report_path = report_path,
        .examples = samples.samples.len,
        .loss = result.loss,
        .mean_reward_margin = result.mean_reward_margin,
        .accuracy = result.accuracy,
        .passed = passed,
    };
}

fn countPreferencePromptOverlaps(
    allocator: std.mem.Allocator,
    train_samples: []const preference_harness.PreferenceSample,
    eval_samples: []const preference_harness.PreferenceSample,
) !usize {
    var train_prompts = std.StringHashMap(void).init(allocator);
    defer train_prompts.deinit();
    for (train_samples) |sample| {
        try train_prompts.put(std.mem.sliceAsBytes(sample.prompt_tokens), {});
    }

    var count: usize = 0;
    for (eval_samples) |eval_sample| {
        count += @intFromBool(train_prompts.contains(std.mem.sliceAsBytes(eval_sample.prompt_tokens)));
    }
    return count;
}

fn runOptimizerBackedQwen2Dpo(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    dataset_path: []const u8,
    report_path: []const u8,
) !void {
    const base_model_dir = recipe.model.path orelse return error.MissingModelPath;
    const adapter = recipe.adapter orelse AdapterConfig{};
    const bootstrap_dir_config = adapter.path;
    const bootstrap_dir = bootstrap_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    defer if (bootstrap_dir_config == null) allocator.free(bootstrap_dir);
    const trained_dir_config = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir;
    const trained_dir = trained_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    defer if (trained_dir_config == null) allocator.free(trained_dir);
    const reference_path = recipe.model.reference_path orelse base_model_dir;
    const backend_kind: qwen2_real_autodiff.BackendKind = .native;
    const max_examples = recipe.dataset.max_examples orelse 32;
    const max_seq_len = recipe.dataset.max_seq_len orelse 512;
    const family = recipe.model.family orelse try inferFamily(recipe);
    const default_target_modules = qwenLoraTargetModulesForFamily(family);
    try validateNonGemmaAdapterOptions(adapter);
    const bootstrap_target_modules = try adapterTargetModulesForQwen(adapter, default_target_modules);

    compat.cwd().access(compat.io(), bootstrap_dir, .{}) catch {
        var bootstrap = try colqwen2.bootstrapLoRABundle(allocator, base_model_dir, bootstrap_dir, .{
            .rank = adapterRank(adapter, .dpo),
            .alpha = adapterAlpha(adapter),
            .base_model_name_or_path = adapter.base_model_name_or_path,
            .target_modules = bootstrap_target_modules,
        });
        defer colqwen2.freeBootstrapSummary(allocator, &bootstrap);
    };

    var session_manager = backends.SessionManager.init(allocator);
    native_backend_choice.configureSessionPreference(&session_manager, try parseRecipeBackendChoice(recipe.backend));
    var model_manager = model_manager_mod.ModelManager.init(allocator, session_manager);
    defer model_manager.deinit();
    const reference_model = try model_manager.loadFromDir(reference_path);

    var samples = try loadDpoTextPreferenceSamples(allocator, io, dataset_path, recipe, reference_model);
    defer samples.deinit();

    var chosen_prepared = try prepareGemmaDpoPreparedExamplesFromSamples(allocator, base_model_dir, samples.samples, max_examples, max_seq_len, .chosen);
    defer gemma4.freePreparedInputsSummary(allocator, &chosen_prepared);
    var rejected_prepared = try prepareGemmaDpoPreparedExamplesFromSamples(allocator, base_model_dir, samples.samples, max_examples, max_seq_len, .rejected);
    defer gemma4.freePreparedInputsSummary(allocator, &rejected_prepared);
    if (chosen_prepared.examples.len != rejected_prepared.examples.len or chosen_prepared.examples.len != samples.samples.len) {
        return error.DpoBatchAlignmentMismatch;
    }

    const graph_config = try qwen2_real_autodiff.loadGraphConfig(allocator, base_model_dir);
    var backend = try qwen2_real_autodiff.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();

    var adapter_inspect = try colqwen2.inspectCheckpoint(allocator, bootstrap_dir);
    defer colqwen2.freeInspectionSummary(allocator, &adapter_inspect);
    const lora_rank = adapter_inspect.lora_rank orelse return error.MissingAdapterConfig;
    const lora_alpha = @as(f32, @floatCast(adapter_inspect.lora_alpha orelse return error.MissingAdapterConfig));
    const target_modules = adapter_inspect.target_modules orelse bootstrap_target_modules;
    const lora_config = ml.graph.lora.LoRAConfig{
        .rank = @intCast(lora_rank),
        .alpha = lora_alpha,
        .target_patterns = target_modules,
    };

    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
        .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
        .grad_accum_steps = @max((recipe.optimizer.gradient_accumulation_steps orelse 1) * 2, 1),
        .hidden_size_hint = graph_config.arch.hidden_size,
        .num_layers_hint = graph_config.arch.num_hidden_layers,
    });
    defer trainer.deinit();

    var ctx = qwen2_real_autodiff.Qwen2AutodiffCtx.init(graph_config);
    const bootstrap_example = qwen2_real_autodiff.findFirstSupervisedExample(chosen_prepared.examples) orelse return error.NoTrainingData;
    try qwen2_real_autodiff.initializeTrainerFromAdapterDir(allocator, &trainer, &ctx, bootstrap_dir, bootstrap_example, @intCast(max_seq_len));

    var ref_scorer = DecoderLogprobScorer{
        .allocator = allocator,
        .model = reference_model,
        .max_seq_len = max_seq_len,
    };

    const epochs = recipe.optimizer.epochs orelse 1;
    var total_loss: f64 = 0.0;
    var total_margin: f64 = 0.0;
    var total_accuracy: f64 = 0.0;
    var examples_seen: usize = 0;
    var single_pc = [_]f32{0};
    var single_pr = [_]f32{0};
    var single_rc = [_]f32{0};
    var single_rr = [_]f32{0};
    var single_cl = [_]u32{0};
    var single_rl = [_]u32{0};
    var single_sft = [_]f32{0};

    var epoch_idx: usize = 0;
    while (epoch_idx < epochs) : (epoch_idx += 1) {
        for (chosen_prepared.examples, rejected_prepared.examples, samples.samples) |*chosen_ex, *rejected_ex, sample| {
            const policy_chosen = try qwen2_real_autodiff.sequenceLogprobForExample(allocator, &trainer, &ctx, chosen_ex, @intCast(max_seq_len));
            const policy_rejected = try qwen2_real_autodiff.sequenceLogprobForExample(allocator, &trainer, &ctx, rejected_ex, @intCast(max_seq_len));

            try DecoderLogprobScorer.modelForward(@ptrCast(&ref_scorer), &.{sample.prompt_tokens}, &.{sample.chosen_tokens}, single_rc[0..1]);
            try DecoderLogprobScorer.modelForward(@ptrCast(&ref_scorer), &.{sample.prompt_tokens}, &.{sample.rejected_tokens}, single_rr[0..1]);

            single_pc[0] = policy_chosen;
            single_pr[0] = policy_rejected;
            single_cl[0] = @intCast(sample.chosen_tokens.len);
            single_rl[0] = @intCast(sample.rejected_tokens.len);
            single_sft[0] = sample.sft_chosen_loss orelse 0;

            var step_result = try preference_loss.pairedPreferenceLoss(allocator, .{
                .policy_chosen_logps = single_pc[0..1],
                .policy_rejected_logps = single_pr[0..1],
                .ref_chosen_logps = single_rc[0..1],
                .ref_rejected_logps = single_rr[0..1],
                .chosen_lengths = single_cl[0..1],
                .rejected_lengths = single_rl[0..1],
                .sft_chosen_loss = single_sft[0..1],
            }, .{
                .kind = .dpo,
                .beta = recipe.preference.beta orelse 0.1,
                .simpo_gamma = recipe.preference.simpo_gamma orelse 0.5,
                .sft_lambda = recipe.preference.sft_lambda orelse 1.0,
                .ipo_tau = recipe.preference.ipo_tau orelse 0.1,
            });
            defer step_result.deinit();
            try scalePreferenceUnitGradients(step_result.grad_chosen, 2);
            try scalePreferenceUnitGradients(step_result.grad_rejected, 2);

            total_loss += step_result.loss;
            total_margin += step_result.mean_reward_margin;
            total_accuracy += step_result.accuracy;
            examples_seen += 1;

            {
                var chosen_input = try qwen2_real_autodiff.makeTrainerInputForLogprobCoeff(allocator, &ctx, chosen_ex, @intCast(max_seq_len), step_result.grad_chosen[0]);
                defer chosen_input.deinit(allocator);
                _ = try trainer.step(chosen_input.trainer_input);
            }

            {
                var rejected_input = try qwen2_real_autodiff.makeTrainerInputForLogprobCoeff(allocator, &ctx, rejected_ex, @intCast(max_seq_len), step_result.grad_rejected[0]);
                defer rejected_input.deinit(allocator);
                _ = try trainer.step(rejected_input.trainer_input);
            }
        }
    }

    try qwen2_real_autodiff.saveTrainerAsQwenAdapterDir(allocator, &trainer, base_model_dir, bootstrap_dir, trained_dir);

    const denom = @as(f64, @floatFromInt(@max(examples_seen, 1)));
    try writeJsonFile(allocator, io, report_path, DpoReport{
        .execution_mode = "train",
        .dataset_format = recipe.dataset.format.?,
        .examples = examples_seen,
        .loss = @floatCast(total_loss / denom),
        .mean_reward_margin = @floatCast(total_margin / denom),
        .accuracy = @floatCast(total_accuracy / denom),
        .beta = recipe.preference.beta orelse 0.1,
        .trained_adapter_dir = trained_dir,
    });
    print("dpo report: {s}\ntrained adapter: {s}\n", .{ report_path, trained_dir });
}

const DpoPreparedSide = enum { chosen, rejected };

fn prepareGemmaDpoPreparedExamplesFromSamples(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    samples: []const preference_harness.PreferenceSample,
    max_examples: usize,
    max_seq_len: usize,
    side: DpoPreparedSide,
) !gemma4.PreparedInputsSummary {
    const limit = @min(samples.len, max_examples);
    const examples = try allocator.alloc(gemma4.PreparedExampleInput, limit);
    var built_count: usize = 0;
    errdefer {
        for (examples[0..built_count]) |*example| freeGemmaPreparedExample(allocator, example);
        allocator.free(examples);
    }

    for (samples[0..limit], 0..) |sample, idx| {
        const completion = switch (side) {
            .chosen => sample.chosen_tokens,
            .rejected => sample.rejected_tokens,
        };
        examples[idx] = try buildGemmaPreparedExampleFromTokens(allocator, sample.prompt_tokens, completion, max_seq_len);
        built_count += 1;
    }

    var max_prompt_tokens: usize = 0;
    var max_response_tokens: usize = 0;
    var max_input_tokens: usize = 0;
    var max_supervised_tokens: usize = 0;
    for (examples[0..limit]) |example| {
        max_prompt_tokens = @max(max_prompt_tokens, example.num_prompt_tokens);
        max_response_tokens = @max(max_response_tokens, example.num_response_tokens);
        max_input_tokens = @max(max_input_tokens, example.num_input_tokens);
        max_supervised_tokens = @max(max_supervised_tokens, example.num_supervised_tokens);
    }

    return .{
        .artifact_family_version = try allocator.dupe(u8, gemma4.artifact_family_version),
        .model_dir = try allocator.dupe(u8, model_dir),
        .max_examples = max_examples,
        .examples_seen = limit,
        .max_seq_len = max_seq_len,
        .max_prompt_tokens = max_prompt_tokens,
        .max_response_tokens = max_response_tokens,
        .max_input_tokens = max_input_tokens,
        .max_supervised_tokens = max_supervised_tokens,
        .examples = examples,
    };
}

fn shouldRunOptimizerBackedGemmaGrpo(recipe: Recipe, format: []const u8) !bool {
    if (!std.mem.eql(u8, format, "text-grpo") and !std.mem.eql(u8, format, "rendered-text-grpo")) return false;
    const family = recipe.model.family orelse try inferFamily(recipe);
    if (!eqlAny(family, &.{ "gemma4", "gemma" })) return false;
    return recipe.artifacts.trained_adapter_dir != null or recipe.artifacts.adapter_dir != null or recipe.adapter != null;
}

fn shouldRunOptimizerBackedQwen2Grpo(recipe: Recipe, format: []const u8) !bool {
    if (!std.mem.eql(u8, format, "text-grpo") and !std.mem.eql(u8, format, "rendered-text-grpo")) return false;
    const family = recipe.model.family orelse try inferFamily(recipe);
    if (!isQwen35Family(family) and !eqlAny(family, &.{ "qwen2", "qwen", "colqwen2", "colqwen", "qwen2vl" })) return false;
    return recipe.artifacts.trained_adapter_dir != null or recipe.artifacts.adapter_dir != null or recipe.adapter != null;
}

fn parseTextRewardMode(value: []const u8) !TextRewardMode {
    if (std.mem.eql(u8, value, "exact-match")) return .exact_match;
    if (std.mem.eql(u8, value, "exact-match-ci")) return .exact_match_ci;
    if (std.mem.eql(u8, value, "prefix-match")) return .prefix_match;
    if (std.mem.eql(u8, value, "token-exact-match")) return .token_exact_match;
    if (std.mem.eql(u8, value, "token-prefix-match")) return .token_prefix_match;
    return error.UnsupportedRewardMode;
}

fn runOptimizerBackedGemmaGrpo(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    dataset_path: []const u8,
    report_path: []const u8,
) !void {
    const base_model_dir = recipe.model.path orelse return error.MissingModelPath;
    const adapter = recipe.adapter orelse AdapterConfig{};
    const bootstrap_dir_config = adapter.path;
    const bootstrap_dir = bootstrap_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    defer if (bootstrap_dir_config == null) allocator.free(bootstrap_dir);
    const trained_dir_config = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir;
    const trained_dir = trained_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    defer if (trained_dir_config == null) allocator.free(trained_dir);
    try requireMissingPreferencePublicationTarget(io, trained_dir);
    try preflightRewardExecutables(allocator, io, recipe);
    const reference_path = recipe.model.reference_path orelse base_model_dir;
    if (!std.mem.eql(u8, reference_path, base_model_dir)) return error.UnsupportedReferencePath;
    const execution = try resolveGemmaPreferenceExecution(recipe.backend);
    const backend_kind = execution.backend_kind;
    try validateGemmaPreferenceEnvironmentContract(backend_kind);
    const execution_policy = train_eval_gemma4_lora_bundle.autodiffExecutionPolicy(backend_kind);
    const max_seq_len = recipe.dataset.max_seq_len orelse 128;
    const group_size = recipe.grpo.group_size orelse 2;
    const max_completion_tokens = recipe.grpo.max_completion_tokens orelse 4;
    if (group_size < 2) return error.InvalidGrpoGroupSize;
    if (max_completion_tokens == 0) return error.InvalidMaxCompletionTokens;
    const coalesce_single_token_groups = max_completion_tokens == 1;
    const batch_single_token_group_scoring = coalesce_single_token_groups and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_BATCH_SINGLE_TOKEN_SCORING", true);
    const batch_multi_token_group_scoring = !coalesce_single_token_groups and
        gemmaGrpoMultiTokenBatchEnabled(group_size);
    const multi_token_backward_batch_size = if (!coalesce_single_token_groups and backend_kind == .metal)
        gemmaGrpoMultiTokenBackwardBatchSize(group_size)
    else
        1;
    const batch_multi_token_group_backward = multi_token_backward_batch_size > 1;
    const sparse_multi_token = !coalesce_single_token_groups and
        backend_kind == .metal and
        !batch_multi_token_group_scoring and
        gemmaGrpoSparseMultiTokenEnabled();
    const physical_micro_batches_per_group: usize = if (coalesce_single_token_groups)
        1
    else if (batch_multi_token_group_backward)
        try std.math.divCeil(usize, group_size, multi_token_backward_batch_size)
    else
        group_size;
    const grad_accum_steps = try preferenceGradAccumSteps(
        recipe.optimizer.gradient_accumulation_steps orelse 1,
        physical_micro_batches_per_group,
    );
    const reward_mode = try parseTextRewardMode(recipe.grpo.reward_mode orelse "exact-match");
    try validateGemmaAdapterOptions(adapter);
    var graph_executor_scope = try train_eval_gemma4_lora_bundle.acquireMetalGraphExecutorScope(backend_kind);
    defer if (graph_executor_scope) |*scope| scope.deinit();
    try train_eval_gemma4_lora_bundle.validateAutodiffBaseArtifactForRecipe(
        allocator,
        base_model_dir,
        backend_kind,
        recipe.model.allow_direct_gguf_training orelse false,
    );
    const direct_gguf_base = try train_eval_gemma4_lora_bundle.autodiffBaseUsesGguf(allocator, base_model_dir);
    if ((recipe.model.allow_direct_gguf_training orelse false) != direct_gguf_base) {
        return error.DirectGgufTrainingAdmissionMismatch;
    }

    compat.cwd().access(compat.io(), bootstrap_dir, .{}) catch {
        var bootstrap = try gemma4.bootstrapLoRABundle(allocator, base_model_dir, bootstrap_dir, .{
            .rank = adapterRank(adapter, .grpo),
            .alpha = adapterAlpha(adapter),
            .base_model_name_or_path = adapter.base_model_name_or_path,
            .target_modules = adapter.target_modules,
            .gemma4_target_preset = gemma4TargetPreset(adapter),
            .target_preset = try gemmaLegacyTargetPreset(adapter),
            .use_dora = adapter.use_dora orelse false,
            .init_lora_weights = adapter.init_lora_weights,
            .initialization_seed = adapter.initialization_seed orelse 0,
        });
        defer gemma4.freeBootstrapSummary(allocator, &bootstrap);
    };

    if (recipe.model.projector_path != null) {
        try runOptimizerBackedGemmaMultimodalGrpo(
            allocator,
            io,
            recipe,
            dataset_path,
            report_path,
            base_model_dir,
            bootstrap_dir,
            trained_dir,
            reference_path,
            backend_kind,
            max_seq_len,
            group_size,
            max_completion_tokens,
            reward_mode,
        );
        return;
    }

    var session_manager = backends.SessionManager.init(allocator);
    native_backend_choice.configureSessionPreference(&session_manager, execution.session_choice);
    var model_manager = model_manager_mod.ModelManager.init(allocator, session_manager);
    defer model_manager.deinit();

    const tokenizer_model = try model_manager.loadFromDir(base_model_dir);

    var prompt_batch = try loadGrpoTextPrompts(allocator, io, dataset_path, recipe, tokenizer_model);
    defer prompt_batch.deinit();
    const epochs = recipe.optimizer.epochs orelse 1;
    const benchmark_enabled = platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_BENCHMARK", false);
    if (benchmark_enabled and recipe.checkpoint != null) return error.PreferenceBenchmarkCheckpointingNotSupported;
    const planned_updates = std.math.mul(usize, prompt_batch.prompts.len, epochs) catch return error.GrpoBenchmarkUpdateCountMismatch;
    if (benchmark_enabled and planned_updates != GrpoBenchmarkRecorder.total_updates) {
        return error.GrpoBenchmarkUpdateCountMismatch;
    }
    var benchmark: ?GrpoBenchmarkRecorder = if (benchmark_enabled) try GrpoBenchmarkRecorder.init(allocator) else null;
    defer if (benchmark) |*recorder| recorder.deinit();

    const graph_config = try gemma4_real_autodiff.loadGraphConfig(allocator, base_model_dir);
    var backend = try gemma4_real_autodiff.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();

    var adapter_inspect = try gemma4.inspectCheckpoint(allocator, bootstrap_dir);
    defer gemma4.freeInspectionSummary(allocator, &adapter_inspect);
    const lora_rank = adapter_inspect.lora_rank orelse return error.MissingAdapterConfig;
    const lora_alpha = @as(f32, @floatCast(adapter_inspect.lora_alpha orelse return error.MissingAdapterConfig));
    const target_modules = adapter_inspect.target_modules orelse (adapter.target_modules orelse gemma4.default_lora_target_modules[0..]);
    const lora_config = ml.graph.lora.LoRAConfig{
        .rank = @intCast(lora_rank),
        .alpha = lora_alpha,
        .target_patterns = target_modules,
        .strict_target_patterns = true,
        .sharing = if (adapter_inspect.recursive_lora_enabled) .by_use else .by_weight,
    };

    const grpo_graph_cache_capacity: u8 = if (batch_multi_token_group_backward) 2 else 1;
    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
        .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
        .grad_accum_steps = grad_accum_steps,
        .seed = recipe.optimizer.seed orelse 42,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
        // Batched backward and exact batch-1 rescoring deliberately use two
        // graph signatures. Retain both so an opt-in campaign measures the
        // batching strategy rather than rebuilding on every GRPO group.
        .graph_cache_capacity = grpo_graph_cache_capacity,
    });
    defer trainer.deinit();

    // The optimizer-backed GRPO loop interleaves eager policy/reference
    // scoring with compiled backward/update frames on one Metal runtime.
    // Coalesced encoders make that mixed ownership timing-dependent: the same
    // inputs can yield a different adapter digest even when the reward trace
    // is identical. Keep ordered encoders for this trainer's complete
    // lifetime; serving and unrelated training routes retain coalescing.
    var grpo_metal_ordering_suspended = false;
    var grpo_metal_row_staging_suspended = false;
    if (backend_kind == .metal) {
        try trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame();
        grpo_metal_ordering_suspended = try trainer.compute_backend.decoderRuntimePushPlannedEncoderCoalescingSuppression();
        if (!grpo_metal_ordering_suspended) return error.PlannedEncoderCoalescingSuppressionUnavailable;
        errdefer trainer.compute_backend.decoderRuntimePopPlannedEncoderCoalescingSuppression() catch {};
        grpo_metal_row_staging_suspended = try trainer.compute_backend.decoderRuntimePushBf16EmbeddingRowStagingSuppression();
        if (!grpo_metal_row_staging_suspended) return error.Bf16EmbeddingRowStagingSuppressionUnavailable;
    }
    defer if (grpo_metal_ordering_suspended or grpo_metal_row_staging_suspended) {
        trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame() catch {
            if (trainer.compute_backend.decoderRuntimeHasActiveFrame()) {
                trainer.compute_backend.decoderRuntimeCancelFrame() catch {};
            }
        };
        if (grpo_metal_row_staging_suspended) {
            trainer.compute_backend.decoderRuntimePopBf16EmbeddingRowStagingSuppression() catch {};
        }
        trainer.compute_backend.decoderRuntimePopPlannedEncoderCoalescingSuppression() catch {};
    };

    var ctx = gemma4_real_autodiff.GemmaAutodiffCtx.init(graph_config);
    ctx.enable_fused_rms_norm_backward = backend_kind == .metal;
    ctx.enable_fused_gqa_attention_backward = backend_kind == .metal and gemma4_real_autodiff.fusedGqaAttentionExperimentEnabled(graph_config);
    ctx.enable_fused_linear_cross_entropy = backend_kind == .metal and
        !direct_gguf_base and
        !platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_LINEAR_CCE", false);
    const bootstrap_prompt = prompt_batch.prompts[0];
    if (bootstrap_prompt.len == 0 or bootstrap_prompt.len >= max_seq_len) return error.NoCompletionBudget;
    const bootstrap_completion = [_]i32{bootstrap_prompt[bootstrap_prompt.len - 1]};
    const bootstrap_example = try buildGemmaPreparedExampleFromTokens(allocator, bootstrap_prompt, &bootstrap_completion, max_seq_len);
    defer freeGemmaPreparedExample(allocator, &bootstrap_example);
    try gemma4_real_autodiff.initializeTrainerFromAdapterDir(allocator, &trainer, &ctx, bootstrap_dir, &bootstrap_example, @intCast(max_seq_len));

    const incremental_kv_enabled = gemmaGrpoIncrementalKvEnabled(recipe);
    if (incremental_kv_enabled and !sparse_multi_token) {
        return error.Gemma4GrpoIncrementalKvRequiresSparseMultiToken;
    }
    const incremental_kv_requested = incremental_kv_enabled;
    const incremental_kv_batch_active = gemmaGrpoIncrementalKvBatchActiveEnabled(recipe);
    const incremental_kv_clone_prompt_tail = gemmaGrpoIncrementalKvClonePromptTailEnabled(recipe);
    const incremental_kv_shadow_exact = gemmaGrpoIncrementalKvShadowExactEnabled(recipe);
    const reward_configuration_digest = try rewardPipelineConfigurationDigestAlloc(allocator, recipe);
    defer allocator.free(reward_configuration_digest);
    const resolved_kl_control = try resolveGrpoKlControl(recipe.grpo);
    var grpo_execution_flags: u64 = 0;
    if (coalesce_single_token_groups) grpo_execution_flags |= @as(u64, 1) << 0;
    if (batch_single_token_group_scoring) grpo_execution_flags |= @as(u64, 1) << 1;
    if (batch_multi_token_group_scoring) grpo_execution_flags |= @as(u64, 1) << 2;
    if (batch_multi_token_group_backward) grpo_execution_flags |= @as(u64, 1) << 3;
    if (sparse_multi_token) grpo_execution_flags |= @as(u64, 1) << 4;
    if (incremental_kv_requested) grpo_execution_flags |= @as(u64, 1) << 5;
    if (incremental_kv_batch_active) grpo_execution_flags |= @as(u64, 1) << 6;
    if (incremental_kv_clone_prompt_tail) grpo_execution_flags |= @as(u64, 1) << 7;
    if (incremental_kv_shadow_exact) grpo_execution_flags |= @as(u64, 1) << 8;
    if (ctx.enable_fused_gqa_attention_backward) grpo_execution_flags |= @as(u64, 1) << 9;
    const metal_numerical_policy = resolveGemmaMetalNumericalPolicy(backend_kind, &ctx);
    const grpo_run_fingerprint = try gemmaPreferenceRunFingerprint(
        allocator,
        io,
        recipe,
        .grpo,
        base_model_dir,
        bootstrap_dir,
        target_modules,
        @intCast(lora_rank),
        lora_alpha,
        adapter_inspect.recursive_lora_enabled,
        backend_kind,
        .{
            .seed = recipe.optimizer.seed orelse 42,
            .max_examples = recipe.dataset.max_examples orelse 32,
            .max_seq_len = max_seq_len,
            .epochs = epochs,
            .learning_rate = recipe.optimizer.learning_rate orelse 0.0001,
            .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
            .requested_gradient_accumulation_steps = recipe.optimizer.gradient_accumulation_steps orelse 1,
            .physical_micro_batches_per_unit = physical_micro_batches_per_group,
            .graph_cache_capacity = grpo_graph_cache_capacity,
            .direct_gguf_base = direct_gguf_base,
            .fused_linear_cross_entropy = ctx.enable_fused_linear_cross_entropy orelse false,
            .execution_flags = grpo_execution_flags,
            .metal_numerical_policy_flags = if (metal_numerical_policy) |policy| policy.fingerprint_flags else 0,
            .metal_sparse_loss_chunk_rows = if (metal_numerical_policy) |policy| policy.sparse_loss_chunk_rows else null,
            .metal_linear_cce_tile_vocab = if (metal_numerical_policy) |policy| policy.linear_cce_tile_vocab else null,
            .grpo_group_size = group_size,
            .grpo_backward_batch_size = multi_token_backward_batch_size,
            .grpo_max_completion_tokens = max_completion_tokens,
            .grpo_clip_epsilon = recipe.grpo.clip_epsilon orelse 0.2,
            .grpo_kl_coef = recipe.grpo.kl_coef orelse 0.04,
            .grpo_train_max_kl = resolved_kl_control.train_max_kl,
            .grpo_adaptive_kl = resolved_kl_control.adaptive,
            .grpo_target_kl = resolved_kl_control.target_kl,
            .grpo_kl_horizon = resolved_kl_control.kl_horizon,
            .grpo_min_kl_coef = resolved_kl_control.min_kl_coef,
            .grpo_max_kl_coef = resolved_kl_control.max_kl_coef,
            .grpo_advantage_eps = recipe.grpo.advantage_eps orelse 1e-8,
            .grpo_normalize_advantage = recipe.grpo.normalize_advantage orelse true,
            .reward_configuration_digest = reward_configuration_digest,
        },
    );
    const grpo_run_fingerprint_text = try formatSha256DigestAlloc(allocator, grpo_run_fingerprint);
    defer allocator.free(grpo_run_fingerprint_text);
    const grpo_checkpoint_path = try preferenceCheckpointPath(allocator, recipe, .grpo);
    defer if (grpo_checkpoint_path) |path| allocator.free(path);
    const grpo_resume_enabled = if (recipe.checkpoint) |checkpoint| checkpoint.resume_path != null else false;
    var grpo_restored = real_autodiff.RestoredTrainingCheckpoint{
        .micro_batch_steps = 0,
        .optimizer_steps = 0,
        .accumulation_micro_batches = 0,
        .configured_accumulation_steps = grad_accum_steps,
        .stochastic_steps = 0,
        .progress = .{},
    };
    var loaded_grpo_state: ?LoadedPreferenceCheckpointState = null;
    defer if (loaded_grpo_state) |*state| state.deinit(allocator);
    var start_epoch: usize = 0;
    var initial_adapter_digest = trainerLoRAParameterDigest(&trainer);
    if (grpo_resume_enabled) {
        const path = grpo_checkpoint_path orelse return error.CheckpointPathRequired;
        grpo_restored = try trainer.loadTrainingCheckpoint(path, &grpo_run_fingerprint);
        loaded_grpo_state = try loadPreferenceCheckpointState(
            allocator,
            io,
            path,
            .grpo,
            &grpo_run_fingerprint,
            grpo_restored,
        );
        start_epoch = std.math.cast(usize, grpo_restored.progress.epoch_index) orelse
            return error.InvalidPreferenceCheckpointState;
        if (start_epoch > epochs) return error.CheckpointBeyondRequestedEpochCount;
        initial_adapter_digest = loaded_grpo_state.?.parsed.value.grpo.?.initial_adapter_digest;
    }

    var incremental_kv_sampler: ?gemma4_real_autodiff.GrpoIncrementalKvSampler = if (incremental_kv_requested)
        try gemma4_real_autodiff.GrpoIncrementalKvSampler.init(
            allocator,
            &trainer,
            &ctx,
            incremental_kv_batch_active,
            incremental_kv_clone_prompt_tail,
        )
    else
        null;
    defer if (incremental_kv_sampler) |*sampler| sampler.deinit();

    const current_base_equivalent_policy = gemmaLoraAdapterIsBaseEquivalent(&trainer);
    const restored_grpo_aggregates = if (loaded_grpo_state) |*state| state.parsed.value.grpo else null;
    if (restored_grpo_aggregates) |state| {
        const expected_groups = std.math.mul(usize, start_epoch, prompt_batch.prompts.len) catch
            return error.InvalidPreferenceCheckpointState;
        const expected_completions = std.math.mul(usize, expected_groups, group_size) catch
            return error.InvalidPreferenceCheckpointState;
        if (state.total_groups != expected_groups or state.total_completions != expected_completions) {
            return error.InvalidPreferenceCheckpointState;
        }
        if (incremental_kv_sampler) |*sampler| {
            const telemetry = state.incremental_kv orelse return error.InvalidPreferenceCheckpointState;
            try sampler.restoreCheckpointTelemetry(telemetry, expected_groups);
        } else if (state.incremental_kv != null) {
            return error.InvalidPreferenceCheckpointState;
        }
    }
    const initial_base_equivalent_policy = if (restored_grpo_aggregates) |state|
        state.initial_base_equivalent_policy
    else
        current_base_equivalent_policy;
    var frozen_lora = try gemma4_real_autodiff.FrozenBaseLoraBindings.init(allocator, &trainer);
    defer frozen_lora.deinit();
    var reference_cache = GemmaGrpoReferenceCache.init(allocator, 1024);
    defer reference_cache.deinit();

    const grpo_minimums = recipe.eval.?.grpo_minimums.?;
    const grpo_baseline_report_path = if (grpo_minimums.min_mean_reward_improvement != null)
        try preferenceBaselineEvaluationReportPath(allocator, recipe, .grpo)
    else
        null;
    defer if (grpo_baseline_report_path) |path| allocator.free(path);
    const grpo_baseline_reward_trace_path = if (grpo_baseline_report_path != null)
        try defaultArtifactPath(allocator, recipe, "grpo_baseline_evaluation_reward_trace.jsonl")
    else
        null;
    defer if (grpo_baseline_reward_trace_path) |path| allocator.free(path);
    const grpo_baseline_exchange_dir = if (grpo_baseline_report_path != null)
        try defaultArtifactPath(allocator, recipe, "grpo-baseline-reward-verifier-exchanges")
    else
        null;
    defer if (grpo_baseline_exchange_dir) |path| allocator.free(path);
    var baseline_recipe = recipe;
    var baseline_reward = recipe.reward orelse RewardConfig{};
    if (grpo_baseline_reward_trace_path) |path| baseline_reward.evaluation_trace_path = path;
    if (grpo_baseline_exchange_dir) |path| baseline_reward.exchange_dir = path;
    if (grpo_baseline_report_path != null) baseline_recipe.reward = baseline_reward;
    const baseline_evaluation: ?GrpoEvaluationSummary = if (grpo_baseline_report_path) |path|
        try evaluateGemmaGrpoHeldout(
            allocator,
            io,
            baseline_recipe,
            tokenizer_model,
            prompt_batch.prompts,
            &trainer,
            &ctx,
            null,
            &frozen_lora,
            max_seq_len,
            group_size,
            max_completion_tokens,
            backend_kind,
            path,
            false,
        )
    else
        null;

    var reward_pipeline = try RewardPipeline.init(
        allocator,
        io,
        recipe,
        tokenizer_model.getTokenizer(),
        prompt_batch.prompt_texts,
        prompt_batch.targets,
        .train,
    );
    defer reward_pipeline.deinit();
    errdefer reward_pipeline.finish() catch {};

    const rewarder = grpo.Rewarder{
        .ctx = &reward_pipeline,
        .call = RewardPipeline.score,
    };
    var cfg = grpo.GRPOConfig{
        .group_size = group_size,
        .clip_epsilon = recipe.grpo.clip_epsilon orelse 0.2,
        .kl_coef = recipe.grpo.kl_coef orelse 0.04,
        .advantage_eps = recipe.grpo.advantage_eps orelse 1e-8,
        .normalize_advantage = recipe.grpo.normalize_advantage orelse true,
    };
    var kl_control = try GrpoKlControl.init(allocator, io, recipe);
    defer kl_control.deinit();
    errdefer kl_control.finish() catch {};

    if (restored_grpo_aggregates) |state| {
        try reward_pipeline.restoreCheckpoint(
            state.reward_call_index,
            state.reward_external_calls,
            state.reward_external_failures,
            state.reward_trace,
        );
        try kl_control.restoreCheckpoint(
            state.kl_current_coef,
            state.kl_admitted_groups,
            state.kl_max_observed_mean,
            state.kl_trace,
        );
        cfg.kl_coef = state.kl_current_coef;
    }

    var total_loss: f64 = if (restored_grpo_aggregates) |state| state.total_loss else 0.0;
    var total_pg_loss: f64 = if (restored_grpo_aggregates) |state| state.total_pg_loss else 0.0;
    var total_kl_loss: f64 = if (restored_grpo_aggregates) |state| state.total_kl_loss else 0.0;
    var total_mean_kl: f64 = if (restored_grpo_aggregates) |state| state.total_mean_kl else 0.0;
    var total_clip_fraction: f64 = if (restored_grpo_aggregates) |state| state.total_clip_fraction else 0.0;
    var total_groups: usize = if (restored_grpo_aggregates) |state| state.total_groups else 0;
    var total_completions: usize = if (restored_grpo_aggregates) |state| state.total_completions else 0;
    var total_tokens: usize = if (restored_grpo_aggregates) |state| state.total_tokens else 0;
    var total_reward: f64 = if (restored_grpo_aggregates) |state| state.total_reward else 0.0;
    var total_reward_squared: f64 = if (restored_grpo_aggregates) |state| state.total_reward_squared else 0.0;
    var total_sampling_seconds: f64 = 0.0;
    var total_policy_rescore_seconds: f64 = 0.0;
    var total_backward_update_seconds: f64 = 0.0;
    var total_reference_scoring_seconds: f64 = 0.0;
    var saw_nonzero_reward_advantage = if (restored_grpo_aggregates) |state| state.saw_nonzero_reward_advantage else false;
    var saw_nonzero_policy_gradient = if (restored_grpo_aggregates) |state| state.saw_nonzero_policy_gradient else false;
    var initial_sampling_rescore_max_abs_error: f32 = if (restored_grpo_aggregates) |state| state.initial_sampling_rescore_max_abs_error else 0.0;
    var initial_policy_reference_max_abs_error: f32 = if (restored_grpo_aggregates) |state| state.initial_policy_reference_max_abs_error else 0.0;
    var captured_initial_logprob_parity = if (restored_grpo_aggregates) |state| state.captured_initial_logprob_parity else false;
    var policy_rescore_completions: usize = if (restored_grpo_aggregates) |state| state.policy_rescore_completions else 0;
    var diagnostic_first_tokens: [8]i32 = if (restored_grpo_aggregates) |state| state.diagnostic_first_tokens else @splat(-1);
    var diagnostic_policy_first_token_logps: [8]f32 = if (restored_grpo_aggregates) |state| state.diagnostic_policy_first_token_logps else @splat(0.0);
    var diagnostic_reference_first_token_logps: [8]f32 = if (restored_grpo_aggregates) |state| state.diagnostic_reference_first_token_logps else @splat(0.0);
    var diagnostic_first_token_count: usize = if (restored_grpo_aggregates) |state| state.diagnostic_first_token_count else 0;

    const top_rank_cap: usize = @min(group_size, 8);
    const eos_id = tokenizer_model.getTokenizer().specialTokens().sep_id;
    var epoch_idx: usize = start_epoch;
    while (epoch_idx < epochs) : (epoch_idx += 1) {
        for (prompt_batch.prompts, 0..) |prompt, prompt_idx| {
            const group_started_ns = if (benchmark_enabled) platform.time.monotonicNs() else 0;
            var group_sampling_rescore_max_abs_error: f32 = 0.0;
            var group_policy_reference_max_abs_error: f32 = 0.0;
            var group_token_count: usize = 0;
            var completions = std.ArrayList(grpo.Completion).empty;
            defer {
                for (completions.items) |completion| {
                    allocator.free(completion.tokens);
                    allocator.free(completion.old_logps);
                    allocator.free(completion.ref_logps);
                }
                completions.deinit(allocator);
            }
            var flat_new_logps = std.ArrayList(f32).empty;
            defer flat_new_logps.deinit(allocator);

            const sampled_token_lists = try allocator.alloc(std.ArrayList(i32), group_size);
            defer allocator.free(sampled_token_lists);
            const sampled_logp_lists = try allocator.alloc(std.ArrayList(f32), group_size);
            defer allocator.free(sampled_logp_lists);
            for (sampled_token_lists, sampled_logp_lists) |*tokens, *logps| {
                tokens.* = .empty;
                logps.* = .empty;
            }
            defer for (sampled_token_lists, sampled_logp_lists) |*tokens, *logps| {
                tokens.deinit(allocator);
                logps.deinit(allocator);
            };
            const sampling_started_ns = platform.time.monotonicNs();
            try sampleGemmaGrpoCompletionGroup(
                allocator,
                &trainer,
                &ctx,
                if (incremental_kv_sampler) |*sampler| sampler else null,
                incremental_kv_shadow_exact,
                prompt,
                @intCast(max_seq_len),
                max_completion_tokens,
                top_rank_cap,
                if (eos_id >= 0) eos_id else null,
                sparse_multi_token,
                sampled_token_lists,
                sampled_logp_lists,
            );
            total_sampling_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - sampling_started_ns)) / std.time.ns_per_s;

            var single_token_policy_rescore_logps: ?[]f32 = null;
            defer if (single_token_policy_rescore_logps) |values| allocator.free(values);
            var single_token_reference_logps: ?[]f32 = null;
            defer if (single_token_reference_logps) |values| allocator.free(values);
            if (batch_single_token_group_scoring) {
                const reference_logps = try allocator.alloc(f32, group_size);
                single_token_reference_logps = reference_logps;
                const reference_started_ns = platform.time.monotonicNs();
                try cachedSingleTokenGroupReferenceLogps(
                    allocator,
                    &trainer,
                    &ctx,
                    prompt,
                    prompt_idx,
                    sampled_token_lists,
                    @intCast(max_seq_len),
                    &frozen_lora,
                    &reference_cache,
                    reference_logps,
                );
                total_reference_scoring_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - reference_started_ns)) / std.time.ns_per_s;

                if (!captured_initial_logprob_parity) {
                    const candidate_token_ids = try allocator.alloc(i32, group_size);
                    defer allocator.free(candidate_token_ids);
                    for (sampled_token_lists, candidate_token_ids) |tokens, *token_id| {
                        if (tokens.items.len != 1) return error.ExpectedSingleTokenCompletion;
                        token_id.* = tokens.items[0];
                    }
                    const policy_logps = try allocator.alloc(f32, group_size);
                    single_token_policy_rescore_logps = policy_logps;
                    const rescore_started_ns = platform.time.monotonicNs();
                    try gemma4_real_autodiff.singleTokenCandidateLogprobsForPrompt(
                        allocator,
                        &trainer,
                        &ctx,
                        prompt,
                        candidate_token_ids,
                        @intCast(max_seq_len),
                        policy_logps,
                    );
                    total_policy_rescore_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - rescore_started_ns)) / std.time.ns_per_s;
                    policy_rescore_completions += group_size;
                }
            }

            var multi_token_policy_rescore_logps: ?CompletionGroupLogps = null;
            defer if (multi_token_policy_rescore_logps) |*values| values.deinit();
            var multi_token_reference_logps: ?CompletionGroupLogps = null;
            defer if (multi_token_reference_logps) |*values| values.deinit();
            if (batch_multi_token_group_scoring) {
                const reference_started_ns = platform.time.monotonicNs();
                multi_token_reference_logps = try batchedMultiTokenGroupReferenceLogps(
                    allocator,
                    &trainer,
                    &ctx,
                    prompt,
                    prompt_idx,
                    sampled_token_lists,
                    @intCast(max_seq_len),
                    &frozen_lora,
                    &reference_cache,
                );
                total_reference_scoring_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - reference_started_ns)) / std.time.ns_per_s;

                if (!captured_initial_logprob_parity) {
                    const rescore_started_ns = platform.time.monotonicNs();
                    multi_token_policy_rescore_logps = try batchedMultiTokenGroupPolicyLogps(
                        allocator,
                        &trainer,
                        &ctx,
                        prompt,
                        sampled_token_lists,
                        @intCast(max_seq_len),
                    );
                    total_policy_rescore_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - rescore_started_ns)) / std.time.ns_per_s;
                    if (multi_token_policy_rescore_logps != null) {
                        policy_rescore_completions += group_size;
                    }
                }
            }

            var completion_idx: usize = 0;
            while (completion_idx < group_size) : (completion_idx += 1) {
                const tokens_owned = try sampled_token_lists[completion_idx].toOwnedSlice(allocator);
                errdefer allocator.free(tokens_owned);
                var diagnostic_slot: ?usize = null;
                if (total_groups == 0 and tokens_owned.len > 0 and diagnostic_first_token_count < diagnostic_first_tokens.len) {
                    diagnostic_slot = diagnostic_first_token_count;
                    diagnostic_first_tokens[diagnostic_slot.?] = tokens_owned[0];
                    diagnostic_first_token_count += 1;
                }
                const old_logps_owned = try sampled_logp_lists[completion_idx].toOwnedSlice(allocator);
                errdefer allocator.free(old_logps_owned);
                const new_logps_owned = try allocator.alloc(f32, tokens_owned.len);
                defer allocator.free(new_logps_owned);
                if (!captured_initial_logprob_parity) {
                    if (single_token_policy_rescore_logps) |batched_logps| {
                        if (new_logps_owned.len != 1) return error.ExpectedSingleTokenCompletion;
                        new_logps_owned[0] = batched_logps[completion_idx];
                    } else if (multi_token_policy_rescore_logps) |batched_logps| {
                        @memcpy(new_logps_owned, batched_logps.rows[completion_idx]);
                    } else if (coalesce_single_token_groups) {
                        const rescore_started_ns = platform.time.monotonicNs();
                        try gemma4_real_autodiff.tokenLogprobsForPromptCompletionSparseRows(
                            allocator,
                            &trainer,
                            &ctx,
                            prompt,
                            tokens_owned,
                            @intCast(max_seq_len),
                            new_logps_owned,
                        );
                        total_policy_rescore_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - rescore_started_ns)) / std.time.ns_per_s;
                        policy_rescore_completions += 1;
                    } else {
                        const rescore_started_ns = platform.time.monotonicNs();
                        if (sparse_multi_token) {
                            try gemma4_real_autodiff.tokenLogprobsForPromptCompletionSparseRows(
                                allocator,
                                &trainer,
                                &ctx,
                                prompt,
                                tokens_owned,
                                @intCast(max_seq_len),
                                new_logps_owned,
                            );
                        } else {
                            try gemma4_real_autodiff.tokenLogprobsForPromptCompletion(
                                allocator,
                                &trainer,
                                &ctx,
                                prompt,
                                tokens_owned,
                                @intCast(max_seq_len),
                                new_logps_owned,
                            );
                        }
                        total_policy_rescore_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - rescore_started_ns)) / std.time.ns_per_s;
                        policy_rescore_completions += 1;
                    }
                } else {
                    @memcpy(new_logps_owned, old_logps_owned);
                }
                const ref_logps_owned = try allocator.alloc(f32, tokens_owned.len);
                errdefer allocator.free(ref_logps_owned);
                if (single_token_reference_logps) |batched_logps| {
                    if (ref_logps_owned.len != 1) return error.ExpectedSingleTokenCompletion;
                    ref_logps_owned[0] = batched_logps[completion_idx];
                } else if (multi_token_reference_logps) |batched_logps| {
                    @memcpy(ref_logps_owned, batched_logps.rows[completion_idx]);
                } else if (!try reference_cache.lookup(prompt_idx, tokens_owned, ref_logps_owned)) {
                    const reference_started_ns = platform.time.monotonicNs();
                    if (coalesce_single_token_groups) {
                        try gemma4_real_autodiff.tokenLogprobsForPromptCompletionSparseRowsFrozenBase(
                            allocator,
                            &trainer,
                            &ctx,
                            prompt,
                            tokens_owned,
                            @intCast(max_seq_len),
                            ref_logps_owned,
                            &frozen_lora,
                        );
                    } else {
                        if (sparse_multi_token) {
                            try gemma4_real_autodiff.tokenLogprobsForPromptCompletionSparseRowsFrozenBase(
                                allocator,
                                &trainer,
                                &ctx,
                                prompt,
                                tokens_owned,
                                @intCast(max_seq_len),
                                ref_logps_owned,
                                &frozen_lora,
                            );
                        } else {
                            try gemma4_real_autodiff.tokenLogprobsForPromptCompletionFrozenBase(
                                allocator,
                                &trainer,
                                &ctx,
                                prompt,
                                tokens_owned,
                                @intCast(max_seq_len),
                                ref_logps_owned,
                                &frozen_lora,
                            );
                        }
                    }
                    total_reference_scoring_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - reference_started_ns)) / std.time.ns_per_s;
                    try reference_cache.insert(prompt_idx, tokens_owned, ref_logps_owned);
                }
                for (old_logps_owned, new_logps_owned, ref_logps_owned) |old_logp, new_logp, ref_logp| {
                    if (!std.math.isFinite(old_logp) or !std.math.isFinite(new_logp) or !std.math.isFinite(ref_logp)) {
                        return error.NonFiniteGrpoLogprob;
                    }
                    group_sampling_rescore_max_abs_error = @max(group_sampling_rescore_max_abs_error, @abs(old_logp - new_logp));
                    group_policy_reference_max_abs_error = @max(group_policy_reference_max_abs_error, @abs(new_logp - ref_logp));
                }
                if (diagnostic_slot) |slot| {
                    diagnostic_policy_first_token_logps[slot] = new_logps_owned[0];
                    diagnostic_reference_first_token_logps[slot] = ref_logps_owned[0];
                }
                try flat_new_logps.appendSlice(allocator, new_logps_owned);
                try completions.append(allocator, .{
                    .prompt_idx = prompt_idx,
                    .tokens = tokens_owned,
                    .old_logps = old_logps_owned,
                    .ref_logps = ref_logps_owned,
                });
                total_tokens += tokens_owned.len;
                group_token_count += tokens_owned.len;
            }

            if (!captured_initial_logprob_parity) {
                if (group_sampling_rescore_max_abs_error > 1e-4) {
                    print("grpo sampling-rescore parity mismatch: max_abs_error={d:.9} tolerance=0.000100000 batched_multi={} sparse_multi={}\n", .{
                        group_sampling_rescore_max_abs_error,
                        batch_multi_token_group_scoring,
                        sparse_multi_token,
                    });
                    return error.GrpoSamplingRescoreParityMismatch;
                }
                if (current_base_equivalent_policy and group_policy_reference_max_abs_error > 1e-4) return error.GrpoInitialReferenceParityMismatch;
                initial_sampling_rescore_max_abs_error = group_sampling_rescore_max_abs_error;
                initial_policy_reference_max_abs_error = group_policy_reference_max_abs_error;
                captured_initial_logprob_parity = true;
            }

            var ga = try grpo.scoreGroup(allocator, rewarder, completions.items);
            defer ga.deinit();
            grpo.computeAdvantages(&ga, completions.items, cfg);
            for (ga.advantages) |advantage| {
                if (advantage != 0.0) saw_nonzero_reward_advantage = true;
            }
            var group_reward: f64 = 0.0;
            var group_reward_squared: f64 = 0.0;
            for (ga.rewards) |reward| {
                total_reward += reward;
                total_reward_squared += @as(f64, reward) * @as(f64, reward);
                group_reward += reward;
                group_reward_squared += @as(f64, reward) * @as(f64, reward);
            }
            const group_reward_denom = @as(f64, @floatFromInt(@max(ga.rewards.len, 1)));
            const group_mean_reward = group_reward / group_reward_denom;
            const group_reward_variance = @max(group_reward_squared / group_reward_denom - group_mean_reward * group_mean_reward, 0.0);

            var loss_result = try grpo.grpoLoss(allocator, completions.items, flat_new_logps.items, ga.advantages, cfg);
            defer loss_result.deinit();
            for (loss_result.grad_new_logps) |gradient| {
                if (gradient != 0.0) saw_nonzero_policy_gradient = true;
            }

            const next_kl_coef = try kl_control.observe(
                total_groups,
                epoch_idx,
                prompt_idx,
                trainer.optimizerSteps(),
                loss_result.mean_kl,
                loss_result.kl_loss,
            );

            total_loss += loss_result.loss;
            total_pg_loss += loss_result.pg_loss;
            total_kl_loss += loss_result.kl_loss;
            total_mean_kl += loss_result.mean_kl;
            total_clip_fraction += loss_result.clip_fraction;
            total_groups += 1;
            total_completions += completions.items.len;

            const backward_started_ns = platform.time.monotonicNs();
            if (coalesce_single_token_groups) {
                var completion_token_ids = std.ArrayList(i32).empty;
                defer completion_token_ids.deinit(allocator);
                for (completions.items) |completion| {
                    if (completion.tokens.len != 1) return error.ExpectedSingleTokenCompletion;
                    try completion_token_ids.append(allocator, completion.tokens[0]);
                }
                var prepared = try buildGemmaPreparedExampleFromTokens(allocator, prompt, completions.items[0].tokens, max_seq_len);
                defer freeGemmaPreparedExample(allocator, &prepared);
                var input = try gemma4_real_autodiff.makeTrainerInputForSingleTokenCompletionGroupLogprobGrads(
                    allocator,
                    &ctx,
                    &prepared,
                    @intCast(max_seq_len),
                    completion_token_ids.items,
                    loss_result.grad_new_logps,
                );
                defer input.deinit(allocator);
                _ = try trainer.step(input.trainer_input);
            } else if (batch_multi_token_group_backward) {
                try scalePreferenceUnitGradients(loss_result.grad_new_logps, physical_micro_batches_per_group);
                var token_offset: usize = 0;
                var completion_start: usize = 0;
                while (completion_start < completions.items.len) {
                    const completion_end = @min(
                        completion_start + multi_token_backward_batch_size,
                        completions.items.len,
                    );
                    const completion_chunk = completions.items[completion_start..completion_end];
                    const prepared = try allocator.alloc(gemma4.PreparedExampleInput, completion_chunk.len);
                    var prepared_count: usize = 0;
                    defer {
                        for (prepared[0..prepared_count]) |*example| freeGemmaPreparedExample(allocator, example);
                        allocator.free(prepared);
                    }
                    const gradient_rows = try allocator.alloc([]const f32, completion_chunk.len);
                    defer allocator.free(gradient_rows);

                    for (completion_chunk, 0..) |completion, chunk_completion_idx| {
                        prepared[chunk_completion_idx] = try buildGemmaPreparedExampleFromTokens(
                            allocator,
                            prompt,
                            completion.tokens,
                            max_seq_len,
                        );
                        prepared_count += 1;
                        gradient_rows[chunk_completion_idx] = loss_result.grad_new_logps[token_offset .. token_offset + completion.tokens.len];
                        token_offset += completion.tokens.len;
                    }

                    var input = try gemma4_real_autodiff.makeTrainerInputForTokenLogprobGradBatch(
                        allocator,
                        &ctx,
                        prepared,
                        @intCast(max_seq_len),
                        gradient_rows,
                    );
                    defer input.deinit(allocator);
                    _ = try trainer.step(input.trainer_input);
                    completion_start = completion_end;
                }
                if (token_offset != loss_result.grad_new_logps.len) return error.GradientShapeMismatch;
            } else {
                try scalePreferenceUnitGradients(loss_result.grad_new_logps, group_size);
                var token_offset: usize = 0;
                for (completions.items) |completion| {
                    var prepared = try buildGemmaPreparedExampleFromTokens(allocator, prompt, completion.tokens, max_seq_len);
                    defer freeGemmaPreparedExample(allocator, &prepared);
                    const grads = loss_result.grad_new_logps[token_offset .. token_offset + completion.tokens.len];
                    var input = try gemma4_real_autodiff.makeTrainerInputForTokenLogprobGrads(
                        allocator,
                        &ctx,
                        &prepared,
                        @intCast(max_seq_len),
                        grads,
                    );
                    defer input.deinit(allocator);
                    _ = try trainer.step(input.trainer_input);
                    token_offset += completion.tokens.len;
                }
            }
            total_backward_update_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - backward_started_ns)) / std.time.ns_per_s;
            if (benchmark) |*recorder| {
                try recorder.record(.{
                    .seconds = @as(f64, @floatFromInt(platform.time.monotonicNs() - group_started_ns)) / std.time.ns_per_s,
                    .loss = loss_result.loss,
                    .pg_loss = loss_result.pg_loss,
                    .kl_loss = loss_result.kl_loss,
                    .mean_reward = @floatCast(group_mean_reward),
                    .reward_stddev = @floatCast(@sqrt(group_reward_variance)),
                    .completion_tokens = group_token_count,
                    .policy_reference_max_abs_error = group_policy_reference_max_abs_error,
                });
            }
            cfg.kl_coef = next_kl_coef;
        }
        const completed_epochs = epoch_idx + 1;
        const checkpoint_every = if (recipe.checkpoint) |checkpoint| checkpoint.every_epochs else null;
        if (checkpoint_every) |every| {
            if (completed_epochs % @as(usize, every) == 0 or completed_epochs == epochs) {
                const path = grpo_checkpoint_path orelse return error.CheckpointPathRequired;
                const checkpoint_incremental_kv = if (incremental_kv_sampler) |*sampler|
                    try sampler.checkpointTelemetry()
                else
                    null;
                try savePreferenceCheckpoint(
                    allocator,
                    io,
                    &trainer,
                    path,
                    .grpo,
                    completed_epochs,
                    total_groups,
                    &grpo_run_fingerprint,
                    .{
                        .task = @tagName(PreferenceTask.grpo),
                        .run_fingerprint_sha256 = grpo_run_fingerprint_text,
                        .epoch_index = completed_epochs,
                        .micro_batch_steps = trainer.microBatchSteps(),
                        .optimizer_steps = trainer.optimizerSteps(),
                        .accumulation_micro_batches = trainer.accumulatedMicroBatches(),
                        .grpo = .{
                            .initial_adapter_digest = initial_adapter_digest,
                            .total_loss = total_loss,
                            .total_pg_loss = total_pg_loss,
                            .total_kl_loss = total_kl_loss,
                            .total_mean_kl = total_mean_kl,
                            .total_clip_fraction = total_clip_fraction,
                            .total_groups = total_groups,
                            .total_completions = total_completions,
                            .total_tokens = total_tokens,
                            .total_reward = total_reward,
                            .total_reward_squared = total_reward_squared,
                            .saw_nonzero_reward_advantage = saw_nonzero_reward_advantage,
                            .saw_nonzero_policy_gradient = saw_nonzero_policy_gradient,
                            .initial_sampling_rescore_max_abs_error = initial_sampling_rescore_max_abs_error,
                            .initial_policy_reference_max_abs_error = initial_policy_reference_max_abs_error,
                            .initial_base_equivalent_policy = initial_base_equivalent_policy,
                            .captured_initial_logprob_parity = captured_initial_logprob_parity,
                            .policy_rescore_completions = policy_rescore_completions,
                            .diagnostic_first_tokens = diagnostic_first_tokens,
                            .diagnostic_policy_first_token_logps = diagnostic_policy_first_token_logps,
                            .diagnostic_reference_first_token_logps = diagnostic_reference_first_token_logps,
                            .diagnostic_first_token_count = diagnostic_first_token_count,
                            .kl_current_coef = kl_control.current_kl_coef,
                            .kl_admitted_groups = kl_control.admitted_groups,
                            .kl_max_observed_mean = kl_control.max_observed_mean_kl,
                            .kl_trace = kl_control.trace.items,
                            .reward_call_index = reward_pipeline.call_index,
                            .reward_external_calls = reward_pipeline.external_calls,
                            .reward_external_failures = reward_pipeline.external_failures,
                            .reward_trace = reward_pipeline.trace.items,
                            .incremental_kv = checkpoint_incremental_kv,
                        },
                    },
                );
            }
        }
    }

    const benchmark_telemetry = if (benchmark) |*recorder| try recorder.finish() else null;

    try reward_pipeline.finish();
    try kl_control.finish();

    if (!saw_nonzero_reward_advantage or !saw_nonzero_policy_gradient) {
        print("grpo zero-signal first completion token ids: {any}\n", .{diagnostic_first_tokens[0..diagnostic_first_token_count]});
    }
    try validateGrpoLearningSignal(saw_nonzero_reward_advantage, saw_nonzero_policy_gradient, total_reward, total_reward_squared, total_completions, total_loss);
    _ = try trainer.flushAccumulatedGradients();
    if (recipe.checkpoint) |checkpoint| if (checkpoint.every_epochs != null) {
        const path = grpo_checkpoint_path orelse return error.CheckpointPathRequired;
        const checkpoint_incremental_kv = if (incremental_kv_sampler) |*sampler|
            try sampler.checkpointTelemetry()
        else
            null;
        try savePreferenceCheckpoint(
            allocator,
            io,
            &trainer,
            path,
            .grpo,
            epochs,
            total_groups,
            &grpo_run_fingerprint,
            .{
                .task = @tagName(PreferenceTask.grpo),
                .run_fingerprint_sha256 = grpo_run_fingerprint_text,
                .epoch_index = epochs,
                .micro_batch_steps = trainer.microBatchSteps(),
                .optimizer_steps = trainer.optimizerSteps(),
                .accumulation_micro_batches = trainer.accumulatedMicroBatches(),
                .grpo = .{
                    .initial_adapter_digest = initial_adapter_digest,
                    .total_loss = total_loss,
                    .total_pg_loss = total_pg_loss,
                    .total_kl_loss = total_kl_loss,
                    .total_mean_kl = total_mean_kl,
                    .total_clip_fraction = total_clip_fraction,
                    .total_groups = total_groups,
                    .total_completions = total_completions,
                    .total_tokens = total_tokens,
                    .total_reward = total_reward,
                    .total_reward_squared = total_reward_squared,
                    .saw_nonzero_reward_advantage = saw_nonzero_reward_advantage,
                    .saw_nonzero_policy_gradient = saw_nonzero_policy_gradient,
                    .initial_sampling_rescore_max_abs_error = initial_sampling_rescore_max_abs_error,
                    .initial_policy_reference_max_abs_error = initial_policy_reference_max_abs_error,
                    .initial_base_equivalent_policy = initial_base_equivalent_policy,
                    .captured_initial_logprob_parity = captured_initial_logprob_parity,
                    .policy_rescore_completions = policy_rescore_completions,
                    .diagnostic_first_tokens = diagnostic_first_tokens,
                    .diagnostic_policy_first_token_logps = diagnostic_policy_first_token_logps,
                    .diagnostic_reference_first_token_logps = diagnostic_reference_first_token_logps,
                    .diagnostic_first_token_count = diagnostic_first_token_count,
                    .kl_current_coef = kl_control.current_kl_coef,
                    .kl_admitted_groups = kl_control.admitted_groups,
                    .kl_max_observed_mean = kl_control.max_observed_mean_kl,
                    .kl_trace = kl_control.trace.items,
                    .reward_call_index = reward_pipeline.call_index,
                    .reward_external_calls = reward_pipeline.external_calls,
                    .reward_external_failures = reward_pipeline.external_failures,
                    .reward_trace = reward_pipeline.trace.items,
                    .incremental_kv = checkpoint_incremental_kv,
                },
            },
        );
    };
    if (trainer.optimizerSteps() == 0) return error.NoOptimizerSteps;
    try validateTrainerAdapterChanged(&trainer, initial_adapter_digest);

    const evaluation_report_path = try preferenceEvaluationReportPath(allocator, recipe, .grpo);
    defer allocator.free(evaluation_report_path);
    const training_incremental_kv = if (incremental_kv_sampler) |*sampler| sampler.telemetry else null;
    if (incremental_kv_sampler) |*sampler| sampler.resetTelemetry();
    // As with DPO, terminal quality admission must not inherit graph, device-
    // allocation, or private-buffer history from the training trajectory.
    // Snapshot the exact final values to host and rebuild the evaluator,
    // frozen-base bindings, and optional incremental sampler on a separately
    // initialized backend with private-buffer reuse disabled.
    const evaluation = evaluation: {
        try trainer.prepareTerminalEvaluationFromHostSnapshot();

        var evaluation_backend = try gemma4_real_autodiff.loadBackendForModelDir(
            allocator,
            base_model_dir,
            backend_kind,
        );
        defer evaluation_backend.deinit();
        var evaluation_trainer = try real_autodiff.RealAutodiffTrainer.init(
            allocator,
            evaluation_backend.backendPtr(),
            .{
                .lora = lora_config,
                .optimizer = .{},
                .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
                .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
                .grad_accum_steps = grad_accum_steps,
                .seed = recipe.optimizer.seed orelse 42,
                .hidden_size_hint = graph_config.hidden_size,
                .num_layers_hint = graph_config.num_hidden_layers,
                .execution_engine = execution_policy.engine,
                .compiled_required = execution_policy.compiled_required,
                .strict_metal_execution = execution_policy.strict_metal_execution,
                .graph_cache_capacity = grpo_graph_cache_capacity,
            },
        );
        defer evaluation_trainer.deinit();

        var evaluation_metal_ordering_suspended = false;
        var evaluation_metal_row_staging_suspended = false;
        if (backend_kind == .metal) {
            try evaluation_trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame();
            evaluation_metal_ordering_suspended = try evaluation_trainer.compute_backend.decoderRuntimePushPlannedEncoderCoalescingSuppression();
            if (!evaluation_metal_ordering_suspended) return error.PlannedEncoderCoalescingSuppressionUnavailable;
            errdefer evaluation_trainer.compute_backend.decoderRuntimePopPlannedEncoderCoalescingSuppression() catch {};
            evaluation_metal_row_staging_suspended = try evaluation_trainer.compute_backend.decoderRuntimePushBf16EmbeddingRowStagingSuppression();
            if (!evaluation_metal_row_staging_suspended) return error.Bf16EmbeddingRowStagingSuppressionUnavailable;
        }
        defer if (evaluation_metal_ordering_suspended or evaluation_metal_row_staging_suspended) {
            evaluation_trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame() catch {
                if (evaluation_trainer.compute_backend.decoderRuntimeHasActiveFrame()) {
                    evaluation_trainer.compute_backend.decoderRuntimeCancelFrame() catch {};
                }
            };
            if (evaluation_metal_row_staging_suspended) {
                evaluation_trainer.compute_backend.decoderRuntimePopBf16EmbeddingRowStagingSuppression() catch {};
            }
            evaluation_trainer.compute_backend.decoderRuntimePopPlannedEncoderCoalescingSuppression() catch {};
        };

        var evaluation_ctx = gemma4_real_autodiff.GemmaAutodiffCtx.init(graph_config);
        evaluation_ctx.enable_fused_rms_norm_backward = ctx.enable_fused_rms_norm_backward;
        evaluation_ctx.enable_fused_gqa_attention_backward = ctx.enable_fused_gqa_attention_backward;
        evaluation_ctx.enable_fused_linear_cross_entropy = ctx.enable_fused_linear_cross_entropy;
        var canonical_eval_reuse_scope = try gemma4_real_autodiff.configureMetalBufferReuseForPreferenceRun(
            &evaluation_trainer,
            false,
            false,
        );
        defer canonical_eval_reuse_scope.deinit();
        try gemma4_real_autodiff.initializeTrainerFromAdapterDir(
            allocator,
            &evaluation_trainer,
            &evaluation_ctx,
            bootstrap_dir,
            &bootstrap_example,
            @intCast(max_seq_len),
        );
        try evaluation_trainer.initializeTerminalEvaluationFromHostSnapshot(&trainer);
        var evaluation_frozen_lora = try gemma4_real_autodiff.FrozenBaseLoraBindings.init(
            allocator,
            &evaluation_trainer,
        );
        defer evaluation_frozen_lora.deinit();
        var evaluation_incremental_kv_sampler: ?gemma4_real_autodiff.GrpoIncrementalKvSampler = if (incremental_kv_requested)
            try gemma4_real_autodiff.GrpoIncrementalKvSampler.init(
                allocator,
                &evaluation_trainer,
                &evaluation_ctx,
                incremental_kv_batch_active,
                incremental_kv_clone_prompt_tail,
            )
        else
            null;
        defer if (evaluation_incremental_kv_sampler) |*sampler| sampler.deinit();
        break :evaluation try evaluateGemmaGrpoHeldout(
            allocator,
            io,
            recipe,
            tokenizer_model,
            prompt_batch.prompts,
            &evaluation_trainer,
            &evaluation_ctx,
            if (evaluation_incremental_kv_sampler) |*sampler| sampler else null,
            &evaluation_frozen_lora,
            max_seq_len,
            group_size,
            max_completion_tokens,
            backend_kind,
            evaluation_report_path,
            true,
        );
    };

    const baseline_relative: ?GrpoBaselineRelativeSummary = if (baseline_evaluation) |baseline| relative: {
        const summary = compareGrpoToBaseline(baseline, evaluation, grpo_minimums);
        if (!summary.passed) return error.GrpoBaselineRelativeEvaluationGateFailed;
        break :relative summary;
    } else null;

    try gemma4_real_autodiff.saveTrainerAsGemmaBundle(allocator, &trainer, base_model_dir, bootstrap_dir, trained_dir);
    try validatePublishedAdapterChanged(allocator, io, bootstrap_dir, trained_dir);

    var grpo_checkpoint_artifact = try preferenceCheckpointArtifactSummary(
        allocator,
        grpo_checkpoint_path,
        .grpo,
        trainer.trainingProgress(),
    );
    defer if (grpo_checkpoint_artifact) |*artifact| artifact.deinit(allocator);

    const denom = @as(f64, @floatFromInt(@max(total_groups, 1)));
    const reward_denom = @as(f64, @floatFromInt(@max(total_completions, 1)));
    const mean_reward = total_reward / reward_denom;
    const reward_variance = @max(total_reward_squared / reward_denom - mean_reward * mean_reward, 0.0);
    try writeJsonFile(allocator, io, report_path, GrpoReport{
        .execution_mode = "train",
        .dataset_format = recipe.dataset.format.?,
        .completions = total_completions,
        .tokens = total_tokens,
        .groups = total_groups,
        .loss = @floatCast(total_loss / denom),
        .pg_loss = @floatCast(total_pg_loss / denom),
        .kl_loss = @floatCast(total_kl_loss / denom),
        .mean_kl = @floatCast(total_mean_kl / denom),
        .clip_fraction = @floatCast(total_clip_fraction / denom),
        .mean_reward = @floatCast(mean_reward),
        .reward_stddev = @floatCast(@sqrt(reward_variance)),
        .training_seed = recipe.optimizer.seed orelse 42,
        .policy_backend = @tagName(backend_kind),
        .optimizer_steps = trainer.optimizerSteps(),
        .micro_batch_steps = trainer.microBatchSteps(),
        .sampling_mode = if (coalesce_single_token_groups)
            "shared-prompt-ranked-sparse-row"
        else if (incremental_kv_sampler != null)
            "shared-page-prompt-ranked-incremental-kv"
        else if (sparse_multi_token)
            "shared-prompt-ranked-sparse-row-each-step"
        else
            "shared-prompt-ranked",
        .policy_logprob_mode = if (batch_single_token_group_scoring)
            "sampling-logprob-reuse-with-batched-initial-rescore"
        else if (batch_multi_token_group_scoring)
            "sampling-logprob-reuse-with-batched-completion-row-initial-rescore"
        else if (sparse_multi_token)
            "sampling-logprob-reuse-with-sparse-completion-row-initial-rescore"
        else
            "sampling-logprob-reuse-with-initial-rescore",
        .policy_rescore_completions = policy_rescore_completions,
        .training_microbatch_mode = if (coalesce_single_token_groups)
            "coalesced-single-token-sparse-weighted-row"
        else if (batch_multi_token_group_backward)
            "batched-independent-row-sparse-weighted-group"
        else
            "per-completion",
        .training_microbatch_batch_size = if (batch_multi_token_group_backward)
            multi_token_backward_batch_size
        else
            1,
        .training_physical_micro_batches_per_group = physical_micro_batches_per_group,
        .sampling_seconds = total_sampling_seconds,
        .incremental_kv = training_incremental_kv,
        .policy_rescore_seconds = total_policy_rescore_seconds,
        .backward_update_seconds = total_backward_update_seconds,
        .reference_mode = if (batch_single_token_group_scoring)
            "compiled-zero-lora-shared-prompt-candidate-row"
        else if (batch_multi_token_group_scoring)
            "compiled-zero-lora-batched-completion-rows"
        else if (sparse_multi_token)
            "compiled-zero-lora-sparse-completion-rows"
        else
            "compiled-zero-lora",
        .reference_scoring_seconds = total_reference_scoring_seconds,
        .reference_cache = reference_cache.telemetry(),
        .initial_logprob_parity = .{
            .sampling_rescore_max_abs_error = initial_sampling_rescore_max_abs_error,
            .policy_reference_max_abs_error = initial_policy_reference_max_abs_error,
            .base_equivalent_policy = initial_base_equivalent_policy,
            .completion_first_token_ids = diagnostic_first_tokens[0..diagnostic_first_token_count],
            .policy_first_token_logps = diagnostic_policy_first_token_logps[0..diagnostic_first_token_count],
            .reference_first_token_logps = diagnostic_reference_first_token_logps[0..diagnostic_first_token_count],
        },
        .kl_control = kl_control.telemetry(),
        .benchmark = benchmark_telemetry,
        .checkpoint_resume = .{
            .enabled = grpo_resume_enabled,
            .start_epoch = start_epoch,
            .checkpoint_path = grpo_checkpoint_path,
            .checkpoint_state_path = if (grpo_checkpoint_artifact) |artifact| artifact.state_path else null,
            .checkpoint_state_sha256 = if (grpo_checkpoint_artifact) |artifact| artifact.state_sha256 else null,
            .checkpoint_epoch = if (grpo_checkpoint_artifact) |artifact| artifact.epoch else null,
            .checkpoint_every_epochs = if (recipe.checkpoint) |checkpoint| checkpoint.every_epochs else null,
            .run_fingerprint_sha256 = grpo_run_fingerprint_text,
            .restored_micro_batch_steps = grpo_restored.micro_batch_steps,
            .restored_optimizer_steps = grpo_restored.optimizer_steps,
            .restored_accumulation_micro_batches = grpo_restored.accumulation_micro_batches,
        },
        .reward_pipeline = reward_pipeline.telemetry(),
        .metal_numerical_policy = metal_numerical_policy,
        .evaluation_execution_policy = canonical_preference_evaluation_policy,
        .baseline_evaluation = baseline_evaluation,
        .baseline_relative = baseline_relative,
        .evaluation = evaluation,
        .trained_adapter_dir = trained_dir,
    });
    print("grpo report: {s}\ntrained adapter: {s}\n", .{ report_path, trained_dir });
}

fn evaluateGemmaGrpoHeldout(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    tokenizer_model: *model_manager_mod.LoadedModel,
    train_prompts: []const []const i32,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_real_autodiff.GemmaAutodiffCtx,
    incremental_kv_sampler: ?*gemma4_real_autodiff.GrpoIncrementalKvSampler,
    frozen_lora: *gemma4_real_autodiff.FrozenBaseLoraBindings,
    max_seq_len: usize,
    group_size: usize,
    max_completion_tokens: usize,
    backend_kind: gemma4_real_autodiff.BackendKind,
    report_path: []const u8,
    enforce_minimums: bool,
) !GrpoEvaluationSummary {
    const eval_path = evalDatasetPath(recipe) orelse return error.MissingPreferenceEvaluationDataset;
    const minimums = recipe.eval.?.grpo_minimums orelse return error.MissingGrpoEvaluationMinimums;
    var eval_recipe = recipe;
    eval_recipe.dataset.max_examples = evalMaxExamples(recipe);
    var prompt_batch = try loadGrpoTextPrompts(allocator, io, eval_path, eval_recipe, tokenizer_model);
    defer prompt_batch.deinit();
    const dataset_fingerprint = try fingerprintPath(allocator, io, "eval_dataset", eval_path);
    defer if (dataset_fingerprint.digest) |digest| allocator.free(digest);
    const policy_adapter_digest = try formatSha256DigestAlloc(allocator, trainerLoRAParameterDigest(trainer));
    defer allocator.free(policy_adapter_digest);
    var reward_pipeline = try RewardPipeline.init(
        allocator,
        io,
        recipe,
        tokenizer_model.getTokenizer(),
        prompt_batch.prompt_texts,
        prompt_batch.targets,
        .evaluation,
    );
    defer reward_pipeline.deinit();
    errdefer reward_pipeline.finish() catch {};

    const overlap_count = try countTokenPromptOverlaps(allocator, train_prompts, prompt_batch.prompts);
    if (overlap_count != 0) {
        try reward_pipeline.finish();
        try writeJsonFile(allocator, io, report_path, GrpoEvaluationReport{
            .status = "failed-prompt-overlap",
            .dataset_path = eval_path,
            .dataset_fingerprint = dataset_fingerprint,
            .policy_adapter_digest = policy_adapter_digest,
            .policy_backend = @tagName(backend_kind),
            .metal_numerical_policy = resolveGemmaMetalNumericalPolicy(backend_kind, ctx),
            .groups = 0,
            .completions = 0,
            .tokens = 0,
            .prompt_overlap_count = overlap_count,
            .mean_reward = 0.0,
            .top_rank_mean_reward = 0.0,
            .positive_reward_group_rate = 0.0,
            .reward_stddev = 0.0,
            .loss = 0.0,
            .pg_loss = 0.0,
            .kl_loss = 0.0,
            .mean_kl = 0.0,
            .clip_fraction = 0.0,
            .minimums = minimums,
            .reference_mode = "compiled-zero-lora",
            .reward_pipeline = reward_pipeline.telemetry(),
        });
        return error.PreferenceTrainEvalPromptOverlap;
    }

    const rewarder = grpo.Rewarder{ .ctx = &reward_pipeline, .call = RewardPipeline.score };
    const cfg = grpo.GRPOConfig{
        .group_size = group_size,
        .clip_epsilon = recipe.grpo.clip_epsilon orelse 0.2,
        .kl_coef = recipe.grpo.kl_coef orelse 0.04,
        .advantage_eps = recipe.grpo.advantage_eps orelse 1e-8,
        .normalize_advantage = recipe.grpo.normalize_advantage orelse true,
    };
    const coalesce = max_completion_tokens == 1;
    const batch_single_token_group_scoring = coalesce and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_BATCH_SINGLE_TOKEN_SCORING", true);
    const batch_multi_token_group_scoring = !coalesce and gemmaGrpoMultiTokenBatchEnabled(group_size);
    const sparse_multi_token = !coalesce and
        backend_kind == .metal and
        !batch_multi_token_group_scoring and
        gemmaGrpoSparseMultiTokenEnabled();
    const top_rank_cap: usize = @min(group_size, 8);
    const eos_id = tokenizer_model.getTokenizer().specialTokens().sep_id;
    var reference_cache = GemmaGrpoReferenceCache.init(allocator, 1024);
    defer reference_cache.deinit();
    var total_loss: f64 = 0.0;
    var total_pg_loss: f64 = 0.0;
    var total_kl_loss: f64 = 0.0;
    var total_mean_kl: f64 = 0.0;
    var total_clip_fraction: f64 = 0.0;
    var total_reward: f64 = 0.0;
    var total_reward_squared: f64 = 0.0;
    var total_top_rank_reward: f64 = 0.0;
    var groups_with_positive_reward: usize = 0;
    var total_groups: usize = 0;
    var total_completions: usize = 0;
    var total_tokens: usize = 0;
    var total_sampling_seconds: f64 = 0.0;
    var total_reference_scoring_seconds: f64 = 0.0;
    var total_reward_loss_seconds: f64 = 0.0;
    const loop_started_ns = platform.time.monotonicNs();

    // Keep live-policy and frozen-reference eager executions in separate
    // phases for the one-token production lane. Besides avoiding repeated
    // model-state switches, this makes the evaluation schedule explicit and
    // deterministic: every sampled token is fixed before a zero-LoRA binding
    // is installed. The storage is only two scalars per completion.
    var sampled_single_token_ids: ?[]i32 = null;
    defer if (sampled_single_token_ids) |values| allocator.free(values);
    var sampled_single_token_logps: ?[]f32 = null;
    defer if (sampled_single_token_logps) |values| allocator.free(values);
    if (batch_single_token_group_scoring) {
        const sampled_count = std.math.mul(usize, prompt_batch.prompts.len, group_size) catch
            return error.InvalidCompletionGroup;
        const token_ids = try allocator.alloc(i32, sampled_count);
        sampled_single_token_ids = token_ids;
        const token_logps = try allocator.alloc(f32, sampled_count);
        sampled_single_token_logps = token_logps;

        for (prompt_batch.prompts, 0..) |prompt, prompt_idx| {
            const sampled_tokens = try allocator.alloc(std.ArrayList(i32), group_size);
            defer allocator.free(sampled_tokens);
            const sampled_logps = try allocator.alloc(std.ArrayList(f32), group_size);
            defer allocator.free(sampled_logps);
            for (sampled_tokens, sampled_logps) |*tokens, *logps| {
                tokens.* = .empty;
                logps.* = .empty;
            }
            defer for (sampled_tokens, sampled_logps) |*tokens, *logps| {
                tokens.deinit(allocator);
                logps.deinit(allocator);
            };

            const sampling_started_ns = platform.time.monotonicNs();
            try sampleGemmaGrpoCompletionGroup(
                allocator,
                trainer,
                ctx,
                incremental_kv_sampler,
                gemmaGrpoIncrementalKvShadowExactEnabled(recipe),
                prompt,
                @intCast(max_seq_len),
                max_completion_tokens,
                top_rank_cap,
                if (eos_id >= 0) eos_id else null,
                sparse_multi_token,
                sampled_tokens,
                sampled_logps,
            );
            total_sampling_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - sampling_started_ns)) / std.time.ns_per_s;

            const group_offset = prompt_idx * group_size;
            for (sampled_tokens, sampled_logps, 0..) |tokens, logps, completion_idx| {
                if (tokens.items.len != 1 or logps.items.len != 1) {
                    return error.ExpectedSingleTokenCompletion;
                }
                token_ids[group_offset + completion_idx] = tokens.items[0];
                token_logps[group_offset + completion_idx] = logps.items[0];
            }
        }
    }

    for (prompt_batch.prompts, 0..) |prompt, prompt_idx| {
        var sampled_tokens = try allocator.alloc(std.ArrayList(i32), group_size);
        defer allocator.free(sampled_tokens);
        var sampled_logps = try allocator.alloc(std.ArrayList(f32), group_size);
        defer allocator.free(sampled_logps);
        for (sampled_tokens, sampled_logps) |*tokens, *logps| {
            tokens.* = .empty;
            logps.* = .empty;
        }
        defer for (sampled_tokens, sampled_logps) |*tokens, *logps| {
            tokens.deinit(allocator);
            logps.deinit(allocator);
        };
        if (sampled_single_token_ids) |token_ids| {
            const token_logps = sampled_single_token_logps orelse return error.InvalidCompletionGroup;
            const group_offset = prompt_idx * group_size;
            for (sampled_tokens, sampled_logps, 0..) |*tokens, *logps, completion_idx| {
                try tokens.append(allocator, token_ids[group_offset + completion_idx]);
                try logps.append(allocator, token_logps[group_offset + completion_idx]);
            }
        } else {
            const sampling_started_ns = platform.time.monotonicNs();
            try sampleGemmaGrpoCompletionGroup(
                allocator,
                trainer,
                ctx,
                incremental_kv_sampler,
                gemmaGrpoIncrementalKvShadowExactEnabled(recipe),
                prompt,
                @intCast(max_seq_len),
                max_completion_tokens,
                top_rank_cap,
                if (eos_id >= 0) eos_id else null,
                sparse_multi_token,
                sampled_tokens,
                sampled_logps,
            );
            total_sampling_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - sampling_started_ns)) / std.time.ns_per_s;
        }

        var single_token_reference_logps: ?[]f32 = null;
        defer if (single_token_reference_logps) |values| allocator.free(values);
        if (batch_single_token_group_scoring) {
            const reference_logps = try allocator.alloc(f32, group_size);
            single_token_reference_logps = reference_logps;
            const reference_started_ns = platform.time.monotonicNs();
            try cachedSingleTokenGroupReferenceLogps(
                allocator,
                trainer,
                ctx,
                prompt,
                prompt_idx,
                sampled_tokens,
                @intCast(max_seq_len),
                frozen_lora,
                &reference_cache,
                reference_logps,
            );
            total_reference_scoring_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - reference_started_ns)) / std.time.ns_per_s;
        }
        var multi_token_reference_logps: ?CompletionGroupLogps = null;
        defer if (multi_token_reference_logps) |*values| values.deinit();
        if (batch_multi_token_group_scoring) {
            const reference_started_ns = platform.time.monotonicNs();
            multi_token_reference_logps = try batchedMultiTokenGroupReferenceLogps(
                allocator,
                trainer,
                ctx,
                prompt,
                prompt_idx,
                sampled_tokens,
                @intCast(max_seq_len),
                frozen_lora,
                &reference_cache,
            );
            total_reference_scoring_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - reference_started_ns)) / std.time.ns_per_s;
        }

        var completions: std.ArrayList(grpo.Completion) = .empty;
        defer {
            for (completions.items) |completion| {
                allocator.free(completion.tokens);
                allocator.free(completion.old_logps);
                allocator.free(completion.ref_logps);
            }
            completions.deinit(allocator);
        }
        var flat_new_logps: std.ArrayList(f32) = .empty;
        defer flat_new_logps.deinit(allocator);
        for (0..group_size) |completion_idx| {
            const tokens_owned = try sampled_tokens[completion_idx].toOwnedSlice(allocator);
            errdefer allocator.free(tokens_owned);
            const policy_logps_owned = try sampled_logps[completion_idx].toOwnedSlice(allocator);
            errdefer allocator.free(policy_logps_owned);
            const reference_logps_owned = try allocator.alloc(f32, tokens_owned.len);
            errdefer allocator.free(reference_logps_owned);
            if (single_token_reference_logps) |batched_logps| {
                if (reference_logps_owned.len != 1) return error.ExpectedSingleTokenCompletion;
                reference_logps_owned[0] = batched_logps[completion_idx];
            } else if (multi_token_reference_logps) |batched_logps| {
                @memcpy(reference_logps_owned, batched_logps.rows[completion_idx]);
            } else if (!try reference_cache.lookup(prompt_idx, tokens_owned, reference_logps_owned)) {
                const reference_started_ns = platform.time.monotonicNs();
                if (coalesce) {
                    try gemma4_real_autodiff.tokenLogprobsForPromptCompletionSparseRowsFrozenBase(
                        allocator,
                        trainer,
                        ctx,
                        prompt,
                        tokens_owned,
                        @intCast(max_seq_len),
                        reference_logps_owned,
                        frozen_lora,
                    );
                } else {
                    if (sparse_multi_token) {
                        try gemma4_real_autodiff.tokenLogprobsForPromptCompletionSparseRowsFrozenBase(
                            allocator,
                            trainer,
                            ctx,
                            prompt,
                            tokens_owned,
                            @intCast(max_seq_len),
                            reference_logps_owned,
                            frozen_lora,
                        );
                    } else {
                        try gemma4_real_autodiff.tokenLogprobsForPromptCompletionFrozenBase(
                            allocator,
                            trainer,
                            ctx,
                            prompt,
                            tokens_owned,
                            @intCast(max_seq_len),
                            reference_logps_owned,
                            frozen_lora,
                        );
                    }
                }
                total_reference_scoring_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - reference_started_ns)) / std.time.ns_per_s;
                try reference_cache.insert(prompt_idx, tokens_owned, reference_logps_owned);
            }
            for (policy_logps_owned, reference_logps_owned) |policy_logp, reference_logp| {
                if (!std.math.isFinite(policy_logp) or !std.math.isFinite(reference_logp)) {
                    return error.NonFiniteGrpoLogprob;
                }
            }
            try flat_new_logps.appendSlice(allocator, policy_logps_owned);
            try completions.append(allocator, .{
                .prompt_idx = prompt_idx,
                .tokens = tokens_owned,
                .old_logps = policy_logps_owned,
                .ref_logps = reference_logps_owned,
            });
            total_tokens += tokens_owned.len;
        }

        const reward_loss_started_ns = platform.time.monotonicNs();
        var advantages = try grpo.scoreGroup(allocator, rewarder, completions.items);
        defer advantages.deinit();
        grpo.computeAdvantages(&advantages, completions.items, cfg);
        if (advantages.rewards.len == 0) return error.EmptyCompletionGroup;
        total_top_rank_reward += advantages.rewards[0];
        var group_has_positive_reward = false;
        for (advantages.rewards) |reward| {
            total_reward += reward;
            total_reward_squared += @as(f64, reward) * @as(f64, reward);
            group_has_positive_reward = group_has_positive_reward or reward > 0.0;
        }
        if (group_has_positive_reward) groups_with_positive_reward += 1;
        var loss = try grpo.grpoLoss(allocator, completions.items, flat_new_logps.items, advantages.advantages, cfg);
        defer loss.deinit();
        total_loss += loss.loss;
        total_pg_loss += loss.pg_loss;
        total_kl_loss += loss.kl_loss;
        total_mean_kl += loss.mean_kl;
        total_clip_fraction += loss.clip_fraction;
        total_groups += 1;
        total_completions += completions.items.len;
        total_reward_loss_seconds += @as(f64, @floatFromInt(platform.time.monotonicNs() - reward_loss_started_ns)) / std.time.ns_per_s;
    }

    const loop_seconds = @as(f64, @floatFromInt(platform.time.monotonicNs() - loop_started_ns)) / std.time.ns_per_s;
    try reward_pipeline.finish();
    const group_denom = @as(f64, @floatFromInt(@max(total_groups, 1)));
    const completion_denom = @as(f64, @floatFromInt(@max(total_completions, 1)));
    const mean_reward = total_reward / completion_denom;
    const top_rank_mean_reward = total_top_rank_reward / group_denom;
    const positive_reward_group_rate = @as(f64, @floatFromInt(groups_with_positive_reward)) / group_denom;
    const reward_variance = @max(total_reward_squared / completion_denom - mean_reward * mean_reward, 0.0);
    const mean_kl = total_kl_loss / group_denom;
    const mean_unweighted_kl = total_mean_kl / group_denom;
    const passed = passesGrpoEvaluationMinimums(
        mean_reward,
        top_rank_mean_reward,
        positive_reward_group_rate,
        mean_kl,
        minimums,
    );
    try writeJsonFile(allocator, io, report_path, GrpoEvaluationReport{
        .status = if (passed) "passed" else "failed-quality-gate",
        .dataset_path = eval_path,
        .dataset_fingerprint = dataset_fingerprint,
        .policy_adapter_digest = policy_adapter_digest,
        .policy_backend = @tagName(backend_kind),
        .metal_numerical_policy = resolveGemmaMetalNumericalPolicy(backend_kind, ctx),
        .groups = total_groups,
        .completions = total_completions,
        .tokens = total_tokens,
        .prompt_overlap_count = 0,
        .mean_reward = @floatCast(mean_reward),
        .top_rank_mean_reward = @floatCast(top_rank_mean_reward),
        .positive_reward_group_rate = @floatCast(positive_reward_group_rate),
        .reward_stddev = @floatCast(@sqrt(reward_variance)),
        .loss = @floatCast(total_loss / group_denom),
        .pg_loss = @floatCast(total_pg_loss / group_denom),
        .kl_loss = @floatCast(mean_kl),
        .mean_kl = @floatCast(mean_unweighted_kl),
        .clip_fraction = @floatCast(total_clip_fraction / group_denom),
        .minimums = minimums,
        .reference_mode = if (batch_single_token_group_scoring)
            "compiled-zero-lora-shared-prompt-candidate-row"
        else if (batch_multi_token_group_scoring)
            "compiled-zero-lora-batched-completion-rows"
        else if (sparse_multi_token)
            "compiled-zero-lora-sparse-completion-rows"
        else
            "compiled-zero-lora",
        .execution_order = if (batch_single_token_group_scoring)
            "policy-sampling-pass-then-frozen-reference-pass"
        else if (batch_multi_token_group_scoring)
            "interleaved-policy-batched-reference"
        else if (sparse_multi_token)
            "interleaved-policy-sparse-reference"
        else
            "interleaved-policy-reference",
        .sampling_seconds = total_sampling_seconds,
        .incremental_kv = if (incremental_kv_sampler) |sampler| sampler.telemetry else null,
        .reference_scoring_seconds = total_reference_scoring_seconds,
        .reward_loss_seconds = total_reward_loss_seconds,
        .loop_seconds = loop_seconds,
        .reward_pipeline = reward_pipeline.telemetry(),
    });
    if (!passed and enforce_minimums) return error.GrpoEvaluationGateFailed;
    return .{
        .report_path = report_path,
        .groups = total_groups,
        .completions = total_completions,
        .mean_reward = @floatCast(mean_reward),
        .top_rank_mean_reward = @floatCast(top_rank_mean_reward),
        .positive_reward_group_rate = @floatCast(positive_reward_group_rate),
        .reward_stddev = @floatCast(@sqrt(reward_variance)),
        .kl_loss = @floatCast(mean_kl),
        .mean_kl = @floatCast(mean_unweighted_kl),
        .sampling_seconds = total_sampling_seconds,
        .reference_scoring_seconds = total_reference_scoring_seconds,
        .reward_loss_seconds = total_reward_loss_seconds,
        .loop_seconds = loop_seconds,
        .passed = passed,
    };
}

fn passesGrpoEvaluationMinimums(
    mean_reward: f64,
    top_rank_mean_reward: f64,
    positive_reward_group_rate: f64,
    mean_kl: f64,
    minimums: GrpoEvalMinimums,
) bool {
    return mean_reward >= minimums.mean_reward and
        top_rank_mean_reward >= minimums.top_rank_mean_reward and
        positive_reward_group_rate >= minimums.positive_reward_group_rate and
        mean_kl <= minimums.max_kl_loss;
}

fn countTokenPromptOverlaps(
    allocator: std.mem.Allocator,
    train_prompts: []const []const i32,
    eval_prompts: []const []const i32,
) !usize {
    var train_prompt_set = std.StringHashMap(void).init(allocator);
    defer train_prompt_set.deinit();
    for (train_prompts) |prompt| {
        try train_prompt_set.put(std.mem.sliceAsBytes(prompt), {});
    }

    var count: usize = 0;
    for (eval_prompts) |eval_prompt| {
        count += @intFromBool(train_prompt_set.contains(std.mem.sliceAsBytes(eval_prompt)));
    }
    return count;
}

fn runOptimizerBackedQwen2Grpo(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    dataset_path: []const u8,
    report_path: []const u8,
) !void {
    const base_model_dir = recipe.model.path orelse return error.MissingModelPath;
    const adapter = recipe.adapter orelse AdapterConfig{};
    const bootstrap_dir_config = adapter.path;
    const bootstrap_dir = bootstrap_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    defer if (bootstrap_dir_config == null) allocator.free(bootstrap_dir);
    const trained_dir_config = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir;
    const trained_dir = trained_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    defer if (trained_dir_config == null) allocator.free(trained_dir);
    const reference_path = recipe.model.reference_path orelse base_model_dir;
    const backend_kind: qwen2_real_autodiff.BackendKind = .native;
    const max_seq_len = recipe.dataset.max_seq_len orelse 128;
    const group_size = recipe.grpo.group_size orelse 2;
    const max_completion_tokens = recipe.grpo.max_completion_tokens orelse 4;
    const reward_mode = try parseTextRewardMode(recipe.grpo.reward_mode orelse "exact-match");
    const family = recipe.model.family orelse try inferFamily(recipe);
    const default_target_modules = qwenLoraTargetModulesForFamily(family);
    try validateNonGemmaAdapterOptions(adapter);
    const bootstrap_target_modules = try adapterTargetModulesForQwen(adapter, default_target_modules);

    compat.cwd().access(compat.io(), bootstrap_dir, .{}) catch {
        var bootstrap = try colqwen2.bootstrapLoRABundle(allocator, base_model_dir, bootstrap_dir, .{
            .rank = adapterRank(adapter, .grpo),
            .alpha = adapterAlpha(adapter),
            .base_model_name_or_path = adapter.base_model_name_or_path,
            .target_modules = bootstrap_target_modules,
        });
        defer colqwen2.freeBootstrapSummary(allocator, &bootstrap);
    };

    var session_manager = backends.SessionManager.init(allocator);
    native_backend_choice.configureSessionPreference(&session_manager, try parseRecipeBackendChoice(recipe.backend));
    var model_manager = model_manager_mod.ModelManager.init(allocator, session_manager);
    defer model_manager.deinit();

    const tokenizer_model = try model_manager.loadFromDir(base_model_dir);
    const reference_model = if (std.mem.eql(u8, reference_path, base_model_dir))
        tokenizer_model
    else
        try model_manager.loadFromDir(reference_path);

    var prompt_batch = try loadGrpoTextPrompts(allocator, io, dataset_path, recipe, tokenizer_model);
    defer prompt_batch.deinit();

    const graph_config = try qwen2_real_autodiff.loadGraphConfig(allocator, base_model_dir);
    var backend = try qwen2_real_autodiff.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();

    var adapter_inspect = try colqwen2.inspectCheckpoint(allocator, bootstrap_dir);
    defer colqwen2.freeInspectionSummary(allocator, &adapter_inspect);
    const lora_rank = adapter_inspect.lora_rank orelse return error.MissingAdapterConfig;
    const lora_alpha = @as(f32, @floatCast(adapter_inspect.lora_alpha orelse return error.MissingAdapterConfig));
    const target_modules = adapter_inspect.target_modules orelse bootstrap_target_modules;
    const lora_config = ml.graph.lora.LoRAConfig{
        .rank = @intCast(lora_rank),
        .alpha = lora_alpha,
        .target_patterns = target_modules,
    };

    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
        .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
        .grad_accum_steps = recipe.optimizer.gradient_accumulation_steps orelse 1,
        .hidden_size_hint = graph_config.arch.hidden_size,
        .num_layers_hint = graph_config.arch.num_hidden_layers,
    });
    defer trainer.deinit();

    var ctx = qwen2_real_autodiff.Qwen2AutodiffCtx.init(graph_config);
    const bootstrap_prompt = prompt_batch.prompts[0];
    if (bootstrap_prompt.len == 0 or bootstrap_prompt.len >= max_seq_len) return error.NoCompletionBudget;
    const bootstrap_completion = [_]i32{bootstrap_prompt[bootstrap_prompt.len - 1]};
    const bootstrap_example = try buildGemmaPreparedExampleFromTokens(allocator, bootstrap_prompt, &bootstrap_completion, max_seq_len);
    defer freeGemmaPreparedExample(allocator, &bootstrap_example);
    try qwen2_real_autodiff.initializeTrainerFromAdapterDir(allocator, &trainer, &ctx, bootstrap_dir, &bootstrap_example, @intCast(max_seq_len));

    var ref_scorer = DecoderLogprobScorer{
        .allocator = allocator,
        .model = reference_model,
        .max_seq_len = max_seq_len,
    };
    var rewarder_ctx = TextRewardCtx{
        .allocator = allocator,
        .tokenizer = tokenizer_model.getTokenizer(),
        .targets = prompt_batch.targets,
        .mode = reward_mode,
    };

    const rewarder = grpo.Rewarder{
        .ctx = &rewarder_ctx,
        .call = TextRewardCtx.score,
    };
    const cfg = grpo.GRPOConfig{
        .group_size = group_size,
        .clip_epsilon = recipe.grpo.clip_epsilon orelse 0.2,
        .kl_coef = recipe.grpo.kl_coef orelse 0.04,
        .advantage_eps = recipe.grpo.advantage_eps orelse 1e-8,
        .normalize_advantage = recipe.grpo.normalize_advantage orelse true,
    };

    var total_loss: f64 = 0.0;
    var total_pg_loss: f64 = 0.0;
    var total_kl_loss: f64 = 0.0;
    var total_clip_fraction: f64 = 0.0;
    var total_groups: usize = 0;
    var total_completions: usize = 0;
    var total_tokens: usize = 0;

    const top_rank_cap: usize = @min(group_size, 8);
    const eos_id = tokenizer_model.getTokenizer().specialTokens().sep_id;
    const epochs = recipe.optimizer.epochs orelse 1;
    var epoch_idx: usize = 0;
    while (epoch_idx < epochs) : (epoch_idx += 1) {
        for (prompt_batch.prompts, 0..) |prompt, prompt_idx| {
            var completions = std.ArrayList(grpo.Completion).empty;
            defer {
                for (completions.items) |completion| {
                    allocator.free(completion.tokens);
                    allocator.free(completion.old_logps);
                    allocator.free(completion.ref_logps);
                }
                completions.deinit(allocator);
            }
            var flat_new_logps = std.ArrayList(f32).empty;
            defer flat_new_logps.deinit(allocator);

            var completion_idx: usize = 0;
            while (completion_idx < group_size) : (completion_idx += 1) {
                var sampled_tokens = std.ArrayList(i32).empty;
                defer sampled_tokens.deinit(allocator);
                var sampled_old_logps = std.ArrayList(f32).empty;
                defer sampled_old_logps.deinit(allocator);
                try qwen2_real_autodiff.sampleCompletionRanked(
                    allocator,
                    &trainer,
                    &ctx,
                    prompt,
                    @intCast(max_seq_len),
                    max_completion_tokens,
                    completion_idx % top_rank_cap,
                    if (eos_id >= 0) eos_id else null,
                    &sampled_tokens,
                    &sampled_old_logps,
                );
                const tokens_owned = try sampled_tokens.toOwnedSlice(allocator);
                errdefer allocator.free(tokens_owned);
                const old_logps_owned = try sampled_old_logps.toOwnedSlice(allocator);
                errdefer allocator.free(old_logps_owned);
                const new_logps_owned = try allocator.alloc(f32, tokens_owned.len);
                defer allocator.free(new_logps_owned);
                try qwen2_real_autodiff.tokenLogprobsForPromptCompletion(
                    allocator,
                    &trainer,
                    &ctx,
                    prompt,
                    tokens_owned,
                    @intCast(max_seq_len),
                    new_logps_owned,
                );
                const ref_logps_owned = try allocator.alloc(f32, tokens_owned.len);
                errdefer allocator.free(ref_logps_owned);
                try DecoderLogprobScorer.tokenLogprobs(@ptrCast(&ref_scorer), prompt, tokens_owned, ref_logps_owned);
                try flat_new_logps.appendSlice(allocator, new_logps_owned);
                try completions.append(allocator, .{
                    .prompt_idx = prompt_idx,
                    .tokens = tokens_owned,
                    .old_logps = old_logps_owned,
                    .ref_logps = ref_logps_owned,
                });
                total_tokens += tokens_owned.len;
            }

            var ga = try grpo.scoreGroup(allocator, rewarder, completions.items);
            defer ga.deinit();
            grpo.computeAdvantages(&ga, completions.items, cfg);

            var loss_result = try grpo.grpoLoss(allocator, completions.items, flat_new_logps.items, ga.advantages, cfg);
            defer loss_result.deinit();
            try scalePreferenceUnitGradients(loss_result.grad_new_logps, group_size);

            total_loss += loss_result.loss;
            total_pg_loss += loss_result.pg_loss;
            total_kl_loss += loss_result.kl_loss;
            total_clip_fraction += loss_result.clip_fraction;
            total_groups += 1;
            total_completions += completions.items.len;

            var token_offset: usize = 0;
            for (completions.items) |completion| {
                var prepared = try buildGemmaPreparedExampleFromTokens(allocator, prompt, completion.tokens, max_seq_len);
                defer freeGemmaPreparedExample(allocator, &prepared);
                const grads = loss_result.grad_new_logps[token_offset .. token_offset + completion.tokens.len];
                var input = try qwen2_real_autodiff.makeTrainerInputForTokenLogprobGrads(allocator, &ctx, &prepared, @intCast(max_seq_len), grads);
                defer input.deinit(allocator);
                _ = try trainer.step(input.trainer_input);
                token_offset += completion.tokens.len;
            }
        }
    }

    try qwen2_real_autodiff.saveTrainerAsQwenAdapterDir(allocator, &trainer, base_model_dir, bootstrap_dir, trained_dir);

    const denom = @as(f64, @floatFromInt(@max(total_groups, 1)));
    try writeJsonFile(allocator, io, report_path, GrpoReport{
        .execution_mode = "train",
        .dataset_format = recipe.dataset.format.?,
        .completions = total_completions,
        .tokens = total_tokens,
        .groups = total_groups,
        .loss = @floatCast(total_loss / denom),
        .pg_loss = @floatCast(total_pg_loss / denom),
        .kl_loss = @floatCast(total_kl_loss / denom),
        .clip_fraction = @floatCast(total_clip_fraction / denom),
        .trained_adapter_dir = trained_dir,
    });
    print("grpo report: {s}\ntrained adapter: {s}\n", .{ report_path, trained_dir });
}

fn runOptimizerBackedGemmaMultimodalGrpo(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    dataset_path: []const u8,
    report_path: []const u8,
    base_model_dir: []const u8,
    bootstrap_dir: []const u8,
    trained_dir: []const u8,
    reference_path: []const u8,
    backend_kind: gemma4_real_autodiff.BackendKind,
    max_seq_len: usize,
    group_size: usize,
    max_completion_tokens: usize,
    reward_mode: TextRewardMode,
) !void {
    const projector_path = recipe.model.projector_path orelse return error.MissingGgufProjector;
    if (!std.mem.eql(u8, reference_path, base_model_dir)) return error.UnsupportedReferencePath;
    const execution_policy = train_eval_gemma4_lora_bundle.autodiffExecutionPolicy(backend_kind);
    const grad_accum_steps = try preferenceGradAccumSteps(recipe.optimizer.gradient_accumulation_steps orelse 1, group_size);

    var session_manager = backends.SessionManager.init(allocator);
    native_backend_choice.configureSessionPreference(&session_manager, switch (backend_kind) {
        .native => .native,
        .metal => .metal,
    });
    var model_manager = model_manager_mod.ModelManager.init(allocator, session_manager);
    defer model_manager.deinit();
    const tokenizer_model = try model_manager.loadFromDir(base_model_dir);

    var prompt_batch = try loadGemmaGrpoPreparedPrompts(allocator, io, dataset_path, recipe, base_model_dir, projector_path);
    defer prompt_batch.deinit();

    const graph_config = try gemma4_real_autodiff.loadGraphConfig(allocator, base_model_dir);
    var backend = try gemma4_real_autodiff.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();

    var adapter_inspect = try gemma4.inspectCheckpoint(allocator, bootstrap_dir);
    defer gemma4.freeInspectionSummary(allocator, &adapter_inspect);
    const lora_rank = adapter_inspect.lora_rank orelse return error.MissingAdapterConfig;
    const lora_alpha = @as(f32, @floatCast(adapter_inspect.lora_alpha orelse return error.MissingAdapterConfig));
    const target_modules = adapter_inspect.target_modules orelse gemma4.default_lora_target_modules[0..];
    const lora_config = ml.graph.lora.LoRAConfig{
        .rank = @intCast(lora_rank),
        .alpha = lora_alpha,
        .target_patterns = target_modules,
        .strict_target_patterns = true,
        .sharing = if (adapter_inspect.recursive_lora_enabled) .by_use else .by_weight,
    };

    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
        .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
        .grad_accum_steps = grad_accum_steps,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
    });
    defer trainer.deinit();
    const tokenizer = try gemma4_mm_real_autodiff.loadTokenizerForModelDir(allocator, base_model_dir);
    var ctx = gemma4_mm_real_autodiff.MultimodalCtx.init(allocator, backend.backendPtr(), graph_config, projector_path, prompt_batch.summaries[0].gguf_projector_sha256.?, tokenizer);
    defer ctx.deinit();
    const bootstrap_prompt = prompt_batch.prompts[0];
    if (bootstrap_prompt.prompt_input_ids.len == 0 or bootstrap_prompt.prompt_input_ids.len >= max_seq_len) return error.NoCompletionBudget;
    const bootstrap_completion = [_]i32{bootstrap_prompt.prompt_input_ids[bootstrap_prompt.prompt_input_ids.len - 1]};
    const bootstrap_example = try buildGemmaPreparedExampleFromPromptExample(allocator, bootstrap_prompt, &bootstrap_completion, max_seq_len);
    defer freeGemmaPreparedExample(allocator, &bootstrap_example);
    try gemma4_mm_real_autodiff.initializeTrainerFromAdapterDir(allocator, &trainer, &ctx, bootstrap_dir, &bootstrap_example, @intCast(max_seq_len));
    const initial_adapter_digest = trainerLoRAParameterDigest(&trainer);

    var ref_trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
        .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
        .grad_accum_steps = 1,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
    });
    defer ref_trainer.deinit();
    const ref_tokenizer = try gemma4_mm_real_autodiff.loadTokenizerForModelDir(allocator, base_model_dir);
    var ref_ctx = gemma4_mm_real_autodiff.MultimodalCtx.init(allocator, backend.backendPtr(), graph_config, projector_path, prompt_batch.summaries[0].gguf_projector_sha256.?, ref_tokenizer);
    defer ref_ctx.deinit();
    try gemma4_mm_real_autodiff.initializeTrainerFromAdapterDir(allocator, &ref_trainer, &ref_ctx, bootstrap_dir, &bootstrap_example, @intCast(max_seq_len));

    var rewarder_ctx = TextRewardCtx{
        .allocator = allocator,
        .tokenizer = tokenizer_model.getTokenizer(),
        .targets = prompt_batch.targets,
        .mode = reward_mode,
    };
    const rewarder = grpo.Rewarder{
        .ctx = &rewarder_ctx,
        .call = TextRewardCtx.score,
    };
    var cfg = grpo.GRPOConfig{
        .group_size = group_size,
        .clip_epsilon = recipe.grpo.clip_epsilon orelse 0.2,
        .kl_coef = recipe.grpo.kl_coef orelse 0.04,
        .advantage_eps = recipe.grpo.advantage_eps orelse 1e-8,
        .normalize_advantage = recipe.grpo.normalize_advantage orelse true,
    };
    var kl_control = try GrpoKlControl.init(allocator, io, recipe);
    defer kl_control.deinit();
    errdefer kl_control.finish() catch {};

    var total_loss: f64 = 0.0;
    var total_pg_loss: f64 = 0.0;
    var total_kl_loss: f64 = 0.0;
    var total_mean_kl: f64 = 0.0;
    var total_clip_fraction: f64 = 0.0;
    var total_groups: usize = 0;
    var total_completions: usize = 0;
    var total_tokens: usize = 0;
    var total_reward: f64 = 0.0;
    var total_reward_squared: f64 = 0.0;
    var saw_nonzero_reward_advantage = false;
    var saw_nonzero_policy_gradient = false;

    const top_rank_cap: usize = @min(group_size, 8);
    const eos_id = tokenizer_model.getTokenizer().specialTokens().sep_id;
    const epochs = recipe.optimizer.epochs orelse 1;
    var epoch_idx: usize = 0;
    while (epoch_idx < epochs) : (epoch_idx += 1) {
        for (prompt_batch.prompts, 0..) |prompt, prompt_idx| {
            var completions = std.ArrayList(grpo.Completion).empty;
            defer {
                for (completions.items) |completion| {
                    allocator.free(completion.tokens);
                    allocator.free(completion.old_logps);
                    allocator.free(completion.ref_logps);
                }
                completions.deinit(allocator);
            }
            var flat_new_logps = std.ArrayList(f32).empty;
            defer flat_new_logps.deinit(allocator);

            var completion_idx: usize = 0;
            while (completion_idx < group_size) : (completion_idx += 1) {
                var sampled_tokens = std.ArrayList(i32).empty;
                defer sampled_tokens.deinit(allocator);
                var sampled_old_logps = std.ArrayList(f32).empty;
                defer sampled_old_logps.deinit(allocator);
                try sampleGemmaMultimodalCompletionRanked(
                    allocator,
                    &trainer,
                    &ctx,
                    prompt,
                    max_seq_len,
                    max_completion_tokens,
                    completion_idx % top_rank_cap,
                    if (eos_id >= 0) eos_id else null,
                    &sampled_tokens,
                    &sampled_old_logps,
                );
                const tokens_owned = try sampled_tokens.toOwnedSlice(allocator);
                errdefer allocator.free(tokens_owned);
                const old_logps_owned = try sampled_old_logps.toOwnedSlice(allocator);
                errdefer allocator.free(old_logps_owned);
                const new_logps_owned = try allocator.alloc(f32, tokens_owned.len);
                defer allocator.free(new_logps_owned);
                try scoreGemmaMultimodalCompletionLogprobs(allocator, &trainer, &ctx, prompt, tokens_owned, max_seq_len, new_logps_owned);
                const ref_logps_owned = try allocator.alloc(f32, tokens_owned.len);
                errdefer allocator.free(ref_logps_owned);
                try scoreGemmaMultimodalCompletionLogprobs(allocator, &ref_trainer, &ref_ctx, prompt, tokens_owned, max_seq_len, ref_logps_owned);
                try flat_new_logps.appendSlice(allocator, new_logps_owned);
                try completions.append(allocator, .{
                    .prompt_idx = prompt_idx,
                    .tokens = tokens_owned,
                    .old_logps = old_logps_owned,
                    .ref_logps = ref_logps_owned,
                });
                total_tokens += tokens_owned.len;
            }

            var ga = try grpo.scoreGroup(allocator, rewarder, completions.items);
            defer ga.deinit();
            grpo.computeAdvantages(&ga, completions.items, cfg);
            for (ga.advantages) |advantage| {
                if (advantage != 0.0) saw_nonzero_reward_advantage = true;
            }
            for (ga.rewards) |reward| {
                total_reward += reward;
                total_reward_squared += @as(f64, reward) * @as(f64, reward);
            }

            var loss_result = try grpo.grpoLoss(allocator, completions.items, flat_new_logps.items, ga.advantages, cfg);
            defer loss_result.deinit();
            for (loss_result.grad_new_logps) |gradient| {
                if (gradient != 0.0) saw_nonzero_policy_gradient = true;
            }
            const next_kl_coef = try kl_control.observe(
                total_groups,
                epoch_idx,
                prompt_idx,
                trainer.optimizerSteps(),
                loss_result.mean_kl,
                loss_result.kl_loss,
            );
            try scalePreferenceUnitGradients(loss_result.grad_new_logps, group_size);

            total_loss += loss_result.loss;
            total_pg_loss += loss_result.pg_loss;
            total_kl_loss += loss_result.kl_loss;
            total_mean_kl += loss_result.mean_kl;
            total_clip_fraction += loss_result.clip_fraction;
            total_groups += 1;
            total_completions += completions.items.len;

            var token_offset: usize = 0;
            for (completions.items) |completion| {
                var prepared = try buildGemmaPreparedExampleFromPromptExample(allocator, prompt, completion.tokens, max_seq_len);
                defer freeGemmaPreparedExample(allocator, &prepared);
                const grads = loss_result.grad_new_logps[token_offset .. token_offset + completion.tokens.len];
                var input = try gemma4_mm_real_autodiff.makeTrainerInputForTokenLogprobGrads(
                    allocator,
                    &ctx,
                    &prepared,
                    @intCast(max_seq_len),
                    grads,
                );
                defer input.deinit(allocator);
                _ = try trainer.step(input.trainer_input);
                token_offset += completion.tokens.len;
            }
            cfg.kl_coef = next_kl_coef;
        }
    }

    try kl_control.finish();
    try validateGrpoLearningSignal(saw_nonzero_reward_advantage, saw_nonzero_policy_gradient, total_reward, total_reward_squared, total_completions, total_loss);
    _ = try trainer.flushAccumulatedGradients();
    if (trainer.optimizerSteps() == 0) return error.NoOptimizerSteps;
    try validateTrainerAdapterChanged(&trainer, initial_adapter_digest);

    try gemma4_real_autodiff.saveTrainerAsGemmaBundle(allocator, &trainer, base_model_dir, bootstrap_dir, trained_dir);
    try validatePublishedAdapterChanged(allocator, io, bootstrap_dir, trained_dir);

    const denom = @as(f64, @floatFromInt(@max(total_groups, 1)));
    const reward_denom = @as(f64, @floatFromInt(@max(total_completions, 1)));
    const mean_reward = total_reward / reward_denom;
    const reward_variance = @max(total_reward_squared / reward_denom - mean_reward * mean_reward, 0.0);
    try writeJsonFile(allocator, io, report_path, GrpoReport{
        .execution_mode = "train",
        .dataset_format = recipe.dataset.format.?,
        .completions = total_completions,
        .tokens = total_tokens,
        .groups = total_groups,
        .loss = @floatCast(total_loss / denom),
        .pg_loss = @floatCast(total_pg_loss / denom),
        .kl_loss = @floatCast(total_kl_loss / denom),
        .mean_kl = @floatCast(total_mean_kl / denom),
        .clip_fraction = @floatCast(total_clip_fraction / denom),
        .mean_reward = @floatCast(mean_reward),
        .reward_stddev = @floatCast(@sqrt(reward_variance)),
        .policy_backend = @tagName(backend_kind),
        .optimizer_steps = trainer.optimizerSteps(),
        .micro_batch_steps = trainer.microBatchSteps(),
        .kl_control = kl_control.telemetry(),
        .trained_adapter_dir = trained_dir,
    });
    print("grpo report: {s}\ntrained adapter: {s}\n", .{ report_path, trained_dir });
}

fn sampleGemmaMultimodalCompletionRanked(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_mm_real_autodiff.MultimodalCtx,
    prompt: *const gemma4.PreparedExampleInput,
    max_seq_len: usize,
    max_completion_tokens: usize,
    rank: usize,
    eos_token_id: ?i32,
    out_tokens: *std.ArrayList(i32),
    out_logps: *std.ArrayList(f32),
) !void {
    var seq = std.ArrayList(i32).empty;
    defer seq.deinit(allocator);

    var step: usize = 0;
    while (step < max_completion_tokens and prompt.prompt_input_ids.len + seq.items.len < max_seq_len) : (step += 1) {
        var prepared = try buildGemmaPreparedExampleFromPromptExample(allocator, prompt, seq.items, max_seq_len);
        defer freeGemmaPreparedExample(allocator, &prepared);
        const logits = try gemma4_mm_real_autodiff.logitsForExample(allocator, trainer, ctx, &prepared, @intCast(max_seq_len));
        defer allocator.free(logits);
        const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
        const row = logits[(prompt.prompt_input_ids.len + seq.items.len - 1) * vocab_size ..][0..vocab_size];
        const token_id = try selectRankedTokenFromLogits(allocator, row, rank);
        try out_tokens.append(allocator, token_id);
        try out_logps.append(allocator, logProbAtToken(row, token_id));
        try seq.append(allocator, token_id);
        if (eos_token_id) |eos_id| if (token_id == eos_id) break;
    }
    if (out_tokens.items.len == 0) return error.EmptyCompletion;
}

fn scoreGemmaMultimodalCompletionLogprobs(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_mm_real_autodiff.MultimodalCtx,
    prompt: *const gemma4.PreparedExampleInput,
    completion: []const i32,
    max_seq_len: usize,
    out_logps: []f32,
) !void {
    if (completion.len != out_logps.len) return error.LogpLenMismatch;
    var prepared = try buildGemmaPreparedExampleFromPromptExample(allocator, prompt, completion, max_seq_len);
    defer freeGemmaPreparedExample(allocator, &prepared);
    const logits = try gemma4_mm_real_autodiff.logitsForExample(allocator, trainer, ctx, &prepared, @intCast(max_seq_len));
    defer allocator.free(logits);
    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    for (completion, 0..) |token_id, idx| {
        const row = logits[(prompt.prompt_input_ids.len + idx - 1) * vocab_size ..][0..vocab_size];
        out_logps[idx] = logProbAtToken(row, token_id);
    }
}

fn buildGemmaPreparedExampleFromTokens(
    allocator: std.mem.Allocator,
    prompt_tokens: []const i32,
    completion_tokens: []const i32,
    max_seq_len: usize,
) !gemma4.PreparedExampleInput {
    if (prompt_tokens.len == 0) return error.EmptyPrompt;
    if (prompt_tokens.len + completion_tokens.len > max_seq_len) return error.SequenceTooLong;

    const prompt_copy = try allocator.dupe(i32, prompt_tokens);
    errdefer allocator.free(prompt_copy);
    const completion_copy = try allocator.dupe(i32, completion_tokens);
    errdefer allocator.free(completion_copy);
    const input_ids = try allocator.alloc(i32, prompt_tokens.len + completion_tokens.len);
    errdefer allocator.free(input_ids);
    @memcpy(input_ids[0..prompt_tokens.len], prompt_tokens);
    @memcpy(input_ids[prompt_tokens.len..], completion_tokens);
    const labels = try allocator.alloc(i32, input_ids.len);
    errdefer allocator.free(labels);
    for (0..prompt_tokens.len) |idx| labels[idx] = -100;
    for (completion_tokens, 0..) |token_id, idx| labels[prompt_tokens.len + idx] = token_id;

    return .{
        .mode = .instruction,
        .prompt_input_ids = prompt_copy,
        .response_input_ids = completion_copy,
        .num_prompt_tokens = prompt_copy.len,
        .num_response_tokens = completion_copy.len,
        .input_ids = input_ids,
        .labels = labels,
        .num_input_tokens = input_ids.len,
        .num_supervised_tokens = completion_copy.len,
    };
}

fn buildGemmaPreparedExampleFromPromptExample(
    allocator: std.mem.Allocator,
    prompt_example: *const gemma4.PreparedExampleInput,
    completion_tokens: []const i32,
    max_seq_len: usize,
) !gemma4.PreparedExampleInput {
    if (prompt_example.prompt_input_ids.len == 0) return error.EmptyPrompt;
    if (prompt_example.prompt_input_ids.len + completion_tokens.len > max_seq_len) return error.SequenceTooLong;

    const prompt_copy = try allocator.dupe(i32, prompt_example.prompt_input_ids);
    errdefer allocator.free(prompt_copy);
    const completion_copy = try allocator.dupe(i32, completion_tokens);
    errdefer allocator.free(completion_copy);
    const input_ids = try allocator.alloc(i32, prompt_copy.len + completion_copy.len);
    errdefer allocator.free(input_ids);
    @memcpy(input_ids[0..prompt_copy.len], prompt_copy);
    @memcpy(input_ids[prompt_copy.len..], completion_copy);
    const labels = try allocator.alloc(i32, input_ids.len);
    errdefer allocator.free(labels);
    for (0..prompt_copy.len) |idx| labels[idx] = -100;
    for (completion_copy, 0..) |token_id, idx| labels[prompt_copy.len + idx] = token_id;

    return .{
        .mode = .instruction,
        .prompt_input_ids = prompt_copy,
        .response_input_ids = completion_copy,
        .num_prompt_tokens = prompt_copy.len,
        .num_response_tokens = completion_copy.len,
        .input_ids = input_ids,
        .labels = labels,
        .num_input_tokens = input_ids.len,
        .num_supervised_tokens = completion_copy.len,
        .image_paths = try dupeStringSlice(allocator, prompt_example.image_paths),
        .audio_paths = try dupeStringSlice(allocator, prompt_example.audio_paths),
        .image_token_counts = try allocator.dupe(usize, prompt_example.image_token_counts),
        .audio_token_counts = try allocator.dupe(usize, prompt_example.audio_token_counts),
    };
}

fn freeGemmaPreparedExample(allocator: std.mem.Allocator, example: *const gemma4.PreparedExampleInput) void {
    allocator.free(example.prompt_input_ids);
    allocator.free(example.response_input_ids);
    allocator.free(example.input_ids);
    allocator.free(example.labels);
    for (example.image_paths) |path| allocator.free(path);
    if (example.image_paths.len > 0) allocator.free(example.image_paths);
    for (example.audio_paths) |path| allocator.free(path);
    if (example.audio_paths.len > 0) allocator.free(example.audio_paths);
    if (example.image_token_counts.len > 0) allocator.free(example.image_token_counts);
    if (example.audio_token_counts.len > 0) allocator.free(example.audio_token_counts);
}

fn dupeStringSlice(allocator: std.mem.Allocator, values: []const []const u8) ![]const []const u8 {
    if (values.len == 0) return &.{};
    const out = try allocator.alloc([]const u8, values.len);
    var copied: usize = 0;
    errdefer {
        for (out[0..copied]) |value| allocator.free(value);
        allocator.free(out);
    }
    for (values, 0..) |value, idx| {
        out[idx] = try allocator.dupe(u8, value);
        copied += 1;
    }
    return out;
}

fn loadDpoScalarJsonl(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !DpoBatchOwned {
    const raw = try readFileMax(allocator, io, path, 256 * 1024 * 1024);
    defer allocator.free(raw);
    var pc: std.ArrayList(f32) = .empty;
    var pr: std.ArrayList(f32) = .empty;
    var rc: std.ArrayList(f32) = .empty;
    var rr: std.ArrayList(f32) = .empty;
    var cl: std.ArrayList(u32) = .empty;
    var rl: std.ArrayList(u32) = .empty;
    var sft: std.ArrayList(f32) = .empty;
    errdefer {
        pc.deinit(allocator);
        pr.deinit(allocator);
        rc.deinit(allocator);
        rr.deinit(allocator);
        cl.deinit(allocator);
        rl.deinit(allocator);
        sft.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;
        const parsed = try std.json.parseFromSliceLeaky(DpoScalarRow, allocator, line, .{ .ignore_unknown_fields = true });
        try pc.append(allocator, parsed.policy_chosen_logp);
        try pr.append(allocator, parsed.policy_rejected_logp);
        try rc.append(allocator, parsed.ref_chosen_logp);
        try rr.append(allocator, parsed.ref_rejected_logp);
        try cl.append(allocator, parsed.chosen_length orelse 0);
        try rl.append(allocator, parsed.rejected_length orelse 0);
        try sft.append(allocator, parsed.sft_chosen_loss orelse 0);
    }
    if (pc.items.len == 0) return error.EmptyBatch;
    return .{
        .policy_chosen_logps = try pc.toOwnedSlice(allocator),
        .policy_rejected_logps = try pr.toOwnedSlice(allocator),
        .ref_chosen_logps = try rc.toOwnedSlice(allocator),
        .ref_rejected_logps = try rr.toOwnedSlice(allocator),
        .chosen_lengths = try cl.toOwnedSlice(allocator),
        .rejected_lengths = try rl.toOwnedSlice(allocator),
        .sft_chosen_loss = try sft.toOwnedSlice(allocator),
    };
}

fn loadDpoTextPreferenceSamples(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    recipe: Recipe,
    policy_model: *model_manager_mod.LoadedModel,
) !DpoPreferenceSamplesOwned {
    const raw = try readFileMax(allocator, io, path, 256 * 1024 * 1024);
    defer allocator.free(raw);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_alloc = arena.allocator();

    var rows: std.ArrayListUnmanaged(DpoTextRow) = .empty;
    errdefer rows.deinit(arena_alloc);
    var lines = std.mem.splitScalar(u8, raw, '\n');
    const max_examples = recipe.dataset.max_examples orelse std.math.maxInt(usize);
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;
        if (rows.items.len >= max_examples) break;
        const parsed = try std.json.parseFromSliceLeaky(DpoTextRow, arena_alloc, line, .{ .ignore_unknown_fields = true });
        try rows.append(arena_alloc, parsed);
    }
    if (rows.items.len == 0) return error.EmptyBatch;

    const samples = try arena_alloc.alloc(preference_harness.PreferenceSample, rows.items.len);
    for (rows.items, 0..) |row, idx| {
        const tokenized = try tokenizeDpoTextRow(arena_alloc, policy_model, recipe, row);
        samples[idx] = .{
            .prompt_tokens = tokenized.prompt_tokens,
            .chosen_tokens = tokenized.chosen_tokens,
            .rejected_tokens = tokenized.rejected_tokens,
            .sft_chosen_loss = row.sft_chosen_loss,
        };
    }
    return .{
        .arena = arena,
        .samples = samples,
    };
}

const TokenizedPreferenceRow = struct {
    prompt_tokens: []const i32,
    chosen_tokens: []const i32,
    rejected_tokens: []const i32,
};

fn tokenizeDpoTextRow(
    allocator: std.mem.Allocator,
    model: *model_manager_mod.LoadedModel,
    recipe: Recipe,
    row: DpoTextRow,
) !TokenizedPreferenceRow {
    const tokenizer = model.getTokenizer();
    const max_seq_len = recipe.dataset.max_seq_len orelse 2048;
    const render_prompt = !std.mem.eql(u8, recipe.dataset.format orelse "text-preference", "rendered-text-preference");
    const prompt_text = if (render_prompt)
        try renderDpoPrompt(allocator, model, row.prompt)
    else
        try allocator.dupe(u8, row.prompt);
    defer allocator.free(prompt_text);

    var prompt_encoded = try generation.encodePromptForGeneration(
        tokenizer,
        allocator,
        prompt_text,
        max_seq_len,
        model.manifest.add_bos_token,
        model.manifest.bos_token,
    );
    defer prompt_encoded.deinit();

    const prompt_len = countAttentionMask(prompt_encoded.attention_mask);
    if (prompt_len == 0) return error.EmptyPrompt;
    const remaining_budget = max_seq_len - prompt_len;
    if (remaining_budget == 0) return error.NoCompletionBudget;

    const chosen_tokens = try tokenizeCompletion(allocator, tokenizer, row.chosen, remaining_budget);
    const rejected_tokens = try tokenizeCompletion(allocator, tokenizer, row.rejected, remaining_budget);
    if (chosen_tokens.len == 0 or rejected_tokens.len == 0) return error.EmptyCompletion;

    const prompt_tokens = try allocator.alloc(i32, prompt_len);
    for (0..prompt_len) |idx| prompt_tokens[idx] = prompt_encoded.ids[idx];
    return .{
        .prompt_tokens = prompt_tokens,
        .chosen_tokens = chosen_tokens,
        .rejected_tokens = rejected_tokens,
    };
}

fn renderDpoPrompt(allocator: std.mem.Allocator, model: *model_manager_mod.LoadedModel, prompt: []const u8) ![]u8 {
    const messages = [_]generation.Message{
        .{ .role = "user", .content = prompt },
    };
    if (model.chat_tmpl) |tmpl| return tmpl.apply(allocator, &messages, true);
    return generation.formatMessages(allocator, &messages);
}

fn tokenizeCompletion(
    allocator: std.mem.Allocator,
    tokenizer: tokenizer_mod.Tokenizer,
    text: []const u8,
    max_tokens: usize,
) ![]const i32 {
    const raw_tokens = try tokenizer.encode(allocator, text);
    defer allocator.free(raw_tokens);
    const token_count = @min(raw_tokens.len, max_tokens);
    if (token_count == 0) return allocator.alloc(i32, 0);
    const out = try allocator.alloc(i32, token_count);
    @memcpy(out, raw_tokens[0..token_count]);
    return out;
}

fn countAttentionMask(mask: []const i32) usize {
    var count: usize = 0;
    for (mask) |value| {
        if (value == 0) break;
        count += 1;
    }
    return count;
}

fn logProbAtToken(logits: []const f32, token_id: i32) f32 {
    if (token_id < 0 or @as(usize, @intCast(token_id)) >= logits.len) return -std.math.inf(f32);
    var max_logit = logits[0];
    for (logits[1..]) |value| {
        if (value > max_logit) max_logit = value;
    }
    var sum_exp: f64 = 0.0;
    for (logits) |value| {
        sum_exp += @exp(@as(f64, value - max_logit));
    }
    const log_z = @as(f64, max_logit) + @log(sum_exp);
    return @as(f32, @floatCast(@as(f64, logits[@intCast(token_id)]) - log_z));
}

fn parseRecipeBackendChoice(value: ?[]const u8) !native_backend_choice.Choice {
    const raw = value orelse return .auto;
    return native_backend_choice.parse(raw) orelse error.InvalidBackend;
}

const GemmaPreferenceExecution = struct {
    backend_kind: gemma4_real_autodiff.BackendKind,
    session_choice: native_backend_choice.Choice,
};

/// Keep Gemma policy training and reference scoring on one explicit backend.
/// `auto` retains the historical CPU-safe recipe behavior; callers requesting
/// Metal get the strict compiled-device training lane and no fallback.
fn resolveGemmaPreferenceExecution(value: ?[]const u8) !GemmaPreferenceExecution {
    const requested = try parseRecipeBackendChoice(value);
    const exact_choice: native_backend_choice.Choice = switch (requested) {
        .auto, .native => .native,
        .metal => .metal,
        .onnx, .cuda, .xla, .webgpu => return error.UnsupportedBackend,
    };
    try native_backend_choice.validate(exact_choice);
    return .{
        .backend_kind = switch (exact_choice) {
            .native => .native,
            .metal => .metal,
            else => unreachable,
        },
        .session_choice = exact_choice,
    };
}

fn metalKernelEnabledUnlessDisabled(name: [*:0]const u8) bool {
    // The bundled BF16 kernels treat an unset, empty, or exact "0" rollback
    // as enabled and every other present value as disabled. Mirror that ABI
    // exactly rather than applying the platform's broader false/no/off parser.
    const raw = platform.env.getenv(name) orelse return true;
    return raw.len == 0 or std.mem.eql(u8, raw, "0");
}

fn metalKernelExplicitlyEnabledUnlessDisabled(enable_name: [*:0]const u8, disable_name: [*:0]const u8) bool {
    return !platform.env.getenvBoolDefault(disable_name, false) and
        platform.env.getenvBoolDefault(enable_name, false);
}

fn gemmaPreferenceEnvironmentNameInScope(name: []const u8) bool {
    return gemma_preference_environment.nameInScope(name);
}

fn gemmaPreferenceEnvironmentNameAttested(name: []const u8) bool {
    return gemma_preference_environment.nameAllowed(name);
}

fn gemmaPreferenceEnvironmentValueIsCanonical(name: []const u8, value: []const u8) bool {
    return gemma_preference_environment.valueIsCanonical(name, value);
}

/// Product preference runs admit only environment controls represented in the
/// report and checkpoint fingerprint. This makes newly added debug/serving
/// switches fail closed until their training semantics are reviewed.
fn validateGemmaPreferenceEnvironmentAssignment(name: []const u8, value: []const u8) !void {
    if (!gemmaPreferenceEnvironmentNameInScope(name)) return;
    if (!gemmaPreferenceEnvironmentNameAttested(name)) {
        return error.UnattestedGemma4PreferenceEnvironmentOverride;
    }
    if (!gemmaPreferenceEnvironmentValueIsCanonical(name, value)) {
        return error.InvalidGemma4PreferenceEnvironmentValue;
    }
}

fn validateGemmaPreferenceEnvironmentContract(backend_kind: gemma4_real_autodiff.BackendKind) !void {
    if (backend_kind != .metal) return;
    const builtin = @import("builtin");
    if (comptime builtin.os.tag == .windows or !builtin.link_libc) return error.UnsupportedBackend;

    var index: usize = 0;
    while (std.c.environ[index]) |entry| : (index += 1) {
        const assignment = std.mem.span(entry);
        const separator = std.mem.indexOfScalar(u8, assignment, '=') orelse continue;
        const name = assignment[0..separator];
        validateGemmaPreferenceEnvironmentAssignment(name, assignment[separator + 1 ..]) catch |err| {
            if (err == error.UnattestedGemma4PreferenceEnvironmentOverride) {
                std.log.err("unattested Gemma 4 preference-training environment override: {s}", .{name});
            } else if (err == error.InvalidGemma4PreferenceEnvironmentValue) {
                std.log.err("non-canonical Gemma 4 preference-training environment value: {s}", .{name});
            }
            return err;
        };
    }
}

fn resolveGemmaMetalNumericalPolicy(
    backend_kind: gemma4_real_autodiff.BackendKind,
    ctx: *const gemma4_real_autodiff.GemmaAutodiffCtx,
) ?GemmaMetalNumericalPolicy {
    if (backend_kind != .metal) return null;
    const eager_policy = metal_compute_mod.resolvedGemmaTrainingNumericalPolicy(ctx.graph_config.vocab_size);
    const executor_policy = metal_partition_executor.resolvedGemmaTrainingNumericalPolicy();
    const residual_add_enabled = if (platform.env.getenvBoolDefault(
        "TERMITE_METAL_DISABLE_RMS_NORM_BACKWARD_RESIDUAL_ADD_FUSION",
        false,
    ))
        false
    else if (platform.env.getenv("TERMITE_METAL_ENABLE_RMS_NORM_BACKWARD_RESIDUAL_ADD_FUSION") != null)
        platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_RMS_NORM_BACKWARD_RESIDUAL_ADD_FUSION", false)
    else
        ctx.graph_config.hidden_size == 1536;
    const gate_up_backward_enabled = if (platform.env.getenvBoolDefault(
        "TERMITE_METAL_DISABLE_GEMMA4_BF16_GATE_UP_BACKWARD_INPUT_SUM",
        false,
    ))
        false
    else
        // Preference Metal training always acquires the strict training graph
        // executor scope, whose product policy enables the qualified shape
        // set when no explicit override is present.
        true;

    var policy = GemmaMetalNumericalPolicy{
        .fingerprint_flags = 0,
        .sparse_loss_chunk_rows = gemma4_real_autodiff.resolvedSparseLossChunkRows(),
        .linear_cce_tile_vocab = eager_policy.linear_cce_tile_vocab,
        .fused_rms_norm_backward = ctx.enable_fused_rms_norm_backward,
        .fused_gqa_attention_backward = ctx.enable_fused_gqa_attention_backward,
        .fused_linear_cross_entropy = ctx.enable_fused_linear_cross_entropy orelse false,
        .sparse_logits_cross_entropy = gemma4_real_autodiff.resolvedSparseLogitsCrossEntropyEnabled(),
        .bf16_tiled32_m16 = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_TILED32_M16"),
        .bf16_simdgroup_mm = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_SIMDGROUP_MM"),
        .bf16_simdgroup_m64 = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_SIMDGROUP_M64"),
        .bf16_forward_simdgroup_m64_packed = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_FORWARD_SIMDGROUP_M64_PACKED"),
        .bf16_simdgroup_m64_prefix_tail = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_SIMDGROUP_M64_PREFIX_TAIL"),
        .bf16_backward_tiled32_m16 = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_BACKWARD_TILED32_M16"),
        .bf16_backward_small_rows = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_BACKWARD_SMALL_ROWS"),
        .bf16_backward_simdgroup_mm = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_MM"),
        .bf16_backward_simdgroup_m64 = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_M64"),
        .bf16_backward_simdgroup_m64_coalesced = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_M64_COALESCED"),
        .bf16_backward_simdgroup_m64_packed = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_M64_PACKED"),
        .rms_norm_backward_simdgroup = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_RMS_NORM_BACKWARD_SIMDGROUP"),
        .rms_norm_backward_residual_add = residual_add_enabled,
        .rms_norm_generated = platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_RMS_NORM_GENERATED", false),
        .linear_cce_f16_grad = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_LINEAR_CCE_F16_GRAD"),
        .linear_cce_logit_cache = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_LINEAR_CCE_LOGIT_CACHE"),
        .linear_cce_f16_mps_backward = metalKernelEnabledUnlessDisabled("TERMITE_METAL_DISABLE_LINEAR_CCE_F16_MPS_BACKWARD"),
        .dense_mps_linear = platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_DENSE_MPS_LINEAR", false),
        .gemma4_bf16_mlp_fusion = metalKernelExplicitlyEnabledUnlessDisabled(
            "TERMITE_METAL_ENABLE_GEMMA4_BF16_MLP_FUSION",
            "TERMITE_METAL_DISABLE_GEMMA4_BF16_MLP_FUSION",
        ),
        .gemma4_gate_up_backward_input_sum = gate_up_backward_enabled,
        .q4_0_linear_rms_add_sumsq = metalKernelExplicitlyEnabledUnlessDisabled(
            "TERMITE_METAL_ENABLE_Q4_0_LINEAR_RMS_ADD_SUMSQ",
            "TERMITE_METAL_DISABLE_Q4_0_LINEAR_RMS_ADD_SUMSQ",
        ),
        .eager_rank1_dot_specialization = eager_policy.rank1_dot_specialization,
        .dense_device_dot_general = eager_policy.dense_device_dot_general,
        .lora_forward_fused_branch = eager_policy.lora_forward_fused_branch,
        .lora_forward_generic_rank16 = eager_policy.lora_forward_generic_rank16,
        .lora_forward_rank1_fused = eager_policy.lora_forward_rank1_fused,
        .reference_quant_linear = eager_policy.reference_quant_linear,
        .quant_backward_force_barriers = eager_policy.quant_backward_force_barriers,
        .contiguous_slice_device_view = eager_policy.contiguous_slice_device_view,
        .partition_fused_patterns = executor_policy.partition_fused_patterns,
        .partition_runtime_commands = executor_policy.partition_runtime_commands,
        .runtime_region_plan = executor_policy.runtime_region_plan,
        .grouped_mps_dot = executor_policy.grouped_mps_dot,
        .gather_promote_input = executor_policy.gather_promote_input,
        .reduce_promote_input = executor_policy.reduce_promote_input,
        .lora_backward_runtime_region = executor_policy.lora_backward_runtime_region,
        .low_rank_lora_backward_runtime_region = executor_policy.low_rank_lora_backward_runtime_region,
        .rank_adapter_backward_runtime_region = executor_policy.rank_adapter_backward_runtime_region,
        .ffn_gelu_backward_runtime_region = executor_policy.ffn_gelu_backward_runtime_region,
        .gated_gelu_backward_runtime_region = executor_policy.gated_gelu_backward_runtime_region,
        .gated_gelu_forward_fusion = executor_policy.gated_gelu_forward_fusion,
        .masked_softmax_runtime_region = executor_policy.masked_softmax_runtime_region,
        .softmax_backward_runtime_region = executor_policy.softmax_backward_runtime_region,
        .graph_rank1_dot_specialization = executor_policy.rank1_dot_specialization,
        .raw_linear_bias_pair_runtime_region = executor_policy.raw_linear_bias_pair_runtime_region,
        .raw_linear_runtime_regions_suppressed = executor_policy.raw_linear_runtime_regions_suppressed,
        .gated_ffn_graph_fusion = executor_policy.gated_ffn_graph_fusion,
        .gemma_gated_mlp_training_graph_fusion = executor_policy.gemma_gated_mlp_training_graph_fusion,
        .attention_output_residual_graph_fusion = executor_policy.attention_output_residual_graph_fusion,
        .grouped_lora_a_r16 = executor_policy.grouped_lora_a_r16,
        .add3_fusion = executor_policy.add3_fusion,
    };
    comptime std.debug.assert(std.meta.fields(GemmaMetalNumericalPolicy).len <= @bitSizeOf(u64));
    inline for (std.meta.fields(GemmaMetalNumericalPolicy), 0..) |field, field_index| {
        if (field.type == bool and @field(policy, field.name)) {
            // The schema string and fingerprint field are non-bools, leaving
            // a stable compact bit for every resolved arithmetic choice.
            policy.fingerprint_flags |= @as(u64, 1) << @intCast(field_index);
        }
    }
    return policy;
}

/// A preference unit is only allowed to cross an optimizer boundary after all
/// of its coupled micro-batches have contributed gradients: chosen+rejected
/// for DPO, or every sampled completion for one GRPO group.
fn preferenceGradAccumSteps(requested_units: u32, micro_batches_per_unit: usize) !u32 {
    if (requested_units == 0 or micro_batches_per_unit == 0) {
        return error.InvalidGradientAccumulationSteps;
    }
    const total = std.math.mul(usize, @as(usize, requested_units), micro_batches_per_unit) catch {
        return error.InvalidGradientAccumulationSteps;
    };
    if (total > std.math.maxInt(u32)) return error.InvalidGradientAccumulationSteps;
    return @intCast(total);
}

/// RealAutodiffTrainer averages every physical micro-batch in an accumulation
/// window. Preference losses are coupled logical units whose gradient is the
/// sum of their chosen/rejected or completion contributions, averaged only
/// across requested logical units. Compensate each physical contribution so
/// the trainer's final mean preserves that objective exactly.
fn scalePreferenceUnitGradients(gradients: []f32, micro_batches_per_unit: usize) !void {
    if (micro_batches_per_unit == 0) return error.InvalidGradientAccumulationSteps;
    const scale: f32 = @floatFromInt(micro_batches_per_unit);
    if (!std.math.isFinite(scale)) return error.InvalidGradientAccumulationSteps;
    for (gradients) |*gradient| gradient.* *= scale;
}

fn validateGrpoLearningSignal(
    saw_nonzero_reward_advantage: bool,
    saw_nonzero_policy_gradient: bool,
    total_reward: f64,
    total_reward_squared: f64,
    total_completions: usize,
    total_loss: f64,
) !void {
    if (saw_nonzero_reward_advantage and saw_nonzero_policy_gradient) return;
    const denom = @as(f64, @floatFromInt(@max(total_completions, 1)));
    const mean = total_reward / denom;
    const variance = @max(total_reward_squared / denom - mean * mean, 0.0);
    print(
        "grpo rejected zero learning signal: completions={d} reward_advantage={} policy_gradient={} mean_reward={d:.6} reward_stddev={d:.6} accumulated_loss={d:.6}\n",
        .{ total_completions, saw_nonzero_reward_advantage, saw_nonzero_policy_gradient, mean, @sqrt(variance), total_loss },
    );
    return error.NoGrpoLearningSignal;
}

test "gemma4 preference execution resolves policy and scoring to one backend" {
    const automatic = try resolveGemmaPreferenceExecution(null);
    try std.testing.expectEqual(gemma4_real_autodiff.BackendKind.native, automatic.backend_kind);
    try std.testing.expectEqual(native_backend_choice.Choice.native, automatic.session_choice);

    const native = try resolveGemmaPreferenceExecution("native");
    try std.testing.expectEqual(gemma4_real_autodiff.BackendKind.native, native.backend_kind);
    try std.testing.expectEqual(native_backend_choice.Choice.native, native.session_choice);

    if (build_options.enable_metal) {
        const metal = try resolveGemmaPreferenceExecution("metal");
        try std.testing.expectEqual(gemma4_real_autodiff.BackendKind.metal, metal.backend_kind);
        try std.testing.expectEqual(native_backend_choice.Choice.metal, metal.session_choice);
    } else {
        try std.testing.expectError(error.BackendUnavailable, resolveGemmaPreferenceExecution("metal"));
    }

    try std.testing.expectError(error.UnsupportedBackend, resolveGemmaPreferenceExecution("cuda"));
    try std.testing.expectError(error.InvalidBackend, resolveGemmaPreferenceExecution("bogus"));
}

test "gemma4 preference gradient accumulation preserves complete DPO pairs and GRPO groups" {
    try std.testing.expectEqual(@as(u32, 2), try preferenceGradAccumSteps(1, 2));
    try std.testing.expectEqual(@as(u32, 12), try preferenceGradAccumSteps(3, 4));
    try std.testing.expectError(error.InvalidGradientAccumulationSteps, preferenceGradAccumSteps(0, 2));
    try std.testing.expectError(error.InvalidGradientAccumulationSteps, preferenceGradAccumSteps(1, 0));
    try std.testing.expectError(error.InvalidGradientAccumulationSteps, preferenceGradAccumSteps(std.math.maxInt(u32), 2));

    var dpo_gradients = [_]f32{ -0.05, 0.05 };
    try scalePreferenceUnitGradients(&dpo_gradients, 2);
    try std.testing.expectEqualSlices(f32, &.{ -0.1, 0.1 }, &dpo_gradients);
    var grpo_gradients = [_]f32{ -0.25, 0.0, 0.25 };
    try scalePreferenceUnitGradients(&grpo_gradients, 4);
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 0.0, 1.0 }, &grpo_gradients);
    try std.testing.expectError(error.InvalidGradientAccumulationSteps, scalePreferenceUnitGradients(&dpo_gradients, 0));
}

test "gemma4 preference baseline-relative gates require strict heldout improvement" {
    const dpo_baseline = DpoEvaluationSummary{
        .report_path = "baseline",
        .examples = 40,
        .loss = 0.7,
        .mean_reward_margin = 0.0,
        .accuracy = 0.5,
        .passed = false,
    };
    var dpo_final = dpo_baseline;
    dpo_final.report_path = "final";
    dpo_final.loss = 0.6;
    dpo_final.mean_reward_margin = 0.1;
    dpo_final.accuracy = 0.6;
    const dpo_minimums = DpoEvalMinimums{
        .accuracy = 0.4,
        .max_loss = 1.0,
        .min_accuracy_improvement = 0.05,
        .min_reward_margin_improvement = 0.05,
        .min_loss_improvement = 0.05,
    };
    try std.testing.expect(compareDpoToBaseline(dpo_baseline, dpo_final, dpo_minimums).passed);
    dpo_final.accuracy = 0.5;
    try std.testing.expect(!compareDpoToBaseline(dpo_baseline, dpo_final, dpo_minimums).passed);

    const grpo_baseline = GrpoEvaluationSummary{
        .report_path = "baseline",
        .groups = 64,
        .completions = 256,
        .mean_reward = 0.1,
        .top_rank_mean_reward = 0.1,
        .positive_reward_group_rate = 0.5,
        .reward_stddev = 0.3,
        .kl_loss = 0.0,
        .mean_kl = 0.0,
        .passed = false,
    };
    var grpo_final = grpo_baseline;
    grpo_final.report_path = "final";
    grpo_final.mean_reward = 0.2;
    grpo_final.top_rank_mean_reward = 0.25;
    grpo_final.positive_reward_group_rate = 0.75;
    const grpo_minimums = GrpoEvalMinimums{
        .mean_reward = 0.125,
        .top_rank_mean_reward = 0.125,
        .positive_reward_group_rate = 0.75,
        .max_kl_loss = 0.004,
        .min_mean_reward_improvement = 0.05,
        .min_top_rank_mean_reward_improvement = 0.05,
        .min_positive_reward_group_rate_improvement = 0.05,
    };
    try std.testing.expect(compareGrpoToBaseline(grpo_baseline, grpo_final, grpo_minimums).passed);
    grpo_final.top_rank_mean_reward = 0.1;
    try std.testing.expect(!compareGrpoToBaseline(grpo_baseline, grpo_final, grpo_minimums).passed);
}

test "compiled DPO loss inversion recovers reward margin" {
    for ([_]f32{ -8.0, -1.25, 0.0, 2.5, 8.0 }) |expected_margin| {
        const loss = -@log(1.0 / (1.0 + @exp(-expected_margin)));
        const actual_margin = try rewardMarginFromDpoLoss(loss);
        try std.testing.expectApproxEqAbs(expected_margin, actual_margin, 2e-3);
    }
    try std.testing.expectError(error.InvalidDpoCompiledLoss, rewardMarginFromDpoLoss(0.0));
    try std.testing.expectError(error.InvalidDpoCompiledLoss, rewardMarginFromDpoLoss(std.math.nan(f32)));
}

test "gemma4 DPO single-token coalescing requires an identical causal prompt" {
    var chosen_prompt = [_]i32{ 1, 2, 3 };
    var rejected_prompt = [_]i32{ 1, 2, 3 };
    var chosen_response = [_]i32{5};
    var rejected_response = [_]i32{7};
    var chosen_input = [_]i32{ 1, 2, 3, 5 };
    var rejected_input = [_]i32{ 1, 2, 3, 7 };
    var chosen_labels = [_]i32{ -100, -100, -100, 5 };
    var rejected_labels = [_]i32{ -100, -100, -100, 7 };
    const chosen = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = &chosen_prompt,
        .response_input_ids = &chosen_response,
        .num_prompt_tokens = 3,
        .num_response_tokens = 1,
        .input_ids = &chosen_input,
        .labels = &chosen_labels,
        .num_input_tokens = 4,
        .num_supervised_tokens = 1,
    };
    var rejected = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = &rejected_prompt,
        .response_input_ids = &rejected_response,
        .num_prompt_tokens = 3,
        .num_response_tokens = 1,
        .input_ids = &rejected_input,
        .labels = &rejected_labels,
        .num_input_tokens = 4,
        .num_supervised_tokens = 1,
    };

    const pair = gemmaDpoSingleTokenPair(&chosen, &rejected) orelse return error.TestExpectedEqual;
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3 }, pair.prompt);
    try std.testing.expectEqual(@as(i32, 5), pair.chosen_token);
    try std.testing.expectEqual(@as(i32, 7), pair.rejected_token);
    try std.testing.expect(allGemmaDpoPairsAreSingleTokenSharedPrompt(&.{chosen}, &.{rejected}));

    rejected.labels[0] = 7;
    try std.testing.expect(gemmaDpoSingleTokenPair(&chosen, &rejected) == null);
    try std.testing.expect(!allGemmaDpoPairsAreSingleTokenSharedPrompt(&.{chosen}, &.{rejected}));
}

test "gemma4 DPO length buckets preserve pair alignment and bound graph signatures" {
    const chosen = [_]gemma4.PreparedExampleInput{
        .{
            .mode = .instruction,
            .prompt_input_ids = &.{},
            .response_input_ids = &.{},
            .num_prompt_tokens = 98,
            .num_response_tokens = 46,
            .num_input_tokens = 144,
            .num_supervised_tokens = 46,
        },
        .{
            .mode = .instruction,
            .prompt_input_ids = &.{},
            .response_input_ids = &.{},
            .num_prompt_tokens = 164,
            .num_response_tokens = 136,
            .num_input_tokens = 300,
            .num_supervised_tokens = 136,
        },
    };
    const rejected = [_]gemma4.PreparedExampleInput{
        .{
            .mode = .instruction,
            .prompt_input_ids = &.{},
            .response_input_ids = &.{},
            .num_prompt_tokens = 98,
            .num_response_tokens = 161,
            .num_input_tokens = 259,
            .num_supervised_tokens = 161,
        },
        .{
            .mode = .instruction,
            .prompt_input_ids = &.{},
            .response_input_ids = &.{},
            .num_prompt_tokens = 164,
            .num_response_tokens = 187,
            .num_input_tokens = 351,
            .num_supervised_tokens = 187,
        },
    };
    const buckets = gemma4_real_autodiff.SequenceLengthBuckets{ .quantum = 16, .minimum = 16 };
    const first = try gemmaDpoPairSchedule(&chosen[0], &rejected[0], 512, buckets);
    try std.testing.expectEqual(@as(u32, 272), first.sequence_length);
    try std.testing.expectEqual(@as(usize, 256), first.weighted_target_rows);
    const second = try gemmaDpoPairSchedule(&chosen[1], &rejected[1], 512, buckets);
    try std.testing.expectEqual(@as(u32, 352), second.sequence_length);
    try std.testing.expectEqual(@as(usize, 256), second.weighted_target_rows);

    const telemetry = try summarizeGemmaDpoPairLengthPolicy(
        std.testing.allocator,
        &chosen,
        &rejected,
        512,
        buckets,
        8,
        "test",
    );
    try std.testing.expectEqualStrings("pair-safe-length-buckets", telemetry.mode);
    try std.testing.expectEqual(@as(usize, 1248), telemetry.scheduled_branch_rows);
    try std.testing.expectEqual(@as(usize, 2048), telemetry.fixed_shape_branch_rows);
    try std.testing.expectEqual(@as(usize, 800), telemetry.padding_rows_avoided);
    try std.testing.expectEqual(@as(usize, 2), telemetry.unique_pair_sequence_lengths);
    try std.testing.expectEqual(@as(?usize, 2), telemetry.unique_pair_graph_signatures);

    const fixed = try summarizeGemmaDpoPairLengthPolicy(
        std.testing.allocator,
        &chosen,
        &rejected,
        512,
        null,
        4,
        "test",
    );
    try std.testing.expectEqualStrings("fixed-pair-padding", fixed.mode);
    try std.testing.expectEqual(@as(usize, 2048), fixed.scheduled_branch_rows);
    try std.testing.expectEqual(@as(usize, 0), fixed.padding_rows_avoided);
    try std.testing.expectEqual(@as(?usize, null), fixed.unique_pair_graph_signatures);

    try std.testing.expectEqual(@as(u8, 1), gemmaDpoGraphCacheCapacity(.{}, true));
    try std.testing.expectEqual(@as(u8, 4), gemmaDpoGraphCacheCapacity(.{}, false));
    try std.testing.expectEqual(
        @as(u8, 4),
        gemmaDpoGraphCacheCapacity(.{
            .runtime = .{ .sequence_length_bucket_quantum = 128 },
        }, false),
    );
    try std.testing.expectEqual(
        @as(u8, 8),
        gemmaDpoGraphCacheCapacity(.{
            .runtime = .{
                .sequence_length_bucket_quantum = 128,
                .graph_cache_capacity = 8,
            },
        }, false),
    );
}

test "gemma4 GRPO learning-signal gate requires reward variation and a policy gradient" {
    try std.testing.expectError(error.NoGrpoLearningSignal, validateGrpoLearningSignal(true, false, 1.0, 1.0, 2, 0.0));
    try std.testing.expectError(error.NoGrpoLearningSignal, validateGrpoLearningSignal(false, true, 0.0, 0.0, 2, 0.5));
    try validateGrpoLearningSignal(true, true, 1.0, 1.0, 2, 0.5);
}

test "gemma4 GRPO heldout gate rejects reward concentrated outside the top rank or in one group" {
    const minimums = GrpoEvalMinimums{
        .mean_reward = 0.1,
        .top_rank_mean_reward = 0.5,
        .positive_reward_group_rate = 0.5,
        .max_kl_loss = 1.0,
    };
    try std.testing.expect(passesGrpoEvaluationMinimums(0.2, 0.5, 0.5, 0.1, minimums));
    try std.testing.expect(!passesGrpoEvaluationMinimums(0.2, 0.25, 0.5, 0.1, minimums));
    try std.testing.expect(!passesGrpoEvaluationMinimums(0.2, 0.5, 0.25, 0.1, minimums));
    try std.testing.expect(!passesGrpoEvaluationMinimums(0.2, 0.5, 0.5, 1.1, minimums));
}

test "gemma4 GRPO KL control persists admitted and rejected pre-update decisions" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const control_recipe = Recipe{
        .grpo = .{
            .kl_coef = 0.04,
            .train_max_kl = 0.1,
            .adaptive_kl = true,
            .target_kl = 0.01,
            .kl_horizon = 100,
            .min_kl_coef = 0.001,
            .max_kl_coef = 1.0,
        },
        .artifacts = .{ .root = root },
    };

    var admitted = try GrpoKlControl.init(allocator, std.testing.io, control_recipe);
    const coefficient_after = try admitted.observe(0, 0, 0, 0, 0.0, 0.0);
    try std.testing.expect(coefficient_after < 0.04);
    try admitted.finish();
    try std.testing.expectEqual(@as(usize, 1), admitted.telemetry().admitted_groups);
    try std.testing.expect(admitted.telemetry().trace_digest != null);
    const admitted_trace_path = try allocator.dupe(u8, admitted.trace_path);
    admitted.deinit();
    defer allocator.free(admitted_trace_path);

    const admitted_trace = try readFileMax(allocator, std.testing.io, admitted_trace_path, 64 * 1024);
    defer allocator.free(admitted_trace);
    try std.testing.expect(std.mem.indexOf(u8, admitted_trace, "\"status\":\"admitted\"") != null);

    var rejected = try GrpoKlControl.init(allocator, std.testing.io, control_recipe);
    defer rejected.deinit();
    try std.testing.expectError(
        error.GrpoTrainKlBudgetExceeded,
        rejected.observe(0, 0, 0, 0, 0.1001, 0.004004),
    );
    try std.testing.expectEqual(@as(usize, 0), rejected.telemetry().admitted_groups);
    try std.testing.expect(rejected.telemetry().trace_digest != null);
    const rejected_trace = try readFileMax(allocator, std.testing.io, rejected.trace_path, 64 * 1024);
    defer allocator.free(rejected_trace);
    try std.testing.expect(std.mem.indexOf(u8, rejected_trace, "\"status\":\"budget-exceeded\"") != null);
}

test "gemma4 GRPO KL checkpoint restores an exact adaptive continuation" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const control_recipe = Recipe{
        .grpo = .{
            .kl_coef = 0.04,
            .train_max_kl = 0.1,
            .adaptive_kl = true,
            .target_kl = 0.01,
            .kl_horizon = 100,
            .min_kl_coef = 0.001,
            .max_kl_coef = 1.0,
        },
        .artifacts = .{ .root = root },
    };

    var uninterrupted = try GrpoKlControl.init(allocator, std.testing.io, control_recipe);
    defer uninterrupted.deinit();
    _ = try uninterrupted.observe(0, 0, 0, 0, 0.005, 0.0002);

    var resumed = try GrpoKlControl.init(allocator, std.testing.io, control_recipe);
    defer resumed.deinit();
    try resumed.restoreCheckpoint(
        uninterrupted.current_kl_coef,
        uninterrupted.admitted_groups,
        uninterrupted.max_observed_mean_kl,
        uninterrupted.trace.items,
    );

    const uninterrupted_next = try uninterrupted.observe(1, 1, 0, 1, 0.02, 0.0008);
    const resumed_next = try resumed.observe(1, 1, 0, 1, 0.02, 0.0008);
    try std.testing.expectEqual(uninterrupted_next, resumed_next);
    try std.testing.expectEqual(uninterrupted.current_kl_coef, resumed.current_kl_coef);
    try std.testing.expectEqual(uninterrupted.admitted_groups, resumed.admitted_groups);
    try std.testing.expectEqual(uninterrupted.max_observed_mean_kl, resumed.max_observed_mean_kl);
    try std.testing.expectEqualSlices(u8, uninterrupted.trace.items, resumed.trace.items);
}

test "gemma4 preference checkpoint embeds its content-addressed sidecar identity" {
    const allocator = std.testing.allocator;
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("checkpoint aggregate state", &digest, .{});
    const progress = real_autodiff.TrainingProgress{
        .epoch_index = 1,
        .examples_seen = 7,
        .order_seed = preferenceCheckpointMagic(.dpo),
        .rng_state = preferenceDigestWords(digest),
    };
    const restored_digest = preferenceDigestFromProgress(progress);
    try std.testing.expectEqualSlices(u8, &digest, &restored_digest);
    try std.testing.expect(preferenceCheckpointMagic(.dpo) != preferenceCheckpointMagic(.grpo));

    const state_path = try preferenceCheckpointStatePath(
        allocator,
        "/tmp/gemma4-dpo-state.safetensors",
        digest,
    );
    defer allocator.free(state_path);
    const digest_hex = std.fmt.bytesToHex(digest, .lower);
    const expected_suffix = try std.fmt.allocPrint(allocator, "{s}.json", .{digest_hex});
    defer allocator.free(expected_suffix);
    try std.testing.expect(std.mem.endsWith(u8, state_path, expected_suffix));

    var artifact = (try preferenceCheckpointArtifactSummary(
        allocator,
        "/tmp/gemma4-dpo-state.safetensors",
        .dpo,
        progress,
    )).?;
    defer artifact.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), artifact.epoch);
    try std.testing.expectEqualStrings(state_path, artifact.state_path);
    try std.testing.expect(std.mem.endsWith(u8, artifact.state_sha256, &digest_hex));
}

const GrpoScalarRow = struct {
    prompt_idx: usize,
    tokens: []const i32,
    old_logps: []const f32,
    ref_logps: []const f32,
    new_logps: []const f32,
    reward: f32,
};

const GrpoBatchOwned = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    completions: []grpo.Completion,
    new_logps: []f32,
    rewards: []f32,

    fn deinit(self: *GrpoBatchOwned) void {
        self.allocator.free(self.completions);
        self.allocator.free(self.new_logps);
        self.allocator.free(self.rewards);
        self.arena.deinit();
    }
};

fn runDirectGrpo(allocator: std.mem.Allocator, io: std.Io, recipe: Recipe, report_path: []const u8) !void {
    const path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const format = recipe.dataset.format orelse return error.MissingDatasetFormat;
    const mode = try parsePreferenceExecutionMode(recipe);
    try validatePreferenceExecutionContract(recipe, .grpo, mode, format);
    if (std.mem.eql(u8, format, "token-logprobs")) {
        var batch = try loadGrpoScalarJsonl(allocator, io, path);
        defer batch.deinit();
        const cfg = grpo.GRPOConfig{
            .group_size = recipe.grpo.group_size orelse 8,
            .clip_epsilon = recipe.grpo.clip_epsilon orelse 0.2,
            .kl_coef = recipe.grpo.kl_coef orelse 0.04,
            .advantage_eps = recipe.grpo.advantage_eps orelse 1e-8,
            .normalize_advantage = recipe.grpo.normalize_advantage orelse true,
        };
        var ga = grpo.GroupAdvantages{
            .allocator = allocator,
            .rewards = try allocator.dupe(f32, batch.rewards),
            .advantages = try allocator.alloc(f32, batch.completions.len),
            .num_groups = countGrpoGroups(batch.completions),
        };
        defer ga.deinit();
        @memset(ga.advantages, 0);
        grpo.computeAdvantages(&ga, batch.completions, cfg);
        var result = try grpo.grpoLoss(allocator, batch.completions, batch.new_logps, ga.advantages, cfg);
        defer result.deinit();
        try writeJsonFile(allocator, io, report_path, GrpoReport{
            .execution_mode = "score",
            .dataset_format = format,
            .completions = batch.completions.len,
            .tokens = batch.new_logps.len,
            .groups = ga.num_groups,
            .loss = result.loss,
            .pg_loss = result.pg_loss,
            .kl_loss = result.kl_loss,
            .clip_fraction = result.clip_fraction,
        });
        print("grpo report: {s}\n", .{report_path});
        return;
    }
    if (mode == .train) {
        if (try shouldRunOptimizerBackedQwen2Grpo(recipe, format)) {
            try runOptimizerBackedQwen2Grpo(allocator, io, recipe, path, report_path);
            return;
        }
        if (try shouldRunOptimizerBackedGemmaGrpo(recipe, format)) {
            try runOptimizerBackedGemmaGrpo(allocator, io, recipe, path, report_path);
            return;
        }
        return error.UnsupportedPreferenceTrainingFamily;
    }

    const policy_path = recipe.model.path orelse return error.MissingModelPath;
    const reference_path = recipe.model.reference_path orelse policy_path;
    const backend_choice = try parseRecipeBackendChoice(recipe.backend);

    var session_manager = backends.SessionManager.init(allocator);
    native_backend_choice.configureSessionPreference(&session_manager, backend_choice);
    var model_manager = model_manager_mod.ModelManager.init(allocator, session_manager);
    defer model_manager.deinit();

    const policy_model = try model_manager.loadFromDir(policy_path);
    const reference_model = if (std.mem.eql(u8, reference_path, policy_path))
        policy_model
    else
        try model_manager.loadFromDir(reference_path);

    var prompt_batch = try loadGrpoTextPrompts(allocator, io, path, recipe, policy_model);
    defer prompt_batch.deinit();

    const reward_mode = try parseTextRewardMode(recipe.grpo.reward_mode orelse "exact-match");

    var sampler = DecoderGrpoSampler{
        .allocator = allocator,
        .model = policy_model,
        .max_seq_len = recipe.dataset.max_seq_len orelse 128,
        .max_completion_tokens = recipe.grpo.max_completion_tokens orelse 4,
    };
    var policy_scorer = DecoderLogprobScorer{
        .allocator = allocator,
        .model = policy_model,
        .max_seq_len = recipe.dataset.max_seq_len orelse 128,
    };
    var ref_scorer = DecoderLogprobScorer{
        .allocator = allocator,
        .model = reference_model,
        .max_seq_len = recipe.dataset.max_seq_len orelse 128,
    };
    var rewarder_ctx = TextRewardCtx{
        .allocator = allocator,
        .tokenizer = policy_model.getTokenizer(),
        .targets = prompt_batch.targets,
        .mode = reward_mode,
    };

    var result = try preference_harness.grpoStep(allocator, prompt_batch.prompts, .{
        .ctx = &sampler,
        .call = DecoderGrpoSampler.sample,
    }, .{
        .ctx = &policy_scorer,
        .call = DecoderLogprobScorer.tokenLogprobs,
    }, .{
        .ctx = &ref_scorer,
        .call = DecoderLogprobScorer.tokenLogprobs,
    }, .{
        .ctx = &rewarder_ctx,
        .call = TextRewardCtx.score,
    }, .{
        .grpo = .{
            .group_size = recipe.grpo.group_size orelse 2,
            .clip_epsilon = recipe.grpo.clip_epsilon orelse 0.2,
            .kl_coef = recipe.grpo.kl_coef orelse 0.04,
            .advantage_eps = recipe.grpo.advantage_eps orelse 1e-8,
            .normalize_advantage = recipe.grpo.normalize_advantage orelse true,
        },
        .num_prompts = prompt_batch.prompts.len,
    });
    defer result.deinit();
    try writeJsonFile(allocator, io, report_path, GrpoReport{
        .execution_mode = "score",
        .dataset_format = format,
        .completions = prompt_batch.prompts.len * (recipe.grpo.group_size orelse 2),
        .tokens = result.grad_new_logps.len,
        .groups = prompt_batch.prompts.len,
        .loss = result.loss,
        .pg_loss = result.pg_loss,
        .kl_loss = result.kl_loss,
        .clip_fraction = result.clip_fraction,
    });
    print("grpo report: {s}\n", .{report_path});
}

fn loadGrpoScalarJsonl(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !GrpoBatchOwned {
    const raw = try readFileMax(allocator, io, path, 256 * 1024 * 1024);
    defer allocator.free(raw);
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const aa = arena.allocator();
    var completions: std.ArrayList(grpo.Completion) = .empty;
    var new_logps: std.ArrayList(f32) = .empty;
    var rewards: std.ArrayList(f32) = .empty;
    errdefer {
        completions.deinit(allocator);
        new_logps.deinit(allocator);
        rewards.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;
        const row = try std.json.parseFromSliceLeaky(GrpoScalarRow, aa, line, .{ .ignore_unknown_fields = true });
        if (row.tokens.len != row.old_logps.len or row.tokens.len != row.ref_logps.len or row.tokens.len != row.new_logps.len) return error.LogpLenMismatch;
        try completions.append(allocator, .{
            .prompt_idx = row.prompt_idx,
            .tokens = row.tokens,
            .old_logps = row.old_logps,
            .ref_logps = row.ref_logps,
        });
        try new_logps.appendSlice(allocator, row.new_logps);
        try rewards.append(allocator, row.reward);
    }
    if (completions.items.len == 0) return error.EmptyBatch;
    return .{
        .allocator = allocator,
        .arena = arena,
        .completions = try completions.toOwnedSlice(allocator),
        .new_logps = try new_logps.toOwnedSlice(allocator),
        .rewards = try rewards.toOwnedSlice(allocator),
    };
}

fn loadGrpoTextPrompts(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    recipe: Recipe,
    policy_model: *model_manager_mod.LoadedModel,
) !GrpoPromptBatchOwned {
    const raw = try readFileMax(allocator, io, path, 256 * 1024 * 1024);
    defer allocator.free(raw);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const aa = arena.allocator();

    var rows: std.ArrayListUnmanaged(GrpoTextRow) = .empty;
    errdefer rows.deinit(aa);
    var lines = std.mem.splitScalar(u8, raw, '\n');
    const max_examples = recipe.dataset.max_examples orelse std.math.maxInt(usize);
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;
        if (rows.items.len >= max_examples) break;
        const parsed = try std.json.parseFromSliceLeaky(GrpoTextRow, aa, line, .{ .ignore_unknown_fields = true });
        try rows.append(aa, parsed);
    }
    if (rows.items.len == 0) return error.EmptyBatch;

    const prompts = try aa.alloc([]const i32, rows.items.len);
    const prompt_texts = try aa.alloc([]const u8, rows.items.len);
    const targets = try aa.alloc([]const u8, rows.items.len);
    for (rows.items, 0..) |row, idx| {
        const tokenized_prompt = try tokenizeGrpoPrompt(aa, policy_model, recipe, row.prompt);
        prompts[idx] = tokenized_prompt;
        prompt_texts[idx] = try aa.dupe(u8, row.prompt);
        // parseFromSliceLeaky may borrow string storage from `raw`, which is
        // released when this loader returns. Keep reward targets in the arena
        // alongside the tokenized prompts so GRPO never scores dangling data.
        targets[idx] = try aa.dupe(u8, row.target);
    }

    return .{
        .arena = arena,
        .prompts = prompts,
        .prompt_texts = prompt_texts,
        .targets = targets,
    };
}

fn loadGemmaGrpoPreparedPrompts(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    recipe: Recipe,
    base_model_dir: []const u8,
    projector_path: []const u8,
) !GemmaPreparedPromptBatchOwned {
    if (std.mem.eql(u8, recipe.dataset.format orelse "text-grpo", "rendered-text-grpo")) {
        return error.UnsupportedRenderedMultimodalGrpo;
    }

    const raw = try readFileMax(allocator, io, path, 256 * 1024 * 1024);
    defer allocator.free(raw);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const aa = arena.allocator();

    var rows: std.ArrayListUnmanaged(GrpoTextRow) = .empty;
    errdefer rows.deinit(aa);
    var lines = std.mem.splitScalar(u8, raw, '\n');
    const max_examples = recipe.dataset.max_examples orelse std.math.maxInt(usize);
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \t\r");
        if (line.len == 0) continue;
        if (rows.items.len >= max_examples) break;
        try rows.append(aa, try std.json.parseFromSliceLeaky(GrpoTextRow, aa, line, .{ .ignore_unknown_fields = true }));
    }
    if (rows.items.len == 0) return error.EmptyBatch;

    const prompts = try allocator.alloc(*const gemma4.PreparedExampleInput, rows.items.len);
    errdefer allocator.free(prompts);
    const summaries = try allocator.alloc(gemma4.PreparedInputsSummary, rows.items.len);
    errdefer allocator.free(summaries);
    const targets = try allocator.alloc([]const u8, rows.items.len);
    errdefer allocator.free(targets);

    var built: usize = 0;
    errdefer {
        for (summaries[0..built]) |*summary| gemma4.freePreparedInputsSummary(allocator, summary);
        for (targets[0..built]) |target| allocator.free(target);
    }

    const max_seq_len = recipe.dataset.max_seq_len orelse 128;
    const max_completion_tokens = recipe.grpo.max_completion_tokens orelse 4;
    if (max_completion_tokens == 0 or max_completion_tokens >= max_seq_len) return error.NoCompletionBudget;
    const prompt_max_seq_len = max_seq_len - max_completion_tokens;
    for (rows.items, 0..) |row, idx| {
        const messages = try allocator.alloc(gemma_chat_data.Message, 1);
        errdefer allocator.free(messages);
        messages[0] = .{ .role = .user, .content = row.prompt };
        const example = gemma_chat_data.Example{
            .messages = messages,
            .image_paths = row.image_paths orelse &.{},
            .audio_paths = row.audio_paths orelse &.{},
        };
        const source = [_]gemma_chat_data.Example{example};
        summaries[idx] = try gemma4.prepareMultimodalInputsFromChatData(allocator, base_model_dir, projector_path, source[0..], 1, prompt_max_seq_len);
        allocator.free(messages);
        if (summaries[idx].examples.len == 0) return error.EmptyPrompt;
        prompts[idx] = &summaries[idx].examples[0];
        targets[idx] = try allocator.dupe(u8, row.target);
        built += 1;
    }
    arena.deinit();

    return .{
        .allocator = allocator,
        .prompts = prompts,
        .summaries = summaries,
        .targets = targets,
    };
}

fn tokenizeGrpoPrompt(
    allocator: std.mem.Allocator,
    model: *model_manager_mod.LoadedModel,
    recipe: Recipe,
    prompt: []const u8,
) ![]const i32 {
    const tokenizer = model.getTokenizer();
    const max_seq_len = recipe.dataset.max_seq_len orelse 128;
    const max_completion_tokens = recipe.grpo.max_completion_tokens orelse 4;
    if (max_completion_tokens == 0 or max_completion_tokens >= max_seq_len) return error.NoCompletionBudget;
    const prompt_max_seq_len = max_seq_len - max_completion_tokens;
    const render_prompt = !std.mem.eql(u8, recipe.dataset.format orelse "text-grpo", "rendered-text-grpo");
    const prompt_text = if (render_prompt)
        try renderDpoPrompt(allocator, model, prompt)
    else
        try allocator.dupe(u8, prompt);
    defer allocator.free(prompt_text);

    var encoded = try generation.encodePromptForGeneration(
        tokenizer,
        allocator,
        prompt_text,
        prompt_max_seq_len,
        model.manifest.add_bos_token,
        model.manifest.bos_token,
    );
    defer encoded.deinit();
    const prompt_len = countAttentionMask(encoded.attention_mask);
    if (prompt_len == 0) return error.EmptyPrompt;
    const out = try allocator.alloc(i32, prompt_len);
    for (0..prompt_len) |idx| out[idx] = encoded.ids[idx];
    return out;
}

fn countGrpoGroups(completions: []const grpo.Completion) usize {
    var max_prompt: usize = 0;
    var any = false;
    for (completions) |completion| {
        if (!any or completion.prompt_idx > max_prompt) max_prompt = completion.prompt_idx;
        any = true;
    }
    return if (any) max_prompt + 1 else 0;
}

fn selectRankedTokenFromLogits(allocator: std.mem.Allocator, logits: []const f32, rank: usize) !i32 {
    const Entry = struct {
        idx: usize,
        value: f32,
    };
    var entries = try allocator.alloc(Entry, logits.len);
    defer allocator.free(entries);
    for (logits, 0..) |value, idx| {
        entries[idx] = .{ .idx = idx, .value = value };
    }
    std.sort.heap(Entry, entries, {}, struct {
        fn lessThan(_: void, lhs: Entry, rhs: Entry) bool {
            return lhs.value > rhs.value;
        }
    }.lessThan);
    return @intCast(entries[@min(rank, entries.len - 1)].idx);
}

fn argv(allocator: std.mem.Allocator, values: []const []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, values);
    return out.toOwnedSlice(allocator);
}

fn appendMany(allocator: std.mem.Allocator, list: *std.ArrayList([]const u8), values: []const []const u8) !void {
    try list.appendSlice(allocator, values);
}

fn fmtInt(allocator: std.mem.Allocator, value: anytype) ![]const u8 {
    return std.fmt.allocPrint(allocator, "{d}", .{value});
}

fn fmtFloat(allocator: std.mem.Allocator, value: anytype) ![]const u8 {
    return std.fmt.allocPrint(allocator, "{d}", .{value});
}

fn appendEntityMinimum(
    allocator: std.mem.Allocator,
    list: *std.ArrayList([]const u8),
    flag: []const u8,
    threshold: f64,
) !void {
    if (!std.math.isFinite(threshold) or threshold < 0 or threshold > 1) return error.InvalidQualityThreshold;
    try appendMany(allocator, list, &.{ flag, try fmtFloat(allocator, threshold) });
}

fn appendFullTaskMinimum(
    allocator: std.mem.Allocator,
    list: *std.ArrayList([]const u8),
    metric: []const u8,
    threshold: f64,
) !void {
    if (!std.math.isFinite(threshold) or threshold <= 0 or threshold > 1) return error.InvalidFullTaskQualityThreshold;
    const value = try std.fmt.allocPrint(allocator, "{s}={d}", .{ metric, threshold });
    try appendMany(allocator, list, &.{ "--min-task-metric", value });
}

fn joinCsv(allocator: std.mem.Allocator, values: []const []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (values, 0..) |value, idx| {
        if (idx != 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, value);
    }
    return out.toOwnedSlice(allocator);
}

fn eqlName(a: []const u8, b: []const u8) bool {
    return std.ascii.eqlIgnoreCase(a, b);
}

fn eqlAny(value: []const u8, names: []const []const u8) bool {
    for (names) |name| if (eqlName(value, name)) return true;
    return false;
}

fn isQwen35Family(family: []const u8) bool {
    return eqlAny(family, &.{
        "qwen3_5",
        "qwen3.5",
        "qwen3-5",
        "qwen35",
        "qwen3_5_text",
        "qwen3.5-text",
        "qwen3-5-text",
        "qwen35_text",
        "chandra",
        "chandra-ocr",
    });
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    return std.ascii.indexOfIgnoreCase(haystack, needle) != null;
}

fn containsQwen35Signal(path: []const u8) bool {
    return containsIgnoreCase(path, "qwen3_5") or
        containsIgnoreCase(path, "qwen3.5") or
        containsIgnoreCase(path, "qwen3-5") or
        containsIgnoreCase(path, "qwen35") or
        containsIgnoreCase(path, "chandra");
}

pub fn freePlan(allocator: std.mem.Allocator, plan: Plan) void {
    freeSteps(allocator, plan.steps);
    allocator.free(plan.steps);
}

fn freeSteps(allocator: std.mem.Allocator, steps: []Step) void {
    for (steps) |step| allocator.free(step.argv);
}

fn installedPackageRoot(allocator: std.mem.Allocator, exe_dir: []const u8) ![]const u8 {
    if (std.mem.endsWith(u8, exe_dir, "/zig-out/bin")) {
        return allocator.dupe(u8, std.fs.path.dirname(std.fs.path.dirname(exe_dir).?).?);
    }
    return allocator.dupe(u8, ".");
}

fn readFileMax(allocator: std.mem.Allocator, io: std.Io, path: []const u8, max_bytes: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_bytes));
}

pub fn usage() void {
    print(
        \\usage: antfly inference finetune run <recipe.json> [--dry-run]
        \\       antfly inference finetune smoke-fast [--out-root <path>] [--case <name>]
        \\
        \\recipe kinds: sft, lora-sft, qlora-sft, dpo, grpo, reranker, vlm-retrieval
        \\common fields: model, dataset, adapter, optimizer, eval, artifacts
        \\
    , .{});
}

fn usageError() error{InvalidArguments} {
    usage();
    return error.InvalidArguments;
}

test "recipe kind accepts taxonomy spellings" {
    try std.testing.expectEqual(RecipeKind.lora_sft, try parseKind("lora-sft"));
    try std.testing.expectEqual(RecipeKind.qlora_sft, try parseKind("qlora_sft"));
    try std.testing.expectEqual(RecipeKind.vlm_retrieval, try parseKind("vlm-retrieval"));
}

test "gemma4 recipe loading rejects unknown fields" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "recipe.json",
        .data =
        \\{"recipe":"lora-sft","model":{"path":"mystery","familly":"gemma4"}}
        ,
    });
    const path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "recipe.json" });
    defer allocator.free(path);
    try std.testing.expectError(error.UnknownField, loadRecipe(allocator, std.testing.io, path));
}

test "gemma4 DPO benchmark recorder enforces and summarizes the fixed protocol" {
    var recorder = try DpoBenchmarkRecorder.init(std.testing.allocator);
    defer recorder.deinit();

    for (0..DpoBenchmarkRecorder.total_updates) |idx| {
        try recorder.record(
            @floatFromInt(idx + 1),
            @as(f32, @floatFromInt(idx)) / 10.0,
        );
    }
    const telemetry = try recorder.finish();
    try std.testing.expectEqual(@as(usize, 1), telemetry.protocol.cold);
    try std.testing.expectEqual(@as(usize, 1), telemetry.protocol.first);
    try std.testing.expectEqual(@as(usize, 3), telemetry.protocol.warmup);
    try std.testing.expectEqual(@as(usize, 20), telemetry.protocol.measured);
    try std.testing.expectEqual(@as(f64, 1.0), telemetry.cold_seconds);
    try std.testing.expectEqual(@as(f64, 2.0), telemetry.first_seconds);
    try std.testing.expectEqual(@as(f64, 15.5), telemetry.median_seconds);
    try std.testing.expectEqual(@as(f64, 15.5), telemetry.mean_seconds);
    try std.testing.expectError(error.DpoBenchmarkUpdateCountMismatch, recorder.record(26.0, 2.5));

    var incomplete = try DpoBenchmarkRecorder.init(std.testing.allocator);
    defer incomplete.deinit();
    try incomplete.record(1.0, 0.6931472);
    try std.testing.expectError(error.DpoBenchmarkUpdateCountMismatch, incomplete.finish());

    var stagnant = try DpoBenchmarkRecorder.init(std.testing.allocator);
    defer stagnant.deinit();
    for (0..DpoBenchmarkRecorder.total_updates) |idx| {
        try stagnant.record(@floatFromInt(idx + 1), 0.6931472);
    }
    try std.testing.expectError(error.DpoBenchmarkNoPolicyMovement, stagnant.finish());
}

test "gemma4 GRPO benchmark recorder enforces protocol and policy movement" {
    var recorder = try GrpoBenchmarkRecorder.init(std.testing.allocator);
    defer recorder.deinit();

    for (0..GrpoBenchmarkRecorder.total_updates) |idx| {
        try recorder.record(.{
            .seconds = @floatFromInt(idx + 1),
            .loss = @as(f32, @floatFromInt(idx)) / 100.0,
            .pg_loss = 0.0,
            .kl_loss = @as(f32, @floatFromInt(idx)) / 100.0,
            .mean_reward = 0.5,
            .reward_stddev = 0.5,
            .completion_tokens = 2,
            .policy_reference_max_abs_error = if (idx == 0) 0.0 else 0.01,
        });
    }
    const telemetry = try recorder.finish();
    try std.testing.expectEqual(@as(usize, 1), telemetry.protocol.cold);
    try std.testing.expectEqual(@as(usize, 1), telemetry.protocol.first);
    try std.testing.expectEqual(@as(usize, 3), telemetry.protocol.warmup);
    try std.testing.expectEqual(@as(usize, 20), telemetry.protocol.measured);
    try std.testing.expectEqual(@as(f64, 1.0), telemetry.cold.seconds);
    try std.testing.expectEqual(@as(f64, 2.0), telemetry.first.seconds);
    try std.testing.expectEqual(@as(f64, 15.5), telemetry.median_seconds);
    try std.testing.expectEqual(@as(f64, 15.5), telemetry.mean_seconds);

    var stagnant = try GrpoBenchmarkRecorder.init(std.testing.allocator);
    defer stagnant.deinit();
    for (0..GrpoBenchmarkRecorder.total_updates) |idx| {
        try stagnant.record(.{
            .seconds = @floatFromInt(idx + 1),
            .loss = 0.0,
            .pg_loss = 0.0,
            .kl_loss = 0.0,
            .mean_reward = 0.5,
            .reward_stddev = 0.5,
            .completion_tokens = 2,
            .policy_reference_max_abs_error = 0.0,
        });
    }
    try std.testing.expectError(error.GrpoBenchmarkNoPolicyMovement, stagnant.finish());
}

test "gemma4 GRPO reference cache uses exact keys and bounded eviction" {
    var cache = GemmaGrpoReferenceCache.init(std.testing.allocator, 2);
    defer cache.deinit();

    var output: [2]f32 = undefined;
    try std.testing.expect(!cache.contains(0, &.{ 10, 11 }));
    try std.testing.expectEqual(@as(usize, 0), cache.telemetry().hits);
    try std.testing.expectEqual(@as(usize, 0), cache.telemetry().misses);
    try std.testing.expect(!try cache.lookup(0, &.{ 10, 11 }, &output));
    try cache.insert(0, &.{ 10, 11 }, &.{ -0.1, -0.2 });
    try std.testing.expect(cache.contains(0, &.{ 10, 11 }));
    try std.testing.expectEqual(@as(usize, 0), cache.telemetry().hits);
    try std.testing.expectEqual(@as(usize, 1), cache.telemetry().misses);
    try std.testing.expect(try cache.lookup(0, &.{ 10, 11 }, &output));
    try std.testing.expectEqualSlices(f32, &.{ -0.1, -0.2 }, &output);

    try cache.insert(0, &.{12}, &.{-0.3});
    try cache.insert(1, &.{13}, &.{-0.4});
    try std.testing.expect(!try cache.lookup(0, &.{ 10, 11 }, &output));
    const telemetry = cache.telemetry();
    try std.testing.expectEqual(@as(usize, 2), telemetry.capacity);
    try std.testing.expectEqual(@as(usize, 2), telemetry.entries);
    try std.testing.expectEqual(@as(usize, 1), telemetry.hits);
    try std.testing.expectEqual(@as(usize, 2), telemetry.misses);
}

test "family inference keeps qwen3_5 and colqwen distinct from qwen2" {
    try std.testing.expectEqualStrings("qwen3_5", inferFamilyFromModelPath("/models/datalab-to/chandra-ocr-2").?);
    try std.testing.expectEqualStrings("qwen3_5", inferFamilyFromModelPath("/models/Qwen3.5-VL").?);
    try std.testing.expectEqualStrings("colqwen2", inferFamilyFromModelPath("/models/vidore/colqwen2-v1.0-hf").?);
    try std.testing.expectEqualStrings("qwen2", inferFamilyFromModelPath("/models/Qwen2-0.5B").?);
}

test "qwen3_5 text preference recipes route to qwen autodiff planner" {
    const sft = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/chandra-ocr-2" },
        .dataset = .{ .path = "/data/train.jsonl" },
        .adapter = .{ .rank = 8, .alpha = 16 },
        .artifacts = .{ .root = "/tmp/qwen35-sft" },
    };
    const sft_plan = try buildPlan(std.heap.page_allocator, sft);
    defer freePlan(std.heap.page_allocator, sft_plan);
    try std.testing.expectEqual(StepKind.direct_sft, sft_plan.steps[0].kind);
    try std.testing.expect(try shouldRunOptimizerBackedQwen35Sft(sft, "text-sft"));

    const dpo = Recipe{
        .recipe = "dpo",
        .execution = .{ .mode = "train" },
        .model = .{ .path = "/models/Qwen3.5-VL" },
        .dataset = .{ .path = "/data/prefs.jsonl", .format = "text-preference" },
        .adapter = .{ .rank = 8, .alpha = 16 },
        .artifacts = .{ .root = "/tmp/qwen35-dpo" },
    };
    const dpo_plan = try buildPlan(std.heap.page_allocator, dpo);
    defer freePlan(std.heap.page_allocator, dpo_plan);
    try std.testing.expectEqual(StepKind.direct_dpo, dpo_plan.steps[0].kind);
    try std.testing.expect(try shouldRunOptimizerBackedQwen2Dpo(dpo, "text-preference"));

    var report_only = dpo;
    report_only.adapter = null;
    report_only.execution.mode = "score";
    report_only.artifacts = .{ .root = "/tmp/qwen35-report" };
    const plan = try buildPlan(std.heap.page_allocator, report_only);
    defer freePlan(std.heap.page_allocator, plan);
    try std.testing.expectEqual(StepKind.direct_dpo, plan.steps[0].kind);
    try std.testing.expect(!try shouldRunOptimizerBackedQwen2Dpo(report_only, "text-preference"));

    const grpo_recipe = Recipe{
        .recipe = "grpo",
        .execution = .{ .mode = "train" },
        .model = .{ .path = "/models/Qwen3.5-VL" },
        .dataset = .{ .path = "/data/prompts.jsonl", .format = "text-grpo" },
        .adapter = .{ .rank = 8, .alpha = 16 },
        .artifacts = .{ .root = "/tmp/qwen35-grpo" },
    };
    const grpo_plan = try buildPlan(std.heap.page_allocator, grpo_recipe);
    defer freePlan(std.heap.page_allocator, grpo_plan);
    try std.testing.expectEqual(StepKind.direct_grpo, grpo_plan.steps[0].kind);
    try std.testing.expect(try shouldRunOptimizerBackedQwen2Grpo(grpo_recipe, "text-grpo"));
}

test "gemma4 lora recipe builds disjoint prepare bootstrap train plan" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl", .eval_path = "/data/eval.jsonl", .max_examples = 4 },
        .adapter = .{ .rank = 4, .alpha = 8 },
        .optimizer = .{ .learning_rate = 0.0002, .epochs = 2 },
        .artifacts = .{ .root = "/tmp/out" },
        .backend = "native",
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try std.testing.expectEqual(@as(usize, 4), plan.steps.len);
    try std.testing.expectEqualStrings("prepare-gemma4-lora-inputs", plan.steps[0].argv[0]);
    try std.testing.expectEqualStrings("prepare-gemma4-lora-inputs", plan.steps[1].argv[0]);
    try std.testing.expectEqualStrings("/data/eval.jsonl", plan.steps[1].argv[2]);
    try std.testing.expectEqualStrings("bootstrap-gemma4-lora", plan.steps[2].argv[0]);
    try std.testing.expectEqualStrings("train-eval-gemma4-lora-bundle", plan.steps[3].argv[0]);
    try expectArgValue(plan.steps[3].argv, "--eval-prepared", plan.steps[1].argv[4]);
}

test "gemma4 lora recipe requires a held-out evaluation dataset" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl" },
        .artifacts = .{ .root = "/tmp/out" },
        .backend = "native",
    };
    try std.testing.expectError(error.MissingEvaluationDataset, buildPlan(std.heap.page_allocator, recipe));
}

test "gemma4 lora recipe defaults to text-all-linear rank16 alpha32" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl", .eval_path = "/data/eval.jsonl" },
        .artifacts = .{ .root = "/tmp/out" },
        .backend = "native",
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try std.testing.expectEqualStrings("16", plan.steps[2].argv[3]);
    try std.testing.expectEqualStrings("32", plan.steps[2].argv[4]);
    try std.testing.expectEqualStrings("--target-preset", plan.steps[2].argv[5]);
    try std.testing.expectEqualStrings("text-all-linear", plan.steps[2].argv[6]);
}

test "gemma4 lora recipe wires the supported heldout example limit" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl", .eval_path = "/data/eval.jsonl", .eval_max_examples = 3 },
        .artifacts = .{ .root = "/tmp/out" },
        .backend = "native",
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try expectArgValue(plan.steps[3].argv, "--eval-max-examples", "3");
}

test "gemma4 lora recipe wires opt-in independent row length buckets" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl", .eval_path = "/data/eval.jsonl", .max_seq_len = 512 },
        .runtime = .{
            .sequence_length_bucket_quantum = 16,
            .sequence_length_bucket_min = 32,
            .graph_cache_capacity = 4,
        },
        .artifacts = .{ .root = "/tmp/out" },
        .backend = "metal",
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try expectArgValue(plan.steps[3].argv, "--sequence-length-bucket-quantum", "16");
    try expectArgValue(plan.steps[3].argv, "--sequence-length-bucket-min", "32");
    try expectArgValue(plan.steps[3].argv, "--graph-cache-capacity", "4");
}

test "gemma4 bootstrap presets execute through runPlan" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const model_dir = try std.fs.path.join(allocator, &.{ root, "model" });
    defer allocator.free(model_dir);
    try std.Io.Dir.cwd().createDirPath(io, model_dir);

    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);
    try writeTextFile(io, config_path,
        \\{"model_type":"gemma4","text_config":{"hidden_size":3,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":3,"intermediate_size":4,"vocab_size":4}}
    );

    const checkpoint_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(checkpoint_path);
    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &.{ 2, 3 }, .data = try makeFilledF32(allocator, 6, 0.1) },
        .{ .name = "model.layers.0.self_attn.k_proj.weight", .shape = &.{ 2, 3 }, .data = try makeFilledF32(allocator, 6, 0.2) },
        .{ .name = "model.layers.0.self_attn.v_proj.weight", .shape = &.{ 2, 3 }, .data = try makeFilledF32(allocator, 6, 0.3) },
        .{ .name = "model.layers.0.self_attn.o_proj.weight", .shape = &.{ 2, 3 }, .data = try makeFilledF32(allocator, 6, 0.4) },
        .{ .name = "model.layers.0.mlp.gate_proj.weight", .shape = &.{ 2, 3 }, .data = try makeFilledF32(allocator, 6, 0.5) },
        .{ .name = "model.layers.0.mlp.up_proj.weight", .shape = &.{ 2, 3 }, .data = try makeFilledF32(allocator, 6, 0.6) },
        .{ .name = "model.layers.0.mlp.down_proj.weight", .shape = &.{ 2, 3 }, .data = try makeFilledF32(allocator, 6, 0.7) },
    });

    const dataset_path = try std.fs.path.join(allocator, &.{ root, "train.jsonl" });
    defer allocator.free(dataset_path);
    try writeTextFile(io, dataset_path, "{}\n");

    const cases = [_]struct {
        name: []const u8,
        requested_preset: ?[]const u8,
        expected_preset: []const u8,
        expected_target_count: usize,
    }{
        .{ .name = "default", .requested_preset = null, .expected_preset = "text-all-linear", .expected_target_count = 7 },
        .{ .name = "peft-qv", .requested_preset = "peft-qv", .expected_preset = "peft-qv", .expected_target_count = 2 },
    };

    for (cases) |case| {
        const case_root = try std.fs.path.join(allocator, &.{ root, case.name });
        defer allocator.free(case_root);
        const recipe = Recipe{
            .recipe = "lora-sft",
            .model = .{ .path = model_dir, .family = "gemma4" },
            .dataset = .{ .path = dataset_path, .eval_path = dataset_path },
            .adapter = .{ .rank = 1, .alpha = 2, .target_preset = case.requested_preset },
            .artifacts = .{ .root = case_root },
            .backend = "native",
        };

        var plan_arena = std.heap.ArenaAllocator.init(allocator);
        defer plan_arena.deinit();
        const generated_plan = try buildPlan(plan_arena.allocator(), recipe);
        const bootstrap_plan = Plan{ .steps = generated_plan.steps[2..3] };
        try std.testing.expectEqualStrings("--target-preset", bootstrap_plan.steps[0].argv[5]);
        try std.testing.expectEqualStrings(case.expected_preset, bootstrap_plan.steps[0].argv[6]);

        const manifest_path = try manifestPath(allocator, recipe);
        defer allocator.free(manifest_path);
        const training_config_path = try defaultArtifactPath(allocator, recipe, "training_config.json");
        defer allocator.free(training_config_path);
        const training_report_path = try defaultArtifactPath(allocator, recipe, "training_report.json");
        defer allocator.free(training_report_path);
        try runPlan(allocator, io, ".", recipe, bootstrap_plan, manifest_path, training_config_path, training_report_path);
        try expectRunStatusFile(allocator, io, manifest_path, "succeeded");

        const adapter_manifest_path = try std.fs.path.join(allocator, &.{ bootstrap_plan.steps[0].argv[2], gemma4.adapter_manifest_file_name });
        defer allocator.free(adapter_manifest_path);
        const raw = try readFileMax(allocator, io, adapter_manifest_path, 1024 * 1024);
        defer allocator.free(raw);
        const parsed = try std.json.parseFromSlice(struct {
            target_preset: []const u8,
            target_modules: []const []const u8,
        }, allocator, raw, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
        defer parsed.deinit();
        try std.testing.expectEqualStrings(case.expected_preset, parsed.value.target_preset);
        try std.testing.expectEqual(case.expected_target_count, parsed.value.target_modules.len);
    }
}

test "gemma4 lora recipe passes supported explicit adapter knobs" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl", .eval_path = "/data/eval.jsonl" },
        .adapter = .{
            .target_modules = &.{ "q_proj", "v_proj" },
            .init_lora_weights = "default",
        },
        .artifacts = .{ .root = "/tmp/out" },
        .backend = "metal",
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try std.testing.expectEqualStrings("--target-modules", plan.steps[2].argv[5]);
    try std.testing.expectEqualStrings("q_proj,v_proj", plan.steps[2].argv[6]);
    try std.testing.expectEqualStrings("--init-lora-weights", plan.steps[2].argv[7]);
    try std.testing.expectEqualStrings("default", plan.steps[2].argv[8]);
}

test "gemma4 recipe kinds fail closed when the requested training semantics are unavailable" {
    const base = Recipe{
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl" },
        .artifacts = .{ .root = "/tmp/out" },
        .backend = "native",
    };
    var full_sft = base;
    full_sft.recipe = "sft";
    try std.testing.expectError(error.Gemma4FullSftNotYetSupported, buildPlan(std.heap.page_allocator, full_sft));

    var qlora = base;
    qlora.recipe = "qlora-sft";
    try std.testing.expectError(error.Gemma4QLoRANotYetSupported, buildPlan(std.heap.page_allocator, qlora));
}

test "gemma4 lora recipe rejects options the trainer cannot honor" {
    const base = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl", .eval_path = "/data/eval.jsonl" },
        .artifacts = .{ .root = "/tmp/out" },
        .backend = "native",
    };

    var recipe = base;
    recipe.model.reference_path = "/models/reference";
    try std.testing.expectError(error.UnsupportedGemma4ModelOption, buildPlan(std.heap.page_allocator, recipe));

    recipe = base;
    recipe.model.projector_path = "/models/projector.gguf";
    try std.testing.expectError(error.Gemma4MultimodalFinetuningNotSupported, buildPlan(std.heap.page_allocator, recipe));

    recipe = base;
    recipe.dataset.format = "messages";
    try std.testing.expectError(error.UnsupportedGemma4DatasetOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.dataset.cache_path = "/tmp/cache";
    try std.testing.expectError(error.UnsupportedGemma4DatasetOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.dataset.train_cache_path = "/tmp/train-cache";
    try std.testing.expectError(error.UnsupportedGemma4DatasetOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.dataset.labels = "labels";
    try std.testing.expectError(error.UnsupportedGemma4DatasetOption, buildPlan(std.heap.page_allocator, recipe));

    recipe = base;
    recipe.adapter = .{ .dropout = 0.05 };
    try std.testing.expectError(error.UnsupportedGemma4AdapterOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.adapter = .{ .layer_name = "model.layers.0" };
    try std.testing.expectError(error.UnsupportedGemma4AdapterOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.adapter = .{ .quantization = "nf4" };
    try std.testing.expectError(error.UnsupportedGemma4AdapterOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.adapter = .{ .use_dora = true };
    try std.testing.expectError(error.DoRAAutodiffNotYetSupported, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.adapter = .{ .init_lora_weights = "pissa" };
    try std.testing.expectError(error.LoRAInitializerRequiresAdjustedBase, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.adapter = .{ .init_lora_weights = "eva" };
    try std.testing.expectError(error.Gemma4RecipeInitializerStatsNotYetSupported, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.adapter = .{ .init_lora_weights = "lora-ga" };
    try std.testing.expectError(error.Gemma4RecipeInitializerStatsNotYetSupported, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.adapter = .{ .rank = 0 };
    try std.testing.expectError(error.InvalidLoRARank, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.adapter = .{ .alpha = 0 };
    try std.testing.expectError(error.InvalidLoRAAlpha, buildPlan(std.heap.page_allocator, recipe));

    recipe = base;
    recipe.optimizer.weight_decay = 0;
    try std.testing.expectError(error.UnsupportedGemma4OptimizerOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.optimizer.lr_scheduler = "cosine";
    try std.testing.expectError(error.UnsupportedGemma4OptimizerOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.optimizer.warmup_steps = 10;
    try std.testing.expectError(error.UnsupportedGemma4OptimizerOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.optimizer.warmup_ratio = 0.1;
    try std.testing.expectError(error.UnsupportedGemma4OptimizerOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.optimizer.num_cycles = 1;
    try std.testing.expectError(error.UnsupportedGemma4OptimizerOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.optimizer.max_steps = 100;
    try std.testing.expectError(error.UnsupportedGemma4OptimizerOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.optimizer.micro_batch_size = 2;
    try std.testing.expectError(error.UnsupportedGemma4OptimizerOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.optimizer.llrd_decay = 0.9;
    try std.testing.expectError(error.UnsupportedGemma4OptimizerOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.optimizer.schedule_free = true;
    try std.testing.expectError(error.UnsupportedGemma4OptimizerOption, buildPlan(std.heap.page_allocator, recipe));

    recipe = base;
    recipe.eval = .{ .every_epochs = 1 };
    try std.testing.expectError(error.UnsupportedGemma4EvalOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.eval = .{ .batch_size = 2 };
    try std.testing.expectError(error.UnsupportedGemma4EvalOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.eval = .{ .early_stopping_patience = 2 };
    try std.testing.expectError(error.UnsupportedGemma4EvalOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.eval = .{ .improvement_threshold = 0.01 };
    try std.testing.expectError(error.UnsupportedGemma4EvalOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.eval = .{ .backend = "metal" };
    try std.testing.expectError(error.UnsupportedGemma4EvalOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.checkpoint = .{ .resume_path = " \t" };
    try std.testing.expectError(error.InvalidGemma4CheckpointPath, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.checkpoint = .{ .every_epochs = 0 };
    try std.testing.expectError(error.InvalidGemma4CheckpointInterval, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.checkpoint = .{ .every_epochs = 2 };
    try std.testing.expectError(error.InvalidGemma4CheckpointInterval, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.checkpoint = .{ .keep_last = 2 };
    try std.testing.expectError(error.UnsupportedGemma4CheckpointOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.runtime = .{ .compiled_required = true };
    try std.testing.expectError(error.UnsupportedGemma4RuntimeOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.runtime = .{ .graph_cache_capacity = 4 };
    try std.testing.expectError(error.Gemma4SequenceLengthBucketQuantumRequired, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.runtime = .{ .sequence_length_bucket_quantum = 16, .graph_cache_capacity = 9 };
    try std.testing.expectError(error.InvalidGemma4GraphCacheCapacity, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.trainer = "surrogate";
    try std.testing.expectError(error.UnsupportedGemma4Trainer, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.preference.beta = 0.1;
    try std.testing.expectError(error.UnsupportedGemma4AlgorithmOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.grpo.group_size = 4;
    try std.testing.expectError(error.UnsupportedGemma4AlgorithmOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.artifacts.materialized_dir = "/tmp/merged";
    try std.testing.expectError(error.UnsupportedGemma4ArtifactOption, buildPlan(std.heap.page_allocator, recipe));
    recipe = base;
    recipe.artifacts.report_path = "/tmp/report.json";
    try std.testing.expectError(error.UnsupportedGemma4ArtifactOption, buildPlan(std.heap.page_allocator, recipe));
}

test "gemma4 lora recipe wires bounded atomic checkpoint and resume controls" {
    const base = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl", .eval_path = "/data/eval.jsonl" },
        .optimizer = .{ .epochs = 3 },
        .artifacts = .{ .root = "/tmp/out" },
        .backend = "native",
    };

    var fresh = base;
    fresh.checkpoint = .{ .every_epochs = 2 };
    const fresh_plan = try buildPlan(std.heap.page_allocator, fresh);
    defer freePlan(std.heap.page_allocator, fresh_plan);
    const fresh_train = fresh_plan.steps[3].argv;
    try expectArgValue(fresh_train, "--checkpoint-path", "/tmp/out/gemma4_trainer_state.safetensors");
    try expectArgValue(fresh_train, "--checkpoint-every-epochs", "2");

    var resumed = base;
    resumed.checkpoint = .{ .every_epochs = 1, .resume_path = "/runs/interrupted/gemma4-state.safetensors" };
    const resumed_plan = try buildPlan(std.heap.page_allocator, resumed);
    defer freePlan(std.heap.page_allocator, resumed_plan);
    const resumed_train = resumed_plan.steps[3].argv;
    try expectArgValue(resumed_train, "--checkpoint-path", "/runs/interrupted/gemma4-state.safetensors");
    try expectArgValue(resumed_train, "--checkpoint-every-epochs", "1");
    try expectArgPresent(resumed_train, "--resume");
}

test "gemma4 recipe keeps bootstrap and immutable training outputs distinct" {
    const base = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl", .eval_path = "/data/eval.jsonl" },
        .backend = "native",
    };

    var artifact_only = base;
    artifact_only.artifacts = .{ .root = "/tmp/out", .adapter_dir = "/tmp/out/final-adapter" };
    const plan = try buildPlan(std.heap.page_allocator, artifact_only);
    defer freePlan(std.heap.page_allocator, plan);
    try std.testing.expectEqualStrings("/tmp/out/adapter-bootstrap", plan.steps[2].argv[2]);
    try std.testing.expectEqualStrings("/tmp/out/final-adapter", plan.steps[3].argv[4]);

    var conflict = base;
    conflict.adapter = .{ .path = "/tmp/out/same" };
    conflict.artifacts = .{ .trained_adapter_dir = "/tmp/out/same" };
    try std.testing.expectError(
        error.Gemma4BootstrapAndTrainingOutputConflict,
        buildPlan(std.heap.page_allocator, conflict),
    );

    var normalized_conflict = base;
    normalized_conflict.adapter = .{ .path = "/tmp/out/seed/../same" };
    normalized_conflict.artifacts = .{ .trained_adapter_dir = "/tmp/out/same" };
    try std.testing.expectError(
        error.Gemma4BootstrapAndTrainingOutputConflict,
        buildPlan(std.heap.page_allocator, normalized_conflict),
    );

    var ancestor_conflict = base;
    ancestor_conflict.adapter = .{ .path = "/tmp/out/seed" };
    ancestor_conflict.artifacts = .{ .root = "/tmp/out", .trained_adapter_dir = "/tmp/out" };
    try std.testing.expectError(
        error.Gemma4BootstrapAndTrainingOutputConflict,
        buildPlan(std.heap.page_allocator, ancestor_conflict),
    );

    var root_conflict = base;
    root_conflict.adapter = .{ .path = "/tmp/seed-outside" };
    root_conflict.artifacts = .{ .root = "/tmp/out", .trained_adapter_dir = "/tmp/out" };
    try std.testing.expectError(
        error.Gemma4OutputConflictsWithArtifactRoot,
        buildPlan(std.heap.page_allocator, root_conflict),
    );

    var prepared_conflict = base;
    prepared_conflict.adapter = .{ .path = "/tmp/seed-outside" };
    prepared_conflict.artifacts = .{
        .prepared_path = "/tmp/final/prepared.json",
        .trained_adapter_dir = "/tmp/final",
    };
    try std.testing.expectError(
        error.Gemma4OutputContainsPlannedArtifact,
        buildPlan(std.heap.page_allocator, prepared_conflict),
    );

    var manifest_conflict = base;
    manifest_conflict.adapter = .{ .path = "/tmp/seed-outside" };
    manifest_conflict.artifacts = .{
        .manifest_path = "/tmp/final/manifest.json",
        .trained_adapter_dir = "/tmp/final",
    };
    try std.testing.expectError(
        error.Gemma4OutputContainsPlannedArtifact,
        buildPlan(std.heap.page_allocator, manifest_conflict),
    );
}

test "gemma4 preference preflight rejects output through symlinked input ancestor" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDir(io, "immutable-model", .default_dir);
    try tmp.dir.symLink(io, "immutable-model", "output-alias", .{});
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const model_path = try std.fs.path.join(allocator, &.{ root, "immutable-model" });
    defer allocator.free(model_path);
    const train_path = try std.fs.path.join(allocator, &.{ root, "train.jsonl" });
    defer allocator.free(train_path);
    const eval_path = try std.fs.path.join(allocator, &.{ root, "eval.jsonl" });
    defer allocator.free(eval_path);
    const output_root = try std.fs.path.join(allocator, &.{ root, "output-alias", "subroot" });
    defer allocator.free(output_root);

    const recipe = Recipe{
        .recipe = "dpo",
        .execution = .{ .mode = "train" },
        .model = .{ .path = model_path, .reference_path = model_path, .family = "gemma4" },
        .dataset = .{ .path = train_path, .eval_path = eval_path, .format = "text-preference", .max_seq_len = 128 },
        .adapter = .{ .rank = 8, .alpha = 16 },
        .optimizer = .{ .learning_rate = 0.0001, .epochs = 1 },
        .eval = .{ .dpo_minimums = .{ .accuracy = 0.5, .max_loss = 2.0 } },
        .artifacts = .{ .root = output_root },
        .backend = "native",
    };
    try std.testing.expectError(
        error.PreferenceArtifactInputConflict,
        buildPlan(allocator, recipe),
    );
    try std.testing.expectError(
        error.FileNotFound,
        tmp.dir.statFile(io, "immutable-model/subroot", .{}),
    );
}

test "gemma4 lora recipe rejects conflicting target selectors" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl", .eval_path = "/data/eval.jsonl" },
        .adapter = .{
            .target_preset = "all-linear",
            .target_modules = &.{"q_proj"},
        },
        .artifacts = .{ .root = "/tmp/out" },
    };
    try std.testing.expectError(error.ConflictingLoRATargetSelection, buildPlan(std.heap.page_allocator, recipe));
}

test "gemma4 lora recipe requires an explicit execution backend" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl", .eval_path = "/data/eval.jsonl" },
        .artifacts = .{ .root = "/tmp/out" },
    };
    try std.testing.expectError(error.MissingBackend, buildPlan(std.heap.page_allocator, recipe));
}

test "qwen adapter target presets map to supported module sets" {
    const all_linear = try adapterTargetModulesForQwen(.{ .target_preset = "all-linear" }, qwen2_real_autodiff.default_lora_target_modules[0..]);
    try std.testing.expectEqual(qwen2_real_autodiff.default_lora_target_modules.len, all_linear.len);
    const attention = try adapterTargetModulesForQwen(.{ .target_preset = "attention-only" }, qwen2_real_autodiff.default_lora_target_modules[0..]);
    try std.testing.expectEqualStrings("q_proj", attention[0]);
    try std.testing.expectEqualStrings("o_proj", attention[3]);
    try std.testing.expectError(error.ConflictingLoRATargetSelection, adapterTargetModulesForQwen(.{ .target_preset = "all-linear", .target_modules = &.{"q_proj"} }, qwen2_real_autodiff.default_lora_target_modules[0..]));
    try std.testing.expectError(error.UnsupportedLoRATargetPreset, adapterTargetModulesForQwen(.{ .target_preset = "moe-experts" }, qwen2_real_autodiff.default_lora_target_modules[0..]));
}

test "generic bootstrap families reject unsupported target preset" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/layoutlmv3", .family = "layoutlmv3" },
        .dataset = .{
            .train_path = "/data/layout-train.jsonl",
        },
        .adapter = .{ .target_preset = "all-linear" },
        .artifacts = .{ .root = "/tmp/layout-run" },
    };
    try std.testing.expectError(error.UnsupportedLoRATargetPreset, buildPlan(std.heap.page_allocator, recipe));
}

test "gliner2 lora recipe routes to autodiff trainer" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gliner2", .family = "gliner2" },
        .dataset = .{ .train_path = "/data/ner-train.jsonl", .max_examples = 8 },
        .optimizer = .{
            .learning_rate = 5e-4,
            .weight_decay = 0.01,
            .lr_scheduler = "cosine_restarts",
            .warmup_ratio = 0.1,
            .warmup_steps = 3,
            .num_cycles = 2,
            .max_steps = 10,
            .epochs = 2,
        },
        .adapter = .{ .rank = 16, .alpha = 32, .dropout = 0.05 },
        .artifacts = .{ .root = "/tmp/gliner2-run", .trained_adapter_dir = "/tmp/gliner2-adapter" },
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try expectStepCommands(plan, &.{ "train-gliner2-autodiff", "validate-gliner2-autodiff-run" });
    const argv_items = plan.steps[0].argv;
    try std.testing.expectEqualStrings("--model-dir", argv_items[1]);
    try std.testing.expectEqualStrings("/models/gliner2", argv_items[2]);
    try std.testing.expectEqualStrings("--train-data", argv_items[3]);
    try std.testing.expectEqualStrings("/data/ner-train.jsonl", argv_items[4]);
    try std.testing.expectEqualStrings("--out-dir", argv_items[5]);
    try std.testing.expectEqualStrings("/tmp/gliner2-adapter", argv_items[6]);
    try std.testing.expectEqualStrings("--objective", argv_items[7]);
    try std.testing.expectEqualStrings("gliner2-total-loss", argv_items[8]);
    try std.testing.expectEqualStrings("--lora-only-trainables", argv_items[9]);
    try expectArgValue(argv_items, "--weight-decay", "0.01");
    try expectArgValue(argv_items, "--lr-scheduler", "cosine_restarts");
    try expectArgValue(argv_items, "--warmup-ratio", "0.1");
    try expectArgValue(argv_items, "--warmup-steps", "3");
    try expectArgValue(argv_items, "--num-cycles", "2");
    try expectArgValue(argv_items, "--max-steps", "10");
    try expectArgValue(argv_items, "--lora-dropout", "0.05");
}

test "gliner2 production recipe wires lifecycle and runtime controls" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gliner2", .family = "gliner2" },
        .dataset = .{
            .train_path = "/data/train.jsonl",
            .labels = "person,organization",
            .max_seq_len = 128,
            .eval_max_examples = 20,
        },
        .optimizer = .{ .gradient_accumulation_steps = 4 },
        .eval = .{
            .path = "/data/eval.jsonl",
            .every_epochs = 2,
            .batch_size = 8,
            .early_stopping_patience = 3,
            .improvement_threshold = 0.001,
            .entity_minimums = .{
                .precision = 0.5,
                .recall = 0.4,
                .f1 = 0.45,
                .exact_match = 0.25,
            },
            .full_task_minimums = .{
                .classifications_micro_f1 = 0.6,
                .classifications_exact_match = 0.5,
                .json_structures_micro_f1 = 0.55,
                .json_structures_exact_match = 0.4,
                .relations_micro_f1 = 0.5,
                .relations_exact_match = 0.35,
                .count_accuracy = 0.7,
            },
        },
        .checkpoint = .{
            .every_epochs = 2,
            .keep_last = 4,
            .resume_path = "/runs/prior/checkpoints/epoch-4.safetensors",
        },
        .runtime = .{
            .compiled_required = true,
            .graph_cache_capacity = 4,
        },
        .backend = "metal",
        .artifacts = .{
            .root = "/runs/gliner2",
            .trained_adapter_dir = "/runs/gliner2/adapter",
            .materialized_dir = "/runs/gliner2/model",
            .validation_report_path = "/runs/gliner2/validation.json",
            .evaluation_report_path = "/runs/gliner2/eval.json",
            .reload_report_path = "/runs/gliner2/reload.json",
        },
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try expectStepCommands(plan, &.{
        "train-gliner2-autodiff",
        "validate-gliner2-autodiff-run",
        "eval-gliner2-autodiff-adapter-dataset",
        "materialize-gliner2-lora",
        "inspect-gliner2-checkpoint",
    });
    const train_args = plan.steps[0].argv;
    try expectArgValue(train_args, "--eval-data", "/data/eval.jsonl");
    try expectArgValue(train_args, "--eval-every-epochs", "2");
    try expectArgValue(train_args, "--eval-batch-size", "8");
    try expectArgValue(train_args, "--early-stopping-patience", "3");
    try expectArgValue(train_args, "--early-stopping-threshold", "0.001");
    try expectArgValue(train_args, "--checkpoint-every-epochs", "2");
    try expectArgValue(train_args, "--checkpoint-keep-last", "4");
    try expectArgValue(train_args, "--resume-checkpoint", "/runs/prior/checkpoints/epoch-4.safetensors");
    try expectArgValue(train_args, "--graph-cache-capacity", "4");
    try expectArgValue(train_args, "--seq-len", "128");
    try expectArgValue(train_args, "--backend", "metal");
    try expectArgPresent(train_args, "--compiled-required");
    try std.testing.expectEqualStrings("/runs/gliner2/validation.json", plan.steps[1].argv[3]);
    try std.testing.expectEqualStrings("/runs/gliner2/eval.json", plan.steps[2].argv[8]);
    const eval_args = plan.steps[2].argv;
    try expectArgValue(eval_args, "--backend", "native");
    try expectArgValue(eval_args, "--min-precision", "0.5");
    try expectArgValue(eval_args, "--min-recall", "0.4");
    try expectArgValue(eval_args, "--min-f1", "0.45");
    try expectArgValue(eval_args, "--min-exact-match", "0.25");
    try expectArgPair(eval_args, "--min-task-metric", "classifications.micro_f1=0.6");
    try expectArgPair(eval_args, "--min-task-metric", "classifications.exact_match=0.5");
    try expectArgPair(eval_args, "--min-task-metric", "json_structures.micro_f1=0.55");
    try expectArgPair(eval_args, "--min-task-metric", "json_structures.exact_match=0.4");
    try expectArgPair(eval_args, "--min-task-metric", "relations.micro_f1=0.5");
    try expectArgPair(eval_args, "--min-task-metric", "relations.exact_match=0.35");
    try expectArgPair(eval_args, "--min-task-metric", "count.accuracy=0.7");
    try std.testing.expect(!containsArg(eval_args, "--compiled-required"));
    try std.testing.expectEqualStrings("/runs/gliner2/model", plan.steps[3].argv[3]);
    try std.testing.expectEqualStrings("/runs/gliner2/reload.json", plan.steps[4].argv[3]);
}

test "gliner2 recipe rejects ungated heldout evaluation" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gliner2", .family = "gliner2" },
        .dataset = .{ .train_path = "/data/train.jsonl" },
        .eval = .{ .path = "/data/eval.jsonl" },
    };
    try std.testing.expectError(error.MissingFullTaskQualityThreshold, buildPlan(std.heap.page_allocator, recipe));
}

test "gliner2 recipe requires entity release floors for heldout evaluation" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gliner2", .family = "gliner2" },
        .dataset = .{ .train_path = "/data/train.jsonl" },
        .eval = .{
            .path = "/data/eval.jsonl",
            .full_task_minimums = .{
                .classifications_micro_f1 = 0.6,
                .classifications_exact_match = 0.5,
                .json_structures_micro_f1 = 0.55,
                .json_structures_exact_match = 0.4,
                .relations_micro_f1 = 0.5,
                .relations_exact_match = 0.35,
                .count_accuracy = 0.7,
            },
        },
    };
    try std.testing.expectError(error.MissingEntityQualityThreshold, buildPlan(std.heap.page_allocator, recipe));
}

test "gliner2 recipe rejects early stopping without heldout data" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/gliner2", .family = "gliner2" },
        .dataset = .{ .train_path = "/data/train.jsonl" },
        .eval = .{ .early_stopping_patience = 2 },
    };
    try std.testing.expectError(error.EarlyStoppingRequiresEvalPath, buildPlan(std.heap.page_allocator, recipe));
}

test "layoutlmv3 token recipe emits train eval positional paths" {
    const recipe = Recipe{
        .recipe = "qlora-sft",
        .model = .{ .path = "/models/layoutlmv3", .family = "layoutlmv3" },
        .dataset = .{
            .train_path = "/data/layout-train.jsonl",
            .eval_path = "/data/layout-eval.jsonl",
            .format = "token",
            .max_examples = 6,
            .eval_max_examples = 3,
        },
        .artifacts = .{ .root = "/tmp/layout-run", .materialized_dir = "/tmp/layout-merged" },
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try expectStepCommands(plan, &.{
        "bootstrap-layoutlmv3-lora",
        "train-eval-layoutlmv3-lora-token",
        "materialize-layoutlmv3-checkpoint",
    });
    try std.testing.expectEqualStrings("/data/layout-train.jsonl", plan.steps[1].argv[3]);
    try std.testing.expectEqualStrings("/data/layout-eval.jsonl", plan.steps[1].argv[4]);
}

test "reranker recipe prepares train and eval pooled caches" {
    const recipe = Recipe{
        .recipe = "reranker",
        .model = .{ .path = "/models/reranker", .family = "reranker" },
        .dataset = .{
            .train_path = "/data/rerank-train.jsonl",
            .eval_path = "/data/rerank-eval.jsonl",
            .max_examples = 10,
            .eval_max_examples = 5,
        },
        .artifacts = .{ .root = "/tmp/rerank-run", .materialized_dir = "/tmp/rerank-materialized" },
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try expectStepCommands(plan, &.{
        "prepare-reranker-pooled-cache",
        "prepare-reranker-pooled-cache",
        "train-eval-reranker-head-cached",
        "materialize-reranker-head",
    });
    try std.testing.expectEqualStrings("/tmp/rerank-run/reranker_train_pooled_cache.json", plan.steps[2].argv[2]);
    try std.testing.expectEqualStrings("/tmp/rerank-run/reranker_eval_pooled_cache.json", plan.steps[2].argv[3]);
}

test "reranker lora recipe requires and routes head input" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/reranker", .family = "reranker" },
        .dataset = .{
            .train_path = "/data/rerank-train.jsonl",
            .eval_path = "/data/rerank-eval.jsonl",
        },
        .artifacts = .{
            .root = "/tmp/rerank-lora-run",
            .report_path = "/tmp/rerank-head",
            .materialized_dir = "/tmp/rerank-lora-merged",
        },
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try expectStepCommands(plan, &.{
        "bootstrap-reranker-lora",
        "prepare-reranker-top-layer-cache",
        "prepare-reranker-top-layer-cache",
        "train-eval-reranker-lora-top-layer-cached-surrogate",
        "materialize-reranker-lora",
    });
    try std.testing.expectEqualStrings("/tmp/rerank-head", plan.steps[3].argv[3]);
}

test "reranker lora recipe routes base model override before module flags" {
    const recipe = Recipe{
        .recipe = "lora-sft",
        .model = .{ .path = "/models/reranker", .family = "reranker" },
        .dataset = .{ .train_path = "/data/rerank-train.jsonl" },
        .adapter = .{
            .base_model_name_or_path = "BAAI/bge-reranker-base",
            .target_modules = &.{"query"},
        },
        .artifacts = .{
            .root = "/tmp/rerank-lora-run",
            .report_path = "/tmp/rerank-head",
        },
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try std.testing.expectEqualStrings("bootstrap-reranker-lora", plan.steps[0].argv[0]);
    try std.testing.expectEqualStrings("16", plan.steps[0].argv[3]);
    try std.testing.expectEqualStrings("32", plan.steps[0].argv[4]);
    try std.testing.expectEqualStrings("BAAI/bge-reranker-base", plan.steps[0].argv[5]);
    try std.testing.expectEqualStrings("--target-modules", plan.steps[0].argv[6]);
    try std.testing.expectEqualStrings("query", plan.steps[0].argv[7]);
}

test "vlm retrieval routes colqwen2 prepared inputs" {
    const recipe = Recipe{
        .recipe = "vlm-retrieval",
        .model = .{ .path = "/models/colqwen2", .family = "colqwen2" },
        .dataset = .{
            .path = "/data/colqwen-root",
            .train_path = "/data/colqwen-examples.jsonl",
            .max_examples = 7,
        },
        .artifacts = .{ .root = "/tmp/colqwen-run" },
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    try expectStepCommands(plan, &.{
        "prepare-colqwen2-inputs",
        "bootstrap-colqwen2-lora",
        "train-eval-colqwen2-lora-bundle",
    });
    try std.testing.expectEqualStrings("/data/colqwen-root", plan.steps[0].argv[2]);
    try std.testing.expectEqualStrings("/data/colqwen-examples.jsonl", plan.steps[0].argv[3]);
}

test "gemma4 full sft fails closed while dpo and grpo build runnable plans" {
    const base = Recipe{
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/train.jsonl" },
        .backend = "native",
    };
    var sft = base;
    sft.recipe = "sft";
    try std.testing.expectError(error.Gemma4FullSftNotYetSupported, buildPlan(std.heap.page_allocator, sft));

    var dpo = base;
    dpo.recipe = "dpo";
    dpo.execution.mode = "train";
    dpo.dataset.format = "text-preference";
    dpo.dataset.eval_path = "/data/eval-preferences.jsonl";
    dpo.eval = .{ .dpo_minimums = .{ .accuracy = 0.5, .max_loss = 2.0 } };
    dpo.adapter = .{ .rank = 8, .alpha = 16 };
    const dpo_plan = try buildPlan(std.heap.page_allocator, dpo);
    defer freePlan(std.heap.page_allocator, dpo_plan);
    try std.testing.expectEqual(StepKind.direct_dpo, dpo_plan.steps[0].kind);

    var grpo_recipe = base;
    grpo_recipe.recipe = "grpo";
    grpo_recipe.execution.mode = "train";
    grpo_recipe.dataset.format = "text-grpo";
    grpo_recipe.dataset.eval_path = "/data/eval-prompts.jsonl";
    grpo_recipe.eval = .{ .grpo_minimums = .{
        .mean_reward = 0.25,
        .top_rank_mean_reward = 0.25,
        .positive_reward_group_rate = 0.25,
        .max_kl_loss = 1.0,
    } };
    grpo_recipe.adapter = .{ .rank = 8, .alpha = 16 };
    const grpo_plan = try buildPlan(std.heap.page_allocator, grpo_recipe);
    defer freePlan(std.heap.page_allocator, grpo_plan);
    try std.testing.expectEqual(StepKind.direct_grpo, grpo_plan.steps[0].kind);
}

test "gemma4 preference recipes require explicit execution intent and dataset format" {
    const base = Recipe{
        .recipe = "dpo",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/preferences.jsonl" },
        .backend = "native",
    };
    try std.testing.expectError(error.MissingDatasetFormat, buildPlan(std.heap.page_allocator, base));

    var missing_mode = base;
    missing_mode.dataset.format = "text-preference";
    try std.testing.expectError(error.MissingPreferenceExecutionMode, buildPlan(std.heap.page_allocator, missing_mode));

    var missing_adapter = missing_mode;
    missing_adapter.execution.mode = "train";
    try std.testing.expectError(error.MissingAdapterTrainingIntent, buildPlan(std.heap.page_allocator, missing_adapter));

    var score_with_adapter = missing_mode;
    score_with_adapter.execution.mode = "score";
    score_with_adapter.adapter = .{ .rank = 8, .alpha = 16 };
    try std.testing.expectError(error.AdapterTrainingRequiresTrainMode, buildPlan(std.heap.page_allocator, score_with_adapter));

    var fixture_train = base;
    fixture_train.dataset.format = "scalar-logprobs";
    fixture_train.execution.mode = "train";
    fixture_train.adapter = .{ .rank = 8, .alpha = 16 };
    try std.testing.expectError(error.PreferenceTrainingRequiresModelDataset, buildPlan(std.heap.page_allocator, fixture_train));
}

test "gemma4 preference training preflight rejects ignored and conflicting options" {
    const valid = Recipe{
        .recipe = "dpo",
        .execution = .{ .mode = "train" },
        .model = .{ .path = "/models/gemma4", .reference_path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/preferences.jsonl", .eval_path = "/data/eval-preferences.jsonl", .format = "text-preference", .max_seq_len = 128 },
        .adapter = .{ .rank = 8, .alpha = 16 },
        .optimizer = .{ .learning_rate = 0.0001, .epochs = 1 },
        .eval = .{ .dpo_minimums = .{ .accuracy = 0.5, .max_loss = 2.0 } },
        .artifacts = .{ .root = "/tmp/gemma4-dpo-contract" },
        .backend = "native",
    };

    const plan = try buildPlan(std.heap.page_allocator, valid);
    defer freePlan(std.heap.page_allocator, plan);
    try std.testing.expectEqualStrings("train", plan.steps[0].name);

    var seeded = valid;
    seeded.optimizer.seed = 991;
    const seeded_plan = try buildPlan(std.heap.page_allocator, seeded);
    defer freePlan(std.heap.page_allocator, seeded_plan);

    var unsupported_seed = valid;
    unsupported_seed.recipe = "reranker";
    unsupported_seed.optimizer.seed = 991;
    try std.testing.expectError(
        error.UnsupportedOptimizerSeed,
        buildPlan(std.heap.page_allocator, unsupported_seed),
    );

    var checkpointed = valid;
    checkpointed.optimizer.epochs = 2;
    checkpointed.checkpoint = .{ .every_epochs = 1 };
    const checkpointed_plan = try buildPlan(std.heap.page_allocator, checkpointed);
    defer freePlan(std.heap.page_allocator, checkpointed_plan);
    const checkpointed_path = try preferenceCheckpointPath(std.testing.allocator, checkpointed, .dpo);
    defer std.testing.allocator.free(checkpointed_path.?);
    try std.testing.expectEqualStrings(
        "/tmp/gemma4-dpo-contract/gemma4_dpo_trainer_state.safetensors",
        checkpointed_path.?,
    );

    var resumed = checkpointed;
    resumed.checkpoint = .{
        .every_epochs = 1,
        .resume_path = "/tmp/gemma4-dpo-resume/state.safetensors",
    };
    const resumed_plan = try buildPlan(std.heap.page_allocator, resumed);
    defer freePlan(std.heap.page_allocator, resumed_plan);

    var empty_checkpoint = valid;
    empty_checkpoint.checkpoint = .{};
    try std.testing.expectError(error.CheckpointIntervalRequired, buildPlan(std.heap.page_allocator, empty_checkpoint));

    var zero_checkpoint = valid;
    zero_checkpoint.checkpoint = .{ .every_epochs = 0 };
    try std.testing.expectError(error.InvalidGemma4CheckpointInterval, buildPlan(std.heap.page_allocator, zero_checkpoint));

    var retained_checkpoint = valid;
    retained_checkpoint.checkpoint = .{ .every_epochs = 1, .keep_last = 2 };
    try std.testing.expectError(error.UnsupportedGemma4CheckpointOption, buildPlan(std.heap.page_allocator, retained_checkpoint));

    var checkpoint_overwrites_dataset = valid;
    checkpoint_overwrites_dataset.checkpoint = .{
        .every_epochs = 1,
        .resume_path = "/data/preferences.jsonl",
    };
    try std.testing.expectError(
        error.PreferenceArtifactInputConflict,
        buildPlan(std.heap.page_allocator, checkpoint_overwrites_dataset),
    );

    var bucketed = valid;
    bucketed.runtime = .{
        .sequence_length_bucket_quantum = 16,
        .sequence_length_bucket_min = 32,
        .graph_cache_capacity = 8,
    };
    const bucketed_plan = try buildPlan(std.heap.page_allocator, bucketed);
    defer freePlan(std.heap.page_allocator, bucketed_plan);
    try std.testing.expectEqualStrings("train", bucketed_plan.steps[0].name);

    var missing_bucket_quantum = valid;
    missing_bucket_quantum.runtime = .{ .graph_cache_capacity = 8 };
    try std.testing.expectError(
        error.Gemma4SequenceLengthBucketQuantumRequired,
        buildPlan(std.heap.page_allocator, missing_bucket_quantum),
    );

    var oversized_graph_cache = valid;
    oversized_graph_cache.runtime = .{ .sequence_length_bucket_quantum = 16, .graph_cache_capacity = 9 };
    try std.testing.expectError(
        error.InvalidGemma4GraphCacheCapacity,
        buildPlan(std.heap.page_allocator, oversized_graph_cache),
    );

    var bad_backend = valid;
    bad_backend.backend = "cuda";
    try std.testing.expectError(error.UnsupportedBackend, buildPlan(std.heap.page_allocator, bad_backend));

    var bad_reference = valid;
    bad_reference.model.reference_path = "/models/other-gemma4";
    try std.testing.expectError(error.UnsupportedReferencePath, buildPlan(std.heap.page_allocator, bad_reference));

    var same_eval = valid;
    same_eval.dataset.eval_path = "/data/preferences.jsonl";
    try std.testing.expectError(error.PreferenceTrainEvalDatasetConflict, buildPlan(std.heap.page_allocator, same_eval));

    var missing_eval_gate = valid;
    missing_eval_gate.eval = null;
    try std.testing.expectError(error.MissingPreferenceEvaluationConfig, buildPlan(std.heap.page_allocator, missing_eval_gate));

    var ignored_optimizer = valid;
    ignored_optimizer.optimizer.max_steps = 10;
    try std.testing.expectError(error.UnsupportedGemma4OptimizerOption, buildPlan(std.heap.page_allocator, ignored_optimizer));

    var unsupported_adapter = valid;
    unsupported_adapter.adapter.?.dropout = 0.1;
    try std.testing.expectError(error.UnsupportedGemma4AdapterOption, buildPlan(std.heap.page_allocator, unsupported_adapter));

    var conflicting_outputs = valid;
    conflicting_outputs.adapter.?.path = "/tmp/gemma4-dpo-same";
    conflicting_outputs.artifacts.trained_adapter_dir = "/tmp/gemma4-dpo-same";
    try std.testing.expectError(error.Gemma4BootstrapAndTrainingOutputConflict, buildPlan(std.heap.page_allocator, conflicting_outputs));
}

test "gemma4 GRPO reward and group contracts fail during planning" {
    const valid = Recipe{
        .recipe = "grpo",
        .execution = .{ .mode = "train" },
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/prompts.jsonl", .eval_path = "/data/eval-prompts.jsonl", .format = "text-grpo", .max_seq_len = 64 },
        .adapter = .{ .rank = 8, .alpha = 16 },
        .grpo = .{ .group_size = 2, .max_completion_tokens = 2, .reward_mode = "prefix-match" },
        .eval = .{ .grpo_minimums = .{
            .mean_reward = 0.25,
            .top_rank_mean_reward = 0.25,
            .positive_reward_group_rate = 0.25,
            .max_kl_loss = 1.0,
        } },
        .artifacts = .{ .root = "/tmp/gemma4-grpo-contract" },
        .backend = "native",
    };
    const plan = try buildPlan(std.heap.page_allocator, valid);
    defer freePlan(std.heap.page_allocator, plan);

    var unsupported_runtime = valid;
    unsupported_runtime.runtime = .{ .sequence_length_bucket_quantum = 16 };
    try std.testing.expectError(
        error.UnsupportedGemma4RuntimeOption,
        buildPlan(std.heap.page_allocator, unsupported_runtime),
    );

    var incremental_native = valid;
    incremental_native.runtime = .{ .grpo_incremental_kv = true };
    try std.testing.expectError(
        error.Gemma4GrpoIncrementalKvRequiresMetal,
        buildPlan(std.heap.page_allocator, incremental_native),
    );

    var incremental_without_admission = valid;
    incremental_without_admission.runtime = .{ .grpo_incremental_kv_batch_active = true };
    try std.testing.expectError(
        error.Gemma4GrpoIncrementalKvRequired,
        buildPlan(std.heap.page_allocator, incremental_without_admission),
    );

    var incremental_checkpointed = valid;
    incremental_checkpointed.backend = "metal";
    incremental_checkpointed.optimizer.epochs = 2;
    incremental_checkpointed.checkpoint = .{ .every_epochs = 1 };
    incremental_checkpointed.runtime = .{
        .grpo_incremental_kv = true,
        .grpo_incremental_kv_batch_active = true,
        .grpo_incremental_kv_clone_prompt_tail = true,
        .grpo_incremental_kv_shadow_exact = true,
    };
    const incremental_checkpointed_plan = try buildPlan(
        std.heap.page_allocator,
        incremental_checkpointed,
    );
    defer freePlan(std.heap.page_allocator, incremental_checkpointed_plan);

    var direct_gguf_incremental = incremental_checkpointed;
    direct_gguf_incremental.model.path = "/models/gemma4.gguf";
    direct_gguf_incremental.model.allow_direct_gguf_training = true;
    try std.testing.expectError(
        error.DirectGgufGrpoIncrementalKvNotQualified,
        buildPlan(std.heap.page_allocator, direct_gguf_incremental),
    );

    var bad_reward = valid;
    bad_reward.grpo.reward_mode = "webhook";
    try std.testing.expectError(error.UnsupportedRewardMode, buildPlan(std.heap.page_allocator, bad_reward));

    var bad_group = valid;
    bad_group.grpo.group_size = 1;
    try std.testing.expectError(error.InvalidGrpoGroupSize, buildPlan(std.heap.page_allocator, bad_group));

    const default_kl = try resolveGrpoKlControl(valid.grpo);
    try std.testing.expectEqual(@as(f32, 0.1), default_kl.train_max_kl);
    try std.testing.expect(!default_kl.adaptive);

    var incomplete_adaptive_kl = valid;
    incomplete_adaptive_kl.grpo.adaptive_kl = true;
    incomplete_adaptive_kl.grpo.target_kl = 0.01;
    try std.testing.expectError(
        error.IncompleteGrpoAdaptiveKlConfig,
        buildPlan(std.heap.page_allocator, incomplete_adaptive_kl),
    );

    var invalid_adaptive_kl = valid;
    invalid_adaptive_kl.grpo = .{
        .group_size = 2,
        .max_completion_tokens = 2,
        .reward_mode = "prefix-match",
        .kl_coef = 0.04,
        .train_max_kl = 0.01,
        .adaptive_kl = true,
        .target_kl = 0.01,
        .kl_horizon = 100,
    };
    try std.testing.expectError(
        error.InvalidGrpoAdaptiveKlConfig,
        buildPlan(std.heap.page_allocator, invalid_adaptive_kl),
    );

    var adaptive_kl = valid;
    adaptive_kl.grpo = .{
        .group_size = 2,
        .max_completion_tokens = 2,
        .reward_mode = "prefix-match",
        .kl_coef = 0.04,
        .train_max_kl = 0.1,
        .adaptive_kl = true,
        .target_kl = 0.01,
        .kl_horizon = 100,
        .min_kl_coef = 0.001,
        .max_kl_coef = 1.0,
    };
    const adaptive_plan = try buildPlan(std.heap.page_allocator, adaptive_kl);
    defer freePlan(std.heap.page_allocator, adaptive_plan);
    const resolved_adaptive_kl = try resolveGrpoKlControl(adaptive_kl.grpo);
    try std.testing.expect(resolved_adaptive_kl.adaptive);
    try std.testing.expectEqual(@as(f32, 0.01), resolved_adaptive_kl.target_kl.?);
    try std.testing.expectEqual(@as(f32, 100), resolved_adaptive_kl.kl_horizon.?);

    adaptive_kl.optimizer.epochs = 2;
    adaptive_kl.checkpoint = .{ .every_epochs = 1 };
    const adaptive_checkpoint_plan = try buildPlan(std.heap.page_allocator, adaptive_kl);
    defer freePlan(std.heap.page_allocator, adaptive_checkpoint_plan);
    const adaptive_checkpoint_path = try preferenceCheckpointPath(std.testing.allocator, adaptive_kl, .grpo);
    defer std.testing.allocator.free(adaptive_checkpoint_path.?);
    try std.testing.expectEqualStrings(
        "/tmp/gemma4-grpo-contract/gemma4_grpo_trainer_state.safetensors",
        adaptive_checkpoint_path.?,
    );

    var checkpoint_with_custom_reward_artifacts = adaptive_kl;
    checkpoint_with_custom_reward_artifacts.grpo.reward_mode = null;
    checkpoint_with_custom_reward_artifacts.reward = .{
        .trace_path = "/tmp/shared-grpo-reward-trace.jsonl",
        .providers = &.{.{
            .name = "exact",
            .kind = "builtin",
            .mode = "exact-match",
        }},
    };
    try std.testing.expectError(
        error.Gemma4GrpoCheckpointCustomRewardArtifactsNotSupported,
        buildPlan(std.heap.page_allocator, checkpoint_with_custom_reward_artifacts),
    );

    var typed = valid;
    typed.grpo.reward_mode = null;
    typed.reward = .{
        .aggregation = "weighted-mean",
        .providers = &.{
            .{ .name = "exact", .kind = "builtin", .mode = "exact-match", .weight = 0.25 },
            .{ .name = "prefix", .kind = "builtin", .mode = "prefix-match", .weight = 0.75 },
        },
    };
    const typed_plan = try buildPlan(std.heap.page_allocator, typed);
    defer freePlan(std.heap.page_allocator, typed_plan);

    var conflicting_reward = typed;
    conflicting_reward.grpo.reward_mode = "exact-match";
    try std.testing.expectError(error.ConflictingRewardConfiguration, buildPlan(std.heap.page_allocator, conflicting_reward));

    var fallback_reward = typed;
    fallback_reward.reward.?.failure_policy = "zero";
    try std.testing.expectError(error.UnsupportedRewardFailurePolicy, buildPlan(std.heap.page_allocator, fallback_reward));

    var duplicate_provider = typed;
    duplicate_provider.reward.?.providers = &.{
        .{ .name = "same", .kind = "builtin", .mode = "exact-match" },
        .{ .name = "same", .kind = "builtin", .mode = "prefix-match" },
    };
    try std.testing.expectError(error.DuplicateRewardProviderName, buildPlan(std.heap.page_allocator, duplicate_provider));

    var unpinned_external = typed;
    unpinned_external.reward.?.providers = &.{
        .{ .name = "judge", .kind = "external-command", .executable_path = "/usr/bin/true" },
    };
    try std.testing.expectError(error.MissingRewardExecutableDigest, buildPlan(std.heap.page_allocator, unpinned_external));

    const zero_digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    var incomplete_model = typed;
    incomplete_model.reward.?.providers = &.{.{
        .name = "judge-model",
        .kind = "model-command",
        .executable_path = "/usr/bin/true",
        .executable_sha256 = zero_digest,
        .min_reward = 0,
        .max_reward = 1,
    }};
    try std.testing.expectError(error.MissingRewardModel, buildPlan(std.heap.page_allocator, incomplete_model));

    var pinned_model = typed;
    pinned_model.reward.?.providers = &.{.{
        .name = "judge-model",
        .kind = "model-command",
        .executable_path = "/usr/bin/true",
        .executable_sha256 = zero_digest,
        .model_path = "/models/reward/model.safetensors",
        .model_sha256 = zero_digest,
        .tokenizer_path = "/models/reward/tokenizer.json",
        .tokenizer_sha256 = zero_digest,
        .chat_template_path = "/models/reward/chat-template.txt",
        .chat_template_sha256 = zero_digest,
        .calibration_dataset_path = "/data/reward-calibration.jsonl",
        .calibration_dataset_sha256 = zero_digest,
        .max_input_tokens = 4096,
        .max_batch_size = 1,
        .min_reward = 0,
        .max_reward = 1,
    }};
    const pinned_model_plan = try buildPlan(std.heap.page_allocator, pinned_model);
    defer freePlan(std.heap.page_allocator, pinned_model_plan);

    var bad_batch_provider = pinned_model.reward.?.providers.?[0];
    bad_batch_provider.max_batch_size = 8;
    var aspirational_batching = pinned_model;
    aspirational_batching.reward.?.providers = &.{bad_batch_provider};
    try std.testing.expectError(error.UnsupportedModelRewardBatchSize, buildPlan(std.heap.page_allocator, aspirational_batching));

    var bad_multimodal = valid;
    bad_multimodal.model.projector_path = "/models/gemma4/mmproj.gguf";
    try std.testing.expectError(
        error.Gemma4MultimodalPreferenceEvaluationNotYetSupported,
        buildPlan(std.heap.page_allocator, bad_multimodal),
    );
}

test "gemma4 GRPO external reward executable preflight rejects digest drift" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const executable_path = try std.fs.path.join(allocator, &.{ root, "reward-verifier" });
    defer allocator.free(executable_path);
    try artifact_publication.writeFileAtomicReplace(allocator, std.testing.io, executable_path, "verifier-v1\n");
    const executable_digest = try sha256FileAlloc(allocator, std.testing.io, executable_path);
    defer allocator.free(executable_digest);

    var providers = [_]RewardProviderConfig{.{
        .name = "judge",
        .kind = "external-command",
        .executable_path = executable_path,
        .executable_sha256 = executable_digest,
    }};
    const recipe = Recipe{ .reward = .{ .providers = &providers } };
    try preflightRewardExecutables(allocator, std.testing.io, recipe);

    providers[0].executable_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    try std.testing.expectError(
        error.RewardExecutableDigestMismatch,
        preflightRewardExecutables(allocator, std.testing.io, recipe),
    );
}

test "gemma4 GRPO model reward preflight and response attestation fail closed" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const artifact_path = try std.fs.path.join(allocator, &.{ root, "pinned-artifact" });
    defer allocator.free(artifact_path);
    try artifact_publication.writeFileAtomicReplace(allocator, std.testing.io, artifact_path, "pinned-v1\n");
    const digest = try sha256FileAlloc(allocator, std.testing.io, artifact_path);
    defer allocator.free(digest);

    var providers = [_]RewardProviderConfig{.{
        .name = "judge-model",
        .kind = "model-command",
        .executable_path = artifact_path,
        .executable_sha256 = digest,
        .model_path = artifact_path,
        .model_sha256 = digest,
        .tokenizer_path = artifact_path,
        .tokenizer_sha256 = digest,
        .chat_template_path = artifact_path,
        .chat_template_sha256 = digest,
        .calibration_dataset_path = artifact_path,
        .calibration_dataset_sha256 = digest,
        .max_input_tokens = 16,
        .max_batch_size = 1,
        .min_reward = 0,
        .max_reward = 1,
    }};
    const recipe = Recipe{ .reward = .{ .providers = &providers } };
    try preflightRewardExecutables(allocator, std.testing.io, recipe);
    try validateModelRewardResponse(providers[0], .{
        .reward = 0.75,
        .input_tokens = 12,
        .model_sha256 = digest,
        .tokenizer_sha256 = digest,
        .chat_template_sha256 = digest,
        .calibration_dataset_sha256 = digest,
    });
    try std.testing.expectError(error.ModelRewardTokenLimitExceeded, validateModelRewardResponse(providers[0], .{
        .reward = 0.75,
        .input_tokens = 17,
        .model_sha256 = digest,
        .tokenizer_sha256 = digest,
        .chat_template_sha256 = digest,
        .calibration_dataset_sha256 = digest,
    }));
    try std.testing.expectError(error.ModelRewardIdentityMismatch, validateModelRewardResponse(providers[0], .{
        .reward = 0.75,
        .input_tokens = 12,
        .model_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        .tokenizer_sha256 = digest,
        .chat_template_sha256 = digest,
        .calibration_dataset_sha256 = digest,
    }));

    try artifact_publication.writeFileAtomicReplace(allocator, std.testing.io, artifact_path, "pinned-v2\n");
    try std.testing.expectError(
        error.RewardExecutableDigestMismatch,
        preflightRewardExecutables(allocator, std.testing.io, recipe),
    );
}

test "gemma4 GRPO model reward subprocess is attested and failures are traced" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const relative_root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(relative_root);
    const root = try std.fs.path.resolve(allocator, &.{relative_root});
    defer allocator.free(root);
    const artifact_path = try std.fs.path.join(allocator, &.{ root, "pinned-model-artifact" });
    defer allocator.free(artifact_path);
    try artifact_publication.writeFileAtomicReplace(allocator, std.testing.io, artifact_path, "pinned-v1\n");
    const artifact_digest = try sha256FileAlloc(allocator, std.testing.io, artifact_path);
    defer allocator.free(artifact_digest);
    const executable_digest = try sha256FileAlloc(allocator, std.testing.io, "/bin/sh");
    defer allocator.free(executable_digest);

    const success_response = try std.json.Stringify.valueAlloc(allocator, ExternalRewardResponse{
        .reward = 0.75,
        .evidence = "fixture-score",
        .input_tokens = 4,
        .model_sha256 = artifact_digest,
        .tokenizer_sha256 = artifact_digest,
        .chat_template_sha256 = artifact_digest,
        .calibration_dataset_sha256 = artifact_digest,
    }, .{});
    defer allocator.free(success_response);
    const script = try std.fmt.allocPrint(allocator, "printf '%s\\n' '{s}'\n", .{success_response});
    defer allocator.free(script);

    const exchange_dir = try std.fs.path.join(allocator, &.{ root, "exchange" });
    try std.Io.Dir.cwd().createDirPath(std.testing.io, exchange_dir);
    const trace_path = try std.fs.path.join(allocator, &.{ root, "reward-trace.jsonl" });
    const configuration_digest = try allocator.dupe(u8, artifact_digest);
    const prompts = [_][]const u8{"Question?"};
    const targets = [_][]const u8{"yes"};
    var pipeline = RewardPipeline{
        .allocator = allocator,
        .io = std.testing.io,
        .tokenizer = undefined,
        .prompt_texts = &prompts,
        .targets = &targets,
        .providers = null,
        .legacy_mode = null,
        .aggregation = "weighted-mean",
        .failure_policy = "fail",
        .phase = .train,
        .trace_path = trace_path,
        .exchange_dir = exchange_dir,
        .configuration_digest = configuration_digest,
        .max_trace_bytes = 1024 * 1024,
    };
    defer pipeline.deinit();

    // The request path appended by the provider protocol becomes sh's $0;
    // the pinned command emits one response and intentionally ignores it.
    const success_args = [_][]const u8{ "-c", script };
    const provider = RewardProviderConfig{
        .name = "judge-model",
        .kind = "model-command",
        .executable_path = "/bin/sh",
        .executable_sha256 = executable_digest,
        .args = &success_args,
        .model_path = artifact_path,
        .model_sha256 = artifact_digest,
        .tokenizer_path = artifact_path,
        .tokenizer_sha256 = artifact_digest,
        .chat_template_path = artifact_path,
        .chat_template_sha256 = artifact_digest,
        .calibration_dataset_path = artifact_path,
        .calibration_dataset_sha256 = artifact_digest,
        .max_input_tokens = 16,
        .max_batch_size = 1,
        .min_reward = 0,
        .max_reward = 1,
    };
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const completion_tokens = [_]i32{ 1, 2 };
    const result = try pipeline.scoreExternalProvider(
        arena.allocator(),
        provider,
        0,
        &completion_tokens,
        "yes",
        "yes",
    );
    try std.testing.expectEqual(@as(f32, 0.75), result.reward);
    try std.testing.expectEqual(@as(?usize, 4), result.input_tokens);
    try std.testing.expectEqual(@as(usize, 1), pipeline.external_calls);
    const request = try readFileMax(allocator, std.testing.io, result.request_path, 64 * 1024);
    defer allocator.free(request);
    try std.testing.expect(std.mem.indexOf(u8, request, "antfly_inference_grpo_reward_request/v2") != null);
    try std.testing.expect(std.mem.indexOf(u8, request, artifact_digest) != null);

    pipeline.call_index = 1;
    const failure_args = [_][]const u8{ "-c", "exit 7" };
    var failing_provider = provider;
    failing_provider.args = &failure_args;
    try std.testing.expectError(error.RewardProviderFailed, pipeline.scoreExternalProvider(
        arena.allocator(),
        failing_provider,
        0,
        &completion_tokens,
        "no",
        "yes",
    ));
    try std.testing.expectEqual(@as(usize, 1), pipeline.external_failures);

    const out_of_bounds_response = try std.json.Stringify.valueAlloc(allocator, ExternalRewardResponse{
        .reward = 2.0,
        .input_tokens = 4,
        .model_sha256 = artifact_digest,
        .tokenizer_sha256 = artifact_digest,
        .chat_template_sha256 = artifact_digest,
        .calibration_dataset_sha256 = artifact_digest,
    }, .{});
    defer allocator.free(out_of_bounds_response);
    const out_of_bounds_script = try std.fmt.allocPrint(allocator, "printf '%s\\n' '{s}'\n", .{out_of_bounds_response});
    defer allocator.free(out_of_bounds_script);
    const out_of_bounds_args = [_][]const u8{ "-c", out_of_bounds_script };
    var out_of_bounds_provider = provider;
    out_of_bounds_provider.args = &out_of_bounds_args;
    pipeline.call_index = 2;
    try std.testing.expectError(error.RewardProviderOutOfBounds, pipeline.scoreExternalProvider(
        arena.allocator(),
        out_of_bounds_provider,
        0,
        &completion_tokens,
        "yes",
        "yes",
    ));
    try std.testing.expectEqual(@as(usize, 2), pipeline.external_failures);
    try pipeline.finish();
    const trace = try readFileMax(allocator, std.testing.io, pipeline.trace_path, 64 * 1024);
    defer allocator.free(trace);
    try std.testing.expect(std.mem.indexOf(u8, trace, "antfly_inference_grpo_reward_failure/v1") != null);
    try std.testing.expect(std.mem.indexOf(u8, trace, "RewardProviderFailed") != null);
    try std.testing.expect(std.mem.indexOf(u8, trace, "RewardProviderOutOfBounds") != null);
    try std.testing.expect(std.mem.indexOf(u8, trace, artifact_digest) != null);
}

test "gemma4 GRPO reward request schema preserves generic provider compatibility" {
    try std.testing.expectEqualStrings(
        "antfly_inference_grpo_reward_request/v1",
        rewardRequestSchemaVersion("external-command"),
    );
    try std.testing.expectEqualStrings(
        "antfly_inference_grpo_reward_request/v2",
        rewardRequestSchemaVersion("model-command"),
    );
}

test "preference evaluators reject token-identical train and heldout prompts" {
    const train_prompt_a = [_]i32{ 1, 2, 3 };
    const train_prompt_b = [_]i32{ 4, 5 };
    const eval_prompt_same = [_]i32{ 1, 2, 3 };
    const eval_prompt_new = [_]i32{ 6, 7 };
    const chosen = [_]i32{8};
    const rejected = [_]i32{9};
    const train_samples = [_]preference_harness.PreferenceSample{
        .{ .prompt_tokens = &train_prompt_a, .chosen_tokens = &chosen, .rejected_tokens = &rejected },
        .{ .prompt_tokens = &train_prompt_b, .chosen_tokens = &chosen, .rejected_tokens = &rejected },
    };
    const eval_samples = [_]preference_harness.PreferenceSample{
        .{ .prompt_tokens = &eval_prompt_same, .chosen_tokens = &chosen, .rejected_tokens = &rejected },
        .{ .prompt_tokens = &eval_prompt_new, .chosen_tokens = &chosen, .rejected_tokens = &rejected },
    };
    try std.testing.expectEqual(
        @as(usize, 1),
        try countPreferencePromptOverlaps(std.testing.allocator, &train_samples, &eval_samples),
    );
    try std.testing.expectEqual(
        @as(usize, 1),
        try countTokenPromptOverlaps(
            std.testing.allocator,
            &.{ &train_prompt_a, &train_prompt_b },
            &.{ &eval_prompt_same, &eval_prompt_new },
        ),
    );
}

test "preference training refuses a stale trained adapter publication target" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const target = try std.fs.path.join(allocator, &.{ root, "adapter-trained" });
    defer allocator.free(target);
    try requireMissingPreferencePublicationTarget(std.testing.io, target);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, target);
    try std.testing.expectError(
        error.PreferenceTrainedAdapterAlreadyExists,
        requireMissingPreferencePublicationTarget(std.testing.io, target),
    );
}

test "gemma4 preference training reports fingerprint bootstrap and trained adapters" {
    const recipe = Recipe{
        .recipe = "dpo",
        .execution = .{ .mode = "train" },
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/preferences.jsonl", .eval_path = "/data/eval-preferences.jsonl", .format = "text-preference" },
        .adapter = .{ .rank = 8, .alpha = 16 },
        .eval = .{ .dpo_minimums = .{ .accuracy = 0.5, .max_loss = 2.0 } },
        .artifacts = .{ .root = "/tmp/gemma4-preference-artifacts" },
        .backend = "native",
    };
    const plan = try buildPlan(std.testing.allocator, recipe);
    defer freePlan(std.testing.allocator, plan);

    var planned: std.ArrayListUnmanaged(PlannedPath) = .empty;
    defer {
        for (planned.items) |item| std.testing.allocator.free(item.path);
        planned.deinit(std.testing.allocator);
    }
    try appendArtifactPathsFromPlan(std.testing.allocator, &planned, recipe, plan);

    var found_bootstrap = false;
    var found_trained = false;
    var found_evaluation = false;
    for (planned.items) |item| {
        if (std.mem.eql(u8, item.label, "adapter_bootstrap")) found_bootstrap = true;
        if (std.mem.eql(u8, item.label, "trained_adapter")) found_trained = true;
        if (std.mem.eql(u8, item.label, "dpo_evaluation")) found_evaluation = true;
    }
    try std.testing.expect(found_bootstrap);
    try std.testing.expect(found_trained);
    try std.testing.expect(found_evaluation);
}

test "run manifest captures recipe plan status" {
    const recipe = Recipe{
        .recipe = "reranker",
        .model = .{ .path = "/models/reranker", .family = "reranker" },
        .dataset = .{ .path = "/data/train.jsonl" },
        .artifacts = .{ .root = "/tmp/manifest-test" },
    };
    const plan = try buildPlan(std.heap.page_allocator, recipe);
    defer freePlan(std.heap.page_allocator, plan);
    const steps = try initStepManifests(std.heap.page_allocator, plan);
    defer std.heap.page_allocator.free(steps);
    steps[0].status = .running;
    const rendered = try std.json.Stringify.valueAlloc(std.heap.page_allocator, RunManifest{
        .status = .running,
        .recipe = recipe,
        .artifact_root = recipe.artifacts.root,
        .steps = steps,
    }, .{ .whitespace = .indent_2 });
    defer std.heap.page_allocator.free(rendered);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "antfly_inference_finetune_recipe_run/v1") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "prepare-reranker-pooled-cache") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "running") != null);
}

test "fast smoke resolves checked-in testdata from current package cwd" {
    const allocator = std.testing.allocator;
    const path = try resolveCwdPath(allocator, std.testing.io, "pkg/inference/testdata/recipe_gemma4_lora.json");
    defer allocator.free(path);
    try std.testing.expect(cwdPathExists(std.testing.io, path));
}

test "fast smoke result cleanup handles an error after an initialized prefix" {
    const Harness = struct {
        fn makeResult(allocator: std.mem.Allocator, index: usize) !FastSmokeCaseResult {
            const manifest_path = try std.fmt.allocPrint(allocator, "/tmp/manifest-{d}.json", .{index});
            errdefer allocator.free(manifest_path);
            const training_report_path = try std.fmt.allocPrint(allocator, "/tmp/report-{d}.json", .{index});
            return .{
                .name = "case",
                .recipe_path = "recipe.json",
                .mode = .execute,
                .status = .succeeded,
                .manifest_path = manifest_path,
                .training_report_path = training_report_path,
            };
        }

        fn failAfter(allocator: std.mem.Allocator, initialized_before_error: usize) !void {
            var results = try allocator.alloc(FastSmokeCaseResult, initialized_before_error + 1);
            var initialized_results: usize = 0;
            defer freeFastSmokeResults(allocator, results, initialized_results);
            while (initialized_results < initialized_before_error) : (initialized_results += 1) {
                results[initialized_results] = try makeResult(allocator, initialized_results);
            }
            return error.InjectedFastSmokeFailure;
        }
    };

    try std.testing.expectError(error.InjectedFastSmokeFailure, Harness.failAfter(std.testing.allocator, 2));
}

test "fast smoke fixture resolution supports repo root and inference directory" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "zig/pkg/inference/testdata");
    try tmp.dir.writeFile(io, .{
        .sub_path = "zig/pkg/inference/testdata/recipe.json",
        .data = "{}",
    });

    const requested = "pkg/inference/testdata/recipe.json";
    const from_repo_root = try resolvePathFromDir(allocator, io, tmp.dir, requested);
    defer allocator.free(from_repo_root);
    try std.testing.expectEqualStrings("zig/pkg/inference/testdata/recipe.json", from_repo_root);
    try std.testing.expect(dirPathExists(tmp.dir, io, from_repo_root));

    var inference_dir = try tmp.dir.openDir(io, "zig/pkg/inference", .{});
    defer inference_dir.close(io);
    const from_inference_dir = try resolvePathFromDir(allocator, io, inference_dir, requested);
    defer allocator.free(from_inference_dir);
    try std.testing.expectEqualStrings("testdata/recipe.json", from_inference_dir);
    try std.testing.expect(dirPathExists(inference_dir, io, from_inference_dir));
}

test "direct command adapter registry covers reranker family steps" {
    try std.testing.expect(isDirectCommandAdapter("prepare-gemma4-lora-inputs"));
    try std.testing.expect(isDirectCommandAdapter("bootstrap-gemma4-lora"));
    try std.testing.expect(isDirectCommandAdapter("train-eval-gemma4-lora-bundle"));
    try std.testing.expect(isDirectCommandAdapter("bootstrap-layoutlmv3-lora"));
    try std.testing.expect(isDirectCommandAdapter("train-eval-layoutlmv3-lora-sequence"));
    try std.testing.expect(isDirectCommandAdapter("train-eval-layoutlmv3-lora-token"));
    try std.testing.expect(isDirectCommandAdapter("materialize-layoutlmv3-checkpoint"));
    try std.testing.expect(isDirectCommandAdapter("bootstrap-reranker-lora"));
    try std.testing.expect(isDirectCommandAdapter("prepare-reranker-top-layer-cache"));
    try std.testing.expect(isDirectCommandAdapter("train-eval-reranker-lora-top-layer-cached-surrogate"));
    try std.testing.expect(isDirectCommandAdapter("materialize-reranker-lora"));
    try std.testing.expect(isDirectCommandAdapter("prepare-colqwen2-inputs"));
    try std.testing.expect(isDirectCommandAdapter("bootstrap-colqwen2-lora"));
    try std.testing.expect(isDirectCommandAdapter("train-eval-colqwen2-lora-bundle"));
    try std.testing.expect(isDirectCommandAdapter("prepare-reranker-pooled-cache"));
    try std.testing.expect(isDirectCommandAdapter("train-eval-reranker-head-cached"));
    try std.testing.expect(isDirectCommandAdapter("materialize-reranker-head"));
}

test "text reward modes score as expected" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scoreTextReward(.exact_match, "yes", "yes"), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scoreTextReward(.exact_match, "yes indeed", "yes"), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scoreTextReward(.exact_match_ci, "Yes", "yes"), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scoreTextReward(.exact_match_ci, "yes indeed", "yes"), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scoreTextReward(.prefix_match, "yes indeed", "yes"), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scoreTextReward(.prefix_match, "indeed yes", "yes"), 1e-6);
}

test "gemma4 token reward modes score control-token completions without decoding" {
    const completion = [_]i32{ 1, 42 };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scoreTokenReward(.token_exact_match, &completion, &completion), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scoreTokenReward(.token_exact_match, &completion, &.{1}), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scoreTokenReward(.token_prefix_match, &completion, &.{1}), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scoreTokenReward(.token_prefix_match, &completion, &.{42}), 1e-6);
}

test "gemma4 preference environment contract fails closed for unknown policy knobs" {
    try std.testing.expect(gemmaPreferenceEnvironmentNameInScope("TERMITE_METAL_DISABLE_FUTURE_TRAINING_FUSION"));
    try std.testing.expect(!gemmaPreferenceEnvironmentNameAttested("TERMITE_METAL_DISABLE_FUTURE_TRAINING_FUSION"));
    try std.testing.expect(gemmaPreferenceEnvironmentNameInScope("ANTFLY_GEMMA4_GRPO_FUTURE_BATCH_ROUTE"));
    try std.testing.expect(!gemmaPreferenceEnvironmentNameAttested("ANTFLY_GEMMA4_GRPO_FUTURE_BATCH_ROUTE"));
    try std.testing.expect(gemmaPreferenceEnvironmentNameInScope("ANTFLY_GEMMA4_PREFERENCE_TRACE"));
    try std.testing.expect(!gemmaPreferenceEnvironmentNameAttested("ANTFLY_GEMMA4_PREFERENCE_TRACE"));
    try std.testing.expect(!gemmaPreferenceEnvironmentNameInScope("HF_HOME"));

    for ([_][]const u8{
        "TERMITE_DISABLE_GRAPH_OUTPUT_OWNED_COPY",
        "TERMITE_DISABLE_GRAPH_OUTPUT_ELISION_OVERRIDE",
        "TERMITE_DISABLE_OUTPUT_HOST_MIRROR_RESYNC",
        "TERMITE_DISABLE_PAGED_KV",
    }) |name| {
        try std.testing.expect(gemmaPreferenceEnvironmentNameInScope(name));
        try std.testing.expect(!gemmaPreferenceEnvironmentNameAttested(name));
        try std.testing.expectError(
            error.UnattestedGemma4PreferenceEnvironmentOverride,
            validateGemmaPreferenceEnvironmentAssignment(name, "1"),
        );
    }
    try std.testing.expect(gemmaPreferenceEnvironmentValueIsCanonical("TERMITE_METAL_DISABLE_BF16_SIMDGROUP_M64", "0"));
    try std.testing.expect(!gemmaPreferenceEnvironmentValueIsCanonical("TERMITE_METAL_DISABLE_BF16_SIMDGROUP_M64", "false"));
    try std.testing.expect(gemmaPreferenceEnvironmentValueIsCanonical("TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_FUSION", "1"));
    try std.testing.expect(!gemmaPreferenceEnvironmentValueIsCanonical("TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_FUSION", "0"));
    try std.testing.expect(gemmaPreferenceEnvironmentValueIsCanonical("TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS", "512"));
    try std.testing.expect(!gemmaPreferenceEnvironmentValueIsCanonical("TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS", "513"));
    try std.testing.expect(gemmaPreferenceEnvironmentValueIsCanonical("TERMITE_DEBUG_DEVICE_GRAD_NORM", "0"));
    try std.testing.expect(!gemmaPreferenceEnvironmentValueIsCanonical("TERMITE_DEBUG_DEVICE_GRAD_NORM", "1"));
    try validateGemmaPreferenceEnvironmentAssignment("TERMITE_DEBUG_DEVICE_GRAD_NORM", "0");
    try std.testing.expectError(
        error.InvalidGemma4PreferenceEnvironmentValue,
        validateGemmaPreferenceEnvironmentAssignment("TERMITE_DEBUG_DEVICE_GRAD_NORM", "1"),
    );
    try validateGemmaPreferenceEnvironmentAssignment("HF_HOME", "/tmp/hf-cache");
}

fn expectStepCommands(plan: Plan, expected: []const []const u8) !void {
    try std.testing.expectEqual(expected.len, plan.steps.len);
    for (expected, 0..) |command, i| {
        try std.testing.expectEqualStrings(command, plan.steps[i].argv[0]);
    }
}

fn expectArgValue(args: []const []const u8, flag: []const u8, expected: []const u8) !void {
    for (args[0..args.len -| 1], 0..) |arg, idx| {
        if (std.mem.eql(u8, arg, flag)) {
            try std.testing.expectEqualStrings(expected, args[idx + 1]);
            return;
        }
    }
    return error.MissingExpectedArgument;
}

fn expectArgPresent(args: []const []const u8, expected: []const u8) !void {
    for (args) |arg| {
        if (std.mem.eql(u8, arg, expected)) return;
    }
    return error.MissingExpectedArgument;
}

fn expectArgPair(args: []const []const u8, flag: []const u8, expected: []const u8) !void {
    for (args[0..args.len -| 1], 0..) |arg, idx| {
        if (std.mem.eql(u8, arg, flag) and std.mem.eql(u8, args[idx + 1], expected)) return;
    }
    return error.MissingExpectedArgument;
}

fn containsArg(args: []const []const u8, expected: []const u8) bool {
    for (args) |arg| {
        if (std.mem.eql(u8, arg, expected)) return true;
    }
    return false;
}
