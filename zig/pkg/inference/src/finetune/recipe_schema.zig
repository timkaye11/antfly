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

//! Serializable recipe and plan types. This module owns no runtime dependencies.

pub const RecipeKind = enum {
    sft,
    lora_sft,
    qlora_sft,
    dpo,
    grpo,
    reranker,
    vlm_retrieval,
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
    /// `sigmoid` (default), `ipo`, or `simpo`. ORPO/CPO require a
    /// differentiable auxiliary SFT term and KTO requires unpaired data, so
    /// those names fail closed in the paired Gemma4 recipe path.
    loss_type: ?[]const u8 = null,
    beta: ?f32 = null,
    label_smoothing: ?f32 = null,
    simpo_gamma: ?f32 = null,
    sft_lambda: ?f32 = null,
    /// IPO regularization parameter. When omitted, `beta` supplies the
    /// standard IPO tau value for compatibility with common trainer APIs.
    ipo_tau: ?f32 = null,
};

pub const GrpoSamplingConfig = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    /// Zero disables top-k filtering.
    top_k: ?usize = null,
};

pub const GrpoConfig = struct {
    group_size: ?usize = null,
    /// Lower PPO clip bound (`1 - clip_epsilon`).
    clip_epsilon: ?f32 = null,
    /// Optional asymmetric upper PPO clip bound (`1 + epsilon_high`).
    epsilon_high: ?f32 = null,
    kl_coef: ?f32 = null,
    /// Fail before optimizer mutation when the unweighted mean token K3
    /// divergence for a sampled group exceeds this bound. Defaults to 0.1.
    train_max_kl: ?f32 = null,
    /// `skip_group` (default) or `abort`.
    train_max_kl_policy: ?[]const u8 = null,
    /// Enables a proportional controller for `kl_coef`. The target and
    /// horizon are required when this is true; coefficient bounds are
    /// optional and default to [0.001, 1.0].
    adaptive_kl: ?bool = null,
    target_kl: ?f32 = null,
    kl_horizon: ?f32 = null,
    min_kl_coef: ?f32 = null,
    max_kl_coef: ?f32 = null,
    advantage_eps: ?f32 = null,
    /// Legacy alias. False is equivalent to `scale_rewards = "none"`.
    normalize_advantage: ?bool = null,
    /// `group` (default), `batch`, or `none`.
    scale_rewards: ?[]const u8 = null,
    /// `grpo`, `bnpo` (default), `dr_grpo`, or `dapo`.
    loss_type: ?[]const u8 = null,
    max_completion_tokens: ?usize = null,
    /// Exclude non-EOS completions that exhaust the generation budget from
    /// the policy/KL loss while retaining them in reward normalization.
    mask_truncated_completions: ?bool = null,
    sampling: ?GrpoSamplingConfig = null,
    reward_mode: ?[]const u8 = null,
};

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
    /// Mid-epoch durable-checkpoint cadence for Gemma4 preference training,
    /// counted in optimizer examples (DPO pairs or GRPO prompt groups) within
    /// each epoch. Requires `every_epochs`, eager sampling, and no
    /// incremental-KV; every other lane rejects it fail-closed.
    every_examples: ?u32 = null,
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
