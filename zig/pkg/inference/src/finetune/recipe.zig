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
const manifest_mod = @import("../models/manifest.zig");
const backends = @import("../backends/backends.zig");
const session_factory = @import("../architectures/session_factory.zig");
const gpt_arch = @import("../architectures/gpt.zig");
const native_backend_choice = @import("../native_backend_choice.zig");
const tokenizer_mod = @import("inference_tokenizer");
const hf_tokenizer = @import("inference_hf_tokenizer");
const compat = @import("../io/compat.zig");
const c_file = @import("../util/c_file.zig");
const command_registry = @import("command_registry.zig");
const ml = @import("ml");
const peft = @import("peft.zig");
const artifact_publication = @import("artifact_publication.zig");
const preference_artifact_contract = @import("preference_artifact_contract.zig");

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

pub const ModelConfig = struct {
    path: ?[]const u8 = null,
    reference_path: ?[]const u8 = null,
    name: ?[]const u8 = null,
    family: ?[]const u8 = null,
    projector_path: ?[]const u8 = null,
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
    use_dora: ?bool = null,
    scaling: ?[]const u8 = null,
};

pub const OptimizerConfig = struct {
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
    advantage_eps: ?f32 = null,
    normalize_advantage: ?bool = null,
    max_completion_tokens: ?usize = null,
    reward_mode: ?[]const u8 = null,
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
};

pub const CheckpointConfig = struct {
    every_epochs: ?u32 = null,
    keep_last: ?u32 = null,
    resume_path: ?[]const u8 = null,
};

pub const RuntimeConfig = struct {
    compiled_required: ?bool = null,
    graph_cache_capacity: ?u8 = null,
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
    eval: ?EvalConfig = null,
    checkpoint: ?CheckpointConfig = null,
    runtime: ?RuntimeConfig = null,
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

const DpoReport = struct {
    schema_version: []const u8 = "antfly_inference_finetune_dpo_report/v1",
    examples: usize,
    loss: f32,
    mean_reward_margin: f32,
    accuracy: f32,
    beta: f32,
    policy_backend: ?[]const u8 = null,
    optimizer_steps: ?u64 = null,
    micro_batch_steps: ?u64 = null,
    device_execution: ?real_autodiff.TrainingExecutionEvidence = null,
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
    input_contract: ?DpoInputContract = null,
    sequence_bucketing: ?GemmaSequenceBucketingTelemetry = null,
    trainable_update: ?TrainableUpdateTelemetry = null,
    preference_session: ?PreferenceSessionRunTelemetry = null,
    benchmark: ?DpoBenchmarkTelemetry = null,
};

const DpoInputContract = struct {
    prompt_input_ids: []const i32,
    chosen_input_ids: []const i32,
    rejected_input_ids: []const i32,
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
    schema_version: []const u8 = "antfly_inference_finetune_grpo_report/v1",
    completions: usize,
    tokens: usize,
    groups: usize,
    loss: f32,
    pg_loss: f32,
    kl_loss: f32,
    clip_fraction: f32,
    mean_reward: ?f32 = null,
    reward_stddev: ?f32 = null,
    reward_mode: ?[]const u8 = null,
    policy_backend: ?[]const u8 = null,
    optimizer_steps: ?u64 = null,
    micro_batch_steps: ?u64 = null,
    device_execution: ?real_autodiff.TrainingExecutionEvidence = null,
    sampling_mode: ?[]const u8 = null,
    policy_logprob_mode: ?[]const u8 = null,
    policy_rescore_completions: ?usize = null,
    training_microbatch_mode: ?[]const u8 = null,
    reference_mode: ?[]const u8 = null,
    reference_scoring_seconds: ?f64 = null,
    reference_cache: ?GrpoReferenceCacheTelemetry = null,
    initial_logprob_parity: ?GrpoInitialLogprobParity = null,
    input_contract: ?GrpoInputContract = null,
    sequence_bucketing: ?GemmaSequenceBucketingTelemetry = null,
    trainable_update: ?TrainableUpdateTelemetry = null,
    preference_session: ?PreferenceSessionRunTelemetry = null,
    benchmark: ?GrpoBenchmarkTelemetry = null,
};

const GrpoInputContract = struct {
    prompt_input_ids: []const i32,
    group_size: usize,
    max_completion_tokens: usize,
    sampling: []const u8,
};

const PreferenceSessionRunTelemetry = struct {
    shared: bool,
    model_admissions: usize,
    run_index: usize,
    reuse_hit: bool,
};

const PreferenceSuiteReport = struct {
    schema_version: []const u8 = "antfly_inference_gemma4_preference_suite/v2",
    status: RunStatus,
    model_path: []const u8,
    backend: []const u8,
    runs_planned: usize,
    runs_completed: usize,
    model_admissions: usize,
    reuse_hits: usize,
    model_admission_seconds: ?f64 = null,
    total_duration_seconds: ?f64 = null,
    runs: []const PreferenceSuiteRunTiming,
};

const PreferenceSuiteRunTiming = struct {
    run_index: usize,
    objective: []const u8,
    duration_seconds: f64,
};

const GemmaSequenceBucketingTelemetry = struct {
    mode: []const u8 = "per-preference-unit-power-of-two-min16-configured-cap",
    configured_max: usize,
    min_required: usize,
    max_required: usize,
    min_bucket: u32,
    max_bucket: u32,
};

const TrainableUpdateTelemetry = struct {
    tensor_count: usize,
    changed_tensor_count: usize,
    max_abs_delta: f32,
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
    trainable_update: TrainableUpdateTelemetry,
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
    grpo_path: []const u8,
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
    if (args.len >= 1 and std.mem.eql(u8, args[0], "run-suite")) {
        return runPreferenceSuite(allocator, io, args[1..]);
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
    const kind = try parseKind(recipe.recipe orelse recipe.kind orelse return error.MissingRecipeKind);
    if (kind == .dpo or kind == .grpo) {
        try validatePreferenceRunArtifacts(allocator, io, recipe, recipe_path);
    }
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
    try runPlan(allocator, io, exe_dir, recipe, plan, manifest_path, training_config_path, training_report_path, null);
}

pub fn loadRecipe(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Parsed(Recipe) {
    const raw = try readFileMax(allocator, io, path, 32 * 1024 * 1024);
    defer allocator.free(raw);
    return std.json.parseFromSlice(Recipe, allocator, raw, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = false,
    });
}

fn validatePreferenceSuiteRecipe(recipe: Recipe) !GemmaPreferenceExecution {
    const kind = try parseKind(recipe.recipe orelse recipe.kind orelse return error.MissingRecipeKind);
    if (kind != .dpo and kind != .grpo) return error.PreferenceSuiteRequiresDpoOrGrpo;

    const family = recipe.model.family orelse try inferFamily(recipe);
    if (!eqlAny(family, &.{ "gemma4", "gemma" })) return error.PreferenceSuiteRequiresGemma4;
    _ = recipe.model.path orelse return error.MissingModelPath;
    // The current multimodal preference runners own their own projector and
    // language-model backend. Admitting one here would be unused and would
    // make the suite's single-admission telemetry false. Keep the contract
    // text-only until those runners accept this same ownership boundary.
    if (recipe.model.projector_path != null) return error.PreferenceSuiteRequiresTextGemma4;
    _ = recipe.artifacts.root orelse return error.PreferenceSuiteRequiresArtifactRoot;
    _ = recipe.artifacts.trained_adapter_dir orelse return error.PreferenceSuiteRequiresTrainedAdapterDir;

    const execution = try resolveGemmaPreferenceExecution(recipe.backend);
    try validateGemmaPreferenceModality(execution.backend_kind, recipe.model.projector_path);
    switch (kind) {
        .dpo => {
            const format = recipe.dataset.format orelse "scalar-logprobs";
            if (!try shouldRunOptimizerBackedGemmaDpo(recipe, format)) {
                return error.PreferenceSuiteRequiresOptimizerBackedGemma4;
            }
        },
        .grpo => {
            const format = recipe.dataset.format orelse "token-logprobs";
            if (!try shouldRunOptimizerBackedGemmaGrpo(recipe, format)) {
                return error.PreferenceSuiteRequiresOptimizerBackedGemma4;
            }
        },
        else => unreachable,
    }
    return execution;
}

const PreferenceSuitePath = preference_artifact_contract.Entry;
const deinitPreferenceSuitePaths = preference_artifact_contract.deinitEntries;
const appendPreferenceSuitePath = preference_artifact_contract.append;

fn appendPreferenceRecipePaths(
    allocator: std.mem.Allocator,
    io: std.Io,
    paths: *std.ArrayList(PreferenceSuitePath),
    recipe: Recipe,
    recipe_path: []const u8,
    recipe_index: usize,
) !void {
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .input, false, "recipe", recipe_path);
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .input, false, "model", recipe.model.path);
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .input, false, "reference_model", recipe.model.reference_path);
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .input, false, "projector", recipe.model.projector_path);
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .input, false, "dataset", recipe.dataset.path);
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .input, false, "train_dataset", recipe.dataset.train_path);
    if (recipe.adapter) |adapter| {
        try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .input, false, "initial_adapter", adapter.path);
    }

    const root = recipe.artifacts.root orelse "antfly-inference-finetune-out";
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .output, true, "artifact_root", root);

    const manifest_path = try manifestPath(allocator, recipe);
    defer allocator.free(manifest_path);
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .output, false, "manifest", manifest_path);

    const training_config_path = try defaultArtifactPath(allocator, recipe, "training_config.json");
    defer allocator.free(training_config_path);
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .output, false, "training_config", training_config_path);

    const training_report_path = try defaultArtifactPath(allocator, recipe, "training_report.json");
    defer allocator.free(training_report_path);
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .output, false, "training_report", training_report_path);

    const kind = try parseKind(recipe.recipe orelse recipe.kind orelse return error.MissingRecipeKind);
    const objective_report_path = switch (kind) {
        .dpo => try dpoReportPath(allocator, recipe),
        .grpo => try grpoReportPath(allocator, recipe),
        else => return error.PreferenceSuiteRequiresDpoOrGrpo,
    };
    defer allocator.free(objective_report_path);
    try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .output, false, "objective_report", objective_report_path);

    if (requestsAdapterTraining(recipe)) {
        if (recipe.adapter == null or recipe.adapter.?.path == null) {
            const configured_bootstrap = recipe.artifacts.adapter_dir;
            const bootstrap_path = configured_bootstrap orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
            defer if (configured_bootstrap == null) allocator.free(bootstrap_path);
            try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .output, false, "adapter_bootstrap", bootstrap_path);
        }
        const configured_trained = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir;
        const trained_path = configured_trained orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
        defer if (configured_trained == null) allocator.free(trained_path);
        try appendPreferenceSuitePath(allocator, io, paths, recipe_index, .output, false, "trained_adapter", trained_path);
    }
}

fn defaultPreferenceSuiteReportPath(allocator: std.mem.Allocator, first_recipe: Recipe) ![]u8 {
    const root = first_recipe.artifacts.root orelse return error.PreferenceSuiteRequiresArtifactRoot;
    return std.fs.path.join(allocator, &.{ std.fs.path.dirname(root) orelse ".", "preference_suite_report.json" });
}

fn validateCollectedPreferencePaths(paths: []const PreferenceSuitePath) !void {
    preference_artifact_contract.validate(paths) catch |err| switch (err) {
        error.PreferenceInputOutputConflict => return error.PreferenceSuiteInputOutputConflict,
        error.PreferenceArtifactConflict => return error.PreferenceSuiteArtifactConflict,
    };
}

fn validatePreferenceRunArtifacts(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    recipe_path: []const u8,
) !void {
    var paths: std.ArrayList(PreferenceSuitePath) = .empty;
    defer deinitPreferenceSuitePaths(allocator, &paths);
    try appendPreferenceRecipePaths(allocator, io, &paths, recipe, recipe_path, 0);
    try validateCollectedPreferencePaths(paths.items);
}

fn validatePreferenceSuiteArtifacts(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipes: []const Recipe,
    recipe_paths: []const []const u8,
    suite_report_path: []const u8,
) !void {
    if (recipes.len != recipe_paths.len) return error.PreferenceSuiteRecipePathCountMismatch;
    var paths: std.ArrayList(PreferenceSuitePath) = .empty;
    defer deinitPreferenceSuitePaths(allocator, &paths);
    for (recipes, recipe_paths, 0..) |recipe, recipe_path, recipe_index| {
        try appendPreferenceRecipePaths(allocator, io, &paths, recipe, recipe_path, recipe_index);
    }
    try appendPreferenceSuitePath(allocator, io, &paths, null, .output, false, "suite_report", suite_report_path);

    try validateCollectedPreferencePaths(paths.items);
}

fn writePreferenceSuiteReport(
    allocator: std.mem.Allocator,
    io: std.Io,
    report_path: []const u8,
    status: RunStatus,
    model_path: []const u8,
    backend_kind: gemma4_real_autodiff.BackendKind,
    runs_planned: usize,
    runs_completed: usize,
    model_admissions: usize,
    reuse_hits: usize,
    model_admission_seconds: ?f64,
    total_duration_seconds: ?f64,
    run_timings: []const PreferenceSuiteRunTiming,
) !void {
    if (std.fs.path.dirname(report_path)) |parent| {
        if (parent.len > 0) try compat.cwd().createDirPath(io, parent);
    }
    try writeJsonFile(allocator, io, report_path, PreferenceSuiteReport{
        .status = status,
        .model_path = model_path,
        .backend = @tagName(backend_kind),
        .runs_planned = runs_planned,
        .runs_completed = runs_completed,
        .model_admissions = model_admissions,
        .reuse_hits = reuse_hits,
        .model_admission_seconds = model_admission_seconds,
        .total_duration_seconds = total_duration_seconds,
        .runs = run_timings,
    });
}

fn runPreferenceSuite(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    var recipe_paths: std.ArrayList([]const u8) = .empty;
    defer recipe_paths.deinit(allocator);
    var report_path_arg: ?[]const u8 = null;
    var dry_run = false;
    var arg_idx: usize = 0;
    while (arg_idx < args.len) : (arg_idx += 1) {
        if (std.mem.eql(u8, args[arg_idx], "--report")) {
            arg_idx += 1;
            if (arg_idx >= args.len or report_path_arg != null) return usageError();
            report_path_arg = args[arg_idx];
        } else if (std.mem.eql(u8, args[arg_idx], "--dry-run") or std.mem.eql(u8, args[arg_idx], "--plan")) {
            dry_run = true;
        } else {
            try recipe_paths.append(allocator, args[arg_idx]);
        }
    }
    if (recipe_paths.items.len < 2) return error.PreferenceSuiteRequiresMultipleRecipes;

    const parsed = try allocator.alloc(std.json.Parsed(Recipe), recipe_paths.items.len);
    var parsed_count: usize = 0;
    defer {
        for (parsed[0..parsed_count]) |*item| item.deinit();
        allocator.free(parsed);
    }
    for (recipe_paths.items, 0..) |path, idx| {
        parsed[idx] = try loadRecipe(allocator, io, path);
        parsed_count += 1;
    }

    const recipes = try allocator.alloc(Recipe, parsed.len);
    defer allocator.free(recipes);
    for (parsed, 0..) |item, idx| recipes[idx] = item.value;

    const first_execution = try validatePreferenceSuiteRecipe(recipes[0]);
    const first_model_path = recipes[0].model.path.?;
    const canonical_model_path = try compat.cwd().realPathFileAlloc(io, first_model_path, allocator);
    defer allocator.free(canonical_model_path);
    for (recipes, 0..) |recipe, idx| {
        const execution = try validatePreferenceSuiteRecipe(recipe);
        if (execution.backend_kind != first_execution.backend_kind) return error.PreferenceSuiteBackendMismatch;
        const candidate_model_path = try compat.cwd().realPathFileAlloc(io, recipe.model.path.?, allocator);
        defer allocator.free(candidate_model_path);
        if (!std.mem.eql(u8, canonical_model_path, candidate_model_path)) return error.PreferenceSuiteModelMismatch;
        if (recipe.model.reference_path) |reference_path| {
            const canonical_reference_path = try compat.cwd().realPathFileAlloc(io, reference_path, allocator);
            defer allocator.free(canonical_reference_path);
            if (!std.mem.eql(u8, canonical_model_path, canonical_reference_path)) {
                return error.UnsupportedReferencePath;
            }
        }

        var plan_arena = std.heap.ArenaAllocator.init(allocator);
        defer plan_arena.deinit();
        const plan = try buildPlan(plan_arena.allocator(), recipe);
        print("preference-suite[{d}/{d}] ", .{ idx + 1, recipes.len });
        try printPlan(io, recipe, plan);
    }
    const owned_report_path = if (report_path_arg) |path|
        try allocator.dupe(u8, path)
    else
        try defaultPreferenceSuiteReportPath(allocator, recipes[0]);
    defer allocator.free(owned_report_path);
    try validatePreferenceSuiteArtifacts(allocator, io, recipes, recipe_paths.items, owned_report_path);
    if (dry_run) return;

    const suite_started_ns = platform.time.monotonicNs();
    const no_run_timings = [_]PreferenceSuiteRunTiming{};
    try writePreferenceSuiteReport(
        allocator,
        io,
        owned_report_path,
        .planned,
        canonical_model_path,
        first_execution.backend_kind,
        recipes.len,
        0,
        0,
        0,
        null,
        null,
        &no_run_timings,
    );

    const admission_started_ns = platform.time.monotonicNs();
    var preference_session = GemmaPreferenceSession.init(
        allocator,
        io,
        canonical_model_path,
        first_execution.backend_kind,
    ) catch |err| {
        writePreferenceSuiteReport(
            allocator,
            io,
            owned_report_path,
            .failed,
            canonical_model_path,
            first_execution.backend_kind,
            recipes.len,
            0,
            0,
            0,
            null,
            @as(f64, @floatFromInt(platform.time.monotonicNs() - suite_started_ns)) / std.time.ns_per_s,
            &no_run_timings,
        ) catch {};
        return err;
    };
    defer preference_session.deinit();
    const model_admission_seconds = @as(f64, @floatFromInt(platform.time.monotonicNs() - admission_started_ns)) / std.time.ns_per_s;
    print("preference session admitted: model={s} backend={s}\n", .{
        preference_session.model_path,
        @tagName(preference_session.backend_kind),
    });

    try writePreferenceSuiteReport(
        allocator,
        io,
        owned_report_path,
        .running,
        preference_session.model_path,
        preference_session.backend_kind,
        recipes.len,
        0,
        1,
        0,
        model_admission_seconds,
        null,
        &no_run_timings,
    );

    const exe_dir = try std.process.executableDirPathAlloc(io, allocator);
    defer allocator.free(exe_dir);
    var runs_completed: usize = 0;
    var run_timings: std.ArrayList(PreferenceSuiteRunTiming) = .empty;
    defer run_timings.deinit(allocator);
    for (recipes) |recipe| {
        var plan_arena = std.heap.ArenaAllocator.init(allocator);
        defer plan_arena.deinit();
        const plan = try buildPlan(plan_arena.allocator(), recipe);
        const manifest_path = try manifestPath(allocator, recipe);
        defer allocator.free(manifest_path);
        const training_config_path = try defaultArtifactPath(allocator, recipe, "training_config.json");
        defer allocator.free(training_config_path);
        const training_report_path = try defaultArtifactPath(allocator, recipe, "training_report.json");
        defer allocator.free(training_report_path);
        const run_started_ns = platform.time.monotonicNs();
        runPlan(
            allocator,
            io,
            exe_dir,
            recipe,
            plan,
            manifest_path,
            training_config_path,
            training_report_path,
            &preference_session,
        ) catch |err| {
            writePreferenceSuiteReport(
                allocator,
                io,
                owned_report_path,
                .failed,
                preference_session.model_path,
                preference_session.backend_kind,
                recipes.len,
                runs_completed,
                1,
                preference_session.reuseHits(),
                model_admission_seconds,
                @as(f64, @floatFromInt(platform.time.monotonicNs() - suite_started_ns)) / std.time.ns_per_s,
                run_timings.items,
            ) catch {};
            return err;
        };
        runs_completed += 1;
        const kind = try parseKind(recipe.recipe orelse recipe.kind orelse return error.MissingRecipeKind);
        try run_timings.append(allocator, .{
            .run_index = runs_completed,
            .objective = @tagName(kind),
            .duration_seconds = @as(f64, @floatFromInt(platform.time.monotonicNs() - run_started_ns)) / std.time.ns_per_s,
        });
        try writePreferenceSuiteReport(
            allocator,
            io,
            owned_report_path,
            .running,
            preference_session.model_path,
            preference_session.backend_kind,
            recipes.len,
            runs_completed,
            1,
            preference_session.reuseHits(),
            model_admission_seconds,
            null,
            run_timings.items,
        );
    }

    try writePreferenceSuiteReport(
        allocator,
        io,
        owned_report_path,
        .succeeded,
        preference_session.model_path,
        preference_session.backend_kind,
        recipes.len,
        runs_completed,
        1,
        preference_session.reuseHits(),
        model_admission_seconds,
        @as(f64, @floatFromInt(platform.time.monotonicNs() - suite_started_ns)) / std.time.ns_per_s,
        run_timings.items,
    );
    print("preference suite report: {s}\n", .{owned_report_path});
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
        .{ .name = "grpo_multimodal_gemma_dry_run", .recipe_path = "pkg/inference/testdata/recipe_grpo_multimodal_gemma_fast.json", .mode = .dry_run },
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
    try runPlan(allocator, io, exe_dir, recipe, plan, manifest_path, training_config_path, training_report_path, null);
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
        .synthetic_gemma_grpo_execute => blk: {
            const assets = try writeSyntheticGemmaSmokeAssets(allocator, io, case_root);
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
    };
}

fn applyFastSmokeRecipeOverrides(recipe: *Recipe, overrides: FastSmokeRecipeOverrides) void {
    if (overrides.model_path) |value| recipe.model.path = value;
    if (overrides.reference_path) |value| recipe.model.reference_path = value;
    if (overrides.dataset_path) |value| recipe.dataset.path = value;
    if (overrides.dataset_format) |value| recipe.dataset.format = value;
    if (overrides.train_path) |value| recipe.dataset.train_path = value;
    if (overrides.eval_path) |value| recipe.dataset.eval_path = value;
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

    if (eqlAny(family, &.{ "gemma4", "gemma" })) {
        switch (kind) {
            .sft => return error.Gemma4FullSftNotYetSupported,
            .qlora_sft => return error.Gemma4QLoRANotYetSupported,
            .dpo, .grpo => if (requestsAdapterTraining(recipe)) {
                try validateGemma4PreferenceRecipeContract(recipe, kind);
            },
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
            eval.full_task_minimums != null)
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
        if (runtime.compiled_required != null or runtime.graph_cache_capacity != null) {
            return error.UnsupportedGemma4RuntimeOption;
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
        recipe.grpo.advantage_eps != null or
        recipe.grpo.normalize_advantage != null or
        recipe.grpo.max_completion_tokens != null or
        recipe.grpo.reward_mode != null)
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

/// Production admission for optimizer-backed Gemma4 preference recipes.
///
/// The shared recipe schema intentionally spans several trainer families. A
/// field being parseable therefore does not mean that this runner implements
/// it. Keep this contract next to the Gemma4 SFT contract and reject every
/// unimplemented semantic before model admission or artifact publication.
fn validateGemma4PreferenceRecipeContract(recipe: Recipe, kind: RecipeKind) !void {
    if (kind != .dpo and kind != .grpo) return error.InvalidPreferenceRecipeKind;
    _ = recipe.model.path orelse return error.MissingModelPath;
    if (kind == .dpo and recipe.model.projector_path != null) {
        return error.Gemma4MultimodalDpoNotSupported;
    }

    if (recipe.dataset.path != null and recipe.dataset.train_path != null) {
        return error.ConflictingPreferenceDatasetPaths;
    }
    _ = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    if (recipe.dataset.eval_path != null or
        recipe.dataset.eval_split != null or
        recipe.dataset.prepared_path != null or
        recipe.dataset.cache_path != null or
        recipe.dataset.train_cache_path != null or
        recipe.dataset.eval_cache_path != null or
        recipe.dataset.labels != null or
        recipe.dataset.eval_max_examples != null or
        !std.mem.eql(u8, recipe.dataset.train_split orelse "train", "train"))
    {
        return error.UnsupportedGemma4PreferenceDatasetOption;
    }
    if (recipe.dataset.max_examples) |max_examples| {
        if (max_examples == 0) return error.InvalidPreferenceMaxExamples;
    }
    const default_max_seq_len: usize = if (kind == .dpo) 512 else 128;
    _ = try gemma4.validateTrainingSequenceLength(
        recipe.dataset.max_seq_len orelse default_max_seq_len,
        std.math.maxInt(u32),
    );
    const format = recipe.dataset.format orelse if (kind == .dpo) "scalar-logprobs" else "token-logprobs";
    switch (kind) {
        .dpo => if (!std.mem.eql(u8, format, "text-preference") and
            !std.mem.eql(u8, format, "rendered-text-preference"))
        {
            return error.UnsupportedGemma4PreferenceFormat;
        },
        .grpo => if (!std.mem.eql(u8, format, "text-grpo") and
            !std.mem.eql(u8, format, "rendered-text-grpo"))
        {
            return error.UnsupportedGemma4PreferenceFormat;
        },
        else => unreachable,
    }

    const adapter = recipe.adapter orelse AdapterConfig{};
    try validateGemmaAdapterOptions(adapter);
    const rank = adapterRank(adapter, kind);
    if (rank == 0 or rank > std.math.maxInt(u32)) return error.InvalidLoRARank;
    const alpha = adapterAlpha(adapter);
    if (!std.math.isFinite(alpha) or alpha <= 0.0) return error.InvalidLoRAAlpha;
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
    if (adapter.path != null and recipe.artifacts.adapter_dir != null) {
        return error.ConflictingGemma4PreferenceAdapterPaths;
    }

    const learning_rate = recipe.optimizer.learning_rate orelse 0.0001;
    if (!std.math.isFinite(learning_rate) or learning_rate <= 0.0) return error.InvalidLearningRate;
    const weight_decay = recipe.optimizer.weight_decay orelse 0.01;
    if (!std.math.isFinite(weight_decay) or weight_decay < 0.0) return error.InvalidWeightDecay;
    const epochs = recipe.optimizer.epochs orelse 1;
    if (epochs == 0) return error.InvalidEpochCount;
    const gradient_accumulation_steps = recipe.optimizer.gradient_accumulation_steps orelse 1;
    if (gradient_accumulation_steps == 0) return error.InvalidGradientAccumulationSteps;
    const max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0;
    if (!std.math.isFinite(max_grad_norm) or max_grad_norm < 0.0) return error.InvalidMaxGradNorm;
    if (recipe.optimizer.lr_scheduler != null or
        recipe.optimizer.warmup_ratio != null or
        recipe.optimizer.warmup_steps != null or
        recipe.optimizer.num_cycles != null or
        recipe.optimizer.max_steps != null or
        recipe.optimizer.micro_batch_size != null or
        recipe.optimizer.llrd_decay != null or
        recipe.optimizer.schedule_free != null)
    {
        return error.UnsupportedGemma4PreferenceOptimizerOption;
    }

    if (recipe.eval != null) return error.UnsupportedGemma4PreferenceEvalOption;
    if (recipe.checkpoint != null) return error.UnsupportedGemma4PreferenceCheckpointOption;
    if (recipe.runtime != null) return error.UnsupportedGemma4PreferenceRuntimeOption;
    if (recipe.trainer) |trainer| {
        if (!std.mem.eql(u8, trainer, "autodiff") and !std.mem.eql(u8, trainer, "auto")) {
            return error.UnsupportedGemma4Trainer;
        }
    }
    if (recipe.artifacts.prepared_path != null or
        recipe.artifacts.materialized_dir != null or
        recipe.artifacts.validation_report_path != null or
        recipe.artifacts.evaluation_report_path != null or
        recipe.artifacts.reload_report_path != null)
    {
        return error.UnsupportedGemma4PreferenceArtifactOption;
    }

    switch (kind) {
        .dpo => {
            const beta = recipe.preference.beta orelse 0.1;
            if (!std.math.isFinite(beta) or beta <= 0.0) return error.InvalidDpoBeta;
            if (recipe.preference.simpo_gamma != null or
                recipe.preference.sft_lambda != null or
                recipe.preference.ipo_tau != null)
            {
                return error.UnsupportedGemma4DpoOption;
            }
            if (recipe.grpo.group_size != null or
                recipe.grpo.clip_epsilon != null or
                recipe.grpo.kl_coef != null or
                recipe.grpo.advantage_eps != null or
                recipe.grpo.normalize_advantage != null or
                recipe.grpo.max_completion_tokens != null or
                recipe.grpo.reward_mode != null)
            {
                return error.UnsupportedGemma4DpoOption;
            }
        },
        .grpo => {
            if (recipe.preference.beta != null or
                recipe.preference.simpo_gamma != null or
                recipe.preference.sft_lambda != null or
                recipe.preference.ipo_tau != null)
            {
                return error.UnsupportedGemma4GrpoOption;
            }
            const group_size = recipe.grpo.group_size orelse 2;
            if (group_size < 2 or group_size > 8) return error.InvalidGrpoGroupSize;
            const clip_epsilon = recipe.grpo.clip_epsilon orelse 0.2;
            if (!std.math.isFinite(clip_epsilon) or clip_epsilon <= 0.0) return error.InvalidGrpoClipEpsilon;
            const kl_coef = recipe.grpo.kl_coef orelse 0.04;
            if (!std.math.isFinite(kl_coef) or kl_coef < 0.0) return error.InvalidGrpoKlCoefficient;
            const advantage_eps = recipe.grpo.advantage_eps orelse 1e-8;
            if (!std.math.isFinite(advantage_eps) or advantage_eps <= 0.0) return error.InvalidGrpoAdvantageEpsilon;
            if ((recipe.grpo.max_completion_tokens orelse 4) == 0) return error.InvalidMaxCompletionTokens;
            _ = try parseTextRewardMode(recipe.grpo.reward_mode orelse "exact-match");
        },
        else => unreachable,
    }
}

fn pathIsSameOrWithin(parent: []const u8, path: []const u8) bool {
    if (std.mem.eql(u8, parent, path)) return true;
    if (parent.len == 0 or path.len <= parent.len or !std.mem.startsWith(u8, path, parent)) return false;
    if (std.fs.path.isSep(parent[parent.len - 1])) return true;
    return std.fs.path.isSep(path[parent.len]);
}

fn pathsOverlap(a: []const u8, b: []const u8) bool {
    return pathIsSameOrWithin(a, b) or pathIsSameOrWithin(b, a);
}

fn preferencePathsReferToSameArtifact(
    allocator: std.mem.Allocator,
    io: std.Io,
    a: []const u8,
    b: []const u8,
) !bool {
    if (std.mem.eql(u8, a, b)) return true;
    const canonical_a = try compat.cwd().realPathFileAlloc(io, a, allocator);
    defer allocator.free(canonical_a);
    const canonical_b = try compat.cwd().realPathFileAlloc(io, b, allocator);
    defer allocator.free(canonical_b);
    return std.mem.eql(u8, canonical_a, canonical_b);
}

fn validateGemma4ArtifactDirectories(
    allocator: std.mem.Allocator,
    recipe: Recipe,
    prepared_path: []const u8,
    eval_prepared_path: []const u8,
    bootstrap_dir: []const u8,
    trained_dir: []const u8,
) !void {
    const resolved_bootstrap = try std.fs.path.resolve(allocator, &.{bootstrap_dir});
    defer allocator.free(resolved_bootstrap);
    const resolved_trained = try std.fs.path.resolve(allocator, &.{trained_dir});
    defer allocator.free(resolved_trained);
    if (pathIsSameOrWithin(resolved_bootstrap, resolved_trained) or
        pathIsSameOrWithin(resolved_trained, resolved_bootstrap))
    {
        return error.Gemma4BootstrapAndTrainingOutputConflict;
    }

    const artifact_root = recipe.artifacts.root orelse "antfly-inference-finetune-out";
    const resolved_root = try std.fs.path.resolve(allocator, &.{artifact_root});
    defer allocator.free(resolved_root);
    if (pathIsSameOrWithin(resolved_bootstrap, resolved_root) or
        pathIsSameOrWithin(resolved_trained, resolved_root))
    {
        return error.Gemma4OutputConflictsWithArtifactRoot;
    }

    const planned_files = [_]?[]const u8{
        prepared_path,
        eval_prepared_path,
        recipe.artifacts.manifest_path,
    };
    for (planned_files) |maybe_path| {
        const path = maybe_path orelse continue;
        const resolved = try std.fs.path.resolve(allocator, &.{path});
        defer allocator.free(resolved);
        if (pathIsSameOrWithin(resolved_bootstrap, resolved) or
            pathIsSameOrWithin(resolved_trained, resolved))
        {
            return error.Gemma4OutputContainsPlannedArtifact;
        }
    }
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

fn buildDpoPlan(allocator: std.mem.Allocator, recipe: Recipe) !Plan {
    _ = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    return .{ .steps = try allocator.dupe(Step, &.{
        .{
            .kind = .direct_dpo,
            .name = "train-eval",
            .argv = try argv(allocator, &.{"antfly-inference-internal-dpo"}),
        },
    }) };
}

fn buildGrpoPlan(allocator: std.mem.Allocator, recipe: Recipe) !Plan {
    _ = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    return .{ .steps = try allocator.dupe(Step, &.{
        .{
            .kind = .direct_grpo,
            .name = "train-eval",
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
    preference_session: ?*GemmaPreferenceSession,
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
                runDirectDpo(allocator, io, recipe, report_path, preference_session) catch |err| {
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
                runDirectGrpo(allocator, io, recipe, report_path, preference_session) catch |err| {
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
            },
            .direct_grpo => {
                const report_path = try grpoReportPath(allocator, recipe);
                defer allocator.free(report_path);
                try appendUniquePlannedPathOwned(allocator, planned, "grpo_report", report_path);
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

fn digestDirectoryAlloc(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8) !DirectoryDigest {
    var entries: std.ArrayListUnmanaged(DirectoryDigestEntry) = .empty;
    errdefer entries.deinit(allocator);
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
            continue;
        }

        const stat = try std.Io.Dir.cwd().statFile(io, child_path, .{});
        const digest = try sha256FileAlloc(allocator, io, child_path);
        try entries.append(allocator, .{
            .relative_path = rel_path,
            .size_bytes = stat.size,
            .digest = digest,
        });
    }
}

fn lessThanDirectoryDigestEntry(_: void, lhs: DirectoryDigestEntry, rhs: DirectoryDigestEntry) bool {
    return std.mem.order(u8, lhs.relative_path, rhs.relative_path) == .lt;
}

fn writeJsonFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, value: anytype) !void {
    const rendered = try std.json.Stringify.valueAlloc(allocator, value, .{ .whitespace = .indent_2 });
    defer allocator.free(rendered);
    try artifact_publication.writeFileAtomicReplace(allocator, io, path, rendered);
}

fn publishGemmaPreferenceBundleAndReport(
    allocator: std.mem.Allocator,
    io: std.Io,
    trainer: *real_autodiff.RealAutodiffTrainer,
    base_model_dir: []const u8,
    bootstrap_dir: []const u8,
    trained_dir: []const u8,
    report_path: []const u8,
    report: anytype,
) !void {
    const rendered = try std.json.Stringify.valueAlloc(allocator, report, .{ .whitespace = .indent_2 });
    defer allocator.free(rendered);
    try gemma4_real_autodiff.saveTrainerAsGemmaBundleWithCompletionEvidence(
        allocator,
        trainer,
        base_model_dir,
        bootstrap_dir,
        trained_dir,
        rendered,
    );
    try artifact_publication.writeFileAtomicReplace(allocator, io, report_path, rendered);
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
    ranked_first,
};

const TextRewardCtx = struct {
    allocator: std.mem.Allocator,
    tokenizer: tokenizer_mod.Tokenizer,
    targets: []const []const u8,
    mode: TextRewardMode,
    group_size: usize,
    ranked_completion_index: usize = 0,

    fn score(
        ctx: *anyopaque,
        prompt_idx: usize,
        completion_tokens: []const i32,
    ) !f32 {
        const self: *TextRewardCtx = @ptrCast(@alignCast(ctx));
        if (self.mode == .ranked_first) {
            return scoreRankedFirst(&self.ranked_completion_index, self.group_size);
        }
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

fn scoreRankedFirst(completion_index: *usize, group_size: usize) !f32 {
    if (group_size < 2 or completion_index.* >= group_size) return error.InvalidGroupSize;
    const reward: f32 = if (completion_index.* == 0) 1.0 else 0.0;
    completion_index.* = if (completion_index.* + 1 == group_size) 0 else completion_index.* + 1;
    return reward;
}

fn scoreTextReward(mode: TextRewardMode, completion_trimmed: []const u8, target_trimmed: []const u8) f32 {
    return switch (mode) {
        .exact_match => blk: {
            if (std.mem.eql(u8, completion_trimmed, target_trimmed)) break :blk 1.0;
            if (std.mem.indexOf(u8, completion_trimmed, target_trimmed) != null) break :blk 0.5;
            break :blk 0.0;
        },
        .exact_match_ci => if (std.ascii.eqlIgnoreCase(completion_trimmed, target_trimmed)) 1.0 else 0.0,
        .prefix_match => if (std.mem.startsWith(u8, completion_trimmed, target_trimmed)) 1.0 else 0.0,
        .token_exact_match, .token_prefix_match, .ranked_first => unreachable,
    };
}

fn scoreTokenReward(mode: TextRewardMode, completion_tokens: []const i32, target_tokens: []const i32) f32 {
    return switch (mode) {
        .token_exact_match => if (std.mem.eql(i32, completion_tokens, target_tokens)) 1.0 else 0.0,
        .token_prefix_match => if (std.mem.startsWith(i32, completion_tokens, target_tokens)) 1.0 else 0.0,
        .exact_match, .exact_match_ci, .prefix_match, .ranked_first => unreachable,
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

fn runDirectDpo(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    report_path: []const u8,
    preference_session: ?*GemmaPreferenceSession,
) !void {
    const path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const format = recipe.dataset.format orelse "scalar-logprobs";
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
            .examples = batch.policy_chosen_logps.len,
            .loss = result.loss,
            .mean_reward_margin = result.mean_reward_margin,
            .accuracy = result.accuracy,
            .beta = recipe.preference.beta orelse 0.1,
        });
        print("dpo report: {s}\n", .{report_path});
        return;
    }
    if (!std.mem.eql(u8, format, "text-preference") and !std.mem.eql(u8, format, "rendered-text-preference")) {
        return error.UnsupportedDpoFormat;
    }
    if (try shouldRunOptimizerBackedQwen2Dpo(recipe, format)) {
        try runOptimizerBackedQwen2Dpo(allocator, io, recipe, path, report_path);
        return;
    }
    if (try shouldRunOptimizerBackedGemmaDpo(recipe, format)) {
        try runOptimizerBackedGemmaDpo(allocator, io, recipe, path, report_path, preference_session);
        return;
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

    var samples = try loadDpoTextPreferenceSamples(
        allocator,
        io,
        path,
        recipe,
        PreferenceTextTokenizerView.fromLoadedModel(policy_model),
    );
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
    selected_margins: ?[]f32,
    precompute_seconds: f64,
    base_equivalent_policy: bool,

    fn deinit(self: *GemmaDpoReferenceCache) void {
        self.allocator.free(self.chosen_logps);
        self.allocator.free(self.rejected_logps);
        if (self.selected_margins) |margins| self.allocator.free(margins);
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

/// Host snapshot of the trainable LoRA tensors used to prove that an
/// optimizer-backed preference run changed policy parameters. CUDA and Metal
/// keep the live optimizer weights device-resident, so callers synchronize
/// them once after the timed training region before asking for a summary.
const TrainableParameterSnapshot = struct {
    allocator: std.mem.Allocator,
    tensors: [][]f32,

    fn init(
        allocator: std.mem.Allocator,
        slots: []const real_autodiff.RealAutodiffTrainer.ParamSlot,
    ) !TrainableParameterSnapshot {
        const tensors = try allocator.alloc([]f32, slots.len);
        var initialized: usize = 0;
        errdefer {
            for (tensors[0..initialized]) |tensor| allocator.free(tensor);
            allocator.free(tensors);
        }
        for (slots, 0..) |slot, idx| {
            tensors[idx] = try allocator.dupe(f32, slot.weights);
            initialized += 1;
        }
        return .{ .allocator = allocator, .tensors = tensors };
    }

    fn deinit(self: *TrainableParameterSnapshot) void {
        for (self.tensors) |tensor| self.allocator.free(tensor);
        self.allocator.free(self.tensors);
        self.* = undefined;
    }

    fn summarize(
        self: *const TrainableParameterSnapshot,
        slots: []const real_autodiff.RealAutodiffTrainer.ParamSlot,
    ) !TrainableUpdateTelemetry {
        if (self.tensors.len != slots.len) return error.TrainableSnapshotShapeMismatch;
        var changed_tensor_count: usize = 0;
        var max_abs_delta: f32 = 0.0;
        for (self.tensors, slots) |initial, slot| {
            if (initial.len != slot.weights.len) return error.TrainableSnapshotShapeMismatch;
            var tensor_changed = false;
            for (initial, slot.weights) |before, after| {
                if (!std.math.isFinite(before) or !std.math.isFinite(after)) {
                    return error.NonFiniteTrainableParameter;
                }
                const delta = @abs(after - before);
                max_abs_delta = @max(max_abs_delta, delta);
                tensor_changed = tensor_changed or delta > 0.0;
            }
            if (tensor_changed) changed_tensor_count += 1;
        }
        return .{
            .tensor_count = slots.len,
            .changed_tensor_count = changed_tensor_count,
            .max_abs_delta = max_abs_delta,
        };
    }
};

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

    fn finish(
        self: *DpoBenchmarkRecorder,
        trainable_update: TrainableUpdateTelemetry,
    ) !DpoBenchmarkTelemetry {
        if (self.updates_seen != total_updates) return error.DpoBenchmarkUpdateCountMismatch;
        if (trainable_update.tensor_count == 0 or
            trainable_update.changed_tensor_count == 0 or
            !(trainable_update.max_abs_delta > 0.0))
        {
            return error.DpoBenchmarkNoPolicyMovement;
        }
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

    fn finish(
        self: *GrpoBenchmarkRecorder,
        trainable_update: TrainableUpdateTelemetry,
    ) !GrpoBenchmarkTelemetry {
        if (self.updates_seen != total_updates) return error.GrpoBenchmarkUpdateCountMismatch;
        // The sampled-token log-probability delta is useful trajectory
        // telemetry, but it is not a reliable parameter-movement oracle. A
        // saturated token can remain numerically unchanged while FP32 LoRA
        // tensors receive a valid update. Prove policy movement directly from
        // the device-resident trainables after synchronizing them to host.
        if (trainable_update.tensor_count == 0 or
            trainable_update.changed_tensor_count == 0 or
            !(trainable_update.max_abs_delta > 0.0))
        {
            return error.GrpoBenchmarkNoPolicyMovement;
        }

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
            .trainable_update = trainable_update,
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

fn shouldCoalesceGemmaDpoSingleTokenPairs(
    backend_kind: gemma4_real_autodiff.BackendKind,
    chosen_examples: []const gemma4.PreparedExampleInput,
    rejected_examples: []const gemma4.PreparedExampleInput,
) bool {
    if (backend_kind != .metal and backend_kind != .cuda) return false;
    return allGemmaDpoPairsAreSingleTokenSharedPrompt(chosen_examples, rejected_examples);
}

/// Keep compiled preference graphs in a bounded family of useful sequence
/// shapes instead of padding every batch-1 update to the dataset ceiling.
/// The configured maximum remains a hard admission limit and an exact final
/// bucket when it is not a power of two.
fn gemmaPreferenceSequenceBucket(required_tokens: usize, configured_max: usize) !u32 {
    if (required_tokens == 0) return error.EmptySequence;
    if (required_tokens > configured_max) return error.SequenceTooLong;
    if (configured_max > std.math.maxInt(u32)) return error.SequenceTooLong;

    var bucket: usize = @min(configured_max, 16);
    while (bucket < required_tokens) {
        const doubled = std.math.mul(usize, bucket, 2) catch configured_max;
        bucket = @min(doubled, configured_max);
    }
    return @intCast(bucket);
}

fn summarizeGemmaDpoSequenceBucketing(
    chosen_examples: []const gemma4.PreparedExampleInput,
    rejected_examples: []const gemma4.PreparedExampleInput,
    configured_max: usize,
) !GemmaSequenceBucketingTelemetry {
    if (chosen_examples.len == 0 or chosen_examples.len != rejected_examples.len) {
        return error.DpoBatchAlignmentMismatch;
    }
    var min_required: usize = std.math.maxInt(usize);
    var max_required: usize = 0;
    var min_bucket: u32 = std.math.maxInt(u32);
    var max_bucket: u32 = 0;
    for (chosen_examples, rejected_examples) |chosen, rejected| {
        const required = @max(chosen.num_input_tokens, rejected.num_input_tokens);
        const bucket = try gemmaPreferenceSequenceBucket(required, configured_max);
        min_required = @min(min_required, required);
        max_required = @max(max_required, required);
        min_bucket = @min(min_bucket, bucket);
        max_bucket = @max(max_bucket, bucket);
    }
    return .{
        .configured_max = configured_max,
        .min_required = min_required,
        .max_required = max_required,
        .min_bucket = min_bucket,
        .max_bucket = max_bucket,
    };
}

fn summarizeGemmaGrpoSequenceBucketing(
    prompts: []const []const i32,
    max_completion_tokens: usize,
    configured_max: usize,
) !GemmaSequenceBucketingTelemetry {
    if (prompts.len == 0) return error.NoTrainingData;
    var min_required: usize = std.math.maxInt(usize);
    var max_required: usize = 0;
    var min_bucket: u32 = std.math.maxInt(u32);
    var max_bucket: u32 = 0;
    for (prompts) |prompt| {
        const required = std.math.add(usize, prompt.len, max_completion_tokens) catch return error.SequenceTooLong;
        const bucket = try gemmaPreferenceSequenceBucket(required, configured_max);
        min_required = @min(min_required, required);
        max_required = @max(max_required, required);
        min_bucket = @min(min_bucket, bucket);
        max_bucket = @max(max_bucket, bucket);
    }
    return .{
        .configured_max = configured_max,
        .min_required = min_required,
        .max_required = max_required,
        .min_bucket = min_bucket,
        .max_bucket = max_bucket,
    };
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

const gemma_dpo_cold_loss_parity_tolerance: f32 = 1e-4;

fn validateGemmaDpoColdLossParity(loss: f32, base_equivalent_policy: bool) !void {
    if (!base_equivalent_policy) return;
    const expected: f32 = @log(@as(f32, 2.0));
    if (!std.math.isFinite(loss) or @abs(loss - expected) > gemma_dpo_cold_loss_parity_tolerance) {
        return error.GemmaDpoColdLossParityMismatch;
    }
}

test "gemma DPO cold loss parity gate is strict only for a base-equivalent policy" {
    const expected: f32 = @log(@as(f32, 2.0));
    try validateGemmaDpoColdLossParity(expected + gemma_dpo_cold_loss_parity_tolerance, true);
    try std.testing.expectError(
        error.GemmaDpoColdLossParityMismatch,
        validateGemmaDpoColdLossParity(expected + 2.0 * gemma_dpo_cold_loss_parity_tolerance, true),
    );
    try validateGemmaDpoColdLossParity(expected + 1.0, false);
}

fn precomputeGemmaDpoBaseReferenceCache(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *gemma4_real_autodiff.GemmaAutodiffCtx,
    chosen_examples: []const gemma4.PreparedExampleInput,
    rejected_examples: []const gemma4.PreparedExampleInput,
    configured_max_seq_len: usize,
    coalesce_single_token_pairs: bool,
    selected_tied_head_reference: bool,
) !GemmaDpoReferenceCache {
    if (chosen_examples.len != rejected_examples.len) return error.DpoBatchAlignmentMismatch;

    const base_equivalent_policy = gemmaLoraAdapterIsBaseEquivalent(trainer);
    var frozen_lora = try gemma4_real_autodiff.FrozenBaseLoraBindings.init(allocator, trainer);
    defer frozen_lora.deinit();

    const chosen_logps = try allocator.alloc(f32, chosen_examples.len);
    errdefer allocator.free(chosen_logps);
    const rejected_logps = try allocator.alloc(f32, rejected_examples.len);
    errdefer allocator.free(rejected_logps);
    const selected_margins: ?[]f32 = if (selected_tied_head_reference)
        try allocator.alloc(f32, chosen_examples.len)
    else
        null;
    errdefer if (selected_margins) |margins| allocator.free(margins);

    const started_ns = platform.time.monotonicNs();
    for (chosen_examples, rejected_examples, 0..) |*chosen, *rejected, idx| {
        const pair_seq_len = try gemmaPreferenceSequenceBucket(
            @max(chosen.num_input_tokens, rejected.num_input_tokens),
            configured_max_seq_len,
        );
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
                pair_seq_len,
                &pair_logps,
                &frozen_lora,
            );
            chosen_logps[idx] = pair_logps[0];
            rejected_logps[idx] = pair_logps[1];
            if (selected_margins) |margins| {
                var selected_input = try gemma4_real_autodiff.makeTrainerInputForSingleTokenDpoPair(
                    allocator,
                    ctx,
                    chosen,
                    rejected,
                    pair_seq_len,
                    0.0,
                    0.0,
                    1.0,
                );
                defer selected_input.deinit(allocator);
                const selected_step = try trainer.evaluate(selected_input.trainer_input);
                margins[idx] = try rewardMarginFromDpoLoss(selected_step.loss);
            }
        } else {
            chosen_logps[idx] = try gemma4_real_autodiff.sequenceLogprobForExampleFrozenBase(
                allocator,
                trainer,
                ctx,
                chosen,
                pair_seq_len,
                &frozen_lora,
            );
            rejected_logps[idx] = try gemma4_real_autodiff.sequenceLogprobForExampleFrozenBase(
                allocator,
                trainer,
                ctx,
                rejected,
                pair_seq_len,
                &frozen_lora,
            );
        }
    }
    const elapsed_ns = platform.time.monotonicNs() - started_ns;

    return .{
        .allocator = allocator,
        .chosen_logps = chosen_logps,
        .rejected_logps = rejected_logps,
        .selected_margins = selected_margins,
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
    preference_session: ?*GemmaPreferenceSession,
) !void {
    const base_model_dir = recipe.model.path orelse return error.MissingModelPath;
    const adapter = recipe.adapter orelse AdapterConfig{};
    const bootstrap_dir_config = adapter.path orelse adapterBootstrapDir(recipe);
    const bootstrap_dir = bootstrap_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    defer if (bootstrap_dir_config == null) allocator.free(bootstrap_dir);
    const trained_dir_config = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir;
    const trained_dir = trained_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    defer if (trained_dir_config == null) allocator.free(trained_dir);
    const reference_path = recipe.model.reference_path orelse base_model_dir;
    if (!try preferencePathsReferToSameArtifact(allocator, io, reference_path, base_model_dir)) {
        return error.UnsupportedReferencePath;
    }
    const execution = try resolveGemmaPreferenceExecution(recipe.backend);
    const backend_kind = execution.backend_kind;
    const execution_policy = train_eval_gemma4_lora_bundle.autodiffExecutionPolicy(backend_kind);
    const max_examples = recipe.dataset.max_examples orelse 32;
    const max_seq_len = recipe.dataset.max_seq_len orelse 512;
    try validateGemmaAdapterOptions(adapter);
    var graph_executor_scope = try train_eval_gemma4_lora_bundle.acquireDeviceGraphExecutorScope(backend_kind);
    defer if (graph_executor_scope) |*scope| scope.deinit();
    try train_eval_gemma4_lora_bundle.validateAutodiffBaseArtifact(allocator, base_model_dir, backend_kind);

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
        });
        defer gemma4.freeBootstrapSummary(allocator, &bootstrap);
    };

    var tokenizer_assets = try OwnedPreferenceTextTokenizer.init(allocator, base_model_dir);
    defer tokenizer_assets.deinit();

    var samples = try loadDpoTextPreferenceSamples(allocator, io, dataset_path, recipe, tokenizer_assets.view());
    defer samples.deinit();

    var chosen_prepared = try prepareGemmaDpoPreparedExamplesFromSamples(allocator, base_model_dir, samples.samples, max_examples, max_seq_len, .chosen);
    defer gemma4.freePreparedInputsSummary(allocator, &chosen_prepared);
    var rejected_prepared = try prepareGemmaDpoPreparedExamplesFromSamples(allocator, base_model_dir, samples.samples, max_examples, max_seq_len, .rejected);
    defer gemma4.freePreparedInputsSummary(allocator, &rejected_prepared);
    if (chosen_prepared.examples.len != rejected_prepared.examples.len or chosen_prepared.examples.len != samples.samples.len) {
        return error.DpoBatchAlignmentMismatch;
    }
    const sequence_bucketing = try summarizeGemmaDpoSequenceBucketing(
        chosen_prepared.examples,
        rejected_prepared.examples,
        max_seq_len,
    );
    // The weighted sparse-row backward is qualified by both strict device
    // executors: CUDA already uses the identical graph for one-token GRPO
    // groups. Native retains the general two-sequence path; its rank-2 dot
    // implementation does not yet support this sparse target graph and must
    // not receive it through a structural fast-path match.
    const coalesce_single_token_pairs = shouldCoalesceGemmaDpoSingleTokenPairs(
        backend_kind,
        chosen_prepared.examples,
        rejected_prepared.examples,
    );
    // The simultaneous whole-objective graph remains outside production until
    // its activation lifetime is below the memory gate. Multi-token Metal DPO
    // uses the qualified detached-gradient path instead.
    const pair_objective_requested = false;
    const batch2_pair_forward_requested = false;
    const detached_pair_gradients_requested = backend_kind == .metal and
        !coalesce_single_token_pairs;
    const slot_bound_outputs = false;
    const completion_cache_enabled = backend_kind == .metal and
        platform.env.getenvBoolDefault("ANTFLY_GEMMA4_DPO_COMPLETION_FENCED_CACHE", true);
    const dpo_checkpoint_config: ?ml.graph.checkpoint.CheckpointConfig = null;

    const graph_config = if (preference_session) |session|
        session.graph_config
    else
        try gemma4_real_autodiff.loadGraphConfig(allocator, base_model_dir);
    // The allocator fences aliases recycled inside a planned encoder and
    // quarantines releases made outside one. The exact E2B topology is
    // production-qualified; other shapes remain fail-closed unless explicitly
    // enabled for research. Setting the variable to 0 is the E2B kill switch.
    const in_frame_buffer_reuse_enabled = backend_kind == .metal and
        platform.env.getenvBoolDefault(
            "ANTFLY_GEMMA4_DPO_IN_FRAME_BUFFER_REUSE",
            gemma4_real_autodiff.qualifiedE2BTrainingTopology(graph_config),
        );
    var owned_backend: ?gemma4_real_autodiff.LoadedBackend = null;
    defer if (owned_backend) |*backend| backend.deinit();
    const backend: *gemma4_real_autodiff.LoadedBackend = if (preference_session) |session| blk: {
        try session.requireCompatible(allocator, io, base_model_dir, backend_kind);
        break :blk &session.backend;
    } else blk: {
        owned_backend = try gemma4_real_autodiff.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
        break :blk &owned_backend.?;
    };
    const preference_session_telemetry = if (preference_session) |session|
        try session.beginRun(allocator, io, base_model_dir, backend_kind)
    else
        null;

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
    const compile_coalesced_pair_objective = coalesce_single_token_pairs;
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
        .optimizer = .{ .weight_decay = recipe.optimizer.weight_decay orelse 0.01 },
        .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
        .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
        .grad_accum_steps = grad_accum_steps,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
        .strict_cuda_execution = execution_policy.strict_cuda_execution,
        .checkpoint_config = dpo_checkpoint_config,
        .metal_slot_bound_outputs = slot_bound_outputs,
        // Per-example power-of-two sequence shapes keep padding bounded. Four
        // hot graphs cover the locally common buckets while preserving the
        // existing memory ceiling for heterogeneous datasets.
        .graph_cache_capacity = 4,
    });
    defer trainer.deinit();

    var ctx = gemma4_real_autodiff.GemmaAutodiffCtx.init(graph_config);
    ctx.enable_fused_rms_norm_backward = backend_kind == .metal;
    ctx.enable_fused_gqa_attention_backward = backend_kind == .metal and gemma4_real_autodiff.fusedGqaAttentionExperimentEnabled(graph_config);
    ctx.enable_fused_linear_cross_entropy = backend_kind == .metal or backend_kind == .cuda;
    const bootstrap_example = gemma4_real_autodiff.findFirstSupervisedExample(chosen_prepared.examples) orelse return error.NoTrainingData;
    const bootstrap_seq_len = try gemmaPreferenceSequenceBucket(bootstrap_example.num_input_tokens, max_seq_len);
    try gemma4_real_autodiff.initializeTrainerFromAdapterDir(allocator, &trainer, &ctx, bootstrap_dir, bootstrap_example, bootstrap_seq_len);

    var trainable_snapshot = try TrainableParameterSnapshot.init(allocator, trainer.lora_params.items);
    defer trainable_snapshot.deinit();

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
        max_seq_len,
        coalesce_single_token_pairs,
        compile_coalesced_pair_objective,
    );
    defer reference_cache.deinit();

    var total_loss: f64 = 0.0;
    var total_margin: f64 = 0.0;
    var total_accuracy: f64 = 0.0;
    var examples_seen: usize = 0;
    var initial_logprob_parity: ?DpoInitialLogprobParity = null;
    var single_pc = [_]f32{0};
    var single_pr = [_]f32{0};
    var single_rc = [_]f32{0};
    var single_rr = [_]f32{0};
    var single_cl = [_]u32{0};
    var single_rl = [_]u32{0};
    var single_sft = [_]f32{0};

    var epoch_idx: usize = 0;
    while (epoch_idx < epochs) : (epoch_idx += 1) {
        for (chosen_prepared.examples, rejected_prepared.examples, samples.samples, 0..) |*chosen_ex, *rejected_ex, sample, sample_idx| {
            const update_started_ns = if (benchmark_enabled) platform.time.monotonicNs() else 0;
            const pair_seq_len = try gemmaPreferenceSequenceBucket(
                @max(chosen_ex.num_input_tokens, rejected_ex.num_input_tokens),
                max_seq_len,
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
                var chosen_raw_input = try gemma4_real_autodiff.makeTrainerInputForLogprobCoeff(
                    allocator,
                    &ctx,
                    chosen_ex,
                    pair_seq_len,
                    1.0,
                );
                defer chosen_raw_input.deinit(allocator);
                const chosen_raw_step = try trainer.step(chosen_raw_input.trainer_input);
                if (chosen_raw_step.optimizer_stepped) return error.DpoDetachedGradientSteppedEarly;
                policy_chosen = chosen_raw_step.loss;
                detached_device_gradients = try trainer.detachAccumulatedDeviceGradients();

                var rejected_raw_input = try gemma4_real_autodiff.makeTrainerInputForLogprobCoeff(
                    allocator,
                    &ctx,
                    rejected_ex,
                    pair_seq_len,
                    1.0,
                );
                defer rejected_raw_input.deinit(allocator);
                const rejected_raw_step = try trainer.step(rejected_raw_input.trainer_input);
                if (rejected_raw_step.optimizer_stepped) return error.DpoDetachedGradientSteppedEarly;
                policy_rejected = rejected_raw_step.loss;
            } else if (!(compile_pair_objective or compile_coalesced_pair_objective) or examples_seen == 0) {
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
                        pair_seq_len,
                        &pair_logps,
                    );
                    policy_chosen = pair_logps[0];
                    policy_rejected = pair_logps[1];
                } else {
                    policy_chosen = try gemma4_real_autodiff.sequenceLogprobForExample(
                        allocator,
                        &trainer,
                        &ctx,
                        chosen_ex,
                        pair_seq_len,
                    );
                    policy_rejected = try gemma4_real_autodiff.sequenceLogprobForExample(
                        allocator,
                        &trainer,
                        &ctx,
                        rejected_ex,
                        pair_seq_len,
                    );
                }
            }

            const audit_reference_chosen = reference_cache.chosen_logps[sample_idx];
            const audit_reference_rejected = reference_cache.rejected_logps[sample_idx];

            if (examples_seen == 0) {
                const max_abs_error = @max(
                    @abs(policy_chosen - audit_reference_chosen),
                    @abs(policy_rejected - audit_reference_rejected),
                );
                initial_logprob_parity = .{
                    .policy_chosen_logp = policy_chosen,
                    .policy_rejected_logp = policy_rejected,
                    .reference_chosen_logp = audit_reference_chosen,
                    .reference_rejected_logp = audit_reference_rejected,
                    .max_abs_error = max_abs_error,
                    .base_equivalent_policy = reference_cache.base_equivalent_policy,
                };
                if (reference_cache.base_equivalent_policy and max_abs_error > 1e-4) {
                    return error.GemmaDpoInitialReferenceParityMismatch;
                }
            }
            if (reference_cache.selected_margins) |margins| {
                single_rc[0] = margins[sample_idx];
                single_rr[0] = 0.0;
            } else {
                single_rc[0] = audit_reference_chosen;
                single_rr[0] = audit_reference_rejected;
            }

            var update_loss: f32 = undefined;
            var update_margin: f32 = undefined;
            var update_accuracy: f32 = undefined;
            if (compile_coalesced_pair_objective) {
                var pair_input = try gemma4_real_autodiff.makeTrainerInputForSingleTokenDpoPair(
                    allocator,
                    &ctx,
                    chosen_ex,
                    rejected_ex,
                    pair_seq_len,
                    single_rc[0],
                    single_rr[0],
                    recipe.preference.beta orelse 0.1,
                );
                defer pair_input.deinit(allocator);
                const pair_step = try trainer.step(pair_input.trainer_input);
                if (examples_seen == 0) {
                    try validateGemmaDpoColdLossParity(pair_step.loss, reference_cache.base_equivalent_policy);
                }
                update_loss = pair_step.loss;
                update_margin = try rewardMarginFromDpoLoss(pair_step.loss);
                update_accuracy = if (update_margin > 0.0) 1.0 else 0.0;
            } else if (compile_pair_objective) {
                var pair_input = try gemma4_real_autodiff.makeTrainerInputForDpoPair(
                    allocator,
                    &ctx,
                    chosen_ex,
                    rejected_ex,
                    pair_seq_len,
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
                        pair_seq_len,
                        &candidate_tokens,
                        &logprob_grads,
                    );
                    defer pair_input.deinit(allocator);
                    _ = try trainer.step(pair_input.trainer_input);
                } else {
                    var chosen_input = try gemma4_real_autodiff.makeTrainerInputForLogprobCoeff(
                        allocator,
                        &ctx,
                        chosen_ex,
                        pair_seq_len,
                        step_result.grad_chosen[0],
                    );
                    defer chosen_input.deinit(allocator);
                    _ = try trainer.step(chosen_input.trainer_input);

                    var rejected_input = try gemma4_real_autodiff.makeTrainerInputForLogprobCoeff(
                        allocator,
                        &ctx,
                        rejected_ex,
                        pair_seq_len,
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
    }

    _ = try trainer.flushAccumulatedGradients();
    if (trainer.optimizerSteps() == 0) return error.NoOptimizerSteps;

    try trainer.syncDeviceTrainablesToHost();
    const trainable_update = try trainable_snapshot.summarize(trainer.lora_params.items);
    if (trainable_update.changed_tensor_count == 0 or !(trainable_update.max_abs_delta > 0.0)) {
        return error.NoDpoPolicyMovement;
    }
    const benchmark_telemetry = if (benchmark) |*recorder|
        try recorder.finish(trainable_update)
    else
        null;

    // Drain the completion-fenced cache before capturing final evidence. The
    // rendered report is then embedded in the immutable adapter transaction
    // before being mirrored to the mutable run-level report path.
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

    const denom = @as(f64, @floatFromInt(@max(examples_seen, 1)));
    const report = DpoReport{
        .examples = examples_seen,
        .loss = @floatCast(total_loss / denom),
        .mean_reward_margin = @floatCast(total_margin / denom),
        .accuracy = @floatCast(total_accuracy / denom),
        .beta = recipe.preference.beta orelse 0.1,
        .policy_backend = @tagName(backend_kind),
        .optimizer_steps = trainer.optimizerSteps(),
        .micro_batch_steps = trainer.microBatchSteps(),
        .device_execution = trainer.trainingExecutionEvidence(),
        .policy_scoring_mode = if (compile_coalesced_pair_objective)
            "initial-parity-only-then-in-graph-selected-logit-margin"
        else if (coalesce_single_token_pairs)
            "shared-prompt-single-row"
        else if (compile_pair_objective)
            "initial-parity-only-then-in-graph"
        else if (detach_pair_gradients)
            "backward-loss-reuse-device-detached"
        else
            "compiled-loss-only-device-reduced",
        .training_microbatch_mode = if (compile_coalesced_pair_objective)
            "compiled-shared-prompt-single-row-selected-logit-dpo"
        else if (coalesce_single_token_pairs)
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
        .input_contract = .{
            .prompt_input_ids = samples.samples[0].prompt_tokens,
            .chosen_input_ids = samples.samples[0].chosen_tokens,
            .rejected_input_ids = samples.samples[0].rejected_tokens,
        },
        .sequence_bucketing = sequence_bucketing,
        .trainable_update = trainable_update,
        .preference_session = preference_session_telemetry,
        .benchmark = benchmark_telemetry,
    };
    try publishGemmaPreferenceBundleAndReport(
        allocator,
        io,
        &trainer,
        base_model_dir,
        bootstrap_dir,
        trained_dir,
        report_path,
        report,
    );
    print("dpo report: {s}\ntrained adapter: {s}\n", .{ report_path, trained_dir });
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
    const bootstrap_dir_config = adapter.path orelse adapterBootstrapDir(recipe);
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

    var samples = try loadDpoTextPreferenceSamples(
        allocator,
        io,
        dataset_path,
        recipe,
        PreferenceTextTokenizerView.fromLoadedModel(reference_model),
    );
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
        .examples = examples_seen,
        .loss = @floatCast(total_loss / denom),
        .mean_reward_margin = @floatCast(total_margin / denom),
        .accuracy = @floatCast(total_accuracy / denom),
        .beta = recipe.preference.beta orelse 0.1,
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
    if (std.mem.eql(u8, value, "ranked-first")) return .ranked_first;
    return error.UnsupportedRewardMode;
}

const max_deterministic_ranked_group_size: usize = 8;

fn validateRankedGrpoGroupSize(group_size: usize) !void {
    if (group_size < 2 or group_size > max_deterministic_ranked_group_size) {
        return error.InvalidGrpoGroupSize;
    }
}

fn runOptimizerBackedGemmaGrpo(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    dataset_path: []const u8,
    report_path: []const u8,
    preference_session: ?*GemmaPreferenceSession,
) !void {
    const base_model_dir = recipe.model.path orelse return error.MissingModelPath;
    const adapter = recipe.adapter orelse AdapterConfig{};
    const bootstrap_dir_config = adapter.path orelse adapterBootstrapDir(recipe);
    const bootstrap_dir = bootstrap_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-bootstrap");
    defer if (bootstrap_dir_config == null) allocator.free(bootstrap_dir);
    const trained_dir_config = recipe.artifacts.trained_adapter_dir orelse recipe.artifacts.adapter_dir;
    const trained_dir = trained_dir_config orelse try defaultArtifactPath(allocator, recipe, "adapter-trained");
    defer if (trained_dir_config == null) allocator.free(trained_dir);
    const reference_path = recipe.model.reference_path orelse base_model_dir;
    if (!try preferencePathsReferToSameArtifact(allocator, io, reference_path, base_model_dir)) {
        return error.UnsupportedReferencePath;
    }
    const execution = try resolveGemmaPreferenceExecution(recipe.backend);
    const backend_kind = execution.backend_kind;
    try validateGemmaPreferenceModality(backend_kind, recipe.model.projector_path);
    const execution_policy = train_eval_gemma4_lora_bundle.autodiffExecutionPolicy(backend_kind);
    const max_seq_len = recipe.dataset.max_seq_len orelse 128;
    const group_size = recipe.grpo.group_size orelse 2;
    const max_completion_tokens = recipe.grpo.max_completion_tokens orelse 4;
    try validateRankedGrpoGroupSize(group_size);
    if (max_completion_tokens == 0) return error.InvalidMaxCompletionTokens;
    const coalesce_single_token_groups = max_completion_tokens == 1;
    const physical_micro_batches_per_group: usize = if (coalesce_single_token_groups) 1 else group_size;
    const preference_units_per_optimizer_u32 = recipe.optimizer.gradient_accumulation_steps orelse 1;
    const preference_units_per_optimizer: usize = @intCast(preference_units_per_optimizer_u32);
    const grad_accum_steps = try preferenceGradAccumSteps(
        preference_units_per_optimizer_u32,
        physical_micro_batches_per_group,
    );
    const reward_mode = try parseTextRewardMode(recipe.grpo.reward_mode orelse "exact-match");
    try validateGemmaAdapterOptions(adapter);
    var graph_executor_scope = try train_eval_gemma4_lora_bundle.acquireDeviceGraphExecutorScope(backend_kind);
    defer if (graph_executor_scope) |*scope| scope.deinit();
    try train_eval_gemma4_lora_bundle.validateAutodiffBaseArtifact(allocator, base_model_dir, backend_kind);

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

    var tokenizer_assets = try OwnedPreferenceTextTokenizer.init(allocator, base_model_dir);
    defer tokenizer_assets.deinit();
    const tokenizer = tokenizer_assets.tokenizer();

    var prompt_batch = try loadGrpoTextPrompts(allocator, io, dataset_path, recipe, tokenizer_assets.view());
    defer prompt_batch.deinit();
    const sequence_bucketing = try summarizeGemmaGrpoSequenceBucketing(
        prompt_batch.prompts,
        max_completion_tokens,
        max_seq_len,
    );
    const epochs = recipe.optimizer.epochs orelse 1;
    const benchmark_enabled = platform.env.getenvBoolDefault("ANTFLY_GEMMA4_GRPO_BENCHMARK", false);
    const planned_updates = std.math.mul(usize, prompt_batch.prompts.len, epochs) catch return error.GrpoBenchmarkUpdateCountMismatch;
    if (benchmark_enabled and planned_updates != GrpoBenchmarkRecorder.total_updates) {
        return error.GrpoBenchmarkUpdateCountMismatch;
    }
    var benchmark: ?GrpoBenchmarkRecorder = if (benchmark_enabled) try GrpoBenchmarkRecorder.init(allocator) else null;
    defer if (benchmark) |*recorder| recorder.deinit();

    const graph_config = if (preference_session) |session|
        session.graph_config
    else
        try gemma4_real_autodiff.loadGraphConfig(allocator, base_model_dir);
    var owned_backend: ?gemma4_real_autodiff.LoadedBackend = null;
    defer if (owned_backend) |*backend| backend.deinit();
    const backend: *gemma4_real_autodiff.LoadedBackend = if (preference_session) |session| blk: {
        try session.requireCompatible(allocator, io, base_model_dir, backend_kind);
        break :blk &session.backend;
    } else blk: {
        owned_backend = try gemma4_real_autodiff.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
        break :blk &owned_backend.?;
    };
    const preference_session_telemetry = if (preference_session) |session|
        try session.beginRun(allocator, io, base_model_dir, backend_kind)
    else
        null;

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

    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{ .weight_decay = recipe.optimizer.weight_decay orelse 0.01 },
        .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
        .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
        .grad_accum_steps = grad_accum_steps,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
        .strict_cuda_execution = execution_policy.strict_cuda_execution,
        .graph_cache_capacity = 4,
    });
    defer trainer.deinit();

    var ctx = gemma4_real_autodiff.GemmaAutodiffCtx.init(graph_config);
    ctx.enable_fused_rms_norm_backward = backend_kind == .metal;
    ctx.enable_fused_gqa_attention_backward = backend_kind == .metal and gemma4_real_autodiff.fusedGqaAttentionExperimentEnabled(graph_config);
    ctx.enable_fused_linear_cross_entropy = backend_kind == .metal or backend_kind == .cuda;
    const bootstrap_prompt = prompt_batch.prompts[0];
    if (bootstrap_prompt.len == 0 or bootstrap_prompt.len >= max_seq_len) return error.NoCompletionBudget;
    const bootstrap_completion = [_]i32{bootstrap_prompt[bootstrap_prompt.len - 1]};
    const bootstrap_example = try buildGemmaPreparedExampleFromTokens(allocator, bootstrap_prompt, &bootstrap_completion, max_seq_len);
    defer freeGemmaPreparedExample(allocator, &bootstrap_example);
    const bootstrap_seq_len = try gemmaPreferenceSequenceBucket(bootstrap_example.num_input_tokens, max_seq_len);
    try gemma4_real_autodiff.initializeTrainerFromAdapterDir(allocator, &trainer, &ctx, bootstrap_dir, &bootstrap_example, bootstrap_seq_len);

    var trainable_snapshot = try TrainableParameterSnapshot.init(allocator, trainer.lora_params.items);
    defer trainable_snapshot.deinit();

    const base_equivalent_policy = gemmaLoraAdapterIsBaseEquivalent(&trainer);
    var frozen_lora = try gemma4_real_autodiff.FrozenBaseLoraBindings.init(allocator, &trainer);
    defer frozen_lora.deinit();
    var reference_cache = GemmaGrpoReferenceCache.init(allocator, 1024);
    defer reference_cache.deinit();
    var rewarder_ctx = TextRewardCtx{
        .allocator = allocator,
        .tokenizer = tokenizer,
        .targets = prompt_batch.targets,
        .mode = reward_mode,
        .group_size = group_size,
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
    var total_reward: f64 = 0.0;
    var total_reward_squared: f64 = 0.0;
    var total_reference_scoring_seconds: f64 = 0.0;
    var saw_nonzero_reward_advantage = false;
    var saw_nonzero_policy_gradient = false;
    var initial_sampling_rescore_max_abs_error: f32 = 0.0;
    var initial_policy_reference_max_abs_error: f32 = 0.0;
    var captured_initial_logprob_parity = false;
    var policy_rescore_completions: usize = 0;
    var diagnostic_first_tokens: [8]i32 = @splat(-1);
    var diagnostic_policy_first_token_logps: [8]f32 = @splat(0.0);
    var diagnostic_reference_first_token_logps: [8]f32 = @splat(0.0);
    var diagnostic_first_token_count: usize = 0;

    const top_rank_cap = group_size;
    const eos_id = tokenizer.specialTokens().sep_id;
    var epoch_idx: usize = 0;
    while (epoch_idx < epochs) : (epoch_idx += 1) {
        for (prompt_batch.prompts, 0..) |prompt, prompt_idx| {
            const group_started_ns = if (benchmark_enabled) platform.time.monotonicNs() else 0;
            const group_required_tokens = std.math.add(usize, prompt.len, max_completion_tokens) catch return error.SequenceTooLong;
            const group_seq_len = try gemmaPreferenceSequenceBucket(group_required_tokens, max_seq_len);
            const optimizer_steps_before_group = trainer.optimizerSteps();
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
            try gemma4_real_autodiff.sampleCompletionGroupRanked(
                allocator,
                &trainer,
                &ctx,
                prompt,
                group_seq_len,
                max_completion_tokens,
                top_rank_cap,
                if (eos_id >= 0) eos_id else null,
                sampled_token_lists,
                sampled_logp_lists,
            );

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
                    if (coalesce_single_token_groups) {
                        try gemma4_real_autodiff.tokenLogprobsForPromptCompletionSparseRows(
                            allocator,
                            &trainer,
                            &ctx,
                            prompt,
                            tokens_owned,
                            group_seq_len,
                            new_logps_owned,
                        );
                    } else {
                        try gemma4_real_autodiff.tokenLogprobsForPromptCompletion(
                            allocator,
                            &trainer,
                            &ctx,
                            prompt,
                            tokens_owned,
                            group_seq_len,
                            new_logps_owned,
                        );
                    }
                    policy_rescore_completions += 1;
                } else {
                    @memcpy(new_logps_owned, old_logps_owned);
                }
                const ref_logps_owned = try allocator.alloc(f32, tokens_owned.len);
                errdefer allocator.free(ref_logps_owned);
                if (!try reference_cache.lookup(prompt_idx, tokens_owned, ref_logps_owned)) {
                    const reference_started_ns = platform.time.monotonicNs();
                    if (coalesce_single_token_groups) {
                        try gemma4_real_autodiff.tokenLogprobsForPromptCompletionSparseRowsFrozenBase(
                            allocator,
                            &trainer,
                            &ctx,
                            prompt,
                            tokens_owned,
                            group_seq_len,
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
                            group_seq_len,
                            ref_logps_owned,
                            &frozen_lora,
                        );
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
                if (group_sampling_rescore_max_abs_error > 1e-4) return error.GrpoSamplingRescoreParityMismatch;
                if (base_equivalent_policy and group_policy_reference_max_abs_error > 1e-4) return error.GrpoInitialReferenceParityMismatch;
                initial_sampling_rescore_max_abs_error = group_sampling_rescore_max_abs_error;
                initial_policy_reference_max_abs_error = group_policy_reference_max_abs_error;
                captured_initial_logprob_parity = true;
            }

            var ga = try grpo.scoreGroup(allocator, rewarder, completions.items);
            defer ga.deinit();
            try grpo.computeAdvantages(&ga, completions.items, cfg);
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

            total_loss += loss_result.loss;
            total_pg_loss += loss_result.pg_loss;
            total_kl_loss += loss_result.kl_loss;
            total_clip_fraction += loss_result.clip_fraction;
            total_groups += 1;
            total_completions += completions.items.len;

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
                    group_seq_len,
                    completion_token_ids.items,
                    loss_result.grad_new_logps,
                );
                defer input.deinit(allocator);
                _ = try trainer.step(input.trainer_input);
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
                        group_seq_len,
                        grads,
                    );
                    defer input.deinit(allocator);
                    _ = try trainer.step(input.trainer_input);
                    token_offset += completion.tokens.len;
                }
            }
            const completed_optimizer_window = total_groups % preference_units_per_optimizer == 0;
            const expected_optimizer_steps = optimizer_steps_before_group +
                @as(u64, @intFromBool(completed_optimizer_window));
            const expected_accumulated_micro_batches: u32 = if (completed_optimizer_window)
                0
            else
                @intCast((total_groups % preference_units_per_optimizer) * physical_micro_batches_per_group);
            if (trainer.optimizerSteps() != expected_optimizer_steps or
                trainer.accumulatedMicroBatches() != expected_accumulated_micro_batches)
            {
                return error.InvalidGrpoGroupOptimizerBoundary;
            }
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
        }
    }

    if (!saw_nonzero_reward_advantage or !saw_nonzero_policy_gradient) {
        print("grpo zero-signal first completion token ids: {any}\n", .{diagnostic_first_tokens[0..diagnostic_first_token_count]});
    }
    try validateGrpoLearningSignal(saw_nonzero_reward_advantage, saw_nonzero_policy_gradient, total_reward, total_reward_squared, total_completions, total_loss);
    _ = try trainer.flushAccumulatedGradients();
    if (trainer.optimizerSteps() == 0) return error.NoOptimizerSteps;

    // The optimizer owns FP32 trainables on the device. Synchronize once,
    // outside every timed benchmark update, and compare against the exact
    // pre-training snapshot. This proves parameter movement even when a
    // saturated sampled token's log-probability is unchanged at f32
    // reporting precision.
    try trainer.syncDeviceTrainablesToHost();
    const trainable_update = try trainable_snapshot.summarize(trainer.lora_params.items);
    const benchmark_telemetry = if (benchmark) |*recorder|
        try recorder.finish(trainable_update)
    else
        null;
    if (trainable_update.changed_tensor_count == 0 or !(trainable_update.max_abs_delta > 0.0)) {
        return error.NoGrpoPolicyMovement;
    }

    const denom = @as(f64, @floatFromInt(@max(total_groups, 1)));
    const reward_denom = @as(f64, @floatFromInt(@max(total_completions, 1)));
    const mean_reward = total_reward / reward_denom;
    const reward_variance = @max(total_reward_squared / reward_denom - mean_reward * mean_reward, 0.0);
    const report = GrpoReport{
        .completions = total_completions,
        .tokens = total_tokens,
        .groups = total_groups,
        .loss = @floatCast(total_loss / denom),
        .pg_loss = @floatCast(total_pg_loss / denom),
        .kl_loss = @floatCast(total_kl_loss / denom),
        .clip_fraction = @floatCast(total_clip_fraction / denom),
        .mean_reward = @floatCast(mean_reward),
        .reward_stddev = @floatCast(@sqrt(reward_variance)),
        .reward_mode = recipe.grpo.reward_mode orelse "exact-match",
        .policy_backend = @tagName(backend_kind),
        .optimizer_steps = trainer.optimizerSteps(),
        .micro_batch_steps = trainer.microBatchSteps(),
        .device_execution = trainer.trainingExecutionEvidence(),
        .sampling_mode = if (coalesce_single_token_groups) "shared-prompt-ranked-sparse-row" else "shared-prompt-ranked",
        .policy_logprob_mode = "sampling-logprob-reuse-with-initial-rescore",
        .policy_rescore_completions = policy_rescore_completions,
        .training_microbatch_mode = if (coalesce_single_token_groups) "coalesced-single-token-sparse-weighted-row" else "per-completion",
        .reference_mode = "compiled-zero-lora",
        .reference_scoring_seconds = total_reference_scoring_seconds,
        .reference_cache = reference_cache.telemetry(),
        .initial_logprob_parity = .{
            .sampling_rescore_max_abs_error = initial_sampling_rescore_max_abs_error,
            .policy_reference_max_abs_error = initial_policy_reference_max_abs_error,
            .base_equivalent_policy = base_equivalent_policy,
            .completion_first_token_ids = diagnostic_first_tokens[0..diagnostic_first_token_count],
            .policy_first_token_logps = diagnostic_policy_first_token_logps[0..diagnostic_first_token_count],
            .reference_first_token_logps = diagnostic_reference_first_token_logps[0..diagnostic_first_token_count],
        },
        .input_contract = .{
            .prompt_input_ids = prompt_batch.prompts[0],
            .group_size = group_size,
            .max_completion_tokens = max_completion_tokens,
            .sampling = "deterministic-ranked-top-k",
        },
        .sequence_bucketing = sequence_bucketing,
        .trainable_update = trainable_update,
        .preference_session = preference_session_telemetry,
        .benchmark = benchmark_telemetry,
    };
    try publishGemmaPreferenceBundleAndReport(
        allocator,
        io,
        &trainer,
        base_model_dir,
        bootstrap_dir,
        trained_dir,
        report_path,
        report,
    );
    print("grpo report: {s}\ntrained adapter: {s}\n", .{ report_path, trained_dir });
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
    const bootstrap_dir_config = adapter.path orelse adapterBootstrapDir(recipe);
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

    var prompt_batch = try loadGrpoTextPrompts(
        allocator,
        io,
        dataset_path,
        recipe,
        PreferenceTextTokenizerView.fromLoadedModel(tokenizer_model),
    );
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
        .group_size = group_size,
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

    const top_rank_cap = group_size;
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
            try grpo.computeAdvantages(&ga, completions.items, cfg);

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
        .completions = total_completions,
        .tokens = total_tokens,
        .groups = total_groups,
        .loss = @floatCast(total_loss / denom),
        .pg_loss = @floatCast(total_pg_loss / denom),
        .kl_loss = @floatCast(total_kl_loss / denom),
        .clip_fraction = @floatCast(total_clip_fraction / denom),
        .reward_mode = recipe.grpo.reward_mode orelse "exact-match",
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
    if (!try preferencePathsReferToSameArtifact(allocator, io, reference_path, base_model_dir)) {
        return error.UnsupportedReferencePath;
    }
    const execution_policy = train_eval_gemma4_lora_bundle.autodiffExecutionPolicy(backend_kind);
    const grad_accum_steps = try preferenceGradAccumSteps(recipe.optimizer.gradient_accumulation_steps orelse 1, group_size);

    var session_manager = backends.SessionManager.init(allocator);
    native_backend_choice.configureSessionPreference(&session_manager, switch (backend_kind) {
        .native => .native,
        .metal => .metal,
        .cuda => .cuda,
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
        .optimizer = .{ .weight_decay = recipe.optimizer.weight_decay orelse 0.01 },
        .lr_schedule = .{ .constant = recipe.optimizer.learning_rate orelse 0.0001 },
        .max_grad_norm = recipe.optimizer.max_grad_norm orelse 1.0,
        .grad_accum_steps = grad_accum_steps,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
        .strict_cuda_execution = execution_policy.strict_cuda_execution,
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

    var trainable_snapshot = try TrainableParameterSnapshot.init(allocator, trainer.lora_params.items);
    defer trainable_snapshot.deinit();

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
        .strict_cuda_execution = execution_policy.strict_cuda_execution,
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
        .group_size = group_size,
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
    var total_reward: f64 = 0.0;
    var total_reward_squared: f64 = 0.0;
    var saw_nonzero_reward_advantage = false;
    var saw_nonzero_policy_gradient = false;

    const top_rank_cap = group_size;
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
            try grpo.computeAdvantages(&ga, completions.items, cfg);
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
            try scalePreferenceUnitGradients(loss_result.grad_new_logps, group_size);

            total_loss += loss_result.loss;
            total_pg_loss += loss_result.pg_loss;
            total_kl_loss += loss_result.kl_loss;
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
        }
    }

    try validateGrpoLearningSignal(saw_nonzero_reward_advantage, saw_nonzero_policy_gradient, total_reward, total_reward_squared, total_completions, total_loss);
    _ = try trainer.flushAccumulatedGradients();
    if (trainer.optimizerSteps() == 0) return error.NoOptimizerSteps;
    try trainer.syncDeviceTrainablesToHost();
    const trainable_update = try trainable_snapshot.summarize(trainer.lora_params.items);
    if (trainable_update.changed_tensor_count == 0 or !(trainable_update.max_abs_delta > 0.0)) {
        return error.NoGrpoPolicyMovement;
    }

    const denom = @as(f64, @floatFromInt(@max(total_groups, 1)));
    const reward_denom = @as(f64, @floatFromInt(@max(total_completions, 1)));
    const mean_reward = total_reward / reward_denom;
    const reward_variance = @max(total_reward_squared / reward_denom - mean_reward * mean_reward, 0.0);
    const report = GrpoReport{
        .completions = total_completions,
        .tokens = total_tokens,
        .groups = total_groups,
        .loss = @floatCast(total_loss / denom),
        .pg_loss = @floatCast(total_pg_loss / denom),
        .kl_loss = @floatCast(total_kl_loss / denom),
        .clip_fraction = @floatCast(total_clip_fraction / denom),
        .mean_reward = @floatCast(mean_reward),
        .reward_stddev = @floatCast(@sqrt(reward_variance)),
        .reward_mode = recipe.grpo.reward_mode orelse "exact-match",
        .policy_backend = @tagName(backend_kind),
        .optimizer_steps = trainer.optimizerSteps(),
        .micro_batch_steps = trainer.microBatchSteps(),
        .trainable_update = trainable_update,
    };
    try publishGemmaPreferenceBundleAndReport(
        allocator,
        io,
        &trainer,
        base_model_dir,
        bootstrap_dir,
        trained_dir,
        report_path,
        report,
    );
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

/// Non-owning tokenizer and prompt-rendering state used by text preference
/// data preparation. Keeping this separate from LoadedModel prevents tokenizing
/// a CUDA training dataset from allocating a second copy of the checkpoint.
const PreferenceTextTokenizerView = struct {
    tokenizer: tokenizer_mod.Tokenizer,
    add_bos_token: bool,
    bos_token: []const u8,
    chat_template: ?*const generation.ChatTemplate,

    fn fromLoadedModel(model: *model_manager_mod.LoadedModel) PreferenceTextTokenizerView {
        return .{
            .tokenizer = model.getTokenizer(),
            .add_bos_token = model.manifest.add_bos_token,
            .bos_token = model.manifest.bos_token,
            .chat_template = model.chat_tmpl,
        };
    }

    fn renderPrompt(self: PreferenceTextTokenizerView, allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
        const messages = [_]generation.Message{
            .{ .role = "user", .content = prompt },
        };
        if (self.chat_template) |tmpl| return tmpl.apply(allocator, &messages, true);
        return generation.formatMessages(allocator, &messages);
    }
};

/// Owns only the assets required to tokenize and render preference data. A
/// full LoadedModel is intentionally not constructed: E4B BF16 alone is about
/// 16 GB, so a redundant CPU policy load can exhaust otherwise healthy CUDA
/// training hosts before the first optimizer step.
const OwnedPreferenceTextTokenizer = struct {
    manifest: manifest_mod.ModelManifest,
    hf_tok: *hf_tokenizer.HfTokenizer,
    chat_template: ?generation.ChatTemplate,

    fn init(allocator: std.mem.Allocator, model_dir: []const u8) !OwnedPreferenceTextTokenizer {
        var manifest = try manifest_mod.loadFromDir(allocator, model_dir);
        errdefer manifest.deinit();

        const tok = try model_manager_mod.loadHuggingFaceTokenizerFromDir(allocator, model_dir);
        errdefer tok.deinitSelf();

        var chat_template: ?generation.ChatTemplate = null;
        errdefer if (chat_template) |*tmpl| tmpl.deinit();
        if (manifest.chat_template) |source| {
            // Preference training must preserve the exact chat rendering used
            // for generation. A malformed configured template is therefore a
            // hard error instead of a silent raw-prompt quality regression.
            chat_template = try generation.ChatTemplate.init(
                allocator,
                source,
                manifest.bos_token,
                manifest.eos_token,
                manifest.unk_token,
                manifest.pad_token,
            );
        }

        return .{
            .manifest = manifest,
            .hf_tok = tok,
            .chat_template = chat_template,
        };
    }

    fn deinit(self: *OwnedPreferenceTextTokenizer) void {
        if (self.chat_template) |*tmpl| tmpl.deinit();
        self.hf_tok.deinitSelf();
        self.manifest.deinit();
        self.* = undefined;
    }

    fn tokenizer(self: *const OwnedPreferenceTextTokenizer) tokenizer_mod.Tokenizer {
        return self.hf_tok.tokenizer();
    }

    fn view(self: *OwnedPreferenceTextTokenizer) PreferenceTextTokenizerView {
        return .{
            .tokenizer = self.tokenizer(),
            .add_bos_token = self.manifest.add_bos_token,
            .bos_token = self.manifest.bos_token,
            .chat_template = if (self.chat_template) |*tmpl| tmpl else null,
        };
    }
};

fn loadDpoTextPreferenceSamples(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    recipe: Recipe,
    tokenizer_assets: PreferenceTextTokenizerView,
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
        const tokenized = try tokenizeDpoTextRow(arena_alloc, tokenizer_assets, recipe, row);
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
    tokenizer_assets: PreferenceTextTokenizerView,
    recipe: Recipe,
    row: DpoTextRow,
) !TokenizedPreferenceRow {
    const tokenizer = tokenizer_assets.tokenizer;
    const max_seq_len = recipe.dataset.max_seq_len orelse 2048;
    const render_prompt = !std.mem.eql(u8, recipe.dataset.format orelse "text-preference", "rendered-text-preference");
    const prompt_text = if (render_prompt)
        try tokenizer_assets.renderPrompt(allocator, row.prompt)
    else
        try allocator.dupe(u8, row.prompt);
    defer allocator.free(prompt_text);

    var prompt_encoded = try generation.encodePromptForGeneration(
        tokenizer,
        allocator,
        prompt_text,
        max_seq_len,
        tokenizer_assets.add_bos_token,
        tokenizer_assets.bos_token,
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
    return PreferenceTextTokenizerView.fromLoadedModel(model).renderPrompt(allocator, prompt);
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
};

fn recordPreferenceSessionRun(runs_started: *usize) PreferenceSessionRunTelemetry {
    runs_started.* += 1;
    return .{
        .shared = true,
        .model_admissions = 1,
        .run_index = runs_started.*,
        .reuse_hit = runs_started.* > 1,
    };
}

/// Process-local ownership boundary for a sequence of compatible Gemma4
/// preference jobs. The admitted model and compute backend stay resident;
/// every DPO/GRPO runner still owns a fresh trainer, graph cache, adapter
/// slots, gradients, and optimizer state.
const GemmaPreferenceSession = struct {
    allocator: std.mem.Allocator,
    model_path: [:0]u8,
    backend_kind: gemma4_real_autodiff.BackendKind,
    graph_config: gpt_arch.Config,
    backend: gemma4_real_autodiff.LoadedBackend,
    runs_started: usize = 0,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        model_path: []const u8,
        backend_kind: gemma4_real_autodiff.BackendKind,
    ) !GemmaPreferenceSession {
        const canonical_model_path = try compat.cwd().realPathFileAlloc(io, model_path, allocator);
        errdefer allocator.free(canonical_model_path);
        const graph_config = try gemma4_real_autodiff.loadGraphConfig(allocator, canonical_model_path);
        var backend = try gemma4_real_autodiff.loadBackendForModelDir(allocator, canonical_model_path, backend_kind);
        errdefer backend.deinit();
        return .{
            .allocator = allocator,
            .model_path = canonical_model_path,
            .backend_kind = backend_kind,
            .graph_config = graph_config,
            .backend = backend,
        };
    }

    fn deinit(self: *GemmaPreferenceSession) void {
        self.backend.deinit();
        self.allocator.free(self.model_path);
        self.* = undefined;
    }

    fn requireCompatible(
        self: *const GemmaPreferenceSession,
        allocator: std.mem.Allocator,
        io: std.Io,
        model_path: []const u8,
        backend_kind: gemma4_real_autodiff.BackendKind,
    ) !void {
        if (backend_kind != self.backend_kind) return error.PreferenceSuiteBackendMismatch;
        const canonical_model_path = try compat.cwd().realPathFileAlloc(io, model_path, allocator);
        defer allocator.free(canonical_model_path);
        if (!std.mem.eql(u8, self.model_path, canonical_model_path)) {
            return error.PreferenceSuiteModelMismatch;
        }
    }

    fn beginRun(
        self: *GemmaPreferenceSession,
        allocator: std.mem.Allocator,
        io: std.Io,
        model_path: []const u8,
        backend_kind: gemma4_real_autodiff.BackendKind,
    ) !PreferenceSessionRunTelemetry {
        try self.requireCompatible(allocator, io, model_path, backend_kind);
        const telemetry = recordPreferenceSessionRun(&self.runs_started);
        print("preference session run {d}: backend={s} reuse_hit={any}\n", .{
            telemetry.run_index,
            @tagName(self.backend_kind),
            telemetry.reuse_hit,
        });
        return telemetry;
    }

    fn reuseHits(self: *const GemmaPreferenceSession) usize {
        return self.runs_started -| 1;
    }
};

/// Keep Gemma policy training and reference scoring on one explicit backend.
/// `auto` retains the historical CPU-safe recipe behavior; callers requesting
/// Metal or CUDA get the strict compiled-device training lane and no fallback.
fn resolveGemmaPreferenceExecution(value: ?[]const u8) !GemmaPreferenceExecution {
    const requested = try parseRecipeBackendChoice(value);
    const exact_choice: native_backend_choice.Choice = switch (requested) {
        .auto, .native => .native,
        .metal => .metal,
        .cuda => .cuda,
        .onnx, .xla, .webgpu => return error.UnsupportedBackend,
    };
    try native_backend_choice.validate(exact_choice);
    return .{
        .backend_kind = switch (exact_choice) {
            .native => .native,
            .metal => .metal,
            .cuda => .cuda,
            else => unreachable,
        },
    };
}

fn validateGemmaPreferenceModality(
    backend_kind: gemma4_real_autodiff.BackendKind,
    projector_path: ?[]const u8,
) !void {
    if (backend_kind == .cuda and projector_path != null) {
        // Text DPO/GRPO is qualified. Multimodal GRPO owns a separate
        // projector graph and remains fail-closed until its CUDA backward
        // primitives and media-residency evidence have an explicit gate.
        return error.Gemma4CudaMultimodalGrpoNotSupported;
    }
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

    const native = try resolveGemmaPreferenceExecution("native");
    try std.testing.expectEqual(gemma4_real_autodiff.BackendKind.native, native.backend_kind);

    if (build_options.enable_metal) {
        const metal = try resolveGemmaPreferenceExecution("metal");
        try std.testing.expectEqual(gemma4_real_autodiff.BackendKind.metal, metal.backend_kind);
    } else {
        try std.testing.expectError(error.BackendUnavailable, resolveGemmaPreferenceExecution("metal"));
    }

    if (build_options.enable_cuda) {
        const cuda = try resolveGemmaPreferenceExecution("cuda");
        try std.testing.expectEqual(gemma4_real_autodiff.BackendKind.cuda, cuda.backend_kind);
        try std.testing.expectError(
            error.Gemma4CudaMultimodalGrpoNotSupported,
            validateGemmaPreferenceModality(.cuda, "projector.gguf"),
        );
        try validateGemmaPreferenceModality(.cuda, null);
    } else {
        try std.testing.expectError(error.BackendUnavailable, resolveGemmaPreferenceExecution("cuda"));
    }
    try std.testing.expectError(error.InvalidBackend, resolveGemmaPreferenceExecution("bogus"));
}

test "gemma4 preference suite admits only optimizer-backed text jobs" {
    const dpo = Recipe{
        .recipe = "dpo",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/dpo.jsonl", .format = "text-preference" },
        .adapter = .{ .rank = 2, .alpha = 4 },
        .artifacts = .{
            .root = "/tmp/preference-suite/dpo",
            .trained_adapter_dir = "/tmp/preference-suite/dpo-trained",
        },
        .backend = "native",
    };
    const dpo_execution = try validatePreferenceSuiteRecipe(dpo);
    try std.testing.expectEqual(gemma4_real_autodiff.BackendKind.native, dpo_execution.backend_kind);

    var grpo_recipe = dpo;
    grpo_recipe.recipe = "grpo";
    grpo_recipe.dataset.format = "text-grpo";
    grpo_recipe.artifacts = .{
        .root = "/tmp/preference-suite/grpo",
        .trained_adapter_dir = "/tmp/preference-suite/grpo-trained",
    };
    _ = try validatePreferenceSuiteRecipe(grpo_recipe);

    var sft = dpo;
    sft.recipe = "lora-sft";
    try std.testing.expectError(error.PreferenceSuiteRequiresDpoOrGrpo, validatePreferenceSuiteRecipe(sft));

    var multimodal = dpo;
    multimodal.model.projector_path = "/models/gemma4/mmproj.gguf";
    try std.testing.expectError(error.PreferenceSuiteRequiresTextGemma4, validatePreferenceSuiteRecipe(multimodal));

    var report_only = dpo;
    report_only.adapter = null;
    report_only.artifacts.trained_adapter_dir = null;
    try std.testing.expectError(error.PreferenceSuiteRequiresTrainedAdapterDir, validatePreferenceSuiteRecipe(report_only));
}

test "gemma4 preference recipes fail closed on ignored or invalid training semantics" {
    const dpo = Recipe{
        .recipe = "dpo",
        .model = .{ .path = "/models/gemma4", .family = "gemma4" },
        .dataset = .{ .path = "/data/dpo.jsonl", .format = "text-preference" },
        .adapter = .{ .rank = 2, .alpha = 4 },
        .optimizer = .{
            .learning_rate = 1e-4,
            .weight_decay = 0.01,
            .epochs = 1,
            .gradient_accumulation_steps = 1,
            .max_grad_norm = 1.0,
        },
        .preference = .{ .beta = 0.1 },
        .artifacts = .{ .root = "/tmp/preference-contract/dpo" },
        .backend = "native",
    };
    try validateGemma4PreferenceRecipeContract(dpo, .dpo);
    const plan = try buildPlan(std.testing.allocator, dpo);
    defer freePlan(std.testing.allocator, plan);

    var invalid = dpo;
    invalid.optimizer.learning_rate = 0.0;
    try std.testing.expectError(error.InvalidLearningRate, buildPlan(std.testing.allocator, invalid));

    invalid = dpo;
    invalid.optimizer.weight_decay = -0.01;
    try std.testing.expectError(error.InvalidWeightDecay, buildPlan(std.testing.allocator, invalid));

    invalid = dpo;
    invalid.optimizer.max_steps = 10;
    try std.testing.expectError(error.UnsupportedGemma4PreferenceOptimizerOption, buildPlan(std.testing.allocator, invalid));

    invalid = dpo;
    invalid.preference.beta = 0.0;
    try std.testing.expectError(error.InvalidDpoBeta, buildPlan(std.testing.allocator, invalid));

    invalid = dpo;
    invalid.preference.simpo_gamma = 0.5;
    try std.testing.expectError(error.UnsupportedGemma4DpoOption, buildPlan(std.testing.allocator, invalid));

    invalid = dpo;
    invalid.checkpoint = .{ .every_epochs = 1 };
    try std.testing.expectError(error.UnsupportedGemma4PreferenceCheckpointOption, buildPlan(std.testing.allocator, invalid));

    invalid = dpo;
    invalid.dataset.train_path = "/data/other.jsonl";
    try std.testing.expectError(error.ConflictingPreferenceDatasetPaths, buildPlan(std.testing.allocator, invalid));

    invalid = dpo;
    invalid.model.projector_path = "/models/gemma4/mmproj.gguf";
    try std.testing.expectError(error.Gemma4MultimodalDpoNotSupported, buildPlan(std.testing.allocator, invalid));

    var grpo_recipe = dpo;
    grpo_recipe.recipe = "grpo";
    grpo_recipe.dataset.format = "text-grpo";
    grpo_recipe.preference = .{};
    grpo_recipe.grpo = .{
        .group_size = 8,
        .clip_epsilon = 0.2,
        .kl_coef = 0.04,
        .advantage_eps = 1e-8,
        .max_completion_tokens = 1,
        .reward_mode = "prefix-match",
    };
    try validateGemma4PreferenceRecipeContract(grpo_recipe, .grpo);

    grpo_recipe.grpo.group_size = 9;
    try std.testing.expectError(error.InvalidGrpoGroupSize, buildPlan(std.testing.allocator, grpo_recipe));
    grpo_recipe.grpo.group_size = 2;
    grpo_recipe.grpo.clip_epsilon = std.math.nan(f32);
    try std.testing.expectError(error.InvalidGrpoClipEpsilon, buildPlan(std.testing.allocator, grpo_recipe));
}

test "gemma4 preference suite output namespaces are pairwise disjoint" {
    const recipes = [_]Recipe{
        .{
            .recipe = "dpo",
            .dataset = .{ .path = "/tmp/preference-suite/a/dpo.jsonl" },
            .artifacts = .{
                .root = "/tmp/preference-suite/a",
                .trained_adapter_dir = "/tmp/preference-suite/a/adapter-trained",
            },
        },
        .{
            .recipe = "grpo",
            .dataset = .{ .path = "/tmp/preference-suite/b/grpo.jsonl" },
            .artifacts = .{
                .root = "/tmp/preference-suite/b",
                .trained_adapter_dir = "/tmp/preference-suite/b/adapter-trained",
            },
        },
    };
    const recipe_paths = [_][]const u8{
        "/tmp/preference-suite/a/recipe.json",
        "/tmp/preference-suite/b/recipe.json",
    };
    try validatePreferenceSuiteArtifacts(
        std.testing.allocator,
        std.testing.io,
        &recipes,
        &recipe_paths,
        "/tmp/preference-suite/preference_suite_report.json",
    );

    var cross_conflict = recipes;
    cross_conflict[1].artifacts.trained_adapter_dir = "/tmp/preference-suite/a/checkpoint";
    try std.testing.expectError(
        error.PreferenceSuiteArtifactConflict,
        validatePreferenceSuiteArtifacts(
            std.testing.allocator,
            std.testing.io,
            &cross_conflict,
            &recipe_paths,
            "/tmp/preference-suite/preference_suite_report.json",
        ),
    );

    var root_conflict = recipes;
    root_conflict[1].artifacts.root = "/tmp/preference-suite/a/nested";
    try std.testing.expectError(
        error.PreferenceSuiteArtifactConflict,
        validatePreferenceSuiteArtifacts(
            std.testing.allocator,
            std.testing.io,
            &root_conflict,
            &recipe_paths,
            "/tmp/preference-suite/preference_suite_report.json",
        ),
    );

    var shared_report = recipes;
    shared_report[0].artifacts.report_path = "/tmp/preference-suite/shared-objective.json";
    shared_report[1].artifacts.report_path = "/tmp/preference-suite/shared-objective.json";
    try std.testing.expectError(
        error.PreferenceSuiteArtifactConflict,
        validatePreferenceSuiteArtifacts(
            std.testing.allocator,
            std.testing.io,
            &shared_report,
            &recipe_paths,
            "/tmp/preference-suite/preference_suite_report.json",
        ),
    );

    var shared_manifest = recipes;
    shared_manifest[0].artifacts.manifest_path = "/tmp/preference-suite/shared-manifest.json";
    shared_manifest[1].artifacts.manifest_path = "/tmp/preference-suite/shared-manifest.json";
    try std.testing.expectError(
        error.PreferenceSuiteArtifactConflict,
        validatePreferenceSuiteArtifacts(
            std.testing.allocator,
            std.testing.io,
            &shared_manifest,
            &recipe_paths,
            "/tmp/preference-suite/preference_suite_report.json",
        ),
    );

    var report_collision = recipes;
    report_collision[0].artifacts.report_path = "/tmp/preference-suite/job-report.json";
    try std.testing.expectError(
        error.PreferenceSuiteArtifactConflict,
        validatePreferenceSuiteArtifacts(
            std.testing.allocator,
            std.testing.io,
            &report_collision,
            &recipe_paths,
            "/tmp/preference-suite/job-report.json",
        ),
    );

    var input_collision = recipes;
    input_collision[0].dataset.path = "/tmp/preference-suite/dataset.jsonl";
    try std.testing.expectError(
        error.PreferenceSuiteInputOutputConflict,
        validatePreferenceSuiteArtifacts(
            std.testing.allocator,
            std.testing.io,
            &input_collision,
            &recipe_paths,
            "/tmp/preference-suite/dataset.jsonl",
        ),
    );
}

test "gemma4 preference suite v2 report keeps admission and per-job timing separate" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const report_path = try std.fs.path.join(allocator, &.{
        ".zig-cache",
        "tmp",
        tmp.sub_path[0..],
        "preference-suite.json",
    });
    defer allocator.free(report_path);
    const timings = [_]PreferenceSuiteRunTiming{
        .{ .run_index = 1, .objective = "dpo", .duration_seconds = 3.5 },
        .{ .run_index = 2, .objective = "grpo", .duration_seconds = 4.5 },
    };
    try writePreferenceSuiteReport(
        allocator,
        io,
        report_path,
        .succeeded,
        "/models/gemma4",
        .cuda,
        2,
        2,
        1,
        1,
        2.25,
        10.5,
        &timings,
    );

    const raw = try readFileMax(allocator, io, report_path, 64 * 1024);
    defer allocator.free(raw);
    var parsed = try std.json.parseFromSlice(PreferenceSuiteReport, allocator, raw, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("antfly_inference_gemma4_preference_suite/v2", parsed.value.schema_version);
    try std.testing.expectEqual(@as(?f64, 2.25), parsed.value.model_admission_seconds);
    try std.testing.expectEqual(@as(?f64, 10.5), parsed.value.total_duration_seconds);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.runs.len);
    try std.testing.expectEqualStrings("grpo", parsed.value.runs[1].objective);
    try std.testing.expectEqual(@as(f64, 4.5), parsed.value.runs[1].duration_seconds);
}

test "gemma4 preference session telemetry records one admission and reuse hits" {
    var runs_started: usize = 0;
    const first = recordPreferenceSessionRun(&runs_started);
    try std.testing.expect(first.shared);
    try std.testing.expectEqual(@as(usize, 1), first.model_admissions);
    try std.testing.expectEqual(@as(usize, 1), first.run_index);
    try std.testing.expect(!first.reuse_hit);

    const second = recordPreferenceSessionRun(&runs_started);
    try std.testing.expectEqual(@as(usize, 1), second.model_admissions);
    try std.testing.expectEqual(@as(usize, 2), second.run_index);
    try std.testing.expect(second.reuse_hit);
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

test "compiled DPO loss inversion recovers reward margin" {
    for ([_]f32{ -8.0, -1.25, 0.0, 2.5, 8.0 }) |expected_margin| {
        const loss = -@log(1.0 / (1.0 + @exp(-expected_margin)));
        const actual_margin = try rewardMarginFromDpoLoss(loss);
        try std.testing.expectApproxEqAbs(expected_margin, actual_margin, 2e-3);
    }
    try std.testing.expectError(error.InvalidDpoCompiledLoss, rewardMarginFromDpoLoss(0.0));
    try std.testing.expectError(error.InvalidDpoCompiledLoss, rewardMarginFromDpoLoss(std.math.nan(f32)));
}

test "gemma4 preference sequence buckets minimize padding within the configured cap" {
    try std.testing.expectEqual(@as(u32, 8), try gemmaPreferenceSequenceBucket(1, 8));
    try std.testing.expectEqual(@as(u32, 16), try gemmaPreferenceSequenceBucket(1, 128));
    try std.testing.expectEqual(@as(u32, 16), try gemmaPreferenceSequenceBucket(16, 128));
    try std.testing.expectEqual(@as(u32, 32), try gemmaPreferenceSequenceBucket(17, 128));
    try std.testing.expectEqual(@as(u32, 64), try gemmaPreferenceSequenceBucket(33, 128));
    try std.testing.expectEqual(@as(u32, 100), try gemmaPreferenceSequenceBucket(90, 100));
    try std.testing.expectError(error.EmptySequence, gemmaPreferenceSequenceBucket(0, 128));
    try std.testing.expectError(error.SequenceTooLong, gemmaPreferenceSequenceBucket(129, 128));
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
    try std.testing.expect(shouldCoalesceGemmaDpoSingleTokenPairs(.cuda, &.{chosen}, &.{rejected}));
    try std.testing.expect(shouldCoalesceGemmaDpoSingleTokenPairs(.metal, &.{chosen}, &.{rejected}));
    try std.testing.expect(!shouldCoalesceGemmaDpoSingleTokenPairs(.native, &.{chosen}, &.{rejected}));
    const sequence_bucketing = try summarizeGemmaDpoSequenceBucketing(&.{chosen}, &.{rejected}, 128);
    try std.testing.expectEqual(@as(usize, 4), sequence_bucketing.min_required);
    try std.testing.expectEqual(@as(usize, 4), sequence_bucketing.max_required);
    try std.testing.expectEqual(@as(u32, 16), sequence_bucketing.min_bucket);
    try std.testing.expectEqual(@as(u32, 16), sequence_bucketing.max_bucket);

    rejected.labels[0] = 7;
    try std.testing.expect(gemmaDpoSingleTokenPair(&chosen, &rejected) == null);
    try std.testing.expect(!allGemmaDpoPairsAreSingleTokenSharedPrompt(&.{chosen}, &.{rejected}));
}

test "gemma4 GRPO learning-signal gate requires reward variation and a policy gradient" {
    try std.testing.expectError(error.NoGrpoLearningSignal, validateGrpoLearningSignal(true, false, 1.0, 1.0, 2, 0.0));
    try std.testing.expectError(error.NoGrpoLearningSignal, validateGrpoLearningSignal(false, true, 0.0, 0.0, 2, 0.5));
    try validateGrpoLearningSignal(true, true, 1.0, 1.0, 2, 0.5);
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

fn runDirectGrpo(
    allocator: std.mem.Allocator,
    io: std.Io,
    recipe: Recipe,
    report_path: []const u8,
    preference_session: ?*GemmaPreferenceSession,
) !void {
    const path = trainDatasetPath(recipe) orelse return error.MissingDatasetPath;
    const format = recipe.dataset.format orelse "token-logprobs";
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
        try grpo.computeAdvantages(&ga, batch.completions, cfg);
        var result = try grpo.grpoLoss(allocator, batch.completions, batch.new_logps, ga.advantages, cfg);
        defer result.deinit();
        try writeJsonFile(allocator, io, report_path, GrpoReport{
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
    if (!std.mem.eql(u8, format, "text-grpo") and !std.mem.eql(u8, format, "rendered-text-grpo")) {
        return error.UnsupportedGrpoFormat;
    }
    try validateRankedGrpoGroupSize(recipe.grpo.group_size orelse 2);
    if (try shouldRunOptimizerBackedQwen2Grpo(recipe, format)) {
        try runOptimizerBackedQwen2Grpo(allocator, io, recipe, path, report_path);
        return;
    }
    if (try shouldRunOptimizerBackedGemmaGrpo(recipe, format)) {
        try runOptimizerBackedGemmaGrpo(allocator, io, recipe, path, report_path, preference_session);
        return;
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

    var prompt_batch = try loadGrpoTextPrompts(
        allocator,
        io,
        path,
        recipe,
        PreferenceTextTokenizerView.fromLoadedModel(policy_model),
    );
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
        .group_size = recipe.grpo.group_size orelse 2,
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
        .completions = prompt_batch.prompts.len * (recipe.grpo.group_size orelse 2),
        .tokens = result.grad_new_logps.len,
        .groups = prompt_batch.prompts.len,
        .loss = result.loss,
        .pg_loss = result.pg_loss,
        .kl_loss = result.kl_loss,
        .clip_fraction = result.clip_fraction,
        .reward_mode = recipe.grpo.reward_mode orelse "exact-match",
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
    tokenizer_assets: PreferenceTextTokenizerView,
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
    const targets = try aa.alloc([]const u8, rows.items.len);
    for (rows.items, 0..) |row, idx| {
        const tokenized_prompt = try tokenizeGrpoPrompt(aa, tokenizer_assets, recipe, row.prompt);
        prompts[idx] = tokenized_prompt;
        // parseFromSliceLeaky may borrow string storage from `raw`, which is
        // released when this loader returns. Keep reward targets in the arena
        // alongside the tokenized prompts so GRPO never scores dangling data.
        targets[idx] = try aa.dupe(u8, row.target);
    }

    return .{
        .arena = arena,
        .prompts = prompts,
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
    tokenizer_assets: PreferenceTextTokenizerView,
    recipe: Recipe,
    prompt: []const u8,
) ![]const i32 {
    const tokenizer = tokenizer_assets.tokenizer;
    const max_seq_len = recipe.dataset.max_seq_len orelse 128;
    const max_completion_tokens = recipe.grpo.max_completion_tokens orelse 4;
    if (max_completion_tokens == 0 or max_completion_tokens >= max_seq_len) return error.NoCompletionBudget;
    const prompt_max_seq_len = max_seq_len - max_completion_tokens;
    const render_prompt = !std.mem.eql(u8, recipe.dataset.format orelse "text-grpo", "rendered-text-grpo");
    const prompt_text = if (render_prompt)
        try tokenizer_assets.renderPrompt(allocator, prompt)
    else
        try allocator.dupe(u8, prompt);
    defer allocator.free(prompt_text);

    var encoded = try generation.encodePromptForGeneration(
        tokenizer,
        allocator,
        prompt_text,
        prompt_max_seq_len,
        tokenizer_assets.add_bos_token,
        tokenizer_assets.bos_token,
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
        \\       antfly inference finetune run-suite [--report <path>] <recipe.json> <recipe.json>... [--dry-run]
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
    const telemetry = try recorder.finish(.{
        .tensor_count = 4,
        .changed_tensor_count = 2,
        .max_abs_delta = 0.001,
    });
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
    try std.testing.expectError(error.DpoBenchmarkUpdateCountMismatch, incomplete.finish(.{
        .tensor_count = 1,
        .changed_tensor_count = 1,
        .max_abs_delta = 0.001,
    }));

    var stagnant = try DpoBenchmarkRecorder.init(std.testing.allocator);
    defer stagnant.deinit();
    for (0..DpoBenchmarkRecorder.total_updates) |idx| {
        try stagnant.record(@floatFromInt(idx + 1), 0.6931472);
    }
    try std.testing.expectError(error.DpoBenchmarkNoPolicyMovement, stagnant.finish(.{
        .tensor_count = 4,
        .changed_tensor_count = 0,
        .max_abs_delta = 0.0,
    }));
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
    const telemetry = try recorder.finish(.{
        .tensor_count = 4,
        .changed_tensor_count = 2,
        .max_abs_delta = 0.001,
    });
    try std.testing.expectEqual(@as(usize, 1), telemetry.protocol.cold);
    try std.testing.expectEqual(@as(usize, 1), telemetry.protocol.first);
    try std.testing.expectEqual(@as(usize, 3), telemetry.protocol.warmup);
    try std.testing.expectEqual(@as(usize, 20), telemetry.protocol.measured);
    try std.testing.expectEqual(@as(f64, 1.0), telemetry.cold.seconds);
    try std.testing.expectEqual(@as(f64, 2.0), telemetry.first.seconds);
    try std.testing.expectEqual(@as(f64, 15.5), telemetry.median_seconds);
    try std.testing.expectEqual(@as(f64, 15.5), telemetry.mean_seconds);
    try std.testing.expectEqual(@as(usize, 2), telemetry.trainable_update.changed_tensor_count);
    try std.testing.expectEqual(@as(f32, 0.001), telemetry.trainable_update.max_abs_delta);

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
    try std.testing.expectError(error.GrpoBenchmarkNoPolicyMovement, stagnant.finish(.{
        .tensor_count = 4,
        .changed_tensor_count = 0,
        .max_abs_delta = 0.0,
    }));
}

test "gemma4 trainable snapshot reports exact LoRA tensor movement" {
    var weights_a = [_]f32{ 0.0, 1.0 };
    var weights_b = [_]f32{ -2.0, 3.0 };
    var gradients_a = [_]f32{ 0.0, 0.0 };
    var gradients_b = [_]f32{ 0.0, 0.0 };
    var dims = [_]i32{2};
    var slots = [_]real_autodiff.RealAutodiffTrainer.ParamSlot{
        .{
            .name = "layer.q_proj.lora_A",
            .weights = &weights_a,
            .grad_accum = &gradients_a,
            .node_id = 1,
            .dims = &dims,
        },
        .{
            .name = "layer.q_proj.lora_B",
            .weights = &weights_b,
            .grad_accum = &gradients_b,
            .node_id = 2,
            .dims = &dims,
        },
    };
    var snapshot = try TrainableParameterSnapshot.init(std.testing.allocator, &slots);
    defer snapshot.deinit();

    weights_b[1] += 0.125;
    const update = try snapshot.summarize(&slots);
    try std.testing.expectEqual(@as(usize, 2), update.tensor_count);
    try std.testing.expectEqual(@as(usize, 1), update.changed_tensor_count);
    try std.testing.expectEqual(@as(f32, 0.125), update.max_abs_delta);
}

test "gemma4 GRPO reference cache uses exact keys and bounded eviction" {
    var cache = GemmaGrpoReferenceCache.init(std.testing.allocator, 2);
    defer cache.deinit();

    var output: [2]f32 = undefined;
    try std.testing.expect(!try cache.lookup(0, &.{ 10, 11 }, &output));
    try cache.insert(0, &.{ 10, 11 }, &.{ -0.1, -0.2 });
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
    report_only.artifacts = .{ .root = "/tmp/qwen35-report" };
    const plan = try buildPlan(std.heap.page_allocator, report_only);
    defer freePlan(std.heap.page_allocator, plan);
    try std.testing.expectEqual(StepKind.direct_dpo, plan.steps[0].kind);
    try std.testing.expect(!try shouldRunOptimizerBackedQwen2Dpo(report_only, "text-preference"));

    const grpo_recipe = Recipe{
        .recipe = "grpo",
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
        try runPlan(allocator, io, ".", recipe, bootstrap_plan, manifest_path, training_config_path, training_report_path, null);
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
    try std.testing.expectError(error.UnsupportedGemma4RuntimeOption, buildPlan(std.heap.page_allocator, recipe));
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
    const dpo_plan = try buildPlan(std.heap.page_allocator, dpo);
    defer freePlan(std.heap.page_allocator, dpo_plan);
    try std.testing.expectEqual(StepKind.direct_dpo, dpo_plan.steps[0].kind);

    var grpo_recipe = base;
    grpo_recipe.recipe = "grpo";
    const grpo_plan = try buildPlan(std.heap.page_allocator, grpo_recipe);
    defer freePlan(std.heap.page_allocator, grpo_plan);
    try std.testing.expectEqual(StepKind.direct_grpo, grpo_plan.steps[0].kind);
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
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), scoreTextReward(.exact_match, "yes indeed", "yes"), 1e-6);
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

test "ranked-first GRPO reward repeats a deterministic one-hot group" {
    var completion_index: usize = 0;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), try scoreRankedFirst(&completion_index, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), try scoreRankedFirst(&completion_index, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), try scoreRankedFirst(&completion_index, 2), 1e-6);
    try std.testing.expectError(error.InvalidGroupSize, scoreRankedFirst(&completion_index, 1));
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
