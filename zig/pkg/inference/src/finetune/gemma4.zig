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

// Gemma4 LoRA data, adapter, and artifact contracts. Production training uses
// the real causal-LM autodiff path in gemma4_real_autodiff.zig through
// gemma4_train_command.zig. The legacy surrogate helpers in this file remain
// available only behind the command's explicit diagnostic trainer mode.
const std = @import("std");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const compat = @import("../io/compat.zig");
const lora = @import("lora.zig");
const lora_init = @import("lora_init.zig");
const peft = @import("peft.zig");
const qlora_nf4 = @import("qlora_nf4.zig");
const graph_bridge = @import("graph_bridge.zig");
const recursive_lora = @import("recursive_lora.zig");
const safetensors = @import("../models/safetensors.zig");
const tensor_mod = @import("../backends/tensor.zig");
const DType = tensor_mod.DType;
const Tensor = tensor_mod.Tensor;
const tensor_access = @import("../models/tensor_access.zig");
const gpt_model = @import("../models/gpt.zig");
const manifest_mod = @import("../models/manifest.zig");
const tensor_store_mod = @import("../models/tensor_store.zig");
const weight_source = @import("../models/weight_source.zig");
const hf_tokenizer = @import("inference_hf_tokenizer");
const sentencepiece = @import("inference_tokenizer").sentencepiece;
const tokenizer_mod = @import("inference_tokenizer");
const c_file = @import("../util/c_file.zig");
const ml = @import("ml");
const optimizers = ml.graph.optimizers;
const native_compute = @import("../ops/native_compute.zig");
const gemma_data = @import("gemma_data.zig");
const gemma_chat_data = @import("gemma_chat_data.zig");
const jsonl_resolve = @import("jsonl_resolve.zig");
const chat_template = @import("chat_template.zig");
const artifact_publication = @import("artifact_publication.zig");
const gemma4_mm = @import("../architectures/gemma4_multimodal.zig");
const gemma4_projector = @import("../architectures/gemma4_projector.zig");
const session_factory = @import("../architectures/session_factory.zig");
const model_manager_mod = @import("../server/model_manager.zig");

pub const artifact_family_version = "gemma4_lora/v1alpha1";
pub const prepared_schema_v2 = "gemma4_prepared/v2";
pub const prepared_schema_v3 = "gemma4_prepared/v3";
pub const prepared_schema_v4 = "gemma4_prepared/v4";
pub const prepared_schema_v5 = "gemma4_prepared/v5";
/// Causal chat tokenization: the rendered Gemma transcript owns its literal
/// BOS token, no implicit EOS is appended, and every assistant turn is
/// supervised even when the tokenizer cannot provide byte offsets.
pub const prepared_schema_v6 = "gemma4_prepared/v6";
/// Until the Metal attention training graph is chunked, admitting longer
/// sequences can create an unbounded quadratic allocation. Keep this explicit
/// and versioned rather than relying on allocator failure as input admission.
pub const max_training_seq_len: usize = 2048;
pub const prepared_chat_template_identity = "antfly_gemma_chat/v1";
pub const checkpoint_file_name = "model.safetensors";
pub const adapter_checkpoint_file_name = "adapter_model.safetensors";
pub const hf_config_file_name = "config.json";
pub const adapter_config_file_name = "adapter_config.json";
pub const adapter_manifest_file_name = "antfly_finetune_manifest.json";
pub const adapter_manifest_schema_v2 = "antfly_gemma4_finetune/v2";
pub const adapter_tensor_key_format_v1 = "antfly_gemma4_adapter_keys/v1";
pub const stock_peft_tensor_key_format_v1 = "stock-peft/v1";
pub const peft_export_manifest_file_name = "antfly_peft_export.json";
pub const peft_export_manifest_schema_v1 = "antfly_gemma4_peft_export/v1";
pub const tokenizer_config_file_name = "tokenizer_config.json";
pub const tokenizer_file_name = "tokenizer.json";
pub const special_tokens_map_file_name = "special_tokens_map.json";

// The legacy merge implementation converts base tensors to f32 and builds a
// complete output buffer. Keep it available for small diagnostic fixtures,
// but fail closed before admitting production-sized or multi-file models.
// E2B/E4B materialization must use the forthcoming streaming writer.
pub const legacy_materialize_max_checkpoint_bytes: u64 = 64 * 1024 * 1024;

pub const default_lora_target_modules = [_][]const u8{
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
};

/// PEFT-style compatibility inventory for query/value-only LoRA.
pub const peft_qv_lora_target_modules = [_][]const u8{
    "q_proj",
    "v_proj",
};

/// The complete trainable text-linear inventory for dense Gemma 4 E-models.
/// The PLE aliases are intentionally specific so a bare `proj` never selects
/// an encoder or projector tensor.
pub const text_all_linear_lora_target_modules = [_][]const u8{
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "per_layer_input.inp_gate",
    "per_layer_input.proj",
    "per_layer_input.per_layer_model_proj",
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

pub const Variant = enum {
    merged,
    adapter_only,
    incomplete,
};

pub const Gemma4LoRATargetPreset = enum {
    /// Hugging Face PEFT compatibility: query and value projections only.
    peft_qv,
    /// Every available text transformer and PLE linear in the checkpoint.
    text_all_linear,
};

pub fn parseGemma4LoRATargetPreset(name: []const u8) ?Gemma4LoRATargetPreset {
    if (std.mem.eql(u8, name, "peft-qv") or std.mem.eql(u8, name, "peft_qv")) return .peft_qv;
    if (std.mem.eql(u8, name, "text-all-linear") or std.mem.eql(u8, name, "text_all_linear")) return .text_all_linear;
    return null;
}

pub const PreparedExampleInput = struct {
    mode: gemma_data.Mode,
    prompt_input_ids: []i32,
    response_input_ids: []i32,
    num_prompt_tokens: usize,
    num_response_tokens: usize,
    input_ids: []i32 = &.{},
    labels: []i32 = &.{},
    num_input_tokens: usize = 0,
    num_supervised_tokens: usize = 0,
    turn_count: usize = 0,
    has_tool_calls: bool = false,
    has_tool_messages: bool = false,
    image_paths: []const []const u8 = &.{},
    audio_paths: []const []const u8 = &.{},
    image_token_counts: []const usize = &.{},
    audio_token_counts: []const usize = &.{},
    teacher_top_k_token_ids: []i32 = &.{},
    teacher_top_k_probs: []f32 = &.{},
    teacher_top_k: usize = 0,
    teacher_temperature: f32 = 1.0,
    was_truncated: bool = false,
    turns_dropped_from_left: usize = 0,
    policy_version: ?[]const u8 = null,
    source_id: ?[]const u8 = null,
    source_group_id: ?[]const u8 = null,
    source_name: ?[]const u8 = null,
    source_record_sha256: ?[]const u8 = null,
    rendered_chat_sha256: ?[]const u8 = null,
    media_content_sha256: []const []const u8 = &.{},
};

pub const PreparedSourceIdentity = struct {
    dataset_path: []const u8,
    split: ?[]const u8 = null,
    revision: ?[]const u8 = null,
};

pub const PreparedInputsSummary = struct {
    artifact_family_version: []const u8,
    model_dir: []const u8,
    schema_version: []const u8 = prepared_schema_v2,
    gguf_projector_path: ?[]const u8 = null,
    gguf_projector_sha256: ?[]const u8 = null,
    gguf_projector_size_bytes: ?u64 = null,
    max_examples: usize,
    examples_seen: usize,
    tokenizer_class: ?[]const u8 = null,
    base_model_sha256: ?[]const u8 = null,
    tokenizer_sha256: ?[]const u8 = null,
    chat_template_sha256: ?[]const u8 = null,
    prepared_examples_sha256: ?[]const u8 = null,
    source_dataset_path: ?[]const u8 = null,
    source_dataset_sha256: ?[]const u8 = null,
    source_split: ?[]const u8 = null,
    source_revision: ?[]const u8 = null,
    max_seq_len: usize = 512,
    max_prompt_tokens: usize = 0,
    max_response_tokens: usize = 0,
    max_input_tokens: usize = 0,
    max_supervised_tokens: usize = 0,
    examples_with_tool_calls: usize = 0,
    examples_with_tool_messages: usize = 0,
    examples_with_multiturn: usize = 0,
    examples_with_images: usize = 0,
    examples_with_audio: usize = 0,
    examples_truncated: usize = 0,
    max_turns_dropped: usize = 0,
    examples: []PreparedExampleInput,
};

pub const SurrogateMetrics = struct {
    examples_seen: usize = 0,
    examples_skipped_no_supervision: usize = 0,
    supervised_tokens_seen: usize = 0,
    average_loss: f64 = 0,
    mse: f64 = 0,
    mae: f64 = 0,
    mean_score: f64 = 0,
};

pub const ProjectorFingerprint = struct {
    path: []const u8,
    sha256: []const u8,
    size_bytes: u64,
};

pub const ModelProvenance = struct {
    base_model_sha256: []const u8,
    tokenizer_sha256: []const u8,
    chat_template_sha256: []const u8,

    pub fn deinit(self: *ModelProvenance, allocator: std.mem.Allocator) void {
        allocator.free(self.base_model_sha256);
        allocator.free(self.tokenizer_sha256);
        allocator.free(self.chat_template_sha256);
        self.* = undefined;
    }
};

const PrepareMediaKind = enum { image, audio };

const PrepareMediaTokenCache = struct {
    items: std.StringHashMapUnmanaged(usize) = .empty,

    fn deinit(self: *PrepareMediaTokenCache, allocator: std.mem.Allocator) void {
        var it = self.items.iterator();
        while (it.next()) |entry| allocator.free(entry.key_ptr.*);
        self.items.deinit(allocator);
        self.* = undefined;
    }
};

pub const TrainEpochOptions = struct {
    learning_rate: f32 = 0.001,
    max_examples: usize = 32,
    layer_name: ?[]const u8 = null,
    max_grad_norm: f32 = 1.0,
    grad_accum_steps: u32 = 1,
    llrd_decay: f32 = 1.0,
    use_schedule_free: bool = false,
    warmup_steps: u32 = 0,
    compute_backend: ?*const @import("../ops/ops.zig").ComputeBackend = null,
    /// Number of DDP replicas. PJRT training falls back to the CPU path when
    /// more than one replica is active because that path has no collectives.
    world_size: u32 = 1,
    ddp_rank: u32 = 0,
    pjrt_lora_steps: if (build_options.enable_pjrt) ?[]?graph_bridge.LoRAPjrtTrainStep else void =
        if (build_options.enable_pjrt) null else {},
};

pub const TrainEpochSummary = struct {
    examples_seen: usize = 0,
    examples_skipped_no_supervision: usize = 0,
    supervised_tokens_seen: usize = 0,
    updates_applied: usize = 0,
    average_loss: f64 = 0,
    mean_score: f64 = 0,
    mean_abs_error: f64 = 0,
    max_grad_norm: f32 = 0,
    llrd_decay: f32 = 0,
    grad_accum_steps: u32 = 0,
};

pub const Config = struct {
    model_type: ?[]const u8 = null,
    hidden_size: ?usize = null,
    num_hidden_layers: ?usize = null,
    num_attention_heads: ?usize = null,
    vocab_size: ?usize = null,
    max_position_embeddings: ?usize = null,
    torch_dtype: ?[]const u8 = null,
    dtype: ?[]const u8 = null,
    text_config: ?TextConfig = null,
};

pub const TextConfig = struct {
    model_type: ?[]const u8 = null,
    hidden_size: ?usize = null,
    num_hidden_layers: ?usize = null,
    num_attention_heads: ?usize = null,
    vocab_size: ?usize = null,
    max_position_embeddings: ?usize = null,
    dtype: ?[]const u8 = null,
    torch_dtype: ?[]const u8 = null,
};

pub const AdapterConfig = struct {
    base_model_name_or_path: ?[]const u8 = null,
    antfly_base_model_sha256: ?[]const u8 = null,
    antfly_tokenizer_sha256: ?[]const u8 = null,
    antfly_chat_template_sha256: ?[]const u8 = null,
    peft_type: ?[]const u8 = null,
    task_type: ?[]const u8 = null,
    r: ?usize = null,
    lora_alpha: ?f64 = null,
    target_modules: ?[]const []const u8 = null,
    target_preset: ?[]const u8 = null,
    use_dora: ?bool = null,
    /// PEFT accepts either a boolean (the common default is `true`) or a
    /// named initializer such as `"pissa"`. Keep the JSON value here so
    /// adapters written by stock PEFT round-trip through inspection.
    init_lora_weights: ?std.json.Value = null,
    recursive_lora: ?recursive_lora.Config = null,
    bias: ?[]const u8 = null,
    fan_in_fan_out: ?bool = null,
    inference_mode: ?bool = null,
    lora_dropout: ?f64 = null,
    modules_to_save: ?[]const []const u8 = null,
    use_rslora: ?bool = null,
};

pub const AdapterManifest = struct {
    schema_version: []const u8,
    status: []const u8,
    artifact_family_version: []const u8,
    /// Antfly persists canonical frozen-weight identities so trainer slots,
    /// GGUF aliases, and sharded Safetensors share one exact namespace. A
    /// stock PEFT checkpoint uses a different wrapper prefix and omits the
    /// frozen `.weight` segment; interchange must translate and verify keys.
    tensor_key_format: []const u8,
    adapter_checkpoint_sha256: []const u8,
    adapter_checkpoint_size_bytes: u64,
    base_model_name_or_path: []const u8,
    base_model_sha256: []const u8,
    tokenizer_sha256: []const u8,
    chat_template_sha256: []const u8,
    target_modules: []const []const u8,
    target_preset: ?[]const u8 = null,
    rank: usize,
    alpha: f32,
    use_dora: bool = false,
    use_rslora: bool = false,
    initializer: ?[]const u8 = null,
    recursive_lora: ?recursive_lora.Config = null,
};

pub const TokenizerConfig = struct {
    model_max_length: ?f64 = null,
    tokenizer_class: ?[]const u8 = null,
};

pub const InspectionSummary = struct {
    artifact_family_version: []const u8,
    variant: Variant,
    model_dir: []const u8,
    checkpoint_path: ?[]const u8 = null,
    gguf_path: ?[]const u8 = null,
    adapter_checkpoint_path: ?[]const u8 = null,
    config_path: ?[]const u8 = null,
    adapter_config_path: ?[]const u8 = null,
    tokenizer_config_path: ?[]const u8 = null,
    tokenizer_path: ?[]const u8 = null,
    special_tokens_map_path: ?[]const u8 = null,
    base_model_name_or_path: ?[]const u8 = null,
    base_model_sha256: ?[]const u8 = null,
    tokenizer_sha256: ?[]const u8 = null,
    chat_template_sha256: ?[]const u8 = null,
    model_type: ?[]const u8 = null,
    hidden_size: ?usize = null,
    num_hidden_layers: ?usize = null,
    num_attention_heads: ?usize = null,
    vocab_size: ?usize = null,
    torch_dtype: ?[]const u8 = null,
    tokenizer_class: ?[]const u8 = null,
    tokenizer_model_max_length: ?usize = null,
    max_position_embeddings: ?usize = null,
    lora_rank: ?usize = null,
    lora_alpha: ?f64 = null,
    peft_type: ?[]const u8 = null,
    task_type: ?[]const u8 = null,
    inference_mode: ?bool = null,
    target_module_count: usize = 0,
    target_modules: ?[]const []const u8 = null,
    target_preset: ?[]const u8 = null,
    use_dora: ?bool = null,
    use_rslora: ?bool = null,
    lora_dropout: ?f64 = null,
    bias: ?[]const u8 = null,
    fan_in_fan_out: ?bool = null,
    modules_to_save_count: usize = 0,
    init_lora_weights: ?[]const u8 = null,
    recursive_lora_enabled: bool = false,
    recursive_source_num_layers: ?usize = null,
    recursive_shared_block_size: ?usize = null,
    recursive_loop_count: ?usize = null,
    recursive_init_strategy: ?[]const u8 = null,
    has_merged_weights: bool = false,
    has_gguf_weights: bool = false,
    has_adapter_weights: bool = false,
    has_tokenizer: bool = false,
};

pub const ArtifactPaths = struct {
    allocator: std.mem.Allocator,
    model_dir: []u8,
    checkpoint_path: ?[]u8 = null,
    gguf_path: ?[]u8 = null,
    adapter_checkpoint_path: ?[]u8 = null,
    config_path: ?[]u8 = null,
    adapter_config_path: ?[]u8 = null,
    adapter_manifest_path: ?[]u8 = null,
    tokenizer_config_path: ?[]u8 = null,
    tokenizer_path: ?[]u8 = null,
    special_tokens_map_path: ?[]u8 = null,

    pub fn deinit(self: *ArtifactPaths) void {
        self.allocator.free(self.model_dir);
        if (self.checkpoint_path) |p| self.allocator.free(p);
        if (self.gguf_path) |p| self.allocator.free(p);
        if (self.adapter_checkpoint_path) |p| self.allocator.free(p);
        if (self.config_path) |p| self.allocator.free(p);
        if (self.adapter_config_path) |p| self.allocator.free(p);
        if (self.adapter_manifest_path) |p| self.allocator.free(p);
        if (self.tokenizer_config_path) |p| self.allocator.free(p);
        if (self.tokenizer_path) |p| self.allocator.free(p);
        if (self.special_tokens_map_path) |p| self.allocator.free(p);
        self.* = undefined;
    }
};

pub const LoRATargetTensor = struct {
    tensor_name: []const u8,
    module_name: []const u8,
    input_dim: usize,
    output_dim: usize,
};

pub const BootstrapOptions = struct {
    rank: usize = 16,
    alpha: f32 = 32.0,
    base_model_name_or_path: ?[]const u8 = null,
    target_modules: ?[]const []const u8 = null,
    /// Gemma 4-specific strict presets. This is separate from the legacy
    /// cross-model PEFT presets so existing callers remain source-compatible.
    gemma4_target_preset: ?Gemma4LoRATargetPreset = null,
    target_preset: ?peft.TargetPreset = null,
    use_dora: bool = false,
    init_lora_weights: ?[]const u8 = null,
    eva_stats_path: ?[]const u8 = null,
    lora_ga_stats_path: ?[]const u8 = null,
    layer_name: ?[]const u8 = null,
    recursive_shared_block_size: ?usize = null,
    recursive_init_strategy: []const u8 = "average_residual_svd",
};

pub const BootstrapSummary = struct {
    artifact_family_version: []const u8,
    model_dir: []const u8,
    output_dir: []const u8,
    checkpoint_path: []const u8,
    adapter_checkpoint_path: []const u8,
    adapter_config_path: []const u8,
    base_model_name_or_path: []const u8,
    lora_rank: usize,
    lora_alpha: f32,
    target_modules: []const []const u8,
    target_preset: ?[]const u8 = null,
    use_dora: bool = false,
    init_lora_weights: ?[]const u8 = null,
    eva_stats_path: ?[]const u8 = null,
    lora_ga_stats_path: ?[]const u8 = null,
    resolved_tensors: []LoRATargetTensor,
};

pub const PeftExportSummary = struct {
    schema_version: []const u8,
    source_adapter_dir: []const u8,
    output_dir: []const u8,
    adapter_checkpoint_path: []const u8,
    adapter_config_path: []const u8,
    export_manifest_path: []const u8,
    tensor_key_format: []const u8,
    tensor_count: usize,
    adapter_checkpoint_size_bytes: u64,
    adapter_checkpoint_sha256: []const u8,
};

const PeftExportManifest = struct {
    schema_version: []const u8,
    status: []const u8,
    source_artifact_family_version: []const u8,
    source_tensor_key_format: []const u8,
    destination_tensor_key_format: []const u8,
    source_adapter_model_sha256: []const u8,
    destination_adapter_model_sha256: []const u8,
    destination_adapter_model_size_bytes: u64,
    adapter_config_sha256: []const u8,
    base_model_name_or_path: []const u8,
    base_model_sha256: []const u8,
    tokenizer_sha256: []const u8,
    chat_template_sha256: []const u8,
    target_preset: []const u8,
    tensor_count: usize,
};

pub const RecursiveCompressedBaseOptions = struct {
    metadata_file_name: []const u8 = "recursive_lora_base_config.json",
};

pub const RecursiveCompressedBaseSummary = struct {
    artifact_family_version: []const u8,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    output_dir: []const u8,
    source_checkpoint_path: []const u8,
    compressed_checkpoint_path: []const u8,
    metadata_path: []const u8,
    source_num_layers: usize,
    shared_block_size: usize,
    loop_count: usize,
    tensors_written: usize,
    tensors_skipped: usize,
    source_checkpoint_bytes: u64,
    compressed_checkpoint_bytes: u64,
    compression_ratio: f64,
};

pub const LoRATensorSummary = struct {
    base_tensor_name: []const u8,
    adapter_a_tensor_name: []const u8,
    adapter_b_tensor_name: []const u8,
    dora_magnitude_tensor_name: ?[]const u8 = null,
    module_name: []const u8,
    loop_index: ?usize = null,
    input_dim: usize,
    output_dim: usize,
    rank: usize,
    adapter_parameter_count: usize,
    dora_magnitude_parameter_count: usize = 0,
};

pub const LoRABundleInspectionSummary = struct {
    artifact_family_version: []const u8,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    base_checkpoint_path: []const u8,
    adapter_checkpoint_path: []const u8,
    adapter_config_path: ?[]const u8 = null,
    base_model_name_or_path: ?[]const u8 = null,
    lora_rank: ?usize = null,
    lora_alpha: ?f64 = null,
    target_module_count: usize = 0,
    target_modules: ?[]const []const u8 = null,
    target_preset: ?[]const u8 = null,
    use_dora: ?bool = null,
    init_lora_weights: ?[]const u8 = null,
    recursive_lora_enabled: bool = false,
    resolved_tensor_count: usize = 0,
    trainable_parameter_count: usize = 0,
    dora_magnitude_tensor_count: usize = 0,
    dora_magnitude_parameter_count: usize = 0,
    tensors: []LoRATensorSummary,
};

pub const MaterializeSummary = struct {
    artifact_family_version: []const u8,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    output_dir: []const u8,
    output_checkpoint_path: []const u8,
    merged_lora_tensor_count: usize,
    merged_dora_tensor_count: usize,
    copied_base_tensor_count: usize,
};

pub const LoadedLoRALayer = struct {
    base_tensor_name: []const u8,
    adapter_a_tensor_name: []const u8,
    adapter_b_tensor_name: []const u8,
    dora_magnitude_tensor_name: ?[]const u8 = null,
    module_name: []const u8,
    input_dim: usize,
    output_dim: usize,
    rank: usize,
    base_weight: []f32,
    adapter_a: []f32,
    adapter_b: []f32,
    dora_magnitude: ?[]f32 = null,
};

pub const LoadedLoRABundle = struct {
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    base_checkpoint_path: []const u8,
    adapter_checkpoint_path: []const u8,
    adapter_config_path: ?[]const u8 = null,
    base_model_name_or_path: ?[]const u8 = null,
    lora_rank: usize,
    lora_alpha: f32,
    use_dora: bool = false,
    recursive_lora_enabled: bool = false,
    target_modules: []const []const u8,
    layers: []LoadedLoRALayer,

    pub fn deinit(self: *LoadedLoRABundle) void {
        self.allocator.free(self.base_model_dir);
        self.allocator.free(self.adapter_model_dir);
        self.allocator.free(self.base_checkpoint_path);
        self.allocator.free(self.adapter_checkpoint_path);
        if (self.adapter_config_path) |p| self.allocator.free(p);
        if (self.base_model_name_or_path) |p| self.allocator.free(p);
        for (self.target_modules) |item| self.allocator.free(item);
        self.allocator.free(self.target_modules);
        for (self.layers) |layer| {
            self.allocator.free(layer.base_tensor_name);
            self.allocator.free(layer.adapter_a_tensor_name);
            self.allocator.free(layer.adapter_b_tensor_name);
            if (layer.dora_magnitude_tensor_name) |name| self.allocator.free(name);
            self.allocator.free(layer.module_name);
            self.allocator.free(layer.base_weight);
            self.allocator.free(layer.adapter_a);
            self.allocator.free(layer.adapter_b);
            if (layer.dora_magnitude) |magnitude| self.allocator.free(magnitude);
        }
        self.allocator.free(self.layers);
        self.* = undefined;
    }
};

const EvalOptions = struct {
    max_examples: usize,
    layer_name: ?[]const u8 = null,
};

const LoRAInitKind = enum {
    default,
    pissa,
    loftq_nf4,
    eva,
    lora_ga,
};

const PreparedInputsSummaryFile = struct {
    summary: PreparedInputsSummary,
};

// ---------------------------------------------------------------------------
// Checkpoint inspection
// ---------------------------------------------------------------------------

pub fn resolveArtifactPaths(allocator: std.mem.Allocator, input: []const u8) !ArtifactPaths {
    const stat = compat.cwd().statFile(compat.io(), input, .{}) catch return error.InputNotFound;
    const model_dir = if (stat.kind == .directory)
        try allocator.dupe(u8, input)
    else
        try allocator.dupe(u8, std.fs.path.dirname(input) orelse ".");
    errdefer allocator.free(model_dir);

    var paths = ArtifactPaths{
        .allocator = allocator,
        .model_dir = model_dir,
        .checkpoint_path = try optionalPathInDir(allocator, model_dir, checkpoint_file_name),
        .gguf_path = null,
        .adapter_checkpoint_path = try optionalPathInDir(allocator, model_dir, adapter_checkpoint_file_name),
        .config_path = try optionalPathInDir(allocator, model_dir, hf_config_file_name),
        .adapter_config_path = try optionalPathInDir(allocator, model_dir, adapter_config_file_name),
        .adapter_manifest_path = try optionalPathInDir(allocator, model_dir, adapter_manifest_file_name),
        .tokenizer_config_path = try optionalPathInDir(allocator, model_dir, tokenizer_config_file_name),
        .tokenizer_path = try optionalPathInDir(allocator, model_dir, tokenizer_file_name),
        .special_tokens_map_path = try optionalPathInDir(allocator, model_dir, special_tokens_map_file_name),
    };

    // Match ModelManifest's selected-artifact order: a canonical dense file
    // wins over a sharded index, which wins over an auto-detected GGUF.
    if (paths.checkpoint_path == null) {
        for (manifest_mod.safetensors_candidates) |candidate| {
            paths.checkpoint_path = try optionalPathInDir(allocator, model_dir, candidate);
            if (paths.checkpoint_path != null) break;
        }
    }
    if (paths.checkpoint_path == null) {
        for (manifest_mod.safetensors_index_candidates) |candidate| {
            paths.checkpoint_path = try optionalPathInDir(allocator, model_dir, candidate);
            if (paths.checkpoint_path != null) break;
        }
    }
    if (stat.kind == .file) {
        if (std.mem.eql(u8, std.fs.path.basename(input), checkpoint_file_name)) {
            if (paths.checkpoint_path) |p| allocator.free(p);
            paths.checkpoint_path = try allocator.dupe(u8, input);
        } else if (std.mem.endsWith(u8, input, ".safetensors.index.json")) {
            if (paths.checkpoint_path) |p| allocator.free(p);
            paths.checkpoint_path = try allocator.dupe(u8, input);
        } else if (std.mem.endsWith(u8, input, ".gguf")) {
            paths.gguf_path = try allocator.dupe(u8, input);
        } else if (std.mem.eql(u8, std.fs.path.basename(input), adapter_checkpoint_file_name)) {
            if (paths.adapter_checkpoint_path) |p| allocator.free(p);
            paths.adapter_checkpoint_path = try allocator.dupe(u8, input);
        }
    }
    if (paths.gguf_path == null) {
        paths.gguf_path = try findDecoderGgufPathInDir(allocator, model_dir);
    }

    return paths;
}

pub fn inspectCheckpoint(allocator: std.mem.Allocator, input: []const u8) !InspectionSummary {
    var paths = try resolveArtifactPaths(allocator, input);
    defer paths.deinit();

    const config_bytes = if (paths.config_path) |p| try c_file.readFile(allocator, p) else null;
    defer if (config_bytes) |b| allocator.free(b);
    const adapter_config_bytes = if (paths.adapter_config_path) |p| try c_file.readFile(allocator, p) else null;
    defer if (adapter_config_bytes) |b| allocator.free(b);
    const adapter_manifest_bytes = if (paths.adapter_manifest_path) |p| try c_file.readFile(allocator, p) else null;
    defer if (adapter_manifest_bytes) |b| allocator.free(b);
    const tokenizer_config_bytes = if (paths.tokenizer_config_path) |p| try c_file.readFile(allocator, p) else null;
    defer if (tokenizer_config_bytes) |b| allocator.free(b);

    var parsed_config = if (config_bytes) |b|
        try std.json.parseFromSlice(Config, allocator, b, .{ .ignore_unknown_fields = true })
    else
        null;
    defer if (parsed_config) |*p| p.deinit();

    var parsed_adapter = if (adapter_config_bytes) |b|
        try std.json.parseFromSlice(AdapterConfig, allocator, b, .{ .ignore_unknown_fields = true })
    else
        null;
    defer if (parsed_adapter) |*p| p.deinit();

    var parsed_adapter_manifest = if (adapter_manifest_bytes) |b|
        try std.json.parseFromSlice(AdapterManifest, allocator, b, .{ .ignore_unknown_fields = false })
    else
        null;
    defer if (parsed_adapter_manifest) |*p| p.deinit();

    var parsed_tokenizer = if (tokenizer_config_bytes) |b|
        try std.json.parseFromSlice(TokenizerConfig, allocator, b, .{ .ignore_unknown_fields = true })
    else
        null;
    defer if (parsed_tokenizer) |*p| p.deinit();

    const config = if (parsed_config) |*p| &p.value else null;
    const adapter_config = if (parsed_adapter) |*p| &p.value else null;
    const adapter_manifest = if (parsed_adapter_manifest) |*p| &p.value else null;
    if (adapter_manifest) |manifest| {
        if (!std.mem.eql(u8, manifest.schema_version, adapter_manifest_schema_v2) or
            !std.mem.eql(u8, manifest.status, "complete") or
            !std.mem.eql(u8, manifest.artifact_family_version, artifact_family_version) or
            !std.mem.eql(u8, manifest.tensor_key_format, adapter_tensor_key_format_v1))
        {
            return error.InvalidAdapterManifest;
        }
        try validateSha256Hex(manifest.base_model_sha256);
        try validateSha256Hex(manifest.tokenizer_sha256);
        try validateSha256Hex(manifest.chat_template_sha256);
        try validateSha256Hex(manifest.adapter_checkpoint_sha256);
        const adapter_checkpoint_path = paths.adapter_checkpoint_path orelse return error.MissingAdapterCheckpoint;
        var adapter_checkpoint = try c_file.MmapRegion.init(allocator, adapter_checkpoint_path);
        defer adapter_checkpoint.deinit();
        if (@as(u64, @intCast(adapter_checkpoint.data.len)) != manifest.adapter_checkpoint_size_bytes) {
            return error.AdapterCheckpointDigestMismatch;
        }
        const adapter_checkpoint_sha256 = try sha256HexAlloc(allocator, adapter_checkpoint.data);
        defer allocator.free(adapter_checkpoint_sha256);
        if (!std.mem.eql(u8, adapter_checkpoint_sha256, manifest.adapter_checkpoint_sha256)) {
            return error.AdapterCheckpointDigestMismatch;
        }
        if (adapter_config) |adapter_cfg| {
            const configured_initializer = try adapterInitializerName(adapter_cfg.init_lora_weights);
            if (adapter_cfg.r == null or adapter_cfg.r.? != manifest.rank or
                !adapterAlphaMatchesManifest(adapter_cfg.lora_alpha, manifest.alpha) or
                adapter_cfg.target_modules == null or
                !orderedStringSlicesEqual(adapter_cfg.target_modules.?, manifest.target_modules) or
                adapter_cfg.base_model_name_or_path == null or
                !std.mem.eql(u8, adapter_cfg.base_model_name_or_path.?, manifest.base_model_name_or_path) or
                (adapter_cfg.use_dora orelse false) != manifest.use_dora or
                (adapter_cfg.use_rslora orelse false) != manifest.use_rslora or
                !optionalStringsEqual(configured_initializer, manifest.initializer))
            {
                return error.AdapterManifestConfigMismatch;
            }
        }
    }
    const tokenizer_config = if (parsed_tokenizer) |*p| &p.value else null;
    const recursive_config = if (adapter_manifest) |manifest|
        manifest.recursive_lora
    else if (adapter_config) |ac|
        ac.recursive_lora
    else
        null;
    const text_config = if (config) |c| c.text_config else null;

    const variant: Variant = if (paths.checkpoint_path != null or paths.gguf_path != null) .merged else if (paths.adapter_checkpoint_path != null) .adapter_only else .incomplete;

    return .{
        .artifact_family_version = try allocator.dupe(u8, artifact_family_version),
        .variant = variant,
        .model_dir = try allocator.dupe(u8, paths.model_dir),
        .checkpoint_path = try dupeOptionalString(allocator, paths.checkpoint_path),
        .gguf_path = try dupeOptionalString(allocator, paths.gguf_path),
        .adapter_checkpoint_path = try dupeOptionalString(allocator, paths.adapter_checkpoint_path),
        .config_path = try dupeOptionalString(allocator, paths.config_path),
        .adapter_config_path = try dupeOptionalString(allocator, paths.adapter_config_path),
        .tokenizer_config_path = try dupeOptionalString(allocator, paths.tokenizer_config_path),
        .tokenizer_path = try dupeOptionalString(allocator, paths.tokenizer_path),
        .special_tokens_map_path = try dupeOptionalString(allocator, paths.special_tokens_map_path),
        .base_model_name_or_path = if (adapter_manifest) |manifest|
            try allocator.dupe(u8, manifest.base_model_name_or_path)
        else if (adapter_config) |ac|
            try dupeOptionalString(allocator, ac.base_model_name_or_path)
        else
            null,
        .base_model_sha256 = if (adapter_manifest) |manifest|
            try allocator.dupe(u8, manifest.base_model_sha256)
        else if (adapter_config) |ac|
            try dupeOptionalString(allocator, ac.antfly_base_model_sha256)
        else
            null,
        .tokenizer_sha256 = if (adapter_manifest) |manifest|
            try allocator.dupe(u8, manifest.tokenizer_sha256)
        else if (adapter_config) |ac|
            try dupeOptionalString(allocator, ac.antfly_tokenizer_sha256)
        else
            null,
        .chat_template_sha256 = if (adapter_manifest) |manifest|
            try allocator.dupe(u8, manifest.chat_template_sha256)
        else if (adapter_config) |ac|
            try dupeOptionalString(allocator, ac.antfly_chat_template_sha256)
        else
            null,
        .model_type = if (text_config) |tc|
            try dupeOptionalString(allocator, tc.model_type orelse if (config) |c| c.model_type else null)
        else if (config) |c|
            try dupeOptionalString(allocator, c.model_type)
        else
            null,
        .hidden_size = if (text_config) |tc| tc.hidden_size orelse if (config) |c| c.hidden_size else null else if (config) |c| c.hidden_size else null,
        .num_hidden_layers = if (text_config) |tc| tc.num_hidden_layers orelse if (config) |c| c.num_hidden_layers else null else if (config) |c| c.num_hidden_layers else null,
        .num_attention_heads = if (text_config) |tc| tc.num_attention_heads orelse if (config) |c| c.num_attention_heads else null else if (config) |c| c.num_attention_heads else null,
        .vocab_size = if (text_config) |tc| tc.vocab_size orelse if (config) |c| c.vocab_size else null else if (config) |c| c.vocab_size else null,
        .torch_dtype = if (text_config) |tc|
            try dupeOptionalString(allocator, tc.torch_dtype orelse tc.dtype orelse if (config) |c| c.torch_dtype orelse c.dtype else null)
        else if (config) |c|
            try dupeOptionalString(allocator, c.torch_dtype orelse c.dtype)
        else
            null,
        .tokenizer_class = if (tokenizer_config) |tc| try dupeOptionalString(allocator, tc.tokenizer_class) else null,
        .tokenizer_model_max_length = if (tokenizer_config) |tc| blk: {
            const v = tc.model_max_length orelse break :blk null;
            // HF uses a sentinel ~1e30 to mean "no limit"; treat anything above usize max as null
            if (v <= 0 or v > @as(f64, @floatFromInt(std.math.maxInt(usize)))) break :blk null;
            break :blk @intFromFloat(v);
        } else null,
        .max_position_embeddings = if (text_config) |tc| tc.max_position_embeddings orelse if (config) |c| c.max_position_embeddings else null else if (config) |c| c.max_position_embeddings else null,
        .lora_rank = if (adapter_config) |ac| ac.r else null,
        .lora_alpha = if (adapter_config) |ac| ac.lora_alpha else null,
        .peft_type = if (adapter_config) |ac| try dupeOptionalString(allocator, ac.peft_type) else null,
        .task_type = if (adapter_config) |ac| try dupeOptionalString(allocator, ac.task_type) else null,
        .inference_mode = if (adapter_config) |ac| ac.inference_mode else null,
        .target_module_count = if (adapter_config) |ac| if (ac.target_modules) |items| items.len else 0 else 0,
        .target_modules = if (adapter_config) |ac| try dupeOptionalStringSlice(allocator, ac.target_modules) else null,
        .target_preset = if (adapter_manifest) |manifest|
            try dupeOptionalString(allocator, manifest.target_preset)
        else if (adapter_config) |ac|
            try dupeOptionalString(allocator, ac.target_preset)
        else
            null,
        .use_dora = if (adapter_config) |ac| ac.use_dora else null,
        .use_rslora = if (adapter_config) |ac| ac.use_rslora else null,
        .lora_dropout = if (adapter_config) |ac| ac.lora_dropout else null,
        .bias = if (adapter_config) |ac| try dupeOptionalString(allocator, ac.bias) else null,
        .fan_in_fan_out = if (adapter_config) |ac| ac.fan_in_fan_out else null,
        .modules_to_save_count = if (adapter_config) |ac| if (ac.modules_to_save) |items| items.len else 0 else 0,
        .init_lora_weights = if (adapter_config) |ac|
            try dupeOptionalString(allocator, try adapterInitializerName(ac.init_lora_weights))
        else
            null,
        .recursive_lora_enabled = if (recursive_config) |rc| rc.enabled else false,
        .recursive_source_num_layers = if (recursive_config) |rc| if (rc.enabled) rc.source_num_layers else null else null,
        .recursive_shared_block_size = if (recursive_config) |rc| if (rc.enabled) rc.shared_block_size else null else null,
        .recursive_loop_count = if (recursive_config) |rc| if (rc.enabled) rc.loop_count else null else null,
        .recursive_init_strategy = if (recursive_config) |rc| if (rc.enabled) try allocator.dupe(u8, rc.init_strategy) else null else null,
        .has_merged_weights = paths.checkpoint_path != null or paths.gguf_path != null,
        .has_gguf_weights = paths.gguf_path != null,
        .has_adapter_weights = paths.adapter_checkpoint_path != null,
        .has_tokenizer = paths.tokenizer_path != null,
    };
}

pub fn freeInspectionSummary(allocator: std.mem.Allocator, summary: *InspectionSummary) void {
    allocator.free(summary.artifact_family_version);
    allocator.free(summary.model_dir);
    if (summary.checkpoint_path) |p| allocator.free(p);
    if (summary.gguf_path) |p| allocator.free(p);
    if (summary.adapter_checkpoint_path) |p| allocator.free(p);
    if (summary.config_path) |p| allocator.free(p);
    if (summary.adapter_config_path) |p| allocator.free(p);
    if (summary.tokenizer_config_path) |p| allocator.free(p);
    if (summary.tokenizer_path) |p| allocator.free(p);
    if (summary.special_tokens_map_path) |p| allocator.free(p);
    if (summary.base_model_name_or_path) |p| allocator.free(p);
    if (summary.base_model_sha256) |p| allocator.free(p);
    if (summary.tokenizer_sha256) |p| allocator.free(p);
    if (summary.chat_template_sha256) |p| allocator.free(p);
    if (summary.model_type) |p| allocator.free(p);
    if (summary.torch_dtype) |p| allocator.free(p);
    if (summary.tokenizer_class) |p| allocator.free(p);
    if (summary.peft_type) |p| allocator.free(p);
    if (summary.task_type) |p| allocator.free(p);
    if (summary.target_preset) |p| allocator.free(p);
    if (summary.bias) |p| allocator.free(p);
    if (summary.init_lora_weights) |p| allocator.free(p);
    if (summary.target_modules) |modules| {
        for (modules) |item| allocator.free(item);
        allocator.free(modules);
    }
    if (summary.recursive_init_strategy) |p| allocator.free(p);
    summary.* = undefined;
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

pub fn bootstrapLoRABundle(
    allocator: std.mem.Allocator,
    model_input: []const u8,
    out_dir: []const u8,
    options: BootstrapOptions,
) !BootstrapSummary {
    if (options.rank == 0 or options.rank > std.math.maxInt(u32)) return error.InvalidLoRARank;
    if (!std.math.isFinite(options.alpha) or options.alpha <= 0) return error.InvalidLoRAAlpha;
    try validateLoRAInitializerBaseCompatibility(options.init_lora_weights);
    var inspect = try inspectCheckpoint(allocator, model_input);
    defer freeInspectionSummary(allocator, &inspect);

    const checkpoint_path = inspect.checkpoint_path orelse inspect.gguf_path orelse return error.MissingMergedCheckpoint;
    const recursive_config = try makeRecursiveConfig(inspect, options);
    try recursive_lora.validate(recursive_config);

    const target_selection = try resolveBootstrapTargetSelection(options);
    const all_resolved_tensors = try inferLoRATargetTensorsForModelInput(allocator, inspect.model_dir, checkpoint_path, target_selection);
    defer freeLoRATargetTensors(allocator, all_resolved_tensors);
    var filtered = std.ArrayListUnmanaged(LoRATargetTensor).empty;
    errdefer {
        for (filtered.items) |item| {
            allocator.free(item.tensor_name);
            allocator.free(item.module_name);
        }
        filtered.deinit(allocator);
    }
    for (all_resolved_tensors) |item| {
        if (!layerMatchesScope(item.tensor_name, options.layer_name)) continue;
        if (recursive_config.enabled) {
            const layer_idx = parseGemma4LayerIndex(item.tensor_name) orelse continue;
            if (layer_idx >= recursive_config.shared_block_size) continue;
        }
        try filtered.append(allocator, .{
            .tensor_name = try allocator.dupe(u8, item.tensor_name),
            .module_name = try allocator.dupe(u8, item.module_name),
            .input_dim = item.input_dim,
            .output_dim = item.output_dim,
        });
    }
    try validateExplicitTargetSelection(target_selection, filtered.items);
    // A preset label means that the persisted target inventory is the exact
    // model-resolved preset. Layer-scoped and recursive adapters intentionally
    // persist only their exact modules; labeling those subsets as the full
    // preset would make provenance and oracle comparisons misleading.
    const persisted_target_preset = if (options.layer_name == null and !recursive_config.enabled)
        targetSelectionPresetName(target_selection)
    else
        null;
    const resolved_tensors = try filtered.toOwnedSlice(allocator);
    errdefer freeLoRATargetTensors(allocator, resolved_tensors);
    if (resolved_tensors.len == 0) return error.NoLoRATargetTensorsResolved;

    // Persist exact module paths from the selected checkpoint schema. This
    // prevents substring presets from later selecting multimodal encoders and
    // records shared-KV omissions exactly as they existed at bootstrap time.
    const resolved_target_modules = try resolvedTargetModulePaths(allocator, resolved_tensors);
    errdefer {
        for (resolved_target_modules) |item| allocator.free(item);
        allocator.free(resolved_target_modules);
    }

    var publication = try artifact_publication.ImmutableDirectoryPublication.init(allocator, compat.io(), out_dir);
    defer publication.deinit();
    try publication.createStaging();

    const staging_adapter_checkpoint_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, adapter_checkpoint_file_name });
    defer allocator.free(staging_adapter_checkpoint_path);
    const staging_adapter_config_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, adapter_config_file_name });
    defer allocator.free(staging_adapter_config_path);
    const staging_adapter_manifest_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, adapter_manifest_file_name });
    defer allocator.free(staging_adapter_manifest_path);

    const base_model_name_or_path = if (options.base_model_name_or_path) |v|
        try allocator.dupe(u8, v)
    else if (inspect.base_model_name_or_path) |v|
        try allocator.dupe(u8, v)
    else
        try allocator.dupe(u8, inspect.model_dir);
    errdefer allocator.free(base_model_name_or_path);

    var provenance = try fingerprintGemma4Model(allocator, inspect.model_dir);
    defer provenance.deinit(allocator);

    try writeBootstrapAdapterCheckpointAtomic(allocator, staging_adapter_checkpoint_path, checkpoint_path, resolved_tensors, options.rank, options.use_dora, options.init_lora_weights, options.eva_stats_path, options.lora_ga_stats_path, recursive_config);
    const adapter_write_options = AdapterConfigWriteOptions{
        .base_model_name_or_path = base_model_name_or_path,
        .base_model_sha256 = provenance.base_model_sha256,
        .tokenizer_sha256 = provenance.tokenizer_sha256,
        .chat_template_sha256 = provenance.chat_template_sha256,
        .rank = options.rank,
        .alpha = options.alpha,
        .target_modules = resolved_target_modules,
        .target_preset = persisted_target_preset,
        .use_dora = options.use_dora,
        .init_lora_weights = options.init_lora_weights,
        .recursive_lora = recursive_config,
    };
    try writeAdapterConfigJson(allocator, staging_adapter_config_path, adapter_write_options);
    try writeAdapterManifestJson(allocator, staging_adapter_manifest_path, adapter_write_options);
    try copySupportingArtifactIfPresent(allocator, inspect.tokenizer_config_path, publication.staging_dir, tokenizer_config_file_name);
    try copySupportingArtifactIfPresent(allocator, inspect.tokenizer_path, publication.staging_dir, tokenizer_file_name);
    try copySupportingArtifactIfPresent(allocator, inspect.special_tokens_map_path, publication.staging_dir, special_tokens_map_file_name);

    // Validate the completed staged artifact before it becomes visible.
    try validateLoRAAdapterInventory(allocator, publication.staging_dir);

    const published_adapter_checkpoint_path = try std.fs.path.join(allocator, &.{ out_dir, adapter_checkpoint_file_name });
    errdefer allocator.free(published_adapter_checkpoint_path);
    const published_adapter_config_path = try std.fs.path.join(allocator, &.{ out_dir, adapter_config_file_name });
    errdefer allocator.free(published_adapter_config_path);
    const summary_artifact_family = try allocator.dupe(u8, artifact_family_version);
    errdefer allocator.free(summary_artifact_family);
    const summary_model_dir = try allocator.dupe(u8, inspect.model_dir);
    errdefer allocator.free(summary_model_dir);
    const summary_output_dir = try allocator.dupe(u8, out_dir);
    errdefer allocator.free(summary_output_dir);
    const summary_checkpoint_path = try allocator.dupe(u8, checkpoint_path);
    errdefer allocator.free(summary_checkpoint_path);
    const summary_target_preset = try dupeOptionalString(allocator, persisted_target_preset);
    errdefer if (summary_target_preset) |value| allocator.free(value);
    const summary_init = try dupeOptionalString(allocator, options.init_lora_weights);
    errdefer if (summary_init) |value| allocator.free(value);
    const summary_eva = try dupeOptionalString(allocator, options.eva_stats_path);
    errdefer if (summary_eva) |value| allocator.free(value);
    const summary_lora_ga = try dupeOptionalString(allocator, options.lora_ga_stats_path);
    errdefer if (summary_lora_ga) |value| allocator.free(value);

    try publication.publish();

    return .{
        .artifact_family_version = summary_artifact_family,
        .model_dir = summary_model_dir,
        .output_dir = summary_output_dir,
        .checkpoint_path = summary_checkpoint_path,
        .adapter_checkpoint_path = published_adapter_checkpoint_path,
        .adapter_config_path = published_adapter_config_path,
        .base_model_name_or_path = base_model_name_or_path,
        .lora_rank = options.rank,
        .lora_alpha = options.alpha,
        .target_modules = resolved_target_modules,
        .target_preset = summary_target_preset,
        .use_dora = options.use_dora,
        .init_lora_weights = summary_init,
        .eva_stats_path = summary_eva,
        .lora_ga_stats_path = summary_lora_ga,
        .resolved_tensors = resolved_tensors,
    };
}

pub fn freeBootstrapSummary(allocator: std.mem.Allocator, summary: *BootstrapSummary) void {
    allocator.free(summary.artifact_family_version);
    allocator.free(summary.model_dir);
    allocator.free(summary.output_dir);
    allocator.free(summary.checkpoint_path);
    allocator.free(summary.adapter_checkpoint_path);
    allocator.free(summary.adapter_config_path);
    allocator.free(summary.base_model_name_or_path);
    for (summary.target_modules) |item| allocator.free(item);
    allocator.free(summary.target_modules);
    if (summary.target_preset) |value| allocator.free(value);
    if (summary.init_lora_weights) |value| allocator.free(value);
    if (summary.eva_stats_path) |value| allocator.free(value);
    if (summary.lora_ga_stats_path) |value| allocator.free(value);
    freeLoRATargetTensors(allocator, summary.resolved_tensors);
    summary.* = undefined;
}

/// Export one validated Antfly Gemma 4 LoRA artifact into the tensor-key
/// layout consumed directly by stock Hugging Face PEFT. The source artifact is
/// never mutated, tensor payload bytes are preserved exactly, and the complete
/// destination directory is published with a no-replace rename.
pub fn exportPeftAdapter(
    allocator: std.mem.Allocator,
    model_input: []const u8,
    adapter_input: []const u8,
    out_dir: []const u8,
) !PeftExportSummary {
    var adapter_inspect = try inspectCheckpoint(allocator, adapter_input);
    defer freeInspectionSummary(allocator, &adapter_inspect);
    const source_checkpoint_path = adapter_inspect.adapter_checkpoint_path orelse return error.MissingAdapterCheckpoint;
    const source_config_path = adapter_inspect.adapter_config_path orelse return error.MissingAdapterConfig;
    const base_model_name_or_path = adapter_inspect.base_model_name_or_path orelse return error.AdapterProvenanceRequired;
    const base_model_sha256 = adapter_inspect.base_model_sha256 orelse return error.AdapterProvenanceRequired;
    const tokenizer_sha256 = adapter_inspect.tokenizer_sha256 orelse return error.AdapterProvenanceRequired;
    const chat_template_sha256 = adapter_inspect.chat_template_sha256 orelse return error.AdapterProvenanceRequired;
    const target_preset = adapter_inspect.target_preset orelse return error.Gemma4PeftExportTargetPresetRequired;
    if (!std.mem.eql(u8, target_preset, "peft-qv") and !std.mem.eql(u8, target_preset, "text-all-linear")) {
        return error.Gemma4PeftExportTargetPresetRequired;
    }
    if (adapter_inspect.use_dora orelse false) return error.Gemma4PeftExportDoRANotSupported;
    if (adapter_inspect.use_rslora orelse false) return error.Gemma4PeftExportRSLoRANotSupported;
    if (adapter_inspect.recursive_lora_enabled) return error.Gemma4PeftExportRecursiveLoRANotSupported;

    var actual_provenance = try fingerprintGemma4Model(allocator, model_input);
    defer actual_provenance.deinit(allocator);
    try validateAdapterModelProvenance(adapter_inspect, actual_provenance);

    var bundle = try inspectLoRABundle(allocator, model_input, adapter_input);
    defer freeLoRABundleInspectionSummary(allocator, &bundle);
    try validateLoRABundleInspection(bundle);
    if (bundle.recursive_lora_enabled) return error.Gemma4PeftExportRecursiveLoRANotSupported;
    if (bundle.use_dora orelse false) return error.Gemma4PeftExportDoRANotSupported;

    var source_reader = try safetensors.MMapReader.openFileAbsolute(allocator, source_checkpoint_path);
    defer source_reader.deinit();
    const source_names = try source_reader.header.tensorNames(allocator);
    defer allocator.free(source_names);
    if (source_names.len != bundle.resolved_tensor_count * 2) return error.AdapterTargetInventoryMismatch;

    var translated = try allocator.alloc(WriteTensorRaw, source_names.len);
    defer allocator.free(translated);
    var translated_names = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (translated_names.items) |name| allocator.free(name);
        translated_names.deinit(allocator);
    }

    for (source_names, 0..) |source_name, idx| {
        const parsed = parseLoRAAdapterTensorName(source_name) orelse return error.UnsupportedPeftExportTensor;
        if (parsed.loop_index != null) return error.Gemma4PeftExportRecursiveLoRANotSupported;
        const module_path = tensorModulePath(parsed.base_tensor_base_name) orelse return error.InvalidLoRATargetTensorName;
        const role = switch (parsed.kind) {
            .a => "lora_A",
            .b => "lora_B",
        };
        const destination_name = try std.fmt.allocPrint(
            allocator,
            "base_model.model.{s}.{s}.weight",
            .{ module_path, role },
        );
        errdefer allocator.free(destination_name);
        try translated_names.append(allocator, destination_name);

        const meta = source_reader.header.tensors.get(source_name) orelse return error.TensorNotFound;
        if (meta.dtype != .f32) return error.UnsupportedAdapterTensorEncoding;
        const absolute_start = std.math.add(u64, source_reader.data_offset, meta.data_start) catch return error.DataOutOfBounds;
        const absolute_end = std.math.add(u64, source_reader.data_offset, meta.data_end) catch return error.DataOutOfBounds;
        if (absolute_start > absolute_end or absolute_end > source_reader.file_bytes.len) return error.DataOutOfBounds;
        translated[idx] = .{
            .name = destination_name,
            .dtype = meta.dtype,
            .shape = meta.shape,
            .raw_bytes = source_reader.file_bytes[@intCast(absolute_start)..@intCast(absolute_end)],
        };
    }
    std.mem.sort(WriteTensorRaw, translated, {}, struct {
        fn lessThan(_: void, lhs: WriteTensorRaw, rhs: WriteTensorRaw) bool {
            return std.mem.lessThan(u8, lhs.name, rhs.name);
        }
    }.lessThan);
    for (translated[1..], 1..) |tensor, idx| {
        if (std.mem.eql(u8, translated[idx - 1].name, tensor.name)) return error.PeftExportTensorKeyCollision;
    }

    const source_config = try c_file.readFile(allocator, source_config_path);
    defer allocator.free(source_config);
    var source_checkpoint = try c_file.MmapRegion.init(allocator, source_checkpoint_path);
    defer source_checkpoint.deinit();
    const source_checkpoint_sha256 = try sha256HexAlloc(allocator, source_checkpoint.data);
    defer allocator.free(source_checkpoint_sha256);

    var publication = try artifact_publication.ImmutableDirectoryPublication.init(allocator, compat.io(), out_dir);
    defer publication.deinit();
    try publication.createStaging();

    const staging_checkpoint_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, adapter_checkpoint_file_name });
    defer allocator.free(staging_checkpoint_path);
    const staging_config_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, adapter_config_file_name });
    defer allocator.free(staging_config_path);
    const staging_manifest_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, peft_export_manifest_file_name });
    defer allocator.free(staging_manifest_path);

    try writeHeaderAndRawTensors(allocator, staging_checkpoint_path, translated);
    try safetensors.validateArtifactSet(allocator, staging_checkpoint_path, null);
    try validatePeftExportCheckpoint(allocator, staging_checkpoint_path, translated);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = staging_config_path, .data = source_config });

    var destination_checkpoint = try c_file.MmapRegion.init(allocator, staging_checkpoint_path);
    defer destination_checkpoint.deinit();
    const destination_checkpoint_sha256 = try sha256HexAlloc(allocator, destination_checkpoint.data);
    errdefer allocator.free(destination_checkpoint_sha256);
    const adapter_config_sha256 = try sha256HexAlloc(allocator, source_config);
    defer allocator.free(adapter_config_sha256);
    try writePeftExportManifestJson(allocator, staging_manifest_path, .{
        .schema_version = peft_export_manifest_schema_v1,
        .status = "complete",
        .source_artifact_family_version = adapter_inspect.artifact_family_version,
        .source_tensor_key_format = adapter_tensor_key_format_v1,
        .destination_tensor_key_format = stock_peft_tensor_key_format_v1,
        .source_adapter_model_sha256 = source_checkpoint_sha256,
        .destination_adapter_model_sha256 = destination_checkpoint_sha256,
        .destination_adapter_model_size_bytes = @intCast(destination_checkpoint.data.len),
        .adapter_config_sha256 = adapter_config_sha256,
        .base_model_name_or_path = base_model_name_or_path,
        .base_model_sha256 = base_model_sha256,
        .tokenizer_sha256 = tokenizer_sha256,
        .chat_template_sha256 = chat_template_sha256,
        .target_preset = target_preset,
        .tensor_count = translated.len,
    });

    const summary_source_adapter_dir = try allocator.dupe(u8, adapter_inspect.model_dir);
    errdefer allocator.free(summary_source_adapter_dir);
    const summary_output_dir = try allocator.dupe(u8, out_dir);
    errdefer allocator.free(summary_output_dir);
    const published_checkpoint_path = try std.fs.path.join(allocator, &.{ out_dir, adapter_checkpoint_file_name });
    errdefer allocator.free(published_checkpoint_path);
    const published_config_path = try std.fs.path.join(allocator, &.{ out_dir, adapter_config_file_name });
    errdefer allocator.free(published_config_path);
    const published_manifest_path = try std.fs.path.join(allocator, &.{ out_dir, peft_export_manifest_file_name });
    errdefer allocator.free(published_manifest_path);

    try publication.publish();
    return .{
        .schema_version = peft_export_manifest_schema_v1,
        .source_adapter_dir = summary_source_adapter_dir,
        .output_dir = summary_output_dir,
        .adapter_checkpoint_path = published_checkpoint_path,
        .adapter_config_path = published_config_path,
        .export_manifest_path = published_manifest_path,
        .tensor_key_format = stock_peft_tensor_key_format_v1,
        .tensor_count = translated.len,
        .adapter_checkpoint_size_bytes = @intCast(destination_checkpoint.data.len),
        .adapter_checkpoint_sha256 = destination_checkpoint_sha256,
    };
}

pub fn freePeftExportSummary(allocator: std.mem.Allocator, summary: *PeftExportSummary) void {
    allocator.free(summary.source_adapter_dir);
    allocator.free(summary.output_dir);
    allocator.free(summary.adapter_checkpoint_path);
    allocator.free(summary.adapter_config_path);
    allocator.free(summary.export_manifest_path);
    allocator.free(summary.adapter_checkpoint_sha256);
    summary.* = undefined;
}

fn validatePeftExportCheckpoint(
    allocator: std.mem.Allocator,
    checkpoint_path: []const u8,
    expected: []const WriteTensorRaw,
) !void {
    var reader = try safetensors.MMapReader.openFileAbsolute(allocator, checkpoint_path);
    defer reader.deinit();
    if (reader.header.tensors.count() != expected.len) return error.PeftExportTensorInventoryMismatch;
    for (expected) |tensor| {
        if (!std.mem.startsWith(u8, tensor.name, "base_model.model.")) return error.InvalidPeftExportTensorName;
        const meta = reader.header.tensors.get(tensor.name) orelse return error.PeftExportTensorInventoryMismatch;
        if (meta.dtype != tensor.dtype or !std.mem.eql(i64, meta.shape, tensor.shape)) {
            return error.PeftExportTensorMetadataMismatch;
        }
        const absolute_start = std.math.add(u64, reader.data_offset, meta.data_start) catch return error.DataOutOfBounds;
        const absolute_end = std.math.add(u64, reader.data_offset, meta.data_end) catch return error.DataOutOfBounds;
        if (absolute_start > absolute_end or absolute_end > reader.file_bytes.len) return error.DataOutOfBounds;
        const actual = reader.file_bytes[@intCast(absolute_start)..@intCast(absolute_end)];
        if (!std.mem.eql(u8, actual, tensor.raw_bytes)) return error.PeftExportTensorPayloadMismatch;
    }
}

fn writePeftExportManifestJson(
    allocator: std.mem.Allocator,
    path: []const u8,
    manifest: PeftExportManifest,
) !void {
    var buffer: std.Io.Writer.Allocating = .init(allocator);
    defer buffer.deinit();
    try std.json.Stringify.value(manifest, .{ .whitespace = .indent_2 }, &buffer.writer);
    try buffer.writer.writeByte('\n');
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = path, .data = buffer.written() });
}

fn makeRecursiveConfig(inspect: InspectionSummary, options: BootstrapOptions) !recursive_lora.Config {
    const shared_block_size = options.recursive_shared_block_size orelse return .{};
    const source_num_layers = inspect.num_hidden_layers orelse return error.InvalidRecursiveLoRAConfig;
    const loop_count = try recursive_lora.inferLoopCount(source_num_layers, shared_block_size);
    return .{
        .enabled = true,
        .source_num_layers = source_num_layers,
        .shared_block_size = shared_block_size,
        .loop_count = loop_count,
        .init_strategy = options.recursive_init_strategy,
    };
}

pub fn materializeRecursiveCompressedBase(
    allocator: std.mem.Allocator,
    base_model_input: []const u8,
    adapter_model_input: []const u8,
    out_dir: []const u8,
    options: RecursiveCompressedBaseOptions,
) !RecursiveCompressedBaseSummary {
    var base_inspect = try inspectCheckpoint(allocator, base_model_input);
    defer freeInspectionSummary(allocator, &base_inspect);
    var adapter_inspect = try inspectCheckpoint(allocator, adapter_model_input);
    defer freeInspectionSummary(allocator, &adapter_inspect);

    if (!adapter_inspect.recursive_lora_enabled) return error.AdapterIsNotRecursiveLoRA;
    const source_num_layers = adapter_inspect.recursive_source_num_layers orelse return error.InvalidRecursiveLoRAConfig;
    const shared_block_size = adapter_inspect.recursive_shared_block_size orelse return error.InvalidRecursiveLoRAConfig;
    const loop_count = adapter_inspect.recursive_loop_count orelse return error.InvalidRecursiveLoRAConfig;
    try recursive_lora.validate(.{
        .enabled = true,
        .source_num_layers = source_num_layers,
        .shared_block_size = shared_block_size,
        .loop_count = loop_count,
        .init_strategy = adapter_inspect.recursive_init_strategy orelse "average_residual_svd",
    });
    try validateLoRAAdapterInventory(allocator, adapter_model_input);

    const checkpoint_path = base_inspect.checkpoint_path orelse return error.MissingMergedCheckpoint;
    if (base_inspect.gguf_path != null) return error.UnsupportedRecursiveCompressedBaseSource;

    var publication = try artifact_publication.ImmutableDirectoryPublication.init(allocator, compat.io(), out_dir);
    defer publication.deinit();

    var access = try openTensorAccessForFile(allocator, checkpoint_path);
    defer access.deinit();
    const names = try access.listNames(allocator);
    defer allocator.free(names);
    std.mem.sort([]const u8, names, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);

    var raw_tensors = std.ArrayListUnmanaged(WriteTensorRaw).empty;
    defer raw_tensors.deinit(allocator);
    var owned_records = std.ArrayListUnmanaged(tensor_access.Record).empty;
    defer {
        for (owned_records.items) |*record| record.deinit();
        owned_records.deinit(allocator);
    }

    var tensors_skipped: usize = 0;
    for (names) |name| {
        if (!keepTensorInRecursiveCompressedBase(name, shared_block_size)) {
            tensors_skipped += 1;
            continue;
        }
        var record = try access.getRecord(allocator, name);
        errdefer record.deinit();
        try raw_tensors.append(allocator, .{
            .name = record.descriptor.name,
            .dtype = denseRecordDType(record.descriptor.encoding) orelse return error.UnsupportedTensorEncoding,
            .shape = record.descriptor.shape,
            .raw_bytes = record.raw_bytes,
        });
        try owned_records.append(allocator, record);
    }
    if (raw_tensors.items.len == 0) return error.NoTensorsSelected;

    try publication.createStaging();
    const staging_checkpoint_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, checkpoint_file_name });
    defer allocator.free(staging_checkpoint_path);
    const staging_metadata_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, options.metadata_file_name });
    defer allocator.free(staging_metadata_path);
    const published_checkpoint_path = try std.fs.path.join(allocator, &.{ out_dir, checkpoint_file_name });
    errdefer allocator.free(published_checkpoint_path);

    try writeHeaderAndRawTensors(allocator, staging_checkpoint_path, raw_tensors.items);
    try copyCompressedBaseSupportFiles(allocator, base_inspect.model_dir, publication.staging_dir);
    const source_checkpoint_bytes = try c_file.fileSize(allocator, checkpoint_path);
    const compressed_checkpoint_bytes = try c_file.fileSize(allocator, staging_checkpoint_path);
    const compression_ratio = if (source_checkpoint_bytes == 0)
        0
    else
        @as(f64, @floatFromInt(compressed_checkpoint_bytes)) / @as(f64, @floatFromInt(source_checkpoint_bytes));

    try writeRecursiveCompressedBaseMetadata(
        allocator,
        staging_metadata_path,
        base_inspect.model_dir,
        adapter_inspect.model_dir,
        checkpoint_path,
        published_checkpoint_path,
        source_num_layers,
        shared_block_size,
        loop_count,
        raw_tensors.items.len,
        tensors_skipped,
        source_checkpoint_bytes,
        compressed_checkpoint_bytes,
        compression_ratio,
    );

    const summary_artifact_family = try allocator.dupe(u8, artifact_family_version);
    errdefer allocator.free(summary_artifact_family);
    const summary_base_model_dir = try allocator.dupe(u8, base_inspect.model_dir);
    errdefer allocator.free(summary_base_model_dir);
    const summary_adapter_model_dir = try allocator.dupe(u8, adapter_inspect.model_dir);
    errdefer allocator.free(summary_adapter_model_dir);
    const summary_output_dir = try allocator.dupe(u8, out_dir);
    errdefer allocator.free(summary_output_dir);
    const summary_source_checkpoint_path = try allocator.dupe(u8, checkpoint_path);
    errdefer allocator.free(summary_source_checkpoint_path);
    const published_metadata_path = try std.fs.path.join(allocator, &.{ out_dir, options.metadata_file_name });
    errdefer allocator.free(published_metadata_path);
    try publication.publish();

    return .{
        .artifact_family_version = summary_artifact_family,
        .base_model_dir = summary_base_model_dir,
        .adapter_model_dir = summary_adapter_model_dir,
        .output_dir = summary_output_dir,
        .source_checkpoint_path = summary_source_checkpoint_path,
        .compressed_checkpoint_path = published_checkpoint_path,
        .metadata_path = published_metadata_path,
        .source_num_layers = source_num_layers,
        .shared_block_size = shared_block_size,
        .loop_count = loop_count,
        .tensors_written = raw_tensors.items.len,
        .tensors_skipped = tensors_skipped,
        .source_checkpoint_bytes = source_checkpoint_bytes,
        .compressed_checkpoint_bytes = compressed_checkpoint_bytes,
        .compression_ratio = compression_ratio,
    };
}

pub fn freeRecursiveCompressedBaseSummary(allocator: std.mem.Allocator, summary: *RecursiveCompressedBaseSummary) void {
    allocator.free(summary.artifact_family_version);
    allocator.free(summary.base_model_dir);
    allocator.free(summary.adapter_model_dir);
    allocator.free(summary.output_dir);
    allocator.free(summary.source_checkpoint_path);
    allocator.free(summary.compressed_checkpoint_path);
    allocator.free(summary.metadata_path);
    summary.* = undefined;
}

// ---------------------------------------------------------------------------
// Load / save LoRA bundle
// ---------------------------------------------------------------------------

/// Validate the adapter checkpoint as a closed inventory. Every configured
/// target must resolve to adapter tensors, every tensor must be configured,
/// A/B tensors must be paired, and DoRA magnitude tensors must match
/// `use_dora`. This deliberately rejects permissive suffix-only inventories:
/// production artifacts persist exact module paths.
pub fn validateLoRAAdapterInventory(
    allocator: std.mem.Allocator,
    adapter_model_input: []const u8,
) !void {
    var inspected = try inspectCheckpoint(allocator, adapter_model_input);
    defer freeInspectionSummary(allocator, &inspected);
    const checkpoint_path = inspected.adapter_checkpoint_path orelse return error.MissingAdapterCheckpoint;
    const configured_targets = inspected.target_modules orelse return error.MissingAdapterTargetInventory;
    if (configured_targets.len == 0) return error.NoLoRATargetTensorsResolved;

    for (configured_targets, 0..) |target, idx| {
        if (target.len == 0 or tensorModulePath(target) != null) return error.InvalidAdapterTargetModule;
        for (configured_targets[0..idx]) |prior| {
            if (std.mem.eql(u8, prior, target)) return error.DuplicateAdapterTargetModule;
        }
    }

    var access = try openTensorAccessForFile(allocator, checkpoint_path);
    defer access.deinit();
    const names = try access.listNames(allocator);
    defer allocator.free(names);

    var resolved_targets = std.ArrayListUnmanaged([]const u8).empty;
    defer resolved_targets.deinit(allocator);
    var adapter_a_count: usize = 0;
    var adapter_b_count: usize = 0;

    for (names) |name| {
        if (parseLoRAAdapterTensorName(name)) |parsed| {
            const counterpart = if (parsed.loop_index) |loop_idx|
                switch (parsed.kind) {
                    .a => try recursive_lora.formatLoopAdapterTensorName(allocator, parsed.base_tensor_base_name, loop_idx, .b),
                    .b => try recursive_lora.formatLoopAdapterTensorName(allocator, parsed.base_tensor_base_name, loop_idx, .a),
                }
            else
                try std.fmt.allocPrint(
                    allocator,
                    "{s}.{s}.weight",
                    .{ parsed.base_tensor_base_name, if (parsed.kind == .a) "lora_B" else "lora_A" },
                );
            defer allocator.free(counterpart);
            if (!stringSliceContains(names, counterpart)) return error.MissingAdapterPair;

            if (parsed.kind == .a) adapter_a_count += 1 else adapter_b_count += 1;
            const module_path = tensorModulePath(parsed.base_tensor_base_name) orelse return error.InvalidLoRATargetTensorName;
            if (!stringSliceContains(resolved_targets.items, module_path)) {
                try resolved_targets.append(allocator, module_path);
            }
            continue;
        }
        if (parseDoRAMagnitudeTensorName(name) != null) continue;
        return error.UnexpectedAdapterTensor;
    }

    if (adapter_a_count == 0 or resolved_targets.items.len == 0) return error.NoLoRATargetTensorsResolved;
    if (adapter_a_count != adapter_b_count) return error.MissingAdapterPair;

    const use_dora = inspected.use_dora orelse false;
    for (resolved_targets.items) |module_path| {
        const base_tensor_name = try std.fmt.allocPrint(allocator, "{s}.weight", .{module_path});
        defer allocator.free(base_tensor_name);
        const magnitude_name = try doraMagnitudeTensorName(allocator, base_tensor_name);
        defer allocator.free(magnitude_name);
        if (stringSliceContains(names, magnitude_name) != use_dora) return error.AdapterDoRAConfigMismatch;
    }
    for (names) |name| {
        const base_tensor_name = parseDoRAMagnitudeTensorName(name) orelse continue;
        const module_path = tensorModulePath(base_tensor_name) orelse return error.InvalidLoRATargetTensorName;
        if (!stringSliceContains(resolved_targets.items, module_path)) return error.UnexpectedAdapterDoRATensor;
    }

    if (configured_targets.len != resolved_targets.items.len) return error.AdapterTargetInventoryMismatch;
    for (configured_targets) |configured| {
        if (!stringSliceContains(resolved_targets.items, configured)) return error.AdapterTargetInventoryMismatch;
    }
    for (resolved_targets.items) |resolved| {
        if (!stringSliceContains(configured_targets, resolved)) return error.AdapterTargetInventoryMismatch;
    }
    try validateAdapterTargetPreset(allocator, inspected);
}

fn validateAdapterTargetPreset(allocator: std.mem.Allocator, inspected: InspectionSummary) !void {
    const preset_name = inspected.target_preset orelse return;
    const selection: BootstrapTargetSelection = if (parseGemma4LoRATargetPreset(preset_name)) |preset|
        .{ .gemma4 = preset }
    else if (peft.parseTargetPreset(preset_name)) |preset|
        .{ .legacy = preset }
    else
        return error.InvalidAdapterTargetPreset;

    const configured_targets = inspected.target_modules orelse return error.MissingAdapterTargetInventory;
    for (configured_targets) |module_path| {
        const tensor_name = try std.fmt.allocPrint(allocator, "{s}.weight", .{module_path});
        defer allocator.free(tensor_name);
        if (!targetMatchesSelection(tensor_name, selection)) return error.AdapterTargetPresetMismatch;
    }
}

pub fn inspectLoRABundle(
    allocator: std.mem.Allocator,
    base_model_input: []const u8,
    adapter_model_input: []const u8,
) !LoRABundleInspectionSummary {
    var base_inspect = try inspectCheckpoint(allocator, base_model_input);
    defer freeInspectionSummary(allocator, &base_inspect);
    var adapter_inspect = try inspectCheckpoint(allocator, adapter_model_input);
    defer freeInspectionSummary(allocator, &adapter_inspect);
    try validateLoRAAdapterInventory(allocator, adapter_model_input);

    const base_checkpoint_path = base_inspect.checkpoint_path orelse base_inspect.gguf_path orelse return error.MissingMergedCheckpoint;
    const adapter_checkpoint_path = adapter_inspect.adapter_checkpoint_path orelse return error.MissingAdapterCheckpoint;

    var base_access = try openTensorAccessForFile(allocator, base_checkpoint_path);
    defer base_access.deinit();
    var adapter_access = try openTensorAccessForFile(allocator, adapter_checkpoint_path);
    defer adapter_access.deinit();
    const base_names = try base_access.listNames(allocator);
    defer allocator.free(base_names);
    if (adapter_inspect.target_preset) |preset| {
        try validateAdapterTargetPresetAgainstBase(
            allocator,
            base_inspect.model_dir,
            preset,
            adapter_inspect.target_modules orelse return error.MissingAdapterTargetInventory,
            base_names,
        );
    }

    var tensors: std.ArrayListUnmanaged(LoRATensorSummary) = .empty;
    errdefer {
        for (tensors.items) |*item| freeLoRATensorSummary(allocator, item);
        tensors.deinit(allocator);
    }

    const adapter_names = try adapter_access.listNames(allocator);
    defer allocator.free(adapter_names);
    for (adapter_names) |adapter_a_name| {
        const parsed = parseLoRAAdapterTensorName(adapter_a_name) orelse continue;
        if (parsed.kind != .a) continue;

        const adapter_b_name = if (parsed.loop_index) |loop_idx|
            try recursive_lora.formatLoopAdapterTensorName(allocator, parsed.base_tensor_base_name, loop_idx, .b)
        else
            try std.fmt.allocPrint(allocator, "{s}.lora_B.weight", .{parsed.base_tensor_base_name});
        defer allocator.free(adapter_b_name);
        const base_tensor_name = parsed.base_tensor_base_name;

        var adapter_a = try adapter_access.getRecord(allocator, adapter_a_name);
        defer adapter_a.deinit();
        var adapter_b = adapter_access.getRecord(allocator, adapter_b_name) catch return error.MissingAdapterPair;
        defer adapter_b.deinit();
        try validateFiniteAdapterTensor(adapter_a);
        try validateFiniteAdapterTensor(adapter_b);
        const source_base_tensor_name = sourceTensorNameForCanonicalAdapterBase(base_names, base_tensor_name) orelse return error.MissingBaseTensorForAdapter;
        var base = base_access.getRecord(allocator, source_base_tensor_name) catch return error.MissingBaseTensorForAdapter;
        defer base.deinit();
        if (adapter_a.descriptor.shape.len != 2 or adapter_b.descriptor.shape.len != 2 or base.descriptor.shape.len != 2) return error.InvalidAdapterTensorShape;
        if (adapter_a.descriptor.shape[1] != base.descriptor.shape[1]) return error.AdapterInputDimMismatch;
        if (adapter_b.descriptor.shape[0] != base.descriptor.shape[0]) return error.AdapterOutputDimMismatch;
        if (adapter_a.descriptor.shape[0] != adapter_b.descriptor.shape[1]) return error.AdapterRankMismatch;

        const maybe_dora_name = try doraMagnitudeTensorName(allocator, base_tensor_name);
        defer allocator.free(maybe_dora_name);
        var dora_name_for_summary: ?[]const u8 = null;
        var dora_parameter_count: usize = 0;
        if (adapter_access.getRecord(allocator, maybe_dora_name)) |record| {
            var magnitude = record;
            defer magnitude.deinit();
            try validateFiniteAdapterTensor(magnitude);
            if (magnitude.descriptor.shape.len != 1) return error.InvalidAdapterTensorShape;
            if (magnitude.descriptor.shape[0] != base.descriptor.shape[0]) return error.AdapterOutputDimMismatch;
            dora_name_for_summary = try allocator.dupe(u8, maybe_dora_name);
            dora_parameter_count = @intCast(magnitude.descriptor.shape[0]);
        } else |err| switch (err) {
            error.TensorNotFound => {},
            else => return err,
        }

        try tensors.append(allocator, .{
            .base_tensor_name = try allocator.dupe(u8, base_tensor_name),
            .adapter_a_tensor_name = try allocator.dupe(u8, adapter_a_name),
            .adapter_b_tensor_name = try allocator.dupe(u8, adapter_b_name),
            .dora_magnitude_tensor_name = dora_name_for_summary,
            .module_name = try allocator.dupe(u8, parsed.module_name),
            .loop_index = parsed.loop_index,
            .input_dim = @intCast(base.descriptor.shape[1]),
            .output_dim = @intCast(base.descriptor.shape[0]),
            .rank = @intCast(adapter_a.descriptor.shape[0]),
            .adapter_parameter_count = @as(usize, @intCast(adapter_a.descriptor.shape[0])) *
                @as(usize, @intCast(adapter_a.descriptor.shape[1])) +
                @as(usize, @intCast(adapter_b.descriptor.shape[0])) *
                    @as(usize, @intCast(adapter_b.descriptor.shape[1])) +
                dora_parameter_count,
            .dora_magnitude_parameter_count = dora_parameter_count,
        });
    }

    std.mem.sort(LoRATensorSummary, tensors.items, {}, struct {
        fn lt(_: void, a: LoRATensorSummary, b: LoRATensorSummary) bool {
            return std.mem.lessThan(u8, a.base_tensor_name, b.base_tensor_name);
        }
    }.lt);

    var trainable_parameter_count: usize = 0;
    var dora_magnitude_tensor_count: usize = 0;
    var dora_magnitude_parameter_count: usize = 0;
    for (tensors.items) |item| {
        trainable_parameter_count += item.adapter_parameter_count;
        if (item.dora_magnitude_tensor_name != null) dora_magnitude_tensor_count += 1;
        dora_magnitude_parameter_count += item.dora_magnitude_parameter_count;
    }

    return .{
        .artifact_family_version = try allocator.dupe(u8, artifact_family_version),
        .base_model_dir = try allocator.dupe(u8, base_inspect.model_dir),
        .adapter_model_dir = try allocator.dupe(u8, adapter_inspect.model_dir),
        .base_checkpoint_path = try allocator.dupe(u8, base_checkpoint_path),
        .adapter_checkpoint_path = try allocator.dupe(u8, adapter_checkpoint_path),
        .adapter_config_path = try dupeOptionalString(allocator, adapter_inspect.adapter_config_path),
        .base_model_name_or_path = try dupeOptionalString(allocator, adapter_inspect.base_model_name_or_path),
        .lora_rank = adapter_inspect.lora_rank,
        .lora_alpha = adapter_inspect.lora_alpha,
        .target_module_count = adapter_inspect.target_module_count,
        .target_modules = try dupeOptionalStringSlice(allocator, adapter_inspect.target_modules),
        .target_preset = try dupeOptionalString(allocator, adapter_inspect.target_preset),
        .use_dora = adapter_inspect.use_dora,
        .init_lora_weights = try dupeOptionalString(allocator, adapter_inspect.init_lora_weights),
        .recursive_lora_enabled = adapter_inspect.recursive_lora_enabled,
        .resolved_tensor_count = tensors.items.len,
        .trainable_parameter_count = trainable_parameter_count,
        .dora_magnitude_tensor_count = dora_magnitude_tensor_count,
        .dora_magnitude_parameter_count = dora_magnitude_parameter_count,
        .tensors = try tensors.toOwnedSlice(allocator),
    };
}

fn validateAdapterTargetPresetAgainstBase(
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    preset_name: []const u8,
    configured_targets: []const []const u8,
    base_tensor_names: []const []const u8,
) !void {
    const selection: BootstrapTargetSelection = if (parseGemma4LoRATargetPreset(preset_name)) |preset|
        .{ .gemma4 = preset }
    else if (peft.parseTargetPreset(preset_name)) |preset|
        .{ .legacy = preset }
    else
        return error.InvalidAdapterTargetPreset;
    const config = try session_factory.loadGptConfigMetadataFromModelDir(allocator, base_model_dir);

    var expected = std.StringHashMapUnmanaged(void).empty;
    defer expected.deinit(allocator);
    var owned = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (owned.items) |item| allocator.free(item);
        owned.deinit(allocator);
    }

    for (base_tensor_names) |tensor_name| {
        if (!targetMatchesSelection(tensor_name, selection)) continue;
        if (!gemma4TargetParticipatesInGraph(tensor_name, config)) continue;
        const module_path = normalizedTargetModulePath(allocator, tensor_name) catch |err| switch (err) {
            error.InvalidLoRATargetTensorName => continue,
            else => return err,
        };
        if (expected.contains(module_path)) {
            allocator.free(module_path);
            continue;
        }
        owned.append(allocator, module_path) catch |err| {
            allocator.free(module_path);
            return err;
        };
        try expected.put(allocator, module_path, {});
    }

    if (configured_targets.len != expected.count()) return error.AdapterTargetPresetMismatch;
    for (configured_targets) |module_path| {
        if (!expected.contains(module_path)) return error.AdapterTargetPresetMismatch;
    }
}

fn validateFiniteAdapterTensor(record: tensor_access.Record) !void {
    const bytes = record.raw_bytes;
    switch (record.descriptor.encoding) {
        .gguf => return error.UnsupportedAdapterTensorEncoding,
        .dense => |dtype| switch (dtype) {
            .f32 => {
                if (bytes.len % @sizeOf(f32) != 0) return error.InvalidAdapterTensorShape;
                var offset: usize = 0;
                while (offset < bytes.len) : (offset += @sizeOf(f32)) {
                    const value: f32 = @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
                    if (!std.math.isFinite(value)) return error.NonFiniteAdapterTensor;
                }
            },
            else => return error.UnsupportedAdapterTensorEncoding,
        },
    }
}

pub fn freeLoRABundleInspectionSummary(allocator: std.mem.Allocator, summary: *LoRABundleInspectionSummary) void {
    allocator.free(summary.artifact_family_version);
    allocator.free(summary.base_model_dir);
    allocator.free(summary.adapter_model_dir);
    allocator.free(summary.base_checkpoint_path);
    allocator.free(summary.adapter_checkpoint_path);
    if (summary.adapter_config_path) |p| allocator.free(p);
    if (summary.base_model_name_or_path) |p| allocator.free(p);
    if (summary.target_preset) |p| allocator.free(p);
    if (summary.init_lora_weights) |p| allocator.free(p);
    if (summary.target_modules) |modules| {
        for (modules) |item| allocator.free(item);
        allocator.free(modules);
    }
    for (summary.tensors) |*item| freeLoRATensorSummary(allocator, item);
    allocator.free(summary.tensors);
    summary.* = undefined;
}

/// Validate the semantic adapter contract after `inspectLoRABundle` has
/// resolved every adapter tensor against the selected base checkpoint.
///
/// This is intentionally separate from `validateLoRAAdapterInventory`: the
/// inventory gate proves that the checkpoint is closed and exactly matches
/// `target_modules`, while inspection proves A/B/base dimensions.  Public
/// train, eval, and adapter-validation entry points should run both before
/// constructing a backend.
pub fn validateLoRABundleInspection(inspected: LoRABundleInspectionSummary) !void {
    try validateLoRAInitializerBaseCompatibility(inspected.init_lora_weights);

    const rank = inspected.lora_rank orelse return error.MissingAdapterConfig;
    const alpha = inspected.lora_alpha orelse return error.MissingAdapterConfig;
    if (rank == 0 or rank > std.math.maxInt(u32)) return error.InvalidLoRARank;
    if (!std.math.isFinite(alpha) or alpha <= 0 or alpha > std.math.floatMax(f32)) return error.InvalidLoRAAlpha;
    if (inspected.tensors.len == 0 or inspected.resolved_tensor_count != inspected.tensors.len) {
        return error.NoLoRATargetTensorsResolved;
    }

    const use_dora = inspected.use_dora orelse false;
    for (inspected.tensors) |tensor| {
        if (tensor.rank != rank) return error.AdapterConfigRankMismatch;
        if ((tensor.dora_magnitude_tensor_name != null) != use_dora) return error.AdapterDoRAConfigMismatch;
    }
}

fn validateGenericLoRABundleInspection(inspected: LoRABundleInspectionSummary) !void {
    try validateLoRABundleInspection(inspected);
    if (inspected.recursive_lora_enabled) return error.RecursiveLoRANotSupportedByGenericLifecycle;
    for (inspected.tensors) |tensor| {
        if (tensor.loop_index != null) return error.RecursiveLoRANotSupportedByGenericLifecycle;
    }
}

fn validateGenericLoadedLoRABundle(bundle: *const LoadedLoRABundle) !void {
    if (bundle.recursive_lora_enabled) return error.RecursiveLoRANotSupportedByGenericLifecycle;
    if (bundle.lora_rank == 0 or bundle.lora_rank > std.math.maxInt(u32)) return error.InvalidLoRARank;
    if (!std.math.isFinite(bundle.lora_alpha) or bundle.lora_alpha <= 0) return error.InvalidLoRAAlpha;

    if (bundle.target_modules.len == 0 or bundle.layers.len == 0) return error.NoLoRATargetTensorsResolved;
    var resolved_target_count: usize = 0;
    for (bundle.layers, 0..) |layer, layer_idx| {
        const parsed = parseLoRAAdapterTensorName(layer.adapter_a_tensor_name) orelse return error.InvalidLoRATargetTensorName;
        if (parsed.loop_index != null) return error.RecursiveLoRANotSupportedByGenericLifecycle;
        if (layer.rank != bundle.lora_rank) return error.AdapterConfigRankMismatch;
        const has_magnitude_name = layer.dora_magnitude_tensor_name != null;
        const has_magnitude = layer.dora_magnitude != null;
        if (has_magnitude_name != has_magnitude or has_magnitude != bundle.use_dora) return error.AdapterDoRAConfigMismatch;

        const module_path = tensorModulePath(layer.base_tensor_name) orelse return error.InvalidLoRATargetTensorName;
        if (!stringSliceContains(bundle.target_modules, module_path)) return error.AdapterTargetInventoryMismatch;
        for (bundle.layers[0..layer_idx]) |prior| {
            if (std.mem.eql(u8, prior.base_tensor_name, layer.base_tensor_name)) return error.DuplicateAdapterTargetTensor;
        }
        resolved_target_count += 1;
    }
    if (resolved_target_count != bundle.target_modules.len) return error.AdapterTargetInventoryMismatch;
    for (bundle.target_modules, 0..) |target, target_idx| {
        if (target.len == 0 or tensorModulePath(target) != null) return error.InvalidAdapterTargetModule;
        for (bundle.target_modules[0..target_idx]) |prior| {
            if (std.mem.eql(u8, prior, target)) return error.DuplicateAdapterTargetModule;
        }
        var found = false;
        for (bundle.layers) |layer| {
            const module_path = tensorModulePath(layer.base_tensor_name) orelse return error.InvalidLoRATargetTensorName;
            if (std.mem.eql(u8, target, module_path)) {
                found = true;
                break;
            }
        }
        if (!found) return error.AdapterTargetInventoryMismatch;
    }
}

pub fn loadLoRABundle(
    allocator: std.mem.Allocator,
    base_model_input: []const u8,
    adapter_model_input: []const u8,
) !LoadedLoRABundle {
    return loadLoRABundleScoped(allocator, base_model_input, adapter_model_input, null);
}

pub fn loadLoRABundleScoped(
    allocator: std.mem.Allocator,
    base_model_input: []const u8,
    adapter_model_input: []const u8,
    layer_name: ?[]const u8,
) !LoadedLoRABundle {
    var inspected = try inspectLoRABundle(allocator, base_model_input, adapter_model_input);
    defer freeLoRABundleInspectionSummary(allocator, &inspected);

    try validateGenericLoRABundleInspection(inspected);

    var scoped_tensor_count: usize = 0;
    for (inspected.tensors) |ts| {
        if (!layerMatchesScope(ts.base_tensor_name, layer_name)) continue;
        scoped_tensor_count += 1;
    }
    const layers = try allocator.alloc(LoadedLoRALayer, scoped_tensor_count);
    var loaded_count: usize = 0;
    errdefer {
        for (layers[0..loaded_count]) |layer| {
            allocator.free(layer.base_tensor_name);
            allocator.free(layer.adapter_a_tensor_name);
            allocator.free(layer.adapter_b_tensor_name);
            if (layer.dora_magnitude_tensor_name) |name| allocator.free(name);
            allocator.free(layer.module_name);
            allocator.free(layer.base_weight);
            allocator.free(layer.adapter_a);
            allocator.free(layer.adapter_b);
            if (layer.dora_magnitude) |magnitude| allocator.free(magnitude);
        }
        allocator.free(layers);
    }

    var base_access = try openTensorAccessForFile(allocator, inspected.base_checkpoint_path);
    defer base_access.deinit();
    var adapter_access = try openTensorAccessForFile(allocator, inspected.adapter_checkpoint_path);
    defer adapter_access.deinit();
    const base_names = try base_access.listNames(allocator);
    defer allocator.free(base_names);

    for (inspected.tensors) |ts| {
        if (!layerMatchesScope(ts.base_tensor_name, layer_name)) continue;
        var a_tensor = try loadTensorAsF32(allocator, adapter_access, ts.adapter_a_tensor_name);
        defer a_tensor.deinit();
        var b_tensor = try loadTensorAsF32(allocator, adapter_access, ts.adapter_b_tensor_name);
        defer b_tensor.deinit();
        const source_base_tensor_name = sourceTensorNameForCanonicalAdapterBase(base_names, ts.base_tensor_name) orelse return error.MissingBaseTensorForAdapter;
        var base_tensor = try loadTensorAsF32(allocator, base_access, source_base_tensor_name);
        defer base_tensor.deinit();
        if (a_tensor.shape.len != 2 or b_tensor.shape.len != 2 or base_tensor.shape.len != 2) {
            return error.InvalidAdapterTensorShape;
        }

        const adapter_a = try allocator.alloc(f32, ts.input_dim * ts.rank);
        errdefer allocator.free(adapter_a);
        transpose2DF32(adapter_a, a_tensor.asFloat32(), ts.rank, ts.input_dim);

        const adapter_b = try allocator.alloc(f32, ts.rank * ts.output_dim);
        errdefer allocator.free(adapter_b);
        transpose2DF32(adapter_b, b_tensor.asFloat32(), ts.output_dim, ts.rank);

        const base_weight = try allocator.alloc(f32, ts.input_dim * ts.output_dim);
        errdefer allocator.free(base_weight);
        transpose2DF32(base_weight, base_tensor.asFloat32(), ts.output_dim, ts.input_dim);

        var dora_magnitude_name: ?[]const u8 = null;
        var dora_magnitude: ?[]f32 = null;
        if (ts.dora_magnitude_tensor_name) |name| {
            var magnitude_tensor = try loadTensorAsF32(allocator, adapter_access, name);
            defer magnitude_tensor.deinit();
            if (magnitude_tensor.shape.len != 1 or magnitude_tensor.shape[0] != @as(i64, @intCast(ts.output_dim))) return error.InvalidAdapterTensorShape;
            const magnitude = try allocator.dupe(f32, magnitude_tensor.asFloat32());
            errdefer allocator.free(magnitude);
            dora_magnitude_name = try allocator.dupe(u8, name);
            errdefer if (dora_magnitude_name) |owned| allocator.free(owned);
            dora_magnitude = magnitude;
        }

        layers[loaded_count] = .{
            .base_tensor_name = try allocator.dupe(u8, ts.base_tensor_name),
            .adapter_a_tensor_name = try allocator.dupe(u8, ts.adapter_a_tensor_name),
            .adapter_b_tensor_name = try allocator.dupe(u8, ts.adapter_b_tensor_name),
            .dora_magnitude_tensor_name = dora_magnitude_name,
            .module_name = try allocator.dupe(u8, ts.module_name),
            .input_dim = ts.input_dim,
            .output_dim = ts.output_dim,
            .rank = ts.rank,
            .base_weight = base_weight,
            .adapter_a = adapter_a,
            .adapter_b = adapter_b,
            .dora_magnitude = dora_magnitude,
        };
        loaded_count += 1;
    }

    return .{
        .allocator = allocator,
        .base_model_dir = try allocator.dupe(u8, inspected.base_model_dir),
        .adapter_model_dir = try allocator.dupe(u8, inspected.adapter_model_dir),
        .base_checkpoint_path = try allocator.dupe(u8, inspected.base_checkpoint_path),
        .adapter_checkpoint_path = try allocator.dupe(u8, inspected.adapter_checkpoint_path),
        .adapter_config_path = try dupeOptionalString(allocator, inspected.adapter_config_path),
        .base_model_name_or_path = try dupeOptionalString(allocator, inspected.base_model_name_or_path),
        .lora_rank = inspected.lora_rank.?,
        .lora_alpha = @floatCast(inspected.lora_alpha.?),
        .use_dora = inspected.use_dora orelse false,
        .recursive_lora_enabled = inspected.recursive_lora_enabled,
        .target_modules = if (layer_name != null)
            try resolvedTargetModulesForLoadedLayers(allocator, layers)
        else if (inspected.target_modules) |items|
            try dupeStringSlice(allocator, items)
        else
            unreachable,
        .layers = layers,
    };
}

fn resolvedTargetModulesForLoadedLayers(
    allocator: std.mem.Allocator,
    layers: []const LoadedLoRALayer,
) ![][]const u8 {
    const targets = try allocator.alloc([]const u8, layers.len);
    errdefer allocator.free(targets);
    var built: usize = 0;
    errdefer for (targets[0..built]) |target| allocator.free(target);
    for (layers, 0..) |layer, idx| {
        const module_path = tensorModulePath(layer.base_tensor_name) orelse return error.InvalidLoRATargetTensorName;
        targets[idx] = try allocator.dupe(u8, module_path);
        built += 1;
    }
    std.mem.sort([]const u8, targets, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    return targets;
}

pub fn saveLoRABundle(bundle: *const LoadedLoRABundle, out_dir: []const u8) !void {
    const allocator = bundle.allocator;
    try validateGenericLoadedLoRABundle(bundle);
    var publication = try artifact_publication.ImmutableDirectoryPublication.init(allocator, compat.io(), out_dir);
    defer publication.deinit();
    try publication.createStaging();
    try writeLoRABundleContents(bundle, publication.staging_dir);
    try validateLoRAAdapterInventory(allocator, publication.staging_dir);
    try publication.publish();
}

/// Write a validated bundle into a caller-owned private staging directory.
/// Whole-run publication uses this so the adapter and reports are exposed by
/// one final rename.
pub fn saveLoRABundleToStaging(bundle: *const LoadedLoRABundle, staging_dir: []const u8) !void {
    try validateGenericLoadedLoRABundle(bundle);
    try writeLoRABundleContents(bundle, staging_dir);
    try validateLoRAAdapterInventory(bundle.allocator, staging_dir);
}

fn writeLoRABundleContents(bundle: *const LoadedLoRABundle, out_dir: []const u8) !void {
    const allocator = bundle.allocator;
    try compat.cwd().createDirPath(compat.io(), out_dir);

    const adapter_checkpoint_path = try std.fs.path.join(allocator, &.{ out_dir, adapter_checkpoint_file_name });
    defer allocator.free(adapter_checkpoint_path);

    var tensors = try allocator.alloc(WriteTensorF32, bundle.layers.len * 3);
    defer allocator.free(tensors);
    var owned_names: std.ArrayListUnmanaged([]const u8) = .empty;
    defer {
        for (owned_names.items) |item| allocator.free(item);
        owned_names.deinit(allocator);
    }
    var owned_shapes: std.ArrayListUnmanaged([]const usize) = .empty;
    defer {
        for (owned_shapes.items) |item| allocator.free(item);
        owned_shapes.deinit(allocator);
    }
    var owned_data: std.ArrayListUnmanaged([]f32) = .empty;
    defer {
        for (owned_data.items) |item| allocator.free(item);
        owned_data.deinit(allocator);
    }

    var tensor_idx: usize = 0;
    for (bundle.layers) |layer| {
        // Transpose back to HuggingFace layout [rank, input_dim] and [output_dim, rank].
        const a_data = try allocator.alloc(f32, layer.adapter_a.len);
        transpose2DF32(a_data, layer.adapter_a, layer.input_dim, layer.rank);
        try owned_data.append(allocator, a_data);

        const b_data = try allocator.alloc(f32, layer.adapter_b.len);
        transpose2DF32(b_data, layer.adapter_b, layer.rank, layer.output_dim);
        try owned_data.append(allocator, b_data);

        const a_name = try allocator.dupe(u8, layer.adapter_a_tensor_name);
        const b_name = try allocator.dupe(u8, layer.adapter_b_tensor_name);
        try owned_names.append(allocator, a_name);
        try owned_names.append(allocator, b_name);

        const a_shape = try allocator.dupe(usize, &.{ layer.rank, layer.input_dim });
        const b_shape = try allocator.dupe(usize, &.{ layer.output_dim, layer.rank });
        try owned_shapes.append(allocator, a_shape);
        try owned_shapes.append(allocator, b_shape);

        tensors[tensor_idx] = .{ .name = a_name, .shape = a_shape, .data = a_data };
        tensor_idx += 1;
        tensors[tensor_idx] = .{ .name = b_name, .shape = b_shape, .data = b_data };
        tensor_idx += 1;

        if (layer.dora_magnitude) |magnitude| {
            const magnitude_data = try allocator.dupe(f32, magnitude);
            try owned_data.append(allocator, magnitude_data);
            const magnitude_name = if (layer.dora_magnitude_tensor_name) |name|
                try allocator.dupe(u8, name)
            else
                try doraMagnitudeTensorName(allocator, layer.base_tensor_name);
            try owned_names.append(allocator, magnitude_name);
            const magnitude_shape = try allocator.dupe(usize, &.{layer.output_dim});
            try owned_shapes.append(allocator, magnitude_shape);
            tensors[tensor_idx] = .{ .name = magnitude_name, .shape = magnitude_shape, .data = magnitude_data };
            tensor_idx += 1;
        }
    }

    try writeHeaderAndTensorsF32(allocator, adapter_checkpoint_path, tensors[0..tensor_idx]);

    // Keep the PEFT adapter config portable and place Antfly provenance in a
    // strict sidecar. Recompute the identity from the selected base rather
    // than trusting metadata carried by the input adapter.
    const adapter_config_path = try std.fs.path.join(allocator, &.{ out_dir, adapter_config_file_name });
    defer allocator.free(adapter_config_path);
    const adapter_manifest_path = try std.fs.path.join(allocator, &.{ out_dir, adapter_manifest_file_name });
    defer allocator.free(adapter_manifest_path);
    const base_name = bundle.base_model_name_or_path orelse bundle.base_model_dir;
    var provenance = try fingerprintGemma4Model(allocator, bundle.base_model_dir);
    defer provenance.deinit(allocator);
    const write_options = AdapterConfigWriteOptions{
        .base_model_name_or_path = base_name,
        .base_model_sha256 = provenance.base_model_sha256,
        .tokenizer_sha256 = provenance.tokenizer_sha256,
        .chat_template_sha256 = provenance.chat_template_sha256,
        .rank = bundle.lora_rank,
        .alpha = bundle.lora_alpha,
        .target_modules = bundle.target_modules,
        .use_dora = bundle.use_dora,
    };
    try writeAdapterConfigJson(allocator, adapter_config_path, write_options);
    try writeAdapterManifestJson(allocator, adapter_manifest_path, write_options);
}

pub fn materializeMergedModel(
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    out_dir: []const u8,
) !MaterializeSummary {
    var base_paths = try resolveArtifactPaths(allocator, base_model_dir);
    defer base_paths.deinit();
    const base_checkpoint_path = base_paths.checkpoint_path orelse base_paths.gguf_path orelse return error.MissingMergedCheckpoint;
    const source_checkpoint_bytes = try c_file.fileSize(allocator, base_checkpoint_path);
    try validateLegacyMaterializationSource(base_checkpoint_path, source_checkpoint_bytes);

    var bundle = try loadLoRABundle(allocator, base_model_dir, adapter_model_dir);
    defer bundle.deinit();

    var publication = try artifact_publication.ImmutableDirectoryPublication.init(allocator, compat.io(), out_dir);
    defer publication.deinit();
    var base_access = try openTensorAccessForFile(allocator, base_checkpoint_path);
    defer base_access.deinit();
    const base_names = try base_access.listNames(allocator);
    defer allocator.free(base_names);

    var merged = std.StringArrayHashMapUnmanaged(Tensor){};
    defer {
        var it = merged.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        merged.deinit(allocator);
    }

    var merged_dora_tensor_count: usize = 0;
    for (bundle.layers) |layer| {
        const merged_weight = try allocator.alloc(f32, layer.base_weight.len);
        defer allocator.free(merged_weight);
        const base_matrix = lora.Matrix{ .rows = layer.input_dim, .cols = layer.output_dim, .data = layer.base_weight };
        const adapter_a = lora.Matrix{ .rows = layer.input_dim, .cols = layer.rank, .data = layer.adapter_a };
        const adapter_b = lora.Matrix{ .rows = layer.rank, .cols = layer.output_dim, .data = layer.adapter_b };
        if (layer.dora_magnitude) |magnitude| {
            lora.doraMergeInto(.{
                .base = base_matrix,
                .adapter_a = adapter_a,
                .adapter_b = adapter_b,
                .magnitude = magnitude,
                .alpha = bundle.lora_alpha,
            }, merged_weight);
            merged_dora_tensor_count += 1;
        } else {
            lora.mergeInto(base_matrix, adapter_a, adapter_b, bundle.lora_alpha, merged_weight);
        }

        const hf_weight = try allocator.alloc(f32, merged_weight.len);
        defer allocator.free(hf_weight);
        transpose2DF32(hf_weight, merged_weight, layer.input_dim, layer.output_dim);

        const source_base_tensor_name = sourceTensorNameForCanonicalAdapterBase(base_names, layer.base_tensor_name) orelse return error.MissingBaseTensorForAdapter;
        const shape = [_]i64{ @as(i64, @intCast(layer.output_dim)), @as(i64, @intCast(layer.input_dim)) };
        const tensor = try Tensor.initFloat32(allocator, source_base_tensor_name, &shape, hf_weight);
        try merged.put(allocator, try allocator.dupe(u8, source_base_tensor_name), tensor);
    }

    try publication.createStaging();
    const staging_checkpoint_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, checkpoint_file_name });
    defer allocator.free(staging_checkpoint_path);
    const bytes = try buildMergedSafetensorsFile(allocator, base_access, base_names, &merged);
    defer allocator.free(bytes);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = staging_checkpoint_path, .data = bytes });

    try copySupportingArtifactIfPresent(allocator, base_paths.config_path, publication.staging_dir, hf_config_file_name);
    try copySupportingArtifactIfPresent(allocator, base_paths.tokenizer_config_path, publication.staging_dir, tokenizer_config_file_name);
    try copySupportingArtifactIfPresent(allocator, base_paths.tokenizer_path, publication.staging_dir, tokenizer_file_name);
    try copySupportingArtifactIfPresent(allocator, base_paths.special_tokens_map_path, publication.staging_dir, special_tokens_map_file_name);

    var adapter_paths = try resolveArtifactPaths(allocator, adapter_model_dir);
    defer adapter_paths.deinit();
    try copySupportingArtifactIfPresent(allocator, adapter_paths.tokenizer_config_path, publication.staging_dir, tokenizer_config_file_name);
    try copySupportingArtifactIfPresent(allocator, adapter_paths.tokenizer_path, publication.staging_dir, tokenizer_file_name);
    try copySupportingArtifactIfPresent(allocator, adapter_paths.special_tokens_map_path, publication.staging_dir, special_tokens_map_file_name);

    var copied_base_tensor_count: usize = 0;
    for (base_names) |name| {
        if (!merged.contains(name)) copied_base_tensor_count += 1;
    }

    const summary_artifact_family = try allocator.dupe(u8, artifact_family_version);
    errdefer allocator.free(summary_artifact_family);
    const summary_base_model_dir = try allocator.dupe(u8, bundle.base_model_dir);
    errdefer allocator.free(summary_base_model_dir);
    const summary_adapter_model_dir = try allocator.dupe(u8, bundle.adapter_model_dir);
    errdefer allocator.free(summary_adapter_model_dir);
    const summary_output_dir = try allocator.dupe(u8, out_dir);
    errdefer allocator.free(summary_output_dir);
    const published_checkpoint_path = try std.fs.path.join(allocator, &.{ out_dir, checkpoint_file_name });
    errdefer allocator.free(published_checkpoint_path);
    try publication.publish();

    return .{
        .artifact_family_version = summary_artifact_family,
        .base_model_dir = summary_base_model_dir,
        .adapter_model_dir = summary_adapter_model_dir,
        .output_dir = summary_output_dir,
        .output_checkpoint_path = published_checkpoint_path,
        .merged_lora_tensor_count = bundle.layers.len,
        .merged_dora_tensor_count = merged_dora_tensor_count,
        .copied_base_tensor_count = copied_base_tensor_count,
    };
}

fn validateLegacyMaterializationSource(checkpoint_path: []const u8, checkpoint_bytes: u64) !void {
    if (std.mem.endsWith(u8, checkpoint_path, ".gguf")) return error.Gemma4GgufMaterializationUnsupported;
    if (std.mem.endsWith(u8, checkpoint_path, ".safetensors.index.json")) return error.Gemma4MaterializationRequiresStreaming;
    if (!std.mem.endsWith(u8, checkpoint_path, ".safetensors")) return error.UnsupportedMaterializationSource;
    if (checkpoint_bytes > legacy_materialize_max_checkpoint_bytes) return error.Gemma4MaterializationRequiresStreaming;
}

pub fn freeMaterializeSummary(allocator: std.mem.Allocator, summary: *MaterializeSummary) void {
    allocator.free(summary.artifact_family_version);
    allocator.free(summary.base_model_dir);
    allocator.free(summary.adapter_model_dir);
    allocator.free(summary.output_dir);
    allocator.free(summary.output_checkpoint_path);
    summary.* = undefined;
}

const LoadedGemmaTokenizer = union(enum) {
    hf: *hf_tokenizer.HfTokenizer,
    sp: *sentencepiece.Processor,

    fn deinit(self: *LoadedGemmaTokenizer, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .hf => |tok| tok.deinitSelf(),
            .sp => |sp| {
                sp.deinit();
                allocator.destroy(sp);
            },
        }
        self.* = undefined;
    }

    fn tokenizer(self: *const LoadedGemmaTokenizer) tokenizer_mod.Tokenizer {
        return switch (self.*) {
            .hf => |tok| tok.tokenizer(),
            .sp => |sp| sp.tokenizer(),
        };
    }
};

fn loadGemmaTokenizerForModelDir(allocator: std.mem.Allocator, model_dir: []const u8) !LoadedGemmaTokenizer {
    const direct_gguf_path = try findDecoderGgufPathInDir(allocator, model_dir);
    defer if (direct_gguf_path) |path| allocator.free(path);
    if (direct_gguf_path) |gguf_path| {
        const has_hf_tokenizer = c_file.fileExistsInDir(allocator, model_dir, "tokenizer.json") or
            c_file.fileExistsInDir(allocator, model_dir, "vocab.txt") or
            c_file.fileExistsInDir(allocator, model_dir, "vocab.json");
        if (!has_hf_tokenizer and !c_file.fileExistsInDir(allocator, model_dir, "tokenizer.model")) {
            const sp = try model_manager_mod.loadSentencePieceTokenizerFromDirOrGguf(allocator, model_dir, gguf_path);
            sp.setPreserveInlineSpecialsAfterLiteralBos(true);
            try model_manager_mod.loadSentencePieceAddedTokens(model_dir, allocator, sp);
            return .{ .sp = sp };
        }
    }

    var manifest = try manifest_mod.loadFromDir(allocator, model_dir);
    defer manifest.deinit();

    const tokenizer_type = blk: {
        if (model_manager_mod.shouldPreferSentencePieceOverride(manifest, model_dir, allocator)) {
            break :blk manifest_mod.TokenizerType.sentencepiece;
        }
        break :blk manifest.tokenizer_type orelse return error.NoTokenizerFound;
    };

    return switch (tokenizer_type) {
        .huggingface => .{
            .hf = try model_manager_mod.loadHuggingFaceTokenizerFromDirOrGguf(allocator, model_dir, manifest.gguf_path),
        },
        .sentencepiece => blk: {
            const sp = try model_manager_mod.loadSentencePieceTokenizerFromDirOrGguf(allocator, model_dir, manifest.gguf_path);
            if (model_manager_mod.shouldEnableGemmaSentencePieceCompat(manifest, model_dir, allocator)) {
                sp.setPreserveInlineSpecialsAfterLiteralBos(true);
            }
            try model_manager_mod.loadSentencePieceAddedTokens(model_dir, allocator, sp);
            break :blk .{ .sp = sp };
        },
    };
}

// ---------------------------------------------------------------------------
// Prepare inputs from text dataset
// ---------------------------------------------------------------------------

pub fn prepareInputsFromData(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    loaded_examples: []const gemma_data.Example,
    max_examples: usize,
    max_seq_len: usize,
) !PreparedInputsSummary {
    const chat_examples = try allocator.alloc(gemma_chat_data.Example, loaded_examples.len);
    var converted_count: usize = 0;
    defer {
        var i: usize = 0;
        while (i < converted_count) : (i += 1) allocator.free(chat_examples[i].messages);
        allocator.free(chat_examples);
    }
    for (loaded_examples, 0..) |ex, idx| {
        chat_examples[idx] = try legacyExampleToChat(allocator, ex);
        converted_count += 1;
    }
    return prepareInputsFromChatData(allocator, model_dir, chat_examples, max_examples, max_seq_len);
}

pub fn prepareInputsFromChatData(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    loaded_examples: []const gemma_chat_data.Example,
    max_examples: usize,
    max_seq_len: usize,
) !PreparedInputsSummary {
    return prepareInputsFromChatDataInternal(
        allocator,
        model_dir,
        loaded_examples,
        max_examples,
        max_seq_len,
        null,
    );
}

pub fn prepareInputsFromChatDataWithSource(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    loaded_examples: []const gemma_chat_data.Example,
    max_examples: usize,
    max_seq_len: usize,
    source: PreparedSourceIdentity,
) !PreparedInputsSummary {
    return prepareInputsFromChatDataInternal(
        allocator,
        model_dir,
        loaded_examples,
        max_examples,
        max_seq_len,
        source,
    );
}

fn prepareInputsFromChatDataInternal(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    loaded_examples: []const gemma_chat_data.Example,
    max_examples: usize,
    max_seq_len: usize,
    source: ?PreparedSourceIdentity,
) !PreparedInputsSummary {
    var inspection = try inspectCheckpoint(allocator, model_dir);
    defer freeInspectionSummary(allocator, &inspection);
    const model_max_positions = try resolveTrainingModelMaxPositions(allocator, model_dir, inspection);
    _ = try validateTrainingSequenceLength(max_seq_len, model_max_positions);
    var provenance = try fingerprintGemma4Model(allocator, model_dir);
    defer provenance.deinit(allocator);

    var loaded_tokenizer = try loadGemmaTokenizerForModelDir(allocator, model_dir);
    defer loaded_tokenizer.deinit(allocator);
    const tok = loaded_tokenizer.tokenizer();

    const tokenizer_config_path = try std.fs.path.join(allocator, &.{ model_dir, tokenizer_config_file_name });
    defer allocator.free(tokenizer_config_path);
    const tokenizer_config_bytes = if (isRegularFilePath(tokenizer_config_path))
        try c_file.readFile(allocator, tokenizer_config_path)
    else
        null;
    defer if (tokenizer_config_bytes) |b| allocator.free(b);
    var parsed_tokenizer_config = if (tokenizer_config_bytes) |b|
        try std.json.parseFromSlice(TokenizerConfig, allocator, b, .{ .ignore_unknown_fields = true })
    else
        null;
    defer if (parsed_tokenizer_config) |*p| p.deinit();
    const tokenizer_class = if (parsed_tokenizer_config) |*p| p.value.tokenizer_class else null;

    const limit = if (max_examples > 0 and max_examples < loaded_examples.len) max_examples else loaded_examples.len;

    const prepared = try allocator.alloc(PreparedExampleInput, limit);
    errdefer allocator.free(prepared);

    var summary = PreparedInputsSummary{
        .artifact_family_version = try allocator.dupe(u8, artifact_family_version),
        .model_dir = try allocator.dupe(u8, model_dir),
        .schema_version = if (source != null) prepared_schema_v6 else prepared_schema_v4,
        .max_examples = max_examples,
        .examples_seen = limit,
        .tokenizer_class = try dupeOptionalString(allocator, tokenizer_class),
        .base_model_sha256 = try allocator.dupe(u8, provenance.base_model_sha256),
        .tokenizer_sha256 = try allocator.dupe(u8, provenance.tokenizer_sha256),
        .chat_template_sha256 = try allocator.dupe(u8, provenance.chat_template_sha256),
        .max_seq_len = max_seq_len,
        .examples = prepared[0..0],
    };
    errdefer freePreparedInputsSummary(allocator, &summary);
    if (source) |identity| try populatePreparedSourceIdentity(allocator, &summary, identity);
    var prepared_count: usize = 0;

    for (loaded_examples[0..limit], 0..) |ex, idx| {
        const item = try tokenizeChatExample(allocator, tok, ex, max_seq_len);
        prepared[idx] = item;
        prepared_count += 1;
        summary.examples = prepared[0..prepared_count];
        summary.max_prompt_tokens = @max(summary.max_prompt_tokens, item.num_prompt_tokens);
        summary.max_response_tokens = @max(summary.max_response_tokens, item.num_response_tokens);
        summary.max_input_tokens = @max(summary.max_input_tokens, item.num_input_tokens);
        summary.max_supervised_tokens = @max(summary.max_supervised_tokens, item.num_supervised_tokens);
        if (item.has_tool_calls) summary.examples_with_tool_calls += 1;
        if (item.has_tool_messages) summary.examples_with_tool_messages += 1;
        if (item.turn_count > 2) summary.examples_with_multiturn += 1;
        if (item.image_paths.len > 0) summary.examples_with_images += 1;
        if (item.audio_paths.len > 0) summary.examples_with_audio += 1;
        if (item.was_truncated) summary.examples_truncated += 1;
        summary.max_turns_dropped = @max(summary.max_turns_dropped, item.turns_dropped_from_left);
    }

    summary.prepared_examples_sha256 = try fingerprintPreparedExamplesForSchemaAlloc(allocator, summary.schema_version, summary.examples);

    return summary;
}

pub fn prepareMultimodalInputsFromChatData(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    gguf_projector_path: []const u8,
    loaded_examples: []const gemma_chat_data.Example,
    max_examples: usize,
    max_seq_len: usize,
) !PreparedInputsSummary {
    return prepareMultimodalInputsFromChatDataInternal(
        allocator,
        model_dir,
        gguf_projector_path,
        loaded_examples,
        max_examples,
        max_seq_len,
        null,
    );
}

pub fn prepareMultimodalInputsFromChatDataWithSource(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    gguf_projector_path: []const u8,
    loaded_examples: []const gemma_chat_data.Example,
    max_examples: usize,
    max_seq_len: usize,
    source: PreparedSourceIdentity,
) !PreparedInputsSummary {
    return prepareMultimodalInputsFromChatDataInternal(
        allocator,
        model_dir,
        gguf_projector_path,
        loaded_examples,
        max_examples,
        max_seq_len,
        source,
    );
}

fn prepareMultimodalInputsFromChatDataInternal(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    gguf_projector_path: []const u8,
    loaded_examples: []const gemma_chat_data.Example,
    max_examples: usize,
    max_seq_len: usize,
    source: ?PreparedSourceIdentity,
) !PreparedInputsSummary {
    var inspection = try inspectCheckpoint(allocator, model_dir);
    defer freeInspectionSummary(allocator, &inspection);
    const model_max_positions = try resolveTrainingModelMaxPositions(allocator, model_dir, inspection);
    _ = try validateTrainingSequenceLength(max_seq_len, model_max_positions);
    var provenance = try fingerprintGemma4Model(allocator, model_dir);
    defer provenance.deinit(allocator);

    var loaded_tokenizer = try loadGemmaTokenizerForModelDir(allocator, model_dir);
    defer loaded_tokenizer.deinit(allocator);
    const tok = loaded_tokenizer.tokenizer();

    const tokenizer_config_path = try std.fs.path.join(allocator, &.{ model_dir, tokenizer_config_file_name });
    defer allocator.free(tokenizer_config_path);
    const tokenizer_config_bytes = if (isRegularFilePath(tokenizer_config_path))
        try c_file.readFile(allocator, tokenizer_config_path)
    else
        null;
    defer if (tokenizer_config_bytes) |b| allocator.free(b);
    var parsed_tokenizer_config = if (tokenizer_config_bytes) |b|
        try std.json.parseFromSlice(TokenizerConfig, allocator, b, .{ .ignore_unknown_fields = true })
    else
        null;
    defer if (parsed_tokenizer_config) |*p| p.deinit();
    const tokenizer_class = if (parsed_tokenizer_config) |*p| p.value.tokenizer_class else null;

    const limit = if (max_examples > 0 and max_examples < loaded_examples.len) max_examples else loaded_examples.len;
    const prepared = try allocator.alloc(PreparedExampleInput, limit);
    errdefer allocator.free(prepared);
    const projector_fingerprint = try fingerprintProjectorFile(allocator, gguf_projector_path);
    defer freeProjectorFingerprint(allocator, &projector_fingerprint);

    var summary = PreparedInputsSummary{
        .artifact_family_version = try allocator.dupe(u8, artifact_family_version),
        .model_dir = try allocator.dupe(u8, model_dir),
        .schema_version = if (source != null) prepared_schema_v6 else prepared_schema_v4,
        .gguf_projector_path = try allocator.dupe(u8, projector_fingerprint.path),
        .gguf_projector_sha256 = try allocator.dupe(u8, projector_fingerprint.sha256),
        .gguf_projector_size_bytes = projector_fingerprint.size_bytes,
        .max_examples = max_examples,
        .examples_seen = limit,
        .tokenizer_class = try dupeOptionalString(allocator, tokenizer_class),
        .base_model_sha256 = try allocator.dupe(u8, provenance.base_model_sha256),
        .tokenizer_sha256 = try allocator.dupe(u8, provenance.tokenizer_sha256),
        .chat_template_sha256 = try allocator.dupe(u8, provenance.chat_template_sha256),
        .max_seq_len = max_seq_len,
        .examples = prepared[0..0],
    };
    errdefer freePreparedInputsSummary(allocator, &summary);
    if (source) |identity| try populatePreparedSourceIdentity(allocator, &summary, identity);
    var prepared_count: usize = 0;

    var dummy_ws = native_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer dummy_ws.resident_weights.deinit(allocator);
    defer dummy_ws.lazy_weights.deinit(allocator);
    var native_engine = native_compute.NativeCompute.init(allocator, &dummy_ws, null);
    const projector_cb = native_engine.computeBackend();
    var media_token_cache = PrepareMediaTokenCache{};
    defer media_token_cache.deinit(allocator);

    for (loaded_examples[0..limit], 0..) |ex, idx| {
        const item = if (ex.image_paths.len == 0 and ex.audio_paths.len == 0)
            try tokenizeChatExample(allocator, tok, ex, max_seq_len)
        else
            try tokenizeMultimodalChatExample(allocator, tok, projector_cb, gguf_projector_path, projector_fingerprint.sha256, &media_token_cache, ex, max_seq_len);
        prepared[idx] = item;
        prepared_count += 1;
        summary.examples = prepared[0..prepared_count];
        summary.max_prompt_tokens = @max(summary.max_prompt_tokens, item.num_prompt_tokens);
        summary.max_response_tokens = @max(summary.max_response_tokens, item.num_response_tokens);
        summary.max_input_tokens = @max(summary.max_input_tokens, item.num_input_tokens);
        summary.max_supervised_tokens = @max(summary.max_supervised_tokens, item.num_supervised_tokens);
        if (item.has_tool_calls) summary.examples_with_tool_calls += 1;
        if (item.has_tool_messages) summary.examples_with_tool_messages += 1;
        if (item.turn_count > 2) summary.examples_with_multiturn += 1;
        if (item.image_paths.len > 0) summary.examples_with_images += 1;
        if (item.audio_paths.len > 0) summary.examples_with_audio += 1;
        if (item.was_truncated) summary.examples_truncated += 1;
        summary.max_turns_dropped = @max(summary.max_turns_dropped, item.turns_dropped_from_left);
    }

    summary.prepared_examples_sha256 = try fingerprintPreparedExamplesForSchemaAlloc(allocator, summary.schema_version, summary.examples);

    return summary;
}

/// Hugging Face bundles expose their context limit through `config.json`, while
/// a standalone GGUF may carry the same contract only as architecture metadata.
/// Preparation must use the exact metadata path that the training graph uses so
/// a one-file deployment checkpoint is not rejected or assigned an invented
/// context limit.
fn resolveTrainingModelMaxPositions(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    inspection: InspectionSummary,
) !usize {
    if (inspection.max_position_embeddings) |value| return value;
    if (inspection.gguf_path == null) return error.MissingModelContextLength;

    const config = try session_factory.loadGptConfigMetadataFromModelDir(allocator, model_dir);
    if (config.family != .gemma) return error.UnsupportedModelFamily;
    if (config.max_position_embeddings == 0) return error.InvalidModelContextLength;
    return @intCast(config.max_position_embeddings);
}

pub fn loadPreparedInputsSummary(allocator: std.mem.Allocator, path: []const u8) !PreparedInputsSummary {
    const raw = try c_file.readFileMax(allocator, path, 128 * 1024 * 1024);
    defer allocator.free(raw);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const parsed = try std.json.parseFromSliceLeaky(PreparedInputsSummaryFile, arena.allocator(), raw, .{
        .ignore_unknown_fields = true,
    });
    var summary = try clonePreparedInputsSummary(allocator, &parsed.summary);
    errdefer freePreparedInputsSummary(allocator, &summary);
    try validatePreparedArtifactIntegrity(allocator, summary);
    return summary;
}

pub fn savePreparedInputsSummary(allocator: std.mem.Allocator, path: []const u8, summary: PreparedInputsSummary) !void {
    var buffer: std.Io.Writer.Allocating = .init(allocator);
    defer buffer.deinit();
    try std.json.Stringify.value(.{ .summary = summary }, .{ .whitespace = .indent_2 }, &buffer.writer);
    try artifact_publication.writeFileImmutable(allocator, compat.io(), path, buffer.written());
}

pub fn freePreparedInputsSummary(allocator: std.mem.Allocator, summary: *const PreparedInputsSummary) void {
    allocator.free(summary.artifact_family_version);
    allocator.free(summary.model_dir);
    if (summary.gguf_projector_path) |p| allocator.free(p);
    if (summary.gguf_projector_sha256) |p| allocator.free(p);
    if (summary.tokenizer_class) |p| allocator.free(p);
    if (summary.base_model_sha256) |p| allocator.free(p);
    if (summary.tokenizer_sha256) |p| allocator.free(p);
    if (summary.chat_template_sha256) |p| allocator.free(p);
    if (summary.prepared_examples_sha256) |p| allocator.free(p);
    if (summary.source_dataset_path) |p| allocator.free(p);
    if (summary.source_dataset_sha256) |p| allocator.free(p);
    if (summary.source_split) |p| allocator.free(p);
    if (summary.source_revision) |p| allocator.free(p);
    for (summary.examples) |*item| freePreparedExampleInput(allocator, item);
    allocator.free(summary.examples);
}

pub fn freeProjectorFingerprint(allocator: std.mem.Allocator, fingerprint: *const ProjectorFingerprint) void {
    allocator.free(fingerprint.path);
    allocator.free(fingerprint.sha256);
}

pub fn fingerprintProjectorFile(allocator: std.mem.Allocator, projector_path: []const u8) !ProjectorFingerprint {
    var mapped = try c_file.MmapRegion.init(allocator, projector_path);
    defer mapped.deinit();
    const sha256 = try sha256HexAlloc(allocator, mapped.data);
    errdefer allocator.free(sha256);
    return .{
        .path = try allocator.dupe(u8, projector_path),
        .sha256 = sha256,
        .size_bytes = mapped.data.len,
    };
}

pub fn sha256HexAlloc(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(bytes);
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
}

/// Fingerprint the exact, resolved JSONL inputs for a dataset split. Absolute
/// paths are deliberately excluded so moving an immutable dataset does not
/// change its identity; the sorted leaf names and file bytes remain part of
/// the digest.
pub fn fingerprintGemmaDatasetSourceAlloc(
    allocator: std.mem.Allocator,
    dataset_path: []const u8,
    split: ?[]const u8,
) ![]const u8 {
    var resolved = try jsonl_resolve.resolveJsonlFiles(allocator, dataset_path, split);
    defer resolved.deinit();
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hashBytes(&hasher, "gemma4_dataset_source/v1");
    hashBytes(&hasher, split orelse "");
    hashLength(&hasher, resolved.paths.len);
    for (resolved.paths) |path| {
        try hashFileInto(&hasher, allocator, "jsonl", path);
    }
    return finishHashAlloc(allocator, &hasher);
}

fn populatePreparedSourceIdentity(
    allocator: std.mem.Allocator,
    summary: *PreparedInputsSummary,
    identity: PreparedSourceIdentity,
) !void {
    summary.source_dataset_path = try allocator.dupe(u8, identity.dataset_path);
    summary.source_split = try dupeOptionalString(allocator, identity.split);
    summary.source_dataset_sha256 = try fingerprintGemmaDatasetSourceAlloc(
        allocator,
        identity.dataset_path,
        identity.split,
    );
    summary.source_revision = try allocator.dupe(
        u8,
        identity.revision orelse summary.source_dataset_sha256.?,
    );
    summary.schema_version = prepared_schema_v6;
}

fn hashLength(hasher: *std.crypto.hash.sha2.Sha256, value: usize) void {
    var encoded: [8]u8 = undefined;
    std.mem.writeInt(u64, &encoded, @intCast(value), .little);
    hasher.update(&encoded);
}

fn hashBytes(hasher: *std.crypto.hash.sha2.Sha256, bytes: []const u8) void {
    hashLength(hasher, bytes.len);
    hasher.update(bytes);
}

fn hashUsize(hasher: *std.crypto.hash.sha2.Sha256, value: usize) void {
    hashLength(hasher, value);
}

fn hashI32Slice(hasher: *std.crypto.hash.sha2.Sha256, values: []const i32) void {
    hashLength(hasher, values.len);
    for (values) |value| {
        var encoded: [4]u8 = undefined;
        std.mem.writeInt(u32, &encoded, @bitCast(value), .little);
        hasher.update(&encoded);
    }
}

fn hashUsizeSlice(hasher: *std.crypto.hash.sha2.Sha256, values: []const usize) void {
    hashLength(hasher, values.len);
    for (values) |value| hashUsize(hasher, value);
}

fn hashF32Slice(hasher: *std.crypto.hash.sha2.Sha256, values: []const f32) void {
    hashLength(hasher, values.len);
    for (values) |value| {
        var encoded: [4]u8 = undefined;
        std.mem.writeInt(u32, &encoded, @bitCast(value), .little);
        hasher.update(&encoded);
    }
}

fn hashStringSlice(hasher: *std.crypto.hash.sha2.Sha256, values: []const []const u8) void {
    hashLength(hasher, values.len);
    for (values) |value| hashBytes(hasher, value);
}

fn updatePreparedExampleHash(
    hasher: *std.crypto.hash.sha2.Sha256,
    example: *const PreparedExampleInput,
    include_training_targets: bool,
    include_source_identity: bool,
) void {
    hashUsize(hasher, @intFromEnum(example.mode));
    hashI32Slice(hasher, example.prompt_input_ids);
    hashI32Slice(hasher, example.response_input_ids);
    hashI32Slice(hasher, example.input_ids);
    hashStringSlice(hasher, example.image_paths);
    hashStringSlice(hasher, example.audio_paths);
    if (include_source_identity) {
        hashBytes(hasher, example.source_id orelse "");
        hashBytes(hasher, example.source_group_id orelse "");
        hashBytes(hasher, example.source_name orelse "");
        hashBytes(hasher, example.source_record_sha256 orelse "");
        hashBytes(hasher, example.rendered_chat_sha256 orelse "");
        hashStringSlice(hasher, example.media_content_sha256);
    }
    hashUsizeSlice(hasher, example.image_token_counts);
    hashUsizeSlice(hasher, example.audio_token_counts);
    if (!include_training_targets) return;
    hashI32Slice(hasher, example.labels);
    hashUsize(hasher, example.num_prompt_tokens);
    hashUsize(hasher, example.num_response_tokens);
    hashUsize(hasher, example.num_input_tokens);
    hashUsize(hasher, example.num_supervised_tokens);
    hashUsize(hasher, example.turn_count);
    hashUsize(hasher, @intFromBool(example.has_tool_calls));
    hashUsize(hasher, @intFromBool(example.has_tool_messages));
    hashI32Slice(hasher, example.teacher_top_k_token_ids);
    hashF32Slice(hasher, example.teacher_top_k_probs);
    hashUsize(hasher, example.teacher_top_k);
    hashF32Slice(hasher, &.{example.teacher_temperature});
    hashUsize(hasher, @intFromBool(example.was_truncated));
    hashUsize(hasher, example.turns_dropped_from_left);
    hashBytes(hasher, example.policy_version orelse "");
}

pub fn fingerprintGemmaChatSourceRecordAlloc(
    allocator: std.mem.Allocator,
    example: gemma_chat_data.Example,
) ![]const u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hashBytes(&hasher, "gemma4_chat_source_record/v1");
    hashBytes(&hasher, example.id orelse "");
    hashBytes(&hasher, example.metadata.source orelse "");
    hashBytes(&hasher, example.metadata.group_id orelse "");
    hashBytes(&hasher, example.metadata.policy_version orelse "");
    hashLength(&hasher, example.messages.len);
    for (example.messages) |message| {
        hashUsize(&hasher, @intFromEnum(message.role));
        hashBytes(&hasher, message.content);
        hashBytes(&hasher, message.tool_call_id orelse "");
        hashBytes(&hasher, message.name orelse "");
        hashLength(&hasher, message.tool_calls.len);
        for (message.tool_calls) |call| {
            hashBytes(&hasher, call.id);
            hashBytes(&hasher, call.name);
            hashBytes(&hasher, call.arguments_json);
        }
    }
    hashLength(&hasher, example.tools.len);
    for (example.tools) |tool| {
        hashBytes(&hasher, tool.name);
        hashBytes(&hasher, tool.description orelse "");
        hashBytes(&hasher, tool.input_schema_json orelse "");
    }
    hashStringSlice(&hasher, example.image_paths);
    hashStringSlice(&hasher, example.audio_paths);
    return finishHashAlloc(allocator, &hasher);
}

fn fingerprintMediaContentAlloc(
    allocator: std.mem.Allocator,
    image_bytes: []const []const u8,
    audio_bytes: []const []const u8,
) ![]const []const u8 {
    const total = try std.math.add(usize, image_bytes.len, audio_bytes.len);
    if (total == 0) return &.{};
    const hashes = try allocator.alloc([]const u8, total);
    var completed: usize = 0;
    errdefer {
        for (hashes[0..completed]) |hash| allocator.free(hash);
        allocator.free(hashes);
    }
    for (image_bytes) |bytes| {
        hashes[completed] = try sha256HexAlloc(allocator, bytes);
        completed += 1;
    }
    for (audio_bytes) |bytes| {
        hashes[completed] = try sha256HexAlloc(allocator, bytes);
        completed += 1;
    }
    return hashes;
}

fn preparedExampleIdentityDigest(example: *const PreparedExampleInput) [std.crypto.hash.sha2.Sha256.digest_length]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hashBytes(&hasher, "gemma4_prepared_example_identity/v1");
    updatePreparedExampleHash(&hasher, example, false, false);
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return digest;
}

pub fn fingerprintPreparedExamplesAlloc(
    allocator: std.mem.Allocator,
    examples: []const PreparedExampleInput,
) ![]const u8 {
    return fingerprintPreparedExamplesVersionedAlloc(
        allocator,
        examples,
        "gemma4_prepared_examples/v1",
        false,
    );
}

pub fn fingerprintPreparedExamplesV2Alloc(
    allocator: std.mem.Allocator,
    examples: []const PreparedExampleInput,
) ![]const u8 {
    return fingerprintPreparedExamplesVersionedAlloc(
        allocator,
        examples,
        "gemma4_prepared_examples/v2",
        true,
    );
}

pub fn fingerprintPreparedExamplesV3Alloc(
    allocator: std.mem.Allocator,
    examples: []const PreparedExampleInput,
) ![]const u8 {
    return fingerprintPreparedExamplesVersionedAlloc(
        allocator,
        examples,
        "gemma4_prepared_examples/v3",
        true,
    );
}

fn fingerprintPreparedExamplesVersionedAlloc(
    allocator: std.mem.Allocator,
    examples: []const PreparedExampleInput,
    domain: []const u8,
    include_source_identity: bool,
) ![]const u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hashBytes(&hasher, domain);
    hashLength(&hasher, examples.len);
    for (examples) |*example| updatePreparedExampleHash(&hasher, example, true, include_source_identity);
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
}

fn fingerprintPreparedExamplesForSchemaAlloc(
    allocator: std.mem.Allocator,
    schema_version: []const u8,
    examples: []const PreparedExampleInput,
) ![]const u8 {
    return if (std.mem.eql(u8, schema_version, prepared_schema_v6))
        fingerprintPreparedExamplesV3Alloc(allocator, examples)
    else if (std.mem.eql(u8, schema_version, prepared_schema_v5))
        fingerprintPreparedExamplesV2Alloc(allocator, examples)
    else
        fingerprintPreparedExamplesAlloc(allocator, examples);
}

pub fn refreshPreparedExamplesFingerprint(
    allocator: std.mem.Allocator,
    summary: *PreparedInputsSummary,
) !void {
    const digest = try fingerprintPreparedExamplesForSchemaAlloc(allocator, summary.schema_version, summary.examples);
    if (summary.prepared_examples_sha256) |old| allocator.free(old);
    summary.prepared_examples_sha256 = digest;
}

fn hashFileInto(
    hasher: *std.crypto.hash.sha2.Sha256,
    allocator: std.mem.Allocator,
    role: []const u8,
    path: []const u8,
) !void {
    var mapped = try c_file.MmapRegion.init(allocator, path);
    defer mapped.deinit();
    hashBytes(hasher, role);
    hashBytes(hasher, std.fs.path.basename(path));
    hashBytes(hasher, mapped.data);
}

fn finishHashAlloc(
    allocator: std.mem.Allocator,
    hasher: *std.crypto.hash.sha2.Sha256,
) ![]const u8 {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
}

pub fn fingerprintGemma4Model(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
) !ModelProvenance {
    var paths = try resolveArtifactPaths(allocator, model_dir);
    defer paths.deinit();

    var base_hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hashBytes(&base_hasher, "gemma4_base_model/v1");
    if (paths.config_path) |path|
        try hashFileInto(&base_hasher, allocator, "config", path)
    else
        hashBytes(&base_hasher, "config_absent");
    if (paths.checkpoint_path) |checkpoint_path| {
        const is_index = std.mem.endsWith(u8, checkpoint_path, ".safetensors.index.json");
        var dependencies = try safetensors.inspectArtifactDependencies(
            allocator,
            if (is_index) null else checkpoint_path,
            if (is_index) checkpoint_path else null,
        );
        defer dependencies.deinit();
        std.sort.heap([]u8, dependencies.paths, {}, struct {
            fn lessThan(_: void, lhs: []u8, rhs: []u8) bool {
                const lhs_base = std.fs.path.basename(lhs);
                const rhs_base = std.fs.path.basename(rhs);
                const order = std.mem.order(u8, lhs_base, rhs_base);
                return if (order == .eq) std.mem.lessThan(u8, lhs, rhs) else order == .lt;
            }
        }.lessThan);
        for (dependencies.paths) |path| try hashFileInto(&base_hasher, allocator, "safetensors", path);
    } else if (paths.gguf_path) |path| {
        try hashFileInto(&base_hasher, allocator, "gguf", path);
    } else return error.MissingMergedCheckpoint;
    const base_digest = try finishHashAlloc(allocator, &base_hasher);
    errdefer allocator.free(base_digest);

    var tokenizer_hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hashBytes(&tokenizer_hasher, "gemma4_tokenizer/v1");
    var tokenizer_file_count: usize = 0;
    const tokenizer_candidates = [_][]const u8{
        tokenizer_file_name,
        "tokenizer.model",
        tokenizer_config_file_name,
        special_tokens_map_file_name,
        "added_tokens.json",
    };
    for (tokenizer_candidates) |file_name| {
        const path = try std.fs.path.join(allocator, &.{ model_dir, file_name });
        defer allocator.free(path);
        if (!isRegularFilePath(path)) continue;
        try hashFileInto(&tokenizer_hasher, allocator, "tokenizer_asset", path);
        tokenizer_file_count += 1;
    }
    // Some GGUF bundles embed their tokenizer. The selected decoder digest is
    // then also the exact tokenizer identity without reading the GGUF twice.
    if (tokenizer_file_count == 0) hashBytes(&tokenizer_hasher, base_digest);
    const tokenizer_digest = try finishHashAlloc(allocator, &tokenizer_hasher);
    errdefer allocator.free(tokenizer_digest);
    const chat_digest = try sha256HexAlloc(allocator, prepared_chat_template_identity);
    errdefer allocator.free(chat_digest);

    return .{
        .base_model_sha256 = base_digest,
        .tokenizer_sha256 = tokenizer_digest,
        .chat_template_sha256 = chat_digest,
    };
}

pub fn populatePreparedProvenance(
    allocator: std.mem.Allocator,
    summary: *PreparedInputsSummary,
    model_dir: []const u8,
) !void {
    var provenance = try fingerprintGemma4Model(allocator, model_dir);
    defer provenance.deinit(allocator);
    if (summary.base_model_sha256) |old| allocator.free(old);
    if (summary.tokenizer_sha256) |old| allocator.free(old);
    if (summary.chat_template_sha256) |old| allocator.free(old);
    summary.base_model_sha256 = try allocator.dupe(u8, provenance.base_model_sha256);
    summary.tokenizer_sha256 = try allocator.dupe(u8, provenance.tokenizer_sha256);
    summary.chat_template_sha256 = try allocator.dupe(u8, provenance.chat_template_sha256);
    summary.schema_version = prepared_schema_v4;
    try refreshPreparedExamplesFingerprint(allocator, summary);
}

fn validateSha256Hex(value: []const u8) !void {
    if (value.len != std.crypto.hash.sha2.Sha256.digest_length * 2) return error.InvalidPreparedFingerprint;
    for (value) |byte| if (!std.ascii.isHex(byte)) return error.InvalidPreparedFingerprint;
}

pub fn validatePreparedArtifactIntegrity(
    allocator: std.mem.Allocator,
    summary: PreparedInputsSummary,
) !void {
    if (!std.mem.eql(u8, summary.artifact_family_version, artifact_family_version)) return error.UnsupportedArtifactFamily;
    const is_v4 = std.mem.eql(u8, summary.schema_version, prepared_schema_v4);
    const is_v5 = std.mem.eql(u8, summary.schema_version, prepared_schema_v5);
    const is_v6 = std.mem.eql(u8, summary.schema_version, prepared_schema_v6);
    if (!is_v4 and !is_v5 and !is_v6) return error.PreparedInputsProvenanceRequired;
    const base_digest = summary.base_model_sha256 orelse return error.PreparedInputsProvenanceRequired;
    const tokenizer_digest = summary.tokenizer_sha256 orelse return error.PreparedInputsProvenanceRequired;
    const chat_digest = summary.chat_template_sha256 orelse return error.PreparedInputsProvenanceRequired;
    const examples_digest = summary.prepared_examples_sha256 orelse return error.PreparedInputsProvenanceRequired;
    try validateSha256Hex(base_digest);
    try validateSha256Hex(tokenizer_digest);
    try validateSha256Hex(chat_digest);
    try validateSha256Hex(examples_digest);
    const actual = try fingerprintPreparedExamplesForSchemaAlloc(allocator, summary.schema_version, summary.examples);
    defer allocator.free(actual);
    if (!std.mem.eql(u8, examples_digest, actual)) return error.PreparedInputsFingerprintMismatch;
    if (is_v5 or is_v6) {
        const source_path = summary.source_dataset_path orelse return error.PreparedSourceProvenanceRequired;
        const source_digest = summary.source_dataset_sha256 orelse return error.PreparedSourceProvenanceRequired;
        const source_revision = summary.source_revision orelse return error.PreparedSourceProvenanceRequired;
        if (std.mem.trim(u8, source_path, " \t\r\n").len == 0 or
            std.mem.trim(u8, source_revision, " \t\r\n").len == 0)
        {
            return error.PreparedSourceProvenanceRequired;
        }
        try validateSha256Hex(source_digest);
        for (summary.examples) |example| {
            const source_id = example.source_id orelse return error.PreparedSourceProvenanceRequired;
            const group_id = example.source_group_id orelse return error.PreparedSourceProvenanceRequired;
            const record_digest = example.source_record_sha256 orelse return error.PreparedSourceProvenanceRequired;
            const rendered_digest = example.rendered_chat_sha256 orelse return error.PreparedSourceProvenanceRequired;
            if (source_id.len == 0 or group_id.len == 0) return error.PreparedSourceProvenanceRequired;
            try validateSha256Hex(record_digest);
            try validateSha256Hex(rendered_digest);
            const expected_media_hashes = try std.math.add(usize, example.image_paths.len, example.audio_paths.len);
            if (example.media_content_sha256.len != expected_media_hashes) return error.PreparedMediaFingerprintMismatch;
            for (example.media_content_sha256) |digest| try validateSha256Hex(digest);
        }
    }
}

pub fn validatePreparedModelProvenance(
    summary: PreparedInputsSummary,
    actual: ModelProvenance,
) !void {
    const base_digest = summary.base_model_sha256 orelse return error.PreparedInputsProvenanceRequired;
    const tokenizer_digest = summary.tokenizer_sha256 orelse return error.PreparedInputsProvenanceRequired;
    const chat_digest = summary.chat_template_sha256 orelse return error.PreparedInputsProvenanceRequired;
    if (!std.mem.eql(u8, base_digest, actual.base_model_sha256)) return error.PreparedBaseModelMismatch;
    if (!std.mem.eql(u8, tokenizer_digest, actual.tokenizer_sha256)) return error.PreparedTokenizerMismatch;
    if (!std.mem.eql(u8, chat_digest, actual.chat_template_sha256)) return error.PreparedChatTemplateMismatch;
}

/// Re-resolve and hash the immutable raw split at train/eval admission. The
/// prepared artifact remains portable for inspection, but a production run
/// must still have the source snapshot available so held-out claims are tied
/// to bytes rather than caller-supplied metadata.
pub fn validatePreparedSourceDatasetProvenance(
    allocator: std.mem.Allocator,
    summary: PreparedInputsSummary,
) !void {
    if (!std.mem.eql(u8, summary.schema_version, prepared_schema_v6)) {
        return error.PreparedSourceProvenanceRequired;
    }
    const source_path = summary.source_dataset_path orelse return error.PreparedSourceProvenanceRequired;
    const expected = summary.source_dataset_sha256 orelse return error.PreparedSourceProvenanceRequired;
    const actual = try fingerprintGemmaDatasetSourceAlloc(allocator, source_path, summary.source_split);
    defer allocator.free(actual);
    if (!std.mem.eql(u8, expected, actual)) return error.PreparedSourceDatasetMismatch;
    if (summary.examples.len == 0) return;

    var loaded = try gemma_chat_data.loadExamples(allocator, source_path, summary.source_split);
    defer loaded.deinit();
    try validatePreparedExamplesBelongToSource(allocator, summary.examples, loaded.examples);
}

const SourceRecordIdentity = struct {
    source_id: []const u8,
    group_id: []const u8,
    source_name: ?[]const u8,
};

fn validatePreparedExamplesBelongToSource(
    allocator: std.mem.Allocator,
    prepared: []const PreparedExampleInput,
    source_examples: []const gemma_chat_data.Example,
) !void {
    var records = std.StringHashMapUnmanaged(SourceRecordIdentity).empty;
    defer records.deinit(allocator);
    var owned_digests = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (owned_digests.items) |digest| allocator.free(digest);
        owned_digests.deinit(allocator);
    }

    try records.ensureTotalCapacity(allocator, @intCast(source_examples.len));
    for (source_examples) |example| {
        const digest = try fingerprintGemmaChatSourceRecordAlloc(allocator, example);
        owned_digests.append(allocator, digest) catch |err| {
            allocator.free(digest);
            return err;
        };
        const entry = records.getOrPutAssumeCapacity(digest);
        if (entry.found_existing) return error.DuplicatePreparedSourceRecord;
        entry.value_ptr.* = .{
            .source_id = example.id orelse digest,
            .group_id = example.metadata.group_id orelse example.id orelse digest,
            .source_name = example.metadata.source,
        };
    }

    var seen = std.StringHashMapUnmanaged(void).empty;
    defer seen.deinit(allocator);
    try seen.ensureTotalCapacity(allocator, @intCast(prepared.len));
    for (prepared) |example| {
        const digest = example.source_record_sha256 orelse return error.PreparedSourceProvenanceRequired;
        const source_id = example.source_id orelse return error.PreparedSourceProvenanceRequired;
        const group_id = example.source_group_id orelse return error.PreparedSourceProvenanceRequired;
        const identity = records.get(digest) orelse return error.PreparedSourceRecordMismatch;
        if (!std.mem.eql(u8, source_id, identity.source_id) or
            !std.mem.eql(u8, group_id, identity.group_id) or
            !optionalStringsEqual(example.source_name, identity.source_name))
        {
            return error.PreparedSourceRecordMismatch;
        }
        const entry = seen.getOrPutAssumeCapacity(digest);
        if (entry.found_existing) return error.DuplicatePreparedSourceRecord;
        entry.value_ptr.* = {};
    }
}

pub fn validateAdapterModelProvenance(
    inspected: InspectionSummary,
    actual: ModelProvenance,
) !void {
    const base_digest = inspected.base_model_sha256 orelse return error.AdapterProvenanceRequired;
    const tokenizer_digest = inspected.tokenizer_sha256 orelse return error.AdapterProvenanceRequired;
    const chat_digest = inspected.chat_template_sha256 orelse return error.AdapterProvenanceRequired;
    if (!std.mem.eql(u8, base_digest, actual.base_model_sha256)) return error.AdapterBaseModelMismatch;
    if (!std.mem.eql(u8, tokenizer_digest, actual.tokenizer_sha256)) return error.AdapterTokenizerMismatch;
    if (!std.mem.eql(u8, chat_digest, actual.chat_template_sha256)) return error.AdapterChatTemplateMismatch;
}

pub fn validatePreparedEvalDisjoint(
    allocator: std.mem.Allocator,
    training_examples: []const PreparedExampleInput,
    eval_examples: []const PreparedExampleInput,
) !void {
    if (eval_examples.len == 0) return error.NoEvaluationData;
    var training = std.AutoHashMapUnmanaged([std.crypto.hash.sha2.Sha256.digest_length]u8, void).empty;
    defer training.deinit(allocator);
    try training.ensureTotalCapacity(allocator, @intCast(training_examples.len));
    for (training_examples) |*example| training.putAssumeCapacity(preparedExampleIdentityDigest(example), {});
    for (eval_examples) |*example| {
        if (training.contains(preparedExampleIdentityDigest(example))) return error.TrainingEvaluationOverlap;
    }

    var source_records = std.StringHashMapUnmanaged(void).empty;
    defer source_records.deinit(allocator);
    var source_groups = std.StringHashMapUnmanaged(void).empty;
    defer source_groups.deinit(allocator);
    for (training_examples) |example| {
        if (example.source_record_sha256) |digest| try source_records.put(allocator, digest, {});
        if (example.source_group_id) |group| try source_groups.put(allocator, group, {});
    }
    for (eval_examples) |example| {
        if (example.source_record_sha256) |digest| {
            if (source_records.contains(digest)) return error.TrainingEvaluationSourceOverlap;
        }
        if (example.source_group_id) |group| {
            if (source_groups.contains(group)) return error.TrainingEvaluationGroupOverlap;
        }
    }
}

pub fn validateTrainingSequenceLength(max_seq_len: usize, model_max_position_embeddings: usize) !u32 {
    if (max_seq_len == 0) return error.InvalidPreparedSequenceLength;
    if (max_seq_len > std.math.maxInt(u32)) return error.PreparedSequenceLengthOverflow;
    if (model_max_position_embeddings == 0) return error.InvalidModelContextLength;
    if (max_seq_len > model_max_position_embeddings) return error.PreparedSequenceExceedsModelContext;
    if (max_seq_len > max_training_seq_len) return error.PreparedSequenceExceedsTrainingLimit;
    return @intCast(max_seq_len);
}

pub fn validatePreparedSequenceAdmission(
    summary: PreparedInputsSummary,
    model_max_position_embeddings: u32,
) !u32 {
    const seq_len = try validateTrainingSequenceLength(summary.max_seq_len, model_max_position_embeddings);
    if (summary.examples_seen != summary.examples.len) return error.PreparedSummaryMismatch;
    if (summary.max_examples > 0 and summary.examples_seen > summary.max_examples) return error.PreparedSummaryMismatch;
    var actual_max_input: usize = 0;
    var actual_max_supervised: usize = 0;
    var actual_max_prompt: usize = 0;
    var actual_max_response: usize = 0;
    var actual_tool_calls: usize = 0;
    var actual_tool_messages: usize = 0;
    var actual_multiturn: usize = 0;
    var actual_images: usize = 0;
    var actual_audio: usize = 0;
    var actual_truncated: usize = 0;
    var actual_max_turns_dropped: usize = 0;
    for (summary.examples) |example| {
        if (example.prompt_input_ids.len != example.num_prompt_tokens or
            example.response_input_ids.len != example.num_response_tokens)
        {
            return error.PreparedSummaryMismatch;
        }
        if (example.input_ids.len != example.num_input_tokens or example.labels.len != example.num_input_tokens) return error.PreparedSummaryMismatch;
        if (example.num_input_tokens > summary.max_seq_len) return error.PreparedSequenceExceedsDeclaredLength;
        var supervised: usize = 0;
        for (example.labels) |label| {
            if (label != -100) supervised += 1;
        }
        if (supervised != example.num_supervised_tokens or supervised == 0) return error.PreparedSummaryMismatch;
        actual_max_prompt = @max(actual_max_prompt, example.num_prompt_tokens);
        actual_max_response = @max(actual_max_response, example.num_response_tokens);
        actual_max_input = @max(actual_max_input, example.num_input_tokens);
        actual_max_supervised = @max(actual_max_supervised, example.num_supervised_tokens);
        if (example.has_tool_calls) actual_tool_calls += 1;
        if (example.has_tool_messages) actual_tool_messages += 1;
        if (example.turn_count > 2) actual_multiturn += 1;
        if (example.image_paths.len > 0) actual_images += 1;
        if (example.audio_paths.len > 0) actual_audio += 1;
        if (example.was_truncated) actual_truncated += 1;
        actual_max_turns_dropped = @max(actual_max_turns_dropped, example.turns_dropped_from_left);
    }
    if (actual_max_prompt != summary.max_prompt_tokens or
        actual_max_response != summary.max_response_tokens or
        actual_max_input != summary.max_input_tokens or
        actual_max_supervised != summary.max_supervised_tokens or
        actual_tool_calls != summary.examples_with_tool_calls or
        actual_tool_messages != summary.examples_with_tool_messages or
        actual_multiturn != summary.examples_with_multiturn or
        actual_images != summary.examples_with_images or
        actual_audio != summary.examples_with_audio or
        actual_truncated != summary.examples_truncated or
        actual_max_turns_dropped != summary.max_turns_dropped)
    {
        return error.PreparedSummaryMismatch;
    }
    return seq_len;
}

pub fn validatePreparedVocabularyAdmission(
    summary: PreparedInputsSummary,
    vocab_size: u32,
) !void {
    if (vocab_size == 0) return error.InvalidModelVocabulary;
    for (summary.examples) |example| {
        var supervised: usize = 0;
        for (example.input_ids) |token_id| {
            if (token_id < 0 or @as(u64, @intCast(token_id)) >= @as(u64, vocab_size)) return error.PreparedTokenOutOfRange;
        }
        for (example.labels) |label| {
            if (label == -100) continue;
            if (label < 0 or @as(u64, @intCast(label)) >= @as(u64, vocab_size)) return error.PreparedLabelOutOfRange;
            supervised += 1;
        }
        if (supervised != example.num_supervised_tokens) return error.PreparedSummaryMismatch;
        for (example.teacher_top_k_token_ids) |token_id| {
            if (token_id < 0 or @as(u64, @intCast(token_id)) >= @as(u64, vocab_size)) return error.PreparedTeacherTokenOutOfRange;
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

pub fn evaluatePreparedExamples(
    allocator: std.mem.Allocator,
    bundle: *const LoadedLoRABundle,
    examples: []const PreparedExampleInput,
    options: EvalOptions,
) !SurrogateMetrics {
    var metrics = SurrogateMetrics{};
    const limit = if (options.max_examples > 0 and options.max_examples < examples.len) options.max_examples else examples.len;
    if (limit == 0) return metrics;

    var total_loss: f64 = 0;
    var total_score: f64 = 0;
    var total_abs_error: f64 = 0;
    var total_token_weight: f64 = 0;

    for (examples[0..limit]) |*example| {
        if (example.num_supervised_tokens == 0) {
            metrics.examples_skipped_no_supervision += 1;
            continue;
        }
        const token_weight: f64 = @floatFromInt(example.num_supervised_tokens);
        const predicted = try scorePreparedExample(allocator, bundle, example, options.layer_name);
        const target = exampleTarget(example);
        const err = predicted - target;
        const loss = 0.5 * err * err;
        total_loss += loss * token_weight;
        total_abs_error += @abs(err) * token_weight;
        total_score += predicted * token_weight;
        metrics.examples_seen += 1;
        metrics.supervised_tokens_seen += example.num_supervised_tokens;
        total_token_weight += token_weight;
    }

    if (total_token_weight > 0) {
        const denom = total_token_weight;
        metrics.average_loss = total_loss / denom;
        metrics.mse = (total_loss * 2.0) / denom;
        metrics.mae = total_abs_error / denom;
        metrics.mean_score = total_score / denom;
    }
    return metrics;
}

pub fn scorePreparedExample(
    allocator: std.mem.Allocator,
    bundle: *const LoadedLoRABundle,
    example: *const PreparedExampleInput,
    layer_name: ?[]const u8,
) !f64 {
    var score: f64 = 0;
    for (bundle.layers) |layer| {
        if (!layerMatchesScope(layer.base_tensor_name, layer_name)) continue;
        score += try scoreLayerExample(allocator, &layer, bundle.lora_alpha, example);
    }
    return score;
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

pub const LoRALayerAdamState = struct {
    allocator: std.mem.Allocator,
    m_a: []f32,
    v_a: []f32,
    m_b: []f32,
    v_b: []f32,
    step: u64,

    pub fn init(alloc: std.mem.Allocator, layer: *const LoadedLoRALayer) !LoRALayerAdamState {
        const m_a = try alloc.alloc(f32, layer.adapter_a.len);
        errdefer alloc.free(m_a);
        const v_a = try alloc.alloc(f32, layer.adapter_a.len);
        errdefer alloc.free(v_a);
        const m_b = try alloc.alloc(f32, layer.adapter_b.len);
        errdefer alloc.free(m_b);
        const v_b = try alloc.alloc(f32, layer.adapter_b.len);
        errdefer alloc.free(v_b);
        @memset(m_a, 0);
        @memset(v_a, 0);
        @memset(m_b, 0);
        @memset(v_b, 0);
        return .{ .allocator = alloc, .m_a = m_a, .v_a = v_a, .m_b = m_b, .v_b = v_b, .step = 0 };
    }

    pub fn deinit(self: *LoRALayerAdamState) void {
        self.allocator.free(self.m_a);
        self.allocator.free(self.v_a);
        self.allocator.free(self.m_b);
        self.allocator.free(self.v_b);
        self.* = undefined;
    }
};

const LoRALayerSFState = struct {
    allocator: std.mem.Allocator,
    z_a: []f32,
    v_a: []f32,
    z_b: []f32,
    v_b: []f32,
    step: u64,

    fn init(alloc: std.mem.Allocator, layer: *const LoadedLoRALayer) !LoRALayerSFState {
        const z_a = try alloc.dupe(f32, layer.adapter_a);
        errdefer alloc.free(z_a);
        const v_a = try alloc.alloc(f32, layer.adapter_a.len);
        errdefer alloc.free(v_a);
        @memset(v_a, 0);
        const z_b = try alloc.dupe(f32, layer.adapter_b);
        errdefer alloc.free(z_b);
        const v_b = try alloc.alloc(f32, layer.adapter_b.len);
        errdefer alloc.free(v_b);
        @memset(v_b, 0);
        return .{ .allocator = alloc, .z_a = z_a, .v_a = v_a, .z_b = z_b, .v_b = v_b, .step = 0 };
    }

    fn deinit(self: *LoRALayerSFState) void {
        self.allocator.free(self.z_a);
        self.allocator.free(self.v_a);
        self.allocator.free(self.z_b);
        self.allocator.free(self.v_b);
        self.* = undefined;
    }
};

pub fn trainPreparedExamplesEpoch(
    allocator: std.mem.Allocator,
    bundle: *LoadedLoRABundle,
    examples: []const PreparedExampleInput,
    options: TrainEpochOptions,
) !TrainEpochSummary {
    var summary = TrainEpochSummary{
        .max_grad_norm = options.max_grad_norm,
        .llrd_decay = options.llrd_decay,
        .grad_accum_steps = options.grad_accum_steps,
    };
    const limit = if (options.max_examples > 0 and options.max_examples < examples.len) options.max_examples else examples.len;
    if (limit == 0) return summary;

    const num_layers = bundle.layers.len;

    const adam_states = try allocator.alloc(LoRALayerAdamState, num_layers);
    var adam_initialized: usize = 0;
    defer {
        var i: usize = 0;
        while (i < adam_initialized) : (i += 1) adam_states[i].deinit();
        allocator.free(adam_states);
    }
    for (bundle.layers, 0..) |*layer, li| {
        adam_states[li] = try LoRALayerAdamState.init(allocator, layer);
        adam_initialized += 1;
    }

    const sf_states = try allocator.alloc(?LoRALayerSFState, num_layers);
    defer allocator.free(sf_states);
    var sf_initialized: usize = 0;
    defer {
        var i: usize = 0;
        while (i < sf_initialized) : (i += 1) {
            if (sf_states[i]) |*s| s.deinit();
        }
    }
    for (bundle.layers, 0..) |*layer, li| {
        if (options.use_schedule_free) {
            sf_states[li] = try LoRALayerSFState.init(allocator, layer);
        } else {
            sf_states[li] = null;
        }
        sf_initialized += 1;
    }

    const accum_grad_a = try allocator.alloc([]f32, num_layers);
    defer allocator.free(accum_grad_a);
    const accum_grad_b = try allocator.alloc([]f32, num_layers);
    defer allocator.free(accum_grad_b);
    var accum_a_initialized: usize = 0;
    var accum_b_initialized: usize = 0;
    defer {
        var i: usize = 0;
        while (i < accum_a_initialized) : (i += 1) allocator.free(accum_grad_a[i]);
    }
    defer {
        var i: usize = 0;
        while (i < accum_b_initialized) : (i += 1) allocator.free(accum_grad_b[i]);
    }
    for (bundle.layers, 0..) |*layer, li| {
        accum_grad_a[li] = try allocator.alloc(f32, layer.adapter_a.len);
        accum_a_initialized += 1;
        accum_grad_b[li] = try allocator.alloc(f32, layer.adapter_b.len);
        accum_b_initialized += 1;
        @memset(accum_grad_a[li], 0);
        @memset(accum_grad_b[li], 0);
    }

    var max_layer_idx: usize = 0;
    for (bundle.layers) |*layer| {
        if (parseGemma4LayerIndex(layer.base_tensor_name)) |li| {
            if (li > max_layer_idx) max_layer_idx = li;
        }
    }

    const accum_steps = if (options.grad_accum_steps == 0) 1 else options.grad_accum_steps;
    var accum_count: u32 = 0;
    var accum_supervised_tokens: usize = 0;

    for (examples[0..limit], 0..) |*example, ex_idx| {
        if (example.num_supervised_tokens == 0) {
            summary.examples_skipped_no_supervision += 1;
            continue;
        }
        const is_last = (ex_idx == limit - 1);
        const token_weight: f64 = @floatFromInt(example.num_supervised_tokens);

        const predicted = try scorePreparedExample(allocator, bundle, example, options.layer_name);
        const target = exampleTarget(example);
        const error_value = predicted - target;
        const loss = 0.5 * error_value * error_value;
        summary.examples_seen += 1;
        summary.supervised_tokens_seen += example.num_supervised_tokens;
        summary.average_loss += loss * token_weight;
        summary.mean_score += predicted * token_weight;
        summary.mean_abs_error += @abs(error_value) * token_weight;

        for (bundle.layers, 0..) |*layer, li| {
            if (!layerMatchesScope(layer.base_tensor_name, options.layer_name)) continue;

            const input_rows: usize = 4;
            const inputs = try buildLayerFeatureRows(allocator, layer.input_dim, input_rows, example);
            defer allocator.free(inputs);
            const probe = try buildProbeVector(allocator, layer.base_tensor_name, layer.output_dim);
            defer allocator.free(probe);
            const output_grads = try allocator.alloc(f32, input_rows * layer.output_dim);
            defer allocator.free(output_grads);

            const row_scale = @as(f32, @floatCast(error_value)) /
                @as(f32, @floatFromInt(input_rows * @max(layer.output_dim, 1)));
            for (0..input_rows) |row_idx| {
                for (0..layer.output_dim) |out_idx| {
                    output_grads[row_idx * layer.output_dim + out_idx] = probe[out_idx] * row_scale;
                }
            }

            var used_pjrt = false;
            if (comptime build_options.enable_pjrt) {
                if (options.world_size <= 1) {
                    if (options.pjrt_lora_steps) |pjrt_steps| {
                        if (pjrt_steps[li]) |*pjrt_step| {
                            if (graph_bridge.computeLoRALinearGradsWithPjrt(
                                allocator,
                                pjrt_step,
                                layer.base_weight,
                                layer.adapter_a,
                                layer.adapter_b,
                                inputs,
                                output_grads,
                            )) |grads| {
                                defer allocator.free(grads.grad_a);
                                defer allocator.free(grads.grad_b);
                                for (accum_grad_a[li], grads.grad_a) |*acc, g| acc.* += g;
                                for (accum_grad_b[li], grads.grad_b) |*acc, g| acc.* += g;
                                used_pjrt = true;
                            } else |_| {}
                        }
                    }
                }
            }
            if (!used_pjrt) {
                const a_mat = lora.Matrix{ .rows = layer.input_dim, .cols = layer.rank, .data = layer.adapter_a };
                const b_mat = lora.Matrix{ .rows = layer.rank, .cols = layer.output_dim, .data = layer.adapter_b };
                lora.accumulateLinearLoRAGradsBackend(
                    options.compute_backend,
                    accum_grad_a[li],
                    accum_grad_b[li],
                    input_rows,
                    layer.input_dim,
                    inputs,
                    layer.output_dim,
                    output_grads,
                    a_mat,
                    b_mat,
                    bundle.lora_alpha,
                );
            }
        }

        accum_count += 1;
        accum_supervised_tokens += example.num_supervised_tokens;

        if (accum_count % accum_steps == 0 or is_last) {
            const eff_world_size: u32 = if (comptime false) options.world_size else 1;
            const token_denom = @max(accum_supervised_tokens, 1);
            const norm_factor = 1.0 / (@as(f32, @floatFromInt(token_denom)) * @as(f32, @floatFromInt(eff_world_size)));
            for (bundle.layers, 0..) |*layer, li| {
                if (!layerMatchesScope(layer.base_tensor_name, options.layer_name)) continue;
                for (accum_grad_a[li]) |*g| g.* *= norm_factor;
                for (accum_grad_b[li]) |*g| g.* *= norm_factor;
            }

            if (options.max_grad_norm > 0) {
                var total_sq: f32 = 0;
                for (bundle.layers, 0..) |*layer, li| {
                    if (!layerMatchesScope(layer.base_tensor_name, options.layer_name)) continue;
                    for (accum_grad_a[li]) |g| total_sq += g * g;
                    for (accum_grad_b[li]) |g| total_sq += g * g;
                }
                const global_norm = @sqrt(total_sq);
                if (global_norm > options.max_grad_norm) {
                    const clip_scale = options.max_grad_norm / (global_norm + 1e-8);
                    for (bundle.layers, 0..) |*layer, li| {
                        if (!layerMatchesScope(layer.base_tensor_name, options.layer_name)) continue;
                        for (accum_grad_a[li]) |*g| g.* *= clip_scale;
                        for (accum_grad_b[li]) |*g| g.* *= clip_scale;
                    }
                }
            }

            for (bundle.layers, 0..) |*layer, li| {
                if (!layerMatchesScope(layer.base_tensor_name, options.layer_name)) continue;

                var layer_lr = options.learning_rate;
                if (options.llrd_decay < 1.0) {
                    const layer_depth = parseGemma4LayerIndex(layer.base_tensor_name) orelse max_layer_idx;
                    const depth_from_top: f32 = @floatFromInt(max_layer_idx - @min(layer_depth, max_layer_idx));
                    layer_lr = options.learning_rate * std.math.pow(f32, options.llrd_decay, depth_from_top);
                }

                if (options.use_schedule_free) {
                    if (sf_states[li]) |*sf| {
                        sf.step += 1;
                        const lr = warmupAdjustedLR(layer_lr, sf.step, options.warmup_steps);
                        applyScheduleFreeInPlace(layer.adapter_a, accum_grad_a[li], sf.z_a, sf.v_a, sf.step, lr);
                        applyScheduleFreeInPlace(layer.adapter_b, accum_grad_b[li], sf.z_b, sf.v_b, sf.step, lr);
                    }
                } else {
                    adam_states[li].step += 1;
                    const lr = warmupAdjustedLR(layer_lr, adam_states[li].step, options.warmup_steps);
                    applyAdamWInPlace(layer.adapter_a, accum_grad_a[li], adam_states[li].m_a, adam_states[li].v_a, adam_states[li].step, lr);
                    applyAdamWInPlace(layer.adapter_b, accum_grad_b[li], adam_states[li].m_b, adam_states[li].v_b, adam_states[li].step, lr);
                }
                summary.updates_applied += 1;
            }

            for (0..num_layers) |li| {
                @memset(accum_grad_a[li], 0);
                @memset(accum_grad_b[li], 0);
            }
            accum_count = 0;
            accum_supervised_tokens = 0;
        }
    }

    if (summary.supervised_tokens_seen > 0) {
        const denom: f64 = @floatFromInt(summary.supervised_tokens_seen);
        summary.average_loss /= denom;
        summary.mean_score /= denom;
        summary.mean_abs_error /= denom;
    }
    return summary;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn tokenizeExample(
    allocator: std.mem.Allocator,
    tok: anytype,
    example: gemma_data.Example,
    max_seq_len: usize,
) !PreparedExampleInput {
    const prompt_text = if (example.mode == .instruction) example.prompt else "";
    const response_text = example.response;

    var prompt_result = try tok.encodeForModel(allocator, prompt_text, max_seq_len);
    defer prompt_result.deinit();
    var response_result = try tok.encodeForModel(allocator, response_text, max_seq_len);
    defer response_result.deinit();

    const prompt_ids = try allocator.dupe(i32, prompt_result.ids);
    errdefer allocator.free(prompt_ids);
    const response_ids = try allocator.dupe(i32, response_result.ids);
    errdefer allocator.free(response_ids);

    return .{
        .mode = example.mode,
        .prompt_input_ids = prompt_ids,
        .response_input_ids = response_ids,
        .num_prompt_tokens = prompt_result.ids.len,
        .num_response_tokens = response_result.ids.len,
    };
}

fn legacyExampleToChat(allocator: std.mem.Allocator, example: gemma_data.Example) !gemma_chat_data.Example {
    if (example.mode == .instruction) {
        const messages = try allocator.alloc(gemma_chat_data.Message, 2);
        messages[0] = .{ .role = .user, .content = example.prompt };
        messages[1] = .{ .role = .assistant, .content = example.response };
        return .{ .messages = messages };
    }
    const messages = try allocator.alloc(gemma_chat_data.Message, 1);
    messages[0] = .{ .role = .assistant, .content = example.response };
    return .{ .messages = messages };
}

fn tokenizeChatExample(
    allocator: std.mem.Allocator,
    tok: anytype,
    example: gemma_chat_data.Example,
    max_seq_len: usize,
) !PreparedExampleInput {
    const selected = try selectRenderableGemmaMessageWindow(allocator, tok, example, max_seq_len);
    defer allocator.free(selected.messages);

    const render_messages = try allocator.alloc(chat_template.Message, selected.messages.len);
    defer allocator.free(render_messages);

    var tool_call_json_bufs = try allocator.alloc(?[]u8, selected.messages.len);
    defer allocator.free(tool_call_json_bufs);
    @memset(tool_call_json_bufs, null);
    defer {
        for (tool_call_json_bufs) |maybe| {
            if (maybe) |buf| allocator.free(buf);
        }
    }

    var has_tool_calls = false;
    var has_tool_messages = false;
    for (selected.messages, 0..) |msg, idx| {
        if (msg.tool_calls.len > 0) {
            has_tool_calls = true;
            tool_call_json_bufs[idx] = try stringifyToolCalls(allocator, msg.tool_calls);
        }
        if (msg.role == .tool) has_tool_messages = true;
        render_messages[idx] = .{
            .role = switch (msg.role) {
                .system => .system,
                .user => .user,
                .assistant => .assistant,
                .tool => .tool,
            },
            .content = msg.content,
            .name = msg.name,
            .tool_call_id = msg.tool_call_id,
            .tool_calls_json = tool_call_json_bufs[idx],
        };
    }

    var rendered = try chat_template.render(allocator, .gemma, render_messages, .{});
    defer rendered.deinit();

    const source_record_sha256 = try fingerprintGemmaChatSourceRecordAlloc(allocator, example);
    errdefer allocator.free(source_record_sha256);
    const rendered_chat_sha256 = try sha256HexAlloc(allocator, rendered.text);
    errdefer allocator.free(rendered_chat_sha256);
    const source_id: ?[]const u8 = try allocator.dupe(u8, example.id orelse source_record_sha256);
    errdefer allocator.free(source_id.?);
    const source_group_id: ?[]const u8 = try allocator.dupe(u8, example.metadata.group_id orelse source_id.?);
    errdefer allocator.free(source_group_id.?);
    const source_name = try dupeOptionalString(allocator, example.metadata.source);
    errdefer if (source_name) |value| allocator.free(value);

    // The Gemma rendering already contains its literal BOS and turn control
    // tokens. Classifier-style encodeForModel would add another BOS plus EOS;
    // causal generation encoding preserves the exact chat-template sequence.
    var encoded = try tok.encodeForGenerationConfigured(allocator, rendered.text, max_seq_len, false);
    defer encoded.deinit();

    const padded_input_ids = try allocator.dupe(i32, encoded.ids);
    errdefer allocator.free(padded_input_ids);

    var padded_labels = if (encoded.offsets) |offsets| blk: {
        const token_offsets = try allocator.alloc(usize, encoded.ids.len);
        defer allocator.free(token_offsets);
        for (offsets, 0..) |off, idx| token_offsets[idx] = off[0];
        break :blk try chat_template.makeCompletionLabels(allocator, padded_input_ids, token_offsets, rendered.assistant_spans, -100);
    } else blk: {
        break :blk try makeCompletionLabelsWithoutOffsets(
            allocator,
            tok,
            rendered.text,
            rendered.assistant_spans,
            padded_input_ids,
            encoded.attention_mask,
            max_seq_len,
        );
    };
    errdefer allocator.free(padded_labels);

    var prompt_count: usize = 0;
    var response_count: usize = 0;
    var input_count: usize = 0;
    for (padded_labels, encoded.attention_mask, 0..) |label, attn, idx| {
        if (attn == 0) {
            padded_labels[idx] = -100;
            continue;
        }
        input_count += 1;
        if (label == -100) {
            prompt_count += 1;
        } else {
            response_count += 1;
        }
    }
    const contiguous_input_count = activeTokenizerTokenCount(encoded.attention_mask) orelse
        return error.TokenOffsetsUnavailable;
    if (contiguous_input_count == 0 or contiguous_input_count != input_count) {
        return error.TokenOffsetsUnavailable;
    }

    // Prepared rows are variable-length causal sequences. Tokenizer APIs pad
    // to max_seq_len for serving, but persisting those padding slots makes
    // num_input_tokens disagree with the serialized arrays and would cause
    // training admission to reject an artifact produced by this command.
    const input_ids = try allocator.dupe(i32, padded_input_ids[0..input_count]);
    errdefer allocator.free(input_ids);
    const labels = try allocator.dupe(i32, padded_labels[0..input_count]);
    errdefer allocator.free(labels);

    const prompt_ids = try allocator.alloc(i32, prompt_count);
    errdefer allocator.free(prompt_ids);
    const response_ids = try allocator.alloc(i32, response_count);
    errdefer allocator.free(response_ids);
    var p_idx: usize = 0;
    var r_idx: usize = 0;
    for (padded_input_ids, padded_labels, encoded.attention_mask) |id, label, attn| {
        if (attn == 0) continue;
        if (label == -100) {
            prompt_ids[p_idx] = id;
            p_idx += 1;
        } else {
            response_ids[r_idx] = id;
            r_idx += 1;
        }
    }
    allocator.free(padded_input_ids);
    allocator.free(padded_labels);

    return .{
        .mode = if (prompt_count > 0) .instruction else .completion,
        .prompt_input_ids = prompt_ids,
        .response_input_ids = response_ids,
        .num_prompt_tokens = prompt_count,
        .num_response_tokens = response_count,
        .input_ids = input_ids,
        .labels = labels,
        .num_input_tokens = input_count,
        .num_supervised_tokens = response_count,
        .turn_count = selected.messages.len,
        .has_tool_calls = has_tool_calls,
        .has_tool_messages = has_tool_messages,
        .image_paths = try cloneStringSlice(allocator, example.image_paths),
        .audio_paths = try cloneStringSlice(allocator, example.audio_paths),
        .was_truncated = selected.turns_dropped_from_left > 0 or encoded.ids.len == max_seq_len and selected.messages.len < example.messages.len,
        .turns_dropped_from_left = selected.turns_dropped_from_left,
        .policy_version = try dupeOptionalString(allocator, example.metadata.policy_version),
        .source_id = source_id,
        .source_group_id = source_group_id,
        .source_name = source_name,
        .source_record_sha256 = source_record_sha256,
        .rendered_chat_sha256 = rendered_chat_sha256,
    };
}

fn tokenizeMultimodalChatExample(
    allocator: std.mem.Allocator,
    tok: anytype,
    cb: @import("../ops/ops.zig").ComputeBackend,
    gguf_projector_path: []const u8,
    gguf_projector_sha256: []const u8,
    media_token_cache: *PrepareMediaTokenCache,
    example: gemma_chat_data.Example,
    max_seq_len: usize,
) !PreparedExampleInput {
    try validateMultimodalExampleShape(example);
    const image_bytes = try loadMediaBytes(allocator, example.image_paths);
    defer freeMediaBytes(allocator, image_bytes);
    const audio_bytes = try loadMediaBytes(allocator, example.audio_paths);
    defer freeMediaBytes(allocator, audio_bytes);

    const image_token_counts = try prepareMediaTokenCounts(allocator, media_token_cache, .image, &cb, gguf_projector_path, gguf_projector_sha256, image_bytes);
    defer allocator.free(image_token_counts);
    const audio_token_counts = try prepareMediaTokenCounts(allocator, media_token_cache, .audio, &cb, gguf_projector_path, gguf_projector_sha256, audio_bytes);
    defer allocator.free(audio_token_counts);

    var expanded = try expandMultimodalExample(
        allocator,
        example,
        image_token_counts,
        audio_token_counts,
    );
    defer freeExpandedMultimodalExample(allocator, &expanded);

    var prepared = try tokenizeChatExample(allocator, tok, expanded, max_seq_len);
    errdefer freePreparedExampleInput(allocator, &prepared);
    prepared.image_token_counts = try cloneUsizeSlice(allocator, image_token_counts);
    prepared.audio_token_counts = try cloneUsizeSlice(allocator, audio_token_counts);
    prepared.media_content_sha256 = try fingerprintMediaContentAlloc(allocator, image_bytes, audio_bytes);
    return prepared;
}

fn prepareMediaTokenCounts(
    allocator: std.mem.Allocator,
    cache: *PrepareMediaTokenCache,
    kind: PrepareMediaKind,
    cb: *const @import("../ops/ops.zig").ComputeBackend,
    gguf_projector_path: []const u8,
    gguf_projector_sha256: []const u8,
    items: []const []const u8,
) ![]usize {
    const counts = try allocator.alloc(usize, items.len);
    errdefer allocator.free(counts);
    for (items, 0..) |bytes, idx| {
        counts[idx] = try cachedPrepareMediaTokenCount(allocator, cache, kind, cb, gguf_projector_path, gguf_projector_sha256, bytes);
    }
    return counts;
}

fn cachedPrepareMediaTokenCount(
    allocator: std.mem.Allocator,
    cache: *PrepareMediaTokenCache,
    kind: PrepareMediaKind,
    cb: *const @import("../ops/ops.zig").ComputeBackend,
    gguf_projector_path: []const u8,
    gguf_projector_sha256: []const u8,
    bytes: []const u8,
) !usize {
    const media_sha256 = try sha256HexAlloc(allocator, bytes);
    defer allocator.free(media_sha256);
    const kind_name = switch (kind) {
        .image => "image",
        .audio => "audio",
    };
    const lookup_key = try std.fmt.allocPrint(allocator, "{s}:{s}:{s}", .{ gguf_projector_sha256, kind_name, media_sha256 });
    defer allocator.free(lookup_key);
    if (cache.items.get(lookup_key)) |tokens| return tokens;

    const tokens = switch (kind) {
        .image => blk: {
            var projected = try gemma4_projector.encodeProjectedImages(cb, allocator, gguf_projector_path, &.{bytes});
            defer projected.deinit();
            if (projected.tokens_per_image.len != 1) return error.InvalidPreparedPrompt;
            break :blk projected.tokens_per_image[0];
        },
        .audio => blk: {
            var projected = try gemma4_projector.encodeProjectedAudio(cb, allocator, gguf_projector_path, &.{bytes});
            defer projected.deinit();
            if (projected.tokens_per_audio.len != 1) return error.InvalidPreparedPrompt;
            break :blk projected.tokens_per_audio[0];
        },
    };
    const owned_key = try allocator.dupe(u8, lookup_key);
    errdefer allocator.free(owned_key);
    const entry = try cache.items.getOrPut(allocator, owned_key);
    if (entry.found_existing) {
        allocator.free(owned_key);
    } else {
        entry.key_ptr.* = owned_key;
        entry.value_ptr.* = tokens;
    }
    return entry.value_ptr.*;
}

fn validateMultimodalExampleShape(example: gemma_chat_data.Example) !void {
    var image_markers: usize = 0;
    var audio_markers: usize = 0;
    for (example.messages) |msg| {
        image_markers += countSubstring(msg.content, "<|image|>");
        audio_markers += countSubstring(msg.content, "<|audio|>");
    }
    if (image_markers != example.image_paths.len) return error.ImagePlaceholderCountMismatch;
    if (audio_markers != example.audio_paths.len) return error.AudioPlaceholderCountMismatch;
}

fn countSubstring(haystack: []const u8, needle: []const u8) usize {
    if (needle.len == 0) return 0;
    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, pos, needle)) |idx| {
        count += 1;
        pos = idx + needle.len;
    }
    return count;
}

fn expandMultimodalExample(
    allocator: std.mem.Allocator,
    example: gemma_chat_data.Example,
    image_token_counts: []const usize,
    audio_token_counts: []const usize,
) !gemma_chat_data.Example {
    const messages = try allocator.alloc(gemma_chat_data.Message, example.messages.len);
    errdefer allocator.free(messages);

    var image_idx: usize = 0;
    var audio_idx: usize = 0;
    for (example.messages, 0..) |msg, idx| {
        const expanded_content = try expandMessageMediaMarkers(
            allocator,
            msg.content,
            image_token_counts,
            &image_idx,
            audio_token_counts,
            &audio_idx,
        );
        messages[idx] = .{
            .role = msg.role,
            .content = expanded_content,
            .tool_call_id = msg.tool_call_id,
            .name = msg.name,
            .tool_calls = msg.tool_calls,
        };
    }
    if (image_idx != image_token_counts.len) return error.ImagePlaceholderCountMismatch;
    if (audio_idx != audio_token_counts.len) return error.AudioPlaceholderCountMismatch;

    return .{
        .id = example.id,
        .messages = messages,
        .tools = example.tools,
        .image_paths = example.image_paths,
        .audio_paths = example.audio_paths,
        .metadata = example.metadata,
    };
}

fn freeExpandedMultimodalExample(allocator: std.mem.Allocator, example: *const gemma_chat_data.Example) void {
    for (example.messages) |msg| allocator.free(msg.content);
    allocator.free(example.messages);
}

fn expandMessageMediaMarkers(
    allocator: std.mem.Allocator,
    content: []const u8,
    image_token_counts: []const usize,
    image_idx: *usize,
    audio_token_counts: []const usize,
    audio_idx: *usize,
) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    var cursor: usize = 0;
    while (cursor < content.len) {
        if (std.mem.startsWith(u8, content[cursor..], "<|image|>")) {
            if (image_idx.* >= image_token_counts.len) return error.ImagePlaceholderCountMismatch;
            try appendExpandedMarker(allocator, &out, "<|image>", "<|image|>", "<image|>", image_token_counts[image_idx.*]);
            image_idx.* += 1;
            cursor += "<|image|>".len;
            continue;
        }
        if (std.mem.startsWith(u8, content[cursor..], "<|audio|>")) {
            if (audio_idx.* >= audio_token_counts.len) return error.AudioPlaceholderCountMismatch;
            try appendExpandedMarker(allocator, &out, "<|audio>", "<|audio|>", "<audio|>", audio_token_counts[audio_idx.*]);
            audio_idx.* += 1;
            cursor += "<|audio|>".len;
            continue;
        }
        try out.append(allocator, content[cursor]);
        cursor += 1;
    }
    return try out.toOwnedSlice(allocator);
}

fn appendExpandedMarker(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    begin_marker: []const u8,
    marker: []const u8,
    end_marker: []const u8,
    token_count: usize,
) !void {
    try out.appendSlice(allocator, begin_marker);
    for (0..token_count) |_| try out.appendSlice(allocator, marker);
    try out.appendSlice(allocator, end_marker);
}

fn loadMediaBytes(allocator: std.mem.Allocator, paths: []const []const u8) ![]const []const u8 {
    if (paths.len == 0) return &.{};
    const out = try allocator.alloc([]const u8, paths.len);
    errdefer allocator.free(out);
    var loaded: usize = 0;
    errdefer {
        for (out[0..loaded]) |item| allocator.free(item);
    }
    for (paths, 0..) |path, idx| {
        out[idx] = try c_file.readFile(allocator, path);
        loaded += 1;
    }
    return out;
}

fn freeMediaBytes(allocator: std.mem.Allocator, data: []const []const u8) void {
    if (data.len == 0) return;
    for (data) |item| allocator.free(item);
    allocator.free(data);
}

fn cloneStringSlice(allocator: std.mem.Allocator, items: []const []const u8) ![]const []const u8 {
    if (items.len == 0) return &.{};
    const out = try allocator.alloc([]const u8, items.len);
    for (items, 0..) |item, idx| out[idx] = try allocator.dupe(u8, item);
    return out;
}

fn cloneUsizeSlice(allocator: std.mem.Allocator, items: []const usize) ![]const usize {
    if (items.len == 0) return &.{};
    return try allocator.dupe(usize, items);
}

fn cloneI32Slice(allocator: std.mem.Allocator, items: []const i32) ![]i32 {
    if (items.len == 0) return &.{};
    return try allocator.dupe(i32, items);
}

fn cloneF32Slice(allocator: std.mem.Allocator, items: []const f32) ![]f32 {
    if (items.len == 0) return &.{};
    return try allocator.dupe(f32, items);
}

fn stringifyToolCalls(allocator: std.mem.Allocator, tool_calls: []const gemma_chat_data.ToolCall) ![]u8 {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    try buf.writer.writeByte('[');
    for (tool_calls, 0..) |tool_call, idx| {
        if (idx != 0) try buf.writer.writeByte(',');
        try buf.writer.writeAll("{\"id\":");
        try std.json.Stringify.value(tool_call.id, .{}, &buf.writer);
        try buf.writer.writeAll(",\"type\":\"function\",\"function\":{\"name\":");
        try std.json.Stringify.value(tool_call.name, .{}, &buf.writer);
        try buf.writer.writeAll(",\"arguments\":");
        try std.json.Stringify.value(tool_call.arguments_json, .{}, &buf.writer);
        try buf.writer.writeAll("}}");
    }
    try buf.writer.writeByte(']');
    return try allocator.dupe(u8, buf.written());
}

fn makeCompletionLabelsWithoutOffsets(
    allocator: std.mem.Allocator,
    tok: anytype,
    rendered_text: []const u8,
    assistant_spans: []const chat_template.AssistantSpan,
    input_ids: []const i32,
    attention_mask: []const i32,
    max_seq_len: usize,
) ![]i32 {
    const labels = try allocator.alloc(i32, input_ids.len);
    errdefer allocator.free(labels);
    @memset(labels, -100);
    if (assistant_spans.len == 0 or input_ids.len != attention_mask.len) {
        return error.TokenOffsetsUnavailable;
    }

    const active_tokens = activeTokenizerTokenCount(attention_mask) orelse return error.TokenOffsetsUnavailable;
    if (active_tokens == 0 or active_tokens > input_ids.len) return error.TokenOffsetsUnavailable;

    var previous_span_end: usize = 0;
    for (assistant_spans) |span| {
        if (span.start < previous_span_end or span.start >= span.end or span.end > rendered_text.len) {
            return error.TokenOffsetsUnavailable;
        }
        const token_start = try tokenBoundaryFromPrefixRetokenization(
            allocator,
            tok,
            rendered_text,
            span.start,
            input_ids[0..active_tokens],
            max_seq_len,
        );
        const token_end = try tokenBoundaryFromPrefixRetokenization(
            allocator,
            tok,
            rendered_text,
            span.end,
            input_ids[0..active_tokens],
            max_seq_len,
        );
        if (token_start >= token_end or token_end > active_tokens) return error.TokenOffsetsUnavailable;
        for (input_ids[token_start..token_end], token_start..) |id, idx| labels[idx] = id;
        previous_span_end = span.end;
    }
    return labels;
}

fn activeTokenizerTokenCount(attention_mask: []const i32) ?usize {
    var count: usize = 0;
    var padding_started = false;
    for (attention_mask) |attn| {
        if (attn == 0) {
            padding_started = true;
        } else {
            if (padding_started or attn != 1) return null;
            count += 1;
        }
    }
    return count;
}

fn tokenBoundaryFromPrefixRetokenization(
    allocator: std.mem.Allocator,
    tok: anytype,
    rendered_text: []const u8,
    byte_offset: usize,
    full_active_ids: []const i32,
    max_seq_len: usize,
) !usize {
    if (byte_offset > rendered_text.len) return error.TokenOffsetsUnavailable;
    var prefix = try tok.encodeForGenerationConfigured(
        allocator,
        rendered_text[0..byte_offset],
        max_seq_len,
        false,
    );
    defer prefix.deinit();
    const prefix_active = activeTokenizerTokenCount(prefix.attention_mask) orelse return error.TokenOffsetsUnavailable;
    if (prefix_active == 0 or prefix_active > prefix.ids.len) return error.TokenOffsetsUnavailable;

    const comparable = @min(prefix_active, full_active_ids.len);
    var common: usize = 0;
    while (common < comparable and prefix.ids[common] == full_active_ids[common]) : (common += 1) {}

    if (byte_offset == rendered_text.len) {
        if (prefix_active != full_active_ids.len or common != full_active_ids.len) {
            return error.TokenOffsetsUnavailable;
        }
        return full_active_ids.len;
    }

    // Some model tokenizers append one terminal token to every encode. It is
    // absent at an interior boundary in the full rendering, so the exact
    // boundary is the common prefix immediately before that terminal token.
    if (common + 1 == prefix_active and
        full_active_ids.len > 0 and
        prefix.ids[prefix_active - 1] == full_active_ids[full_active_ids.len - 1])
    {
        return common;
    }
    // Tokenizers without an automatic terminal token may be prefix-stable.
    if (common == prefix_active) return common;
    return error.TokenOffsetsUnavailable;
}

fn freePreparedExampleInput(allocator: std.mem.Allocator, item: *const PreparedExampleInput) void {
    allocator.free(item.prompt_input_ids);
    allocator.free(item.response_input_ids);
    allocator.free(item.input_ids);
    allocator.free(item.labels);
    for (item.image_paths) |path| allocator.free(path);
    if (item.image_paths.len > 0) allocator.free(item.image_paths);
    for (item.audio_paths) |path| allocator.free(path);
    if (item.audio_paths.len > 0) allocator.free(item.audio_paths);
    if (item.image_token_counts.len > 0) allocator.free(item.image_token_counts);
    if (item.audio_token_counts.len > 0) allocator.free(item.audio_token_counts);
    if (item.teacher_top_k_token_ids.len > 0) allocator.free(item.teacher_top_k_token_ids);
    if (item.teacher_top_k_probs.len > 0) allocator.free(item.teacher_top_k_probs);
    if (item.policy_version) |p| allocator.free(p);
    if (item.source_id) |p| allocator.free(p);
    if (item.source_group_id) |p| allocator.free(p);
    if (item.source_name) |p| allocator.free(p);
    if (item.source_record_sha256) |p| allocator.free(p);
    if (item.rendered_chat_sha256) |p| allocator.free(p);
    for (item.media_content_sha256) |hash| allocator.free(hash);
    if (item.media_content_sha256.len > 0) allocator.free(item.media_content_sha256);
}

fn clonePreparedInputsSummary(allocator: std.mem.Allocator, source: *const PreparedInputsSummary) !PreparedInputsSummary {
    const normalized_schema = try normalizePreparedSchemaVersion(source.schema_version);
    const examples = try allocator.alloc(PreparedExampleInput, source.examples.len);
    var cloned_count: usize = 0;
    errdefer {
        for (examples[0..cloned_count]) |*item| freePreparedExampleInput(allocator, item);
        allocator.free(examples);
    }
    for (source.examples, 0..) |item, idx| {
        examples[idx] = .{
            .mode = item.mode,
            .prompt_input_ids = try allocator.dupe(i32, item.prompt_input_ids),
            .response_input_ids = try allocator.dupe(i32, item.response_input_ids),
            .num_prompt_tokens = item.num_prompt_tokens,
            .num_response_tokens = item.num_response_tokens,
            .input_ids = try allocator.dupe(i32, item.input_ids),
            .labels = try allocator.dupe(i32, item.labels),
            .num_input_tokens = item.num_input_tokens,
            .num_supervised_tokens = item.num_supervised_tokens,
            .turn_count = item.turn_count,
            .has_tool_calls = item.has_tool_calls,
            .has_tool_messages = item.has_tool_messages,
            .image_paths = try cloneStringSlice(allocator, item.image_paths),
            .audio_paths = try cloneStringSlice(allocator, item.audio_paths),
            .image_token_counts = try cloneUsizeSlice(allocator, item.image_token_counts),
            .audio_token_counts = try cloneUsizeSlice(allocator, item.audio_token_counts),
            .teacher_top_k_token_ids = try cloneI32Slice(allocator, item.teacher_top_k_token_ids),
            .teacher_top_k_probs = try cloneF32Slice(allocator, item.teacher_top_k_probs),
            .teacher_top_k = item.teacher_top_k,
            .teacher_temperature = item.teacher_temperature,
            .was_truncated = item.was_truncated,
            .turns_dropped_from_left = item.turns_dropped_from_left,
            .policy_version = try dupeOptionalString(allocator, item.policy_version),
            .source_id = try dupeOptionalString(allocator, item.source_id),
            .source_group_id = try dupeOptionalString(allocator, item.source_group_id),
            .source_name = try dupeOptionalString(allocator, item.source_name),
            .source_record_sha256 = try dupeOptionalString(allocator, item.source_record_sha256),
            .rendered_chat_sha256 = try dupeOptionalString(allocator, item.rendered_chat_sha256),
            .media_content_sha256 = try cloneStringSlice(allocator, item.media_content_sha256),
        };
        cloned_count += 1;
    }

    return .{
        .artifact_family_version = try allocator.dupe(u8, source.artifact_family_version),
        .model_dir = try allocator.dupe(u8, source.model_dir),
        .schema_version = normalized_schema,
        .gguf_projector_path = try dupeOptionalString(allocator, source.gguf_projector_path),
        .gguf_projector_sha256 = try dupeOptionalString(allocator, source.gguf_projector_sha256),
        .gguf_projector_size_bytes = source.gguf_projector_size_bytes,
        .max_examples = source.max_examples,
        .examples_seen = source.examples_seen,
        .tokenizer_class = try dupeOptionalString(allocator, source.tokenizer_class),
        .base_model_sha256 = try dupeOptionalString(allocator, source.base_model_sha256),
        .tokenizer_sha256 = try dupeOptionalString(allocator, source.tokenizer_sha256),
        .chat_template_sha256 = try dupeOptionalString(allocator, source.chat_template_sha256),
        .prepared_examples_sha256 = try dupeOptionalString(allocator, source.prepared_examples_sha256),
        .source_dataset_path = try dupeOptionalString(allocator, source.source_dataset_path),
        .source_dataset_sha256 = try dupeOptionalString(allocator, source.source_dataset_sha256),
        .source_split = try dupeOptionalString(allocator, source.source_split),
        .source_revision = try dupeOptionalString(allocator, source.source_revision),
        .max_seq_len = source.max_seq_len,
        .max_prompt_tokens = source.max_prompt_tokens,
        .max_response_tokens = source.max_response_tokens,
        .max_input_tokens = source.max_input_tokens,
        .max_supervised_tokens = source.max_supervised_tokens,
        .examples_with_tool_calls = source.examples_with_tool_calls,
        .examples_with_tool_messages = source.examples_with_tool_messages,
        .examples_with_multiturn = source.examples_with_multiturn,
        .examples_with_images = source.examples_with_images,
        .examples_with_audio = source.examples_with_audio,
        .examples_truncated = source.examples_truncated,
        .max_turns_dropped = source.max_turns_dropped,
        .examples = examples,
    };
}

fn normalizePreparedSchemaVersion(schema_version: []const u8) ![]const u8 {
    if (std.mem.eql(u8, schema_version, prepared_schema_v2)) return prepared_schema_v2;
    if (std.mem.eql(u8, schema_version, prepared_schema_v3)) return prepared_schema_v3;
    if (std.mem.eql(u8, schema_version, prepared_schema_v4)) return prepared_schema_v4;
    if (std.mem.eql(u8, schema_version, prepared_schema_v5)) return prepared_schema_v5;
    if (std.mem.eql(u8, schema_version, prepared_schema_v6)) return prepared_schema_v6;
    return error.UnsupportedPreparedInputsSchema;
}

fn scoreLayerExample(
    allocator: std.mem.Allocator,
    layer: *const LoadedLoRALayer,
    alpha: f32,
    example: *const PreparedExampleInput,
) !f64 {
    const input_rows: usize = 4;
    const inputs = try buildLayerFeatureRows(allocator, layer.input_dim, input_rows, example);
    defer allocator.free(inputs);
    const probe = try buildProbeVector(allocator, layer.base_tensor_name, layer.output_dim);
    defer allocator.free(probe);
    const scale = lora.effectiveScale(alpha, layer.rank);

    var total: f64 = 0;
    for (0..input_rows) |row_idx| {
        const row = inputs[row_idx * layer.input_dim .. (row_idx + 1) * layer.input_dim];
        var row_score: f64 = 0;
        for (0..layer.output_dim) |j| {
            var merged: f64 = 0;
            for (0..layer.input_dim) |i| {
                merged += @as(f64, row[i]) * @as(f64, layer.base_weight[j * layer.input_dim + i]);
            }
            row_score += merged * probe[j];
        }

        var tmp_rank = try allocator.alloc(f32, layer.rank);
        defer allocator.free(tmp_rank);
        @memset(tmp_rank, 0.0);
        for (0..layer.input_dim) |i| {
            const x = row[i];
            const a_row = layer.adapter_a[i * layer.rank .. (i + 1) * layer.rank];
            for (a_row, 0..) |a, r| tmp_rank[r] += x * a;
        }
        for (0..layer.rank) |r| {
            const scaled = @as(f64, tmp_rank[r] * scale);
            const b_row = layer.adapter_b[r * layer.output_dim .. (r + 1) * layer.output_dim];
            for (b_row, 0..) |b, j| row_score += scaled * b * probe[j];
        }
        total += row_score / @as(f64, @floatFromInt(@max(layer.output_dim, 1)));
    }
    return total / @as(f64, @floatFromInt(input_rows));
}

fn buildLayerFeatureRows(
    allocator: std.mem.Allocator,
    input_dim: usize,
    input_rows: usize,
    example: *const PreparedExampleInput,
) ![]f32 {
    const rows = try allocator.alloc(f32, input_rows * input_dim);
    @memset(rows, 0.0);
    if (input_rows == 0 or input_dim == 0) return rows;
    const prompt_ids = if (example.prompt_input_ids.len > 0) example.prompt_input_ids else example.input_ids;
    const response_ids = if (example.response_input_ids.len > 0) example.response_input_ids else example.labels;
    // Row 0: prompt tokens
    hashTokenIdsIntoRow(rows[0..input_dim], prompt_ids, 1.0);
    // Row 1: response tokens
    if (input_rows > 1) hashNonIgnoreTokenIdsIntoRow(rows[input_dim .. input_dim * 2], response_ids, 0.8);
    // Row 2: combined prompt + response
    if (input_rows > 2) {
        hashTokenIdsIntoRow(rows[input_dim * 2 .. input_dim * 3], prompt_ids, 0.5);
        hashNonIgnoreTokenIdsIntoRow(rows[input_dim * 2 .. input_dim * 3], response_ids, 0.5);
    }
    // Row 3: causal transitions across supervised assistant tokens.
    if (input_rows > 3) {
        hashSupervisedTokenTransitionsIntoRow(
            rows[input_dim * 3 .. input_dim * 4],
            example.input_ids,
            example.labels,
            1.0,
        );
    }
    return rows;
}

fn hashTokenIdsIntoRow(row: []f32, ids: []const i32, scale: f32) void {
    if (row.len == 0) return;
    for (ids, 0..) |id, idx| {
        const id_bits: u32 = @bitCast(id);
        const hash_seed = (@as(u64, id_bits) *% 0x9E3779B185EBCA87) ^ (@as(u64, idx) *% 1315423911);
        const pos: usize = @intCast(hash_seed % row.len);
        row[pos] += scale;
    }
}

fn hashNonIgnoreTokenIdsIntoRow(row: []f32, ids: []const i32, scale: f32) void {
    if (row.len == 0) return;
    var idx: usize = 0;
    for (ids) |id| {
        if (id == -100) continue;
        const id_bits: u32 = @bitCast(id);
        const hash_seed = (@as(u64, id_bits) *% 0x9E3779B185EBCA87) ^ (@as(u64, idx) *% 1315423911);
        const pos: usize = @intCast(hash_seed % row.len);
        row[pos] += scale;
        idx += 1;
    }
}

fn hashSupervisedTokenTransitionsIntoRow(
    row: []f32,
    input_ids: []const i32,
    labels: []const i32,
    scale: f32,
) void {
    if (row.len == 0 or input_ids.len < 2 or labels.len != input_ids.len) return;
    var transition_idx: usize = 0;
    var i: usize = 1;
    while (i < input_ids.len) : (i += 1) {
        if (labels[i] == -100) continue;
        const prev_bits: u32 = @bitCast(input_ids[i - 1]);
        const next_bits: u32 = @bitCast(input_ids[i]);
        const hash_seed = (@as(u64, prev_bits) *% 0x9E3779B185EBCA87) ^
            (@as(u64, next_bits) *% 0xC2B2AE3D27D4EB4F) ^
            (@as(u64, transition_idx) *% 0x165667B19E3779F9);
        const pos: usize = @intCast(hash_seed % row.len);
        row[pos] += scale;
        transition_idx += 1;
    }
}

const SelectedGemmaMessages = struct {
    messages: []gemma_chat_data.Message,
    turns_dropped_from_left: usize,
};

fn selectRenderableGemmaMessageWindow(
    allocator: std.mem.Allocator,
    tok: anytype,
    example: gemma_chat_data.Example,
    max_seq_len: usize,
) !SelectedGemmaMessages {
    if (example.messages.len == 0) {
        return .{ .messages = try allocator.alloc(gemma_chat_data.Message, 0), .turns_dropped_from_left = 0 };
    }

    const last_assistant_idx = findLastAssistantMessageIndex(example.messages) orelse example.messages.len - 1;
    var start_idx: usize = 0;
    while (start_idx <= last_assistant_idx) : (start_idx += 1) {
        const window = example.messages[start_idx..];
        if (!containsAssistantMessage(window)) continue;
        if (try renderedGemmaMessagesFitWithinBudget(allocator, tok, window, max_seq_len)) {
            return .{
                .messages = try allocator.dupe(gemma_chat_data.Message, window),
                .turns_dropped_from_left = start_idx,
            };
        }
    }

    const fallback_start = if (last_assistant_idx < example.messages.len) last_assistant_idx else example.messages.len - 1;
    return .{
        .messages = try allocator.dupe(gemma_chat_data.Message, example.messages[fallback_start..]),
        .turns_dropped_from_left = fallback_start,
    };
}

fn renderedGemmaMessagesFitWithinBudget(
    allocator: std.mem.Allocator,
    tok: anytype,
    messages: []const gemma_chat_data.Message,
    max_seq_len: usize,
) !bool {
    const render_messages = try allocator.alloc(chat_template.Message, messages.len);
    defer allocator.free(render_messages);

    var tool_call_json_bufs = try allocator.alloc(?[]u8, messages.len);
    defer allocator.free(tool_call_json_bufs);
    @memset(tool_call_json_bufs, null);
    defer {
        for (tool_call_json_bufs) |maybe| {
            if (maybe) |buf| allocator.free(buf);
        }
    }

    for (messages, 0..) |msg, idx| {
        if (msg.tool_calls.len > 0) {
            tool_call_json_bufs[idx] = try stringifyToolCalls(allocator, msg.tool_calls);
        }
        render_messages[idx] = .{
            .role = switch (msg.role) {
                .system => .system,
                .user => .user,
                .assistant => .assistant,
                .tool => .tool,
            },
            .content = msg.content,
            .name = msg.name,
            .tool_call_id = msg.tool_call_id,
            .tool_calls_json = tool_call_json_bufs[idx],
        };
    }

    var rendered = try chat_template.render(allocator, .gemma, render_messages, .{});
    defer rendered.deinit();
    const encoded = try tok.encode(allocator, rendered.text);
    defer allocator.free(encoded);
    return encoded.len <= max_seq_len;
}

fn containsAssistantMessage(messages: []const gemma_chat_data.Message) bool {
    for (messages) |msg| {
        if (msg.role == .assistant) return true;
    }
    return false;
}

fn findLastAssistantMessageIndex(messages: []const gemma_chat_data.Message) ?usize {
    var idx = messages.len;
    while (idx > 0) {
        idx -= 1;
        if (messages[idx].role == .assistant) return idx;
    }
    return null;
}

fn buildProbeVector(allocator: std.mem.Allocator, layer_name: []const u8, output_dim: usize) ![]f32 {
    const probe = try allocator.alloc(f32, output_dim);
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(layer_name);
    const base = hasher.final();
    for (probe, 0..) |*value, idx| {
        const angle: f32 = @floatFromInt((base % 997) + idx + 1);
        value.* = @sin(angle * 0.017) * 0.5 + @cos(angle * 0.009) * 0.5;
    }
    return probe;
}

fn exampleTarget(_: *const PreparedExampleInput) f64 {
    return 1.0;
}

fn layerMatchesScope(layer_base_tensor_name: []const u8, layer_name: ?[]const u8) bool {
    const selector = layer_name orelse return true;
    if (parseLayerSelectorIndex(selector)) |want_idx| {
        return parseGemma4LayerIndex(layer_base_tensor_name) == want_idx;
    }
    return std.mem.indexOf(u8, layer_base_tensor_name, selector) != null;
}

fn parseGemma4LayerIndex(tensor_name: []const u8) ?usize {
    if (parseGemma4TextLayerTensorName(tensor_name)) |parsed| return parsed.layer_index;

    // Preserve adapter/checkpoint compatibility with names that add an outer
    // PEFT prefix (for example `base_model.model.model.layers.*`). Target
    // inventory discovery itself uses the stricter text-root parser above.
    for ([_][]const u8{ "model.layers.", "blk." }) |prefix| {
        const prefix_index = std.mem.indexOf(u8, tensor_name, prefix) orelse continue;
        const digits = tensor_name[prefix_index + prefix.len ..];
        var end: usize = 0;
        while (end < digits.len and std.ascii.isDigit(digits[end])) : (end += 1) {}
        if (end == 0) continue;
        return std.fmt.parseUnsigned(usize, digits[0..end], 10) catch null;
    }
    return null;
}

fn keepTensorInRecursiveCompressedBase(tensor_name: []const u8, shared_block_size: usize) bool {
    const layer_idx = parseGemma4LayerIndex(tensor_name) orelse return true;
    return layer_idx < shared_block_size;
}

fn parseLayerSelectorIndex(selector: []const u8) ?usize {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, selector, prefix)) return null;
    const digits = selector[prefix.len..];
    var end: usize = 0;
    while (end < digits.len and std.ascii.isDigit(digits[end])) : (end += 1) {}
    if (end == 0) return null;
    if (end != digits.len) return null;
    return std.fmt.parseUnsigned(usize, digits, 10) catch null;
}

fn warmupAdjustedLR(base_lr: f32, step: u64, warmup_steps: u32) f32 {
    if (warmup_steps == 0 or step >= warmup_steps) return base_lr;
    return base_lr * @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(warmup_steps));
}

fn applyAdamWInPlace(params: []f32, grads: []const f32, m: []f32, v: []f32, step: u64, lr: f32) void {
    const t: f32 = @floatFromInt(step);
    const beta1: f32 = 0.9;
    const beta2: f32 = 0.999;
    const eps: f32 = 1e-8;
    const wd: f32 = 0.01;
    const bc1 = 1.0 - std.math.pow(f32, beta1, t);
    const bc2 = 1.0 - std.math.pow(f32, beta2, t);
    for (params, grads, m, v) |*p, g, *mi, *vi| {
        mi.* = beta1 * mi.* + (1.0 - beta1) * g;
        vi.* = beta2 * vi.* + (1.0 - beta2) * g * g;
        p.* -= lr * (mi.* / bc1 / (@sqrt(vi.* / bc2) + eps) + wd * p.*);
    }
}

fn applyScheduleFreeInPlace(params: []f32, grads: []const f32, z: []f32, v: []f32, step: u64, lr: f32) void {
    const t: f32 = @floatFromInt(step);
    const beta2: f32 = 0.999;
    const epsilon: f32 = 1e-8;
    const weight_decay: f32 = 0.01;
    const c = @min(@as(f32, 0.9), 1.0 / t);
    for (params, grads, z, v) |*x, g, *zi, *vi| {
        vi.* = beta2 * vi.* + (1.0 - beta2) * g * g;
        const v_hat = vi.* / (1.0 - std.math.pow(f32, beta2, t));
        zi.* = zi.* - lr * g / (@sqrt(v_hat) + epsilon) - lr * weight_decay * zi.*;
        x.* = (1.0 - c) * x.* + c * zi.*;
    }
}

test "hash supervised token transitions ignores masked labels" {
    var row: [32]f32 = [_]f32{0} ** 32;
    const input_ids = [_]i32{ 10, 11, 12, 13, 14 };
    const labels = [_]i32{ -100, -100, 12, -100, 14 };
    hashSupervisedTokenTransitionsIntoRow(&row, &input_ids, &labels, 1.0);

    var non_zero: usize = 0;
    var total: f32 = 0;
    for (row) |value| {
        if (value != 0) non_zero += 1;
        total += value;
    }
    try std.testing.expect(non_zero > 0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), total, 1e-6);
}

test "selectRenderableGemmaMessageWindow drops oldest turns first" {
    const allocator = std.testing.allocator;
    var messages = [_]gemma_chat_data.Message{
        .{ .role = .system, .content = "system" },
        .{ .role = .user, .content = "first user turn with enough words to overflow budget" },
        .{ .role = .assistant, .content = "first assistant answer with enough words to overflow budget" },
        .{ .role = .user, .content = "keep me" },
        .{ .role = .assistant, .content = "keep me too" },
    };
    const example = gemma_chat_data.Example{ .messages = messages[0..] };
    const tok = TestWhitespaceTokenizer{};

    const selected = try selectRenderableGemmaMessageWindow(allocator, tok, example, 12);
    defer allocator.free(selected.messages);

    try std.testing.expectEqual(@as(usize, 3), selected.turns_dropped_from_left);
    try std.testing.expectEqual(@as(usize, 2), selected.messages.len);
    try std.testing.expectEqual(gemma_chat_data.Role.user, selected.messages[0].role);
    try std.testing.expectEqual(gemma_chat_data.Role.assistant, selected.messages[1].role);
}

test "Gemma4 no-offset tokenizer supervises every assistant turn" {
    const allocator = std.testing.allocator;
    const tok = TestNoOffsetGenerationTokenizer{};
    const rendered_text = "uAAxxBB";
    const spans = [_]chat_template.AssistantSpan{
        .{ .start = 1, .end = 3 },
        .{ .start = 5, .end = rendered_text.len },
    };
    var encoded = try tok.encodeForGenerationConfigured(allocator, rendered_text, 16, false);
    defer encoded.deinit();
    const labels = try makeCompletionLabelsWithoutOffsets(
        allocator,
        tok,
        rendered_text,
        &spans,
        encoded.ids,
        encoded.attention_mask,
        16,
    );
    defer allocator.free(labels);

    // The fake causal tokenizer emits one token per byte, then padding. Both
    // A bytes and both B bytes are supervised, with interleaved user bytes
    // and every padding position masked.
    try std.testing.expectEqualSlices(
        i32,
        &.{ -100, 'A', 'A', -100, -100, 'B', 'B', -100, -100, -100, -100, -100, -100, -100, -100, -100 },
        labels,
    );
}

test "Gemma4 prepared causal rows omit tokenizer padding" {
    const allocator = std.testing.allocator;
    const tok = TestNoOffsetGenerationTokenizer{};
    var messages = [_]gemma_chat_data.Message{
        .{ .role = .user, .content = "u" },
        .{ .role = .assistant, .content = "a" },
    };
    var prepared = try tokenizeChatExample(
        allocator,
        tok,
        .{ .messages = messages[0..] },
        256,
    );
    defer freePreparedExampleInput(allocator, &prepared);

    try std.testing.expect(prepared.num_input_tokens > 0);
    try std.testing.expect(prepared.num_input_tokens < 256);
    try std.testing.expectEqual(prepared.num_input_tokens, prepared.input_ids.len);
    try std.testing.expectEqual(prepared.num_input_tokens, prepared.labels.len);
    try std.testing.expectEqual(prepared.num_supervised_tokens, prepared.response_input_ids.len);
}

test "expandMessageMediaMarkers expands image and audio runs" {
    const allocator = std.testing.allocator;
    var image_idx: usize = 0;
    var audio_idx: usize = 0;
    const expanded = try expandMessageMediaMarkers(allocator, "look <|image|> then <|audio|>", &.{2}, &image_idx, &.{3}, &audio_idx);
    defer allocator.free(expanded);
    try std.testing.expectEqualStrings("look <|image><|image|><|image|><image|> then <|audio><|audio|><|audio|><|audio|><audio|>", expanded);
    try std.testing.expectEqual(@as(usize, 1), image_idx);
    try std.testing.expectEqual(@as(usize, 1), audio_idx);
}

test "validateMultimodalExampleShape catches placeholder mismatch" {
    const allocator = std.testing.allocator;
    const messages = try allocator.alloc(gemma_chat_data.Message, 1);
    defer allocator.free(messages);
    messages[0] = .{ .role = .user, .content = "missing markers" };
    const ex = gemma_chat_data.Example{
        .messages = messages,
        .image_paths = &.{"img.png"},
    };
    try std.testing.expectError(error.ImagePlaceholderCountMismatch, validateMultimodalExampleShape(ex));
}

test "normalizePreparedSchemaVersion accepts supported versions" {
    try std.testing.expectEqualStrings(prepared_schema_v2, try normalizePreparedSchemaVersion(prepared_schema_v2));
    try std.testing.expectEqualStrings(prepared_schema_v3, try normalizePreparedSchemaVersion(prepared_schema_v3));
    try std.testing.expectEqualStrings(prepared_schema_v4, try normalizePreparedSchemaVersion(prepared_schema_v4));
    try std.testing.expectEqualStrings(prepared_schema_v5, try normalizePreparedSchemaVersion(prepared_schema_v5));
    try std.testing.expectEqualStrings(prepared_schema_v6, try normalizePreparedSchemaVersion(prepared_schema_v6));
}

test "normalizePreparedSchemaVersion rejects unknown version" {
    try std.testing.expectError(error.UnsupportedPreparedInputsSchema, normalizePreparedSchemaVersion("gemma4_prepared/v999"));
}

test "Gemma4 training sequence admission is bounded before graph construction" {
    try std.testing.expectEqual(@as(u32, 1), try validateTrainingSequenceLength(1, 4096));
    try std.testing.expectEqual(@as(u32, max_training_seq_len), try validateTrainingSequenceLength(max_training_seq_len, 4096));
    try std.testing.expectError(error.InvalidPreparedSequenceLength, validateTrainingSequenceLength(0, 4096));
    try std.testing.expectError(error.InvalidModelContextLength, validateTrainingSequenceLength(1, 0));
    try std.testing.expectError(error.PreparedSequenceExceedsModelContext, validateTrainingSequenceLength(65, 64));
    try std.testing.expectError(error.PreparedSequenceExceedsTrainingLimit, validateTrainingSequenceLength(max_training_seq_len + 1, 4096));
    if (std.math.maxInt(usize) > std.math.maxInt(u32)) {
        try std.testing.expectError(error.PreparedSequenceLengthOverflow, validateTrainingSequenceLength(@as(usize, std.math.maxInt(u32)) + 1, std.math.maxInt(u32)));
    }
}

test "prepared v4 integrity detects content mutation and heldout overlap" {
    const allocator = std.testing.allocator;
    var train_ids = [_]i32{ 1, 2, 3 };
    var train_labels = [_]i32{ -100, -100, 3 };
    var eval_ids = [_]i32{ 4, 5, 6 };
    var eval_labels = [_]i32{ -100, -100, 6 };
    var train_examples = [_]PreparedExampleInput{.{
        .mode = .instruction,
        .prompt_input_ids = train_ids[0..2],
        .response_input_ids = train_ids[2..],
        .num_prompt_tokens = 2,
        .num_response_tokens = 1,
        .input_ids = &train_ids,
        .labels = &train_labels,
        .num_input_tokens = 3,
        .num_supervised_tokens = 1,
    }};
    var eval_examples = [_]PreparedExampleInput{.{
        .mode = .instruction,
        .prompt_input_ids = eval_ids[0..2],
        .response_input_ids = eval_ids[2..],
        .num_prompt_tokens = 2,
        .num_response_tokens = 1,
        .input_ids = &eval_ids,
        .labels = &eval_labels,
        .num_input_tokens = 3,
        .num_supervised_tokens = 1,
    }};
    const digest = try fingerprintPreparedExamplesAlloc(allocator, &train_examples);
    defer allocator.free(digest);
    const valid = PreparedInputsSummary{
        .artifact_family_version = artifact_family_version,
        .model_dir = "/model",
        .schema_version = prepared_schema_v4,
        .base_model_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        .tokenizer_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        .chat_template_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        .prepared_examples_sha256 = digest,
        .max_examples = 1,
        .examples_seen = 1,
        .max_seq_len = 3,
        .max_prompt_tokens = 2,
        .max_response_tokens = 1,
        .max_input_tokens = 3,
        .max_supervised_tokens = 1,
        .examples = &train_examples,
    };
    try validatePreparedArtifactIntegrity(allocator, valid);
    _ = try validatePreparedSequenceAdmission(valid, 16);
    try validatePreparedEvalDisjoint(allocator, &train_examples, &eval_examples);
    try std.testing.expectError(error.TrainingEvaluationOverlap, validatePreparedEvalDisjoint(allocator, &train_examples, &train_examples));

    train_ids[2] = 9;
    try std.testing.expectError(error.PreparedInputsFingerprintMismatch, validatePreparedArtifactIntegrity(allocator, valid));
}

test "prepared v5 binds source provenance groups and vocabulary" {
    const allocator = std.testing.allocator;
    var train_ids = [_]i32{ 1, 2 };
    var train_labels = [_]i32{ -100, 2 };
    var eval_ids = [_]i32{ 3, 4 };
    var eval_labels = [_]i32{ -100, 4 };
    var train_examples = [_]PreparedExampleInput{.{
        .mode = .instruction,
        .prompt_input_ids = train_ids[0..1],
        .response_input_ids = train_ids[1..],
        .num_prompt_tokens = 1,
        .num_response_tokens = 1,
        .input_ids = &train_ids,
        .labels = &train_labels,
        .num_input_tokens = 2,
        .num_supervised_tokens = 1,
        .turn_count = 2,
        .source_id = "train-1",
        .source_group_id = "group-train",
        .source_record_sha256 = "1111111111111111111111111111111111111111111111111111111111111111",
        .rendered_chat_sha256 = "2222222222222222222222222222222222222222222222222222222222222222",
    }};
    var eval_examples = [_]PreparedExampleInput{.{
        .mode = .instruction,
        .prompt_input_ids = eval_ids[0..1],
        .response_input_ids = eval_ids[1..],
        .num_prompt_tokens = 1,
        .num_response_tokens = 1,
        .input_ids = &eval_ids,
        .labels = &eval_labels,
        .num_input_tokens = 2,
        .num_supervised_tokens = 1,
        .turn_count = 2,
        .source_id = "eval-1",
        .source_group_id = "group-eval",
        .source_record_sha256 = "3333333333333333333333333333333333333333333333333333333333333333",
        .rendered_chat_sha256 = "4444444444444444444444444444444444444444444444444444444444444444",
    }};
    const digest = try fingerprintPreparedExamplesV2Alloc(allocator, &train_examples);
    defer allocator.free(digest);
    const summary = PreparedInputsSummary{
        .artifact_family_version = artifact_family_version,
        .model_dir = "/model",
        .schema_version = prepared_schema_v5,
        .base_model_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        .tokenizer_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        .chat_template_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        .prepared_examples_sha256 = digest,
        .source_dataset_path = "/dataset",
        .source_dataset_sha256 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        .source_split = "train",
        .source_revision = "revision-1",
        .max_examples = 1,
        .examples_seen = 1,
        .max_seq_len = 2,
        .max_prompt_tokens = 1,
        .max_response_tokens = 1,
        .max_input_tokens = 2,
        .max_supervised_tokens = 1,
        .examples = &train_examples,
    };
    try validatePreparedArtifactIntegrity(allocator, summary);
    try std.testing.expectError(
        error.PreparedSourceProvenanceRequired,
        validatePreparedSourceDatasetProvenance(allocator, summary),
    );
    var relabeled_v6 = summary;
    relabeled_v6.schema_version = prepared_schema_v6;
    try std.testing.expectError(
        error.PreparedInputsFingerprintMismatch,
        validatePreparedArtifactIntegrity(allocator, relabeled_v6),
    );
    _ = try validatePreparedSequenceAdmission(summary, 16);
    try validatePreparedVocabularyAdmission(summary, 8);
    try validatePreparedEvalDisjoint(allocator, &train_examples, &eval_examples);

    eval_examples[0].source_group_id = "group-train";
    try std.testing.expectError(
        error.TrainingEvaluationGroupOverlap,
        validatePreparedEvalDisjoint(allocator, &train_examples, &eval_examples),
    );
    train_labels[1] = -1;
    try std.testing.expectError(error.PreparedLabelOutOfRange, validatePreparedVocabularyAdmission(summary, 8));
}

test "prepared v6 revalidates raw dataset bytes at admission" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "train.jsonl", .data = "{\"id\":\"row-1\"}\n" });
    const dataset_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "train.jsonl" });
    defer allocator.free(dataset_path);
    const source_digest = try fingerprintGemmaDatasetSourceAlloc(allocator, dataset_path, null);
    defer allocator.free(source_digest);
    var no_examples: [0]PreparedExampleInput = .{};
    const summary = PreparedInputsSummary{
        .artifact_family_version = artifact_family_version,
        .model_dir = "/model",
        .schema_version = prepared_schema_v6,
        .source_dataset_path = dataset_path,
        .source_dataset_sha256 = source_digest,
        .source_revision = source_digest,
        .max_examples = 0,
        .examples_seen = 0,
        .max_seq_len = 1,
        .examples = &no_examples,
    };
    try validatePreparedSourceDatasetProvenance(allocator, summary);

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "train.jsonl", .data = "{\"id\":\"row-2\"}\n" });
    try std.testing.expectError(
        error.PreparedSourceDatasetMismatch,
        validatePreparedSourceDatasetProvenance(allocator, summary),
    );
}

test "prepared v6 rows must belong to the bound raw source split" {
    const allocator = std.testing.allocator;
    var messages = [_]gemma_chat_data.Message{
        .{ .role = .user, .content = "question" },
        .{ .role = .assistant, .content = "answer" },
    };
    const source_examples = [_]gemma_chat_data.Example{.{
        .id = "row-1",
        .messages = &messages,
        .metadata = .{ .source = "fixture", .group_id = "group-1" },
    }};
    const record_digest = try fingerprintGemmaChatSourceRecordAlloc(allocator, source_examples[0]);
    defer allocator.free(record_digest);
    var prompt_ids = [_]i32{1};
    var response_ids = [_]i32{2};
    var prepared = [_]PreparedExampleInput{.{
        .mode = .instruction,
        .prompt_input_ids = &prompt_ids,
        .response_input_ids = &response_ids,
        .num_prompt_tokens = 1,
        .num_response_tokens = 1,
        .source_id = "row-1",
        .source_group_id = "group-1",
        .source_name = "fixture",
        .source_record_sha256 = record_digest,
    }};
    try validatePreparedExamplesBelongToSource(allocator, &prepared, &source_examples);

    prepared[0].source_group_id = "forged-group";
    try std.testing.expectError(
        error.PreparedSourceRecordMismatch,
        validatePreparedExamplesBelongToSource(allocator, &prepared, &source_examples),
    );
}

const TestWhitespaceTokenizer = struct {
    const EncodeResult = struct {
        ids: []i32,

        fn deinit(self: *EncodeResult) void {
            std.testing.allocator.free(self.ids);
            self.* = undefined;
        }
    };
    fn encode(_: TestWhitespaceTokenizer, allocator: std.mem.Allocator, text: []const u8) ![]i32 {
        var tokens = std.ArrayList(i32).empty;
        errdefer tokens.deinit(allocator);
        var it = std.mem.tokenizeAny(u8, text, " \n\t\r");
        var idx: i32 = 0;
        while (it.next() != null) : (idx += 1) {
            try tokens.append(allocator, idx + 1);
        }
        return tokens.toOwnedSlice(allocator);
    }
};

const TestNoOffsetGenerationTokenizer = struct {
    const EncodeResult = struct {
        allocator: std.mem.Allocator,
        ids: []i32,
        attention_mask: []i32,
        offsets: ?[]const [2]u32 = null,

        fn deinit(self: *EncodeResult) void {
            self.allocator.free(self.ids);
            self.allocator.free(self.attention_mask);
            self.* = undefined;
        }
    };

    fn encode(
        _: TestNoOffsetGenerationTokenizer,
        allocator: std.mem.Allocator,
        text: []const u8,
    ) ![]i32 {
        const ids = try allocator.alloc(i32, text.len);
        for (text, 0..) |byte, idx| ids[idx] = @intCast(byte);
        return ids;
    }

    fn encodeForGenerationConfigured(
        _: TestNoOffsetGenerationTokenizer,
        allocator: std.mem.Allocator,
        text: []const u8,
        max_seq_len: usize,
        add_bos_token: bool,
    ) !EncodeResult {
        if (add_bos_token) return error.TestUnexpectedBos;
        if (text.len > max_seq_len) return error.TestSequenceTooLong;
        const ids = try allocator.alloc(i32, max_seq_len);
        errdefer allocator.free(ids);
        const attention_mask = try allocator.alloc(i32, max_seq_len);
        errdefer allocator.free(attention_mask);
        @memset(ids, 0);
        @memset(attention_mask, 0);
        for (text, 0..) |byte, idx| {
            ids[idx] = @intCast(byte);
            attention_mask[idx] = 1;
        }
        return .{
            .allocator = allocator,
            .ids = ids,
            .attention_mask = attention_mask,
        };
    }
};

// ---------------------------------------------------------------------------
// Safetensors I/O helpers
// ---------------------------------------------------------------------------

const WriteTensorF32 = struct {
    name: []const u8,
    shape: []const usize,
    data: []const f32,
};

const WriteTensorRaw = struct {
    name: []const u8,
    dtype: tensor_mod.DType,
    shape: []const i64,
    raw_bytes: []const u8,
};

const BootstrapTargetSelection = union(enum) {
    gemma4: Gemma4LoRATargetPreset,
    legacy: peft.TargetPreset,
    explicit: []const []const u8,
};

const Gemma4TextLinearKind = enum {
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    gate_proj,
    up_proj,
    down_proj,
    ple_input_gate,
    ple_projection,
    ple_model_projection,
};

const Gemma4TextLayerTensorName = struct {
    layer_index: usize,
    suffix: []const u8,
    is_gguf: bool,
};

fn resolveBootstrapTargetSelection(options: BootstrapOptions) !BootstrapTargetSelection {
    const selection_count = @intFromBool(options.target_modules != null) +
        @intFromBool(options.gemma4_target_preset != null) +
        @intFromBool(options.target_preset != null);
    if (selection_count > 1) return error.ConflictingLoRATargetSelection;
    if (options.target_modules) |modules| return .{ .explicit = modules };
    if (options.gemma4_target_preset) |preset| return .{ .gemma4 = preset };
    if (options.target_preset) |preset| return .{ .legacy = preset };
    return .{ .gemma4 = .text_all_linear };
}

pub fn gemma4LoRATargetPresetName(preset: Gemma4LoRATargetPreset) []const u8 {
    return switch (preset) {
        .peft_qv => "peft-qv",
        .text_all_linear => "text-all-linear",
    };
}

fn targetSelectionPresetName(selection: BootstrapTargetSelection) ?[]const u8 {
    return switch (selection) {
        .gemma4 => |preset| gemma4LoRATargetPresetName(preset),
        .legacy => |preset| targetPresetName(preset),
        .explicit => null,
    };
}

fn inferLoRATargetTensorsForModelInput(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    weights_path: []const u8,
    selection: BootstrapTargetSelection,
) ![]LoRATargetTensor {
    const config = try session_factory.loadGptConfigMetadataFromModelDir(allocator, model_dir);

    const discovered = if (!std.mem.endsWith(u8, weights_path, ".gguf"))
        try inferLoRATargetTensors(weights_path, allocator, selection)
    else blk: {
        break :blk try inferGgufLoRATargetTensors(allocator, weights_path, selection);
    };
    defer freeLoRATargetTensors(allocator, discovered);

    var targets: std.ArrayListUnmanaged(LoRATargetTensor) = .empty;
    errdefer {
        for (targets.items) |item| {
            allocator.free(item.tensor_name);
            allocator.free(item.module_name);
        }
        targets.deinit(allocator);
    }
    for (discovered) |item| {
        if (!gemma4TargetParticipatesInGraph(item.tensor_name, config)) continue;
        try targets.append(allocator, .{
            .tensor_name = try allocator.dupe(u8, item.tensor_name),
            .module_name = try allocator.dupe(u8, item.module_name),
            .output_dim = item.output_dim,
            .input_dim = item.input_dim,
        });
    }
    try validateExplicitTargetSelection(selection, targets.items);
    return targets.toOwnedSlice(allocator);
}

fn inferGgufLoRATargetTensors(
    allocator: std.mem.Allocator,
    weights_path: []const u8,
    selection: BootstrapTargetSelection,
) ![]LoRATargetTensor {
    var targets: std.ArrayListUnmanaged(LoRATargetTensor) = .empty;
    errdefer {
        for (targets.items) |item| {
            allocator.free(item.tensor_name);
            allocator.free(item.module_name);
        }
        targets.deinit(allocator);
    }

    // GGUF target discovery only needs tensor metadata. Prefer libc's allocator
    // when available to avoid Debug allocator overhead on large headers.
    const gguf_allocator = platform.allocator.processAllocator(allocator);
    var access = try tensor_access.GgufAccess.initAbsolute(gguf_allocator, weights_path);
    defer access.tensorAccess().deinit();

    for (access.store.parsed.tensors) |tensor| {
        const tensor_name = tensor.name;
        const module_name = moduleNameForTensorWithSelection(tensor_name, selection) orelse continue;
        if (!targetMatchesSelection(tensor_name, selection)) continue;
        if (tensor.dimensions.len != 2) continue;
        try targets.append(allocator, .{
            .tensor_name = try allocator.dupe(u8, tensor_name),
            .module_name = try allocator.dupe(u8, module_name),
            .output_dim = @intCast(tensor.dimensions[1]),
            .input_dim = @intCast(tensor.dimensions[0]),
        });
    }

    std.mem.sort(LoRATargetTensor, targets.items, {}, struct {
        fn lt(_: void, a: LoRATargetTensor, b: LoRATargetTensor) bool {
            return std.mem.lessThan(u8, a.tensor_name, b.tensor_name);
        }
    }.lt);
    try validateExplicitTargetSelection(selection, targets.items);
    return targets.toOwnedSlice(allocator);
}

fn gemma4TargetParticipatesInGraph(tensor_name: []const u8, config: gpt_model.Config) bool {
    const parsed = parseGemma4TextLayerTensorName(tensor_name) orelse return true;
    if (parsed.layer_index >= config.num_hidden_layers) return false;

    const kind = gemma4TextLinearKindForTensor(tensor_name) orelse return true;
    if ((kind == .k_proj or kind == .v_proj) and config.layerSharesKv(parsed.layer_index)) return false;
    if (kind == .v_proj and config.layerOmitsVProj(parsed.layer_index)) return false;
    return true;
}

fn inferLoRATargetTensors(
    checkpoint_path: []const u8,
    allocator: std.mem.Allocator,
    selection: BootstrapTargetSelection,
) ![]LoRATargetTensor {
    var access = try openTensorAccessForFile(allocator, checkpoint_path);
    defer access.deinit();
    const names = try access.listNames(allocator);
    defer allocator.free(names);

    var targets: std.ArrayListUnmanaged(LoRATargetTensor) = .empty;
    errdefer {
        for (targets.items) |item| {
            allocator.free(item.tensor_name);
            allocator.free(item.module_name);
        }
        targets.deinit(allocator);
    }

    for (names) |tensor_name| {
        const module_name = moduleNameForTensorWithSelection(tensor_name, selection) orelse continue;
        if (!targetMatchesSelection(tensor_name, selection)) continue;
        var record = try access.getRecord(allocator, tensor_name);
        defer record.deinit();
        if (record.descriptor.shape.len != 2) continue;
        try targets.append(allocator, .{
            .tensor_name = try allocator.dupe(u8, tensor_name),
            .module_name = try allocator.dupe(u8, module_name),
            .output_dim = @intCast(record.descriptor.shape[0]),
            .input_dim = @intCast(record.descriptor.shape[1]),
        });
    }

    std.mem.sort(LoRATargetTensor, targets.items, {}, struct {
        fn lt(_: void, a: LoRATargetTensor, b: LoRATargetTensor) bool {
            return std.mem.lessThan(u8, a.tensor_name, b.tensor_name);
        }
    }.lt);
    try validateExplicitTargetSelection(selection, targets.items);
    return targets.toOwnedSlice(allocator);
}

fn moduleNameForTensor(tensor_name: []const u8) ?[]const u8 {
    const kind = gemma4TextLinearKindForTensor(tensor_name) orelse return null;
    return gemma4TextLinearKindName(kind);
}

fn moduleNameForTensorWithSelection(tensor_name: []const u8, selection: BootstrapTargetSelection) ?[]const u8 {
    switch (selection) {
        .legacy => |preset| if (preset == .moe_experts and isGemma4TextLayerTensor(tensor_name) and peft.matchesMoEExpertTensor(tensor_name)) return "moe_expert",
        else => {},
    }
    return moduleNameForTensor(tensor_name);
}

fn targetMatchesSelection(tensor_name: []const u8, selection: BootstrapTargetSelection) bool {
    const maybe_kind = gemma4TextLinearKindForTensor(tensor_name);
    return switch (selection) {
        .gemma4 => |preset| if (maybe_kind) |kind| switch (preset) {
            .peft_qv => kind == .q_proj or kind == .v_proj,
            .text_all_linear => true,
        } else false,
        .legacy => |preset| switch (preset) {
            .all_linear => maybe_kind != null,
            .attention_only => if (maybe_kind) |kind| isAttentionLinearKind(kind) else false,
            .mlp_only => if (maybe_kind) |kind| isMlpLinearKind(kind) else false,
            .moe_experts => isGemma4TextLayerTensor(tensor_name) and peft.matchesMoEExpertTensor(tensor_name),
        },
        .explicit => |requested_modules| if (maybe_kind) |kind| blk: {
            for (requested_modules) |requested| {
                if (explicitTargetMatchesTensor(requested, tensor_name, kind)) break :blk true;
            }
            break :blk false;
        } else false,
    };
}

fn gemma4TextLinearKindForTensor(tensor_name: []const u8) ?Gemma4TextLinearKind {
    var canonical_buf: [256]u8 = undefined;
    const canonical_name = session_factory.canonicalizeGemma4LegacyWeightKey(tensor_name, &canonical_buf) orelse tensor_name;
    if (isGemma4PleModelProjectionTensor(canonical_name)) return .ple_model_projection;

    const parsed = parseGemma4TextLayerTensorName(canonical_name) orelse return null;
    const suffix = parsed.suffix;
    if (std.mem.eql(u8, suffix, "self_attn.q_proj.weight") or parsed.is_gguf and std.mem.eql(u8, suffix, "attn_q.weight")) return .q_proj;
    if (std.mem.eql(u8, suffix, "self_attn.k_proj.weight") or parsed.is_gguf and std.mem.eql(u8, suffix, "attn_k.weight")) return .k_proj;
    if (std.mem.eql(u8, suffix, "self_attn.v_proj.weight") or parsed.is_gguf and std.mem.eql(u8, suffix, "attn_v.weight")) return .v_proj;
    if (std.mem.eql(u8, suffix, "self_attn.o_proj.weight") or parsed.is_gguf and std.mem.eql(u8, suffix, "attn_output.weight")) return .o_proj;
    if (std.mem.eql(u8, suffix, "mlp.gate_proj.weight") or parsed.is_gguf and std.mem.eql(u8, suffix, "ffn_gate.weight")) return .gate_proj;
    if (std.mem.eql(u8, suffix, "mlp.up_proj.weight") or parsed.is_gguf and std.mem.eql(u8, suffix, "ffn_up.weight")) return .up_proj;
    if (std.mem.eql(u8, suffix, "mlp.down_proj.weight") or parsed.is_gguf and std.mem.eql(u8, suffix, "ffn_down.weight")) return .down_proj;
    if (std.mem.eql(u8, suffix, "per_layer_input.inp_gate.weight") or parsed.is_gguf and std.mem.eql(u8, suffix, "inp_gate.weight")) return .ple_input_gate;
    if (std.mem.eql(u8, suffix, "per_layer_input.proj.weight") or parsed.is_gguf and std.mem.eql(u8, suffix, "proj.weight")) return .ple_projection;
    return null;
}

fn gemma4TextLinearKindName(kind: Gemma4TextLinearKind) []const u8 {
    return switch (kind) {
        .q_proj => "q_proj",
        .k_proj => "k_proj",
        .v_proj => "v_proj",
        .o_proj => "o_proj",
        .gate_proj => "gate_proj",
        .up_proj => "up_proj",
        .down_proj => "down_proj",
        .ple_input_gate => "per_layer_input.inp_gate",
        .ple_projection => "per_layer_input.proj",
        .ple_model_projection => "per_layer_input.per_layer_model_proj",
    };
}

fn isAttentionLinearKind(kind: Gemma4TextLinearKind) bool {
    return switch (kind) {
        .q_proj, .k_proj, .v_proj, .o_proj => true,
        else => false,
    };
}

fn isMlpLinearKind(kind: Gemma4TextLinearKind) bool {
    return switch (kind) {
        .gate_proj, .up_proj, .down_proj => true,
        else => false,
    };
}

fn isGemma4PleModelProjectionTensor(tensor_name: []const u8) bool {
    const names = [_][]const u8{
        "per_layer_model_proj.weight",
        "model.per_layer_input.per_layer_model_proj.weight",
        "model.language_model.per_layer_input.per_layer_model_proj.weight",
        "language_model.model.per_layer_input.per_layer_model_proj.weight",
        "language_model.per_layer_input.per_layer_model_proj.weight",
        "vlm.model.language_model.per_layer_input.per_layer_model_proj.weight",
    };
    inline for (names) |name| {
        if (std.mem.eql(u8, tensor_name, name)) return true;
    }
    return false;
}

fn parseGemma4TextLayerTensorName(tensor_name: []const u8) ?Gemma4TextLayerTensorName {
    const normalized_prefixes = [_][]const u8{
        "model.language_model.layers.",
        "language_model.model.layers.",
        "language_model.layers.",
        "model.layers.",
        "layers.",
    };
    inline for (normalized_prefixes) |prefix| {
        if (parseGemma4TextLayerTensorAfterPrefix(tensor_name, prefix, false)) |parsed| return parsed;
    }
    return parseGemma4TextLayerTensorAfterPrefix(tensor_name, "blk.", true);
}

fn parseGemma4TextLayerTensorAfterPrefix(
    tensor_name: []const u8,
    prefix: []const u8,
    is_gguf: bool,
) ?Gemma4TextLayerTensorName {
    if (!std.mem.startsWith(u8, tensor_name, prefix)) return null;
    const rest = tensor_name[prefix.len..];
    const separator = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    if (separator == 0 or separator + 1 >= rest.len) return null;
    const layer_index = std.fmt.parseInt(usize, rest[0..separator], 10) catch return null;
    return .{
        .layer_index = layer_index,
        .suffix = rest[separator + 1 ..],
        .is_gguf = is_gguf,
    };
}

fn isGemma4TextLayerTensor(tensor_name: []const u8) bool {
    return parseGemma4TextLayerTensorName(tensor_name) != null;
}

fn explicitTargetMatchesTensor(
    requested: []const u8,
    tensor_name: []const u8,
    kind: Gemma4TextLinearKind,
) bool {
    if (std.mem.eql(u8, requested, gemma4TextLinearKindName(kind))) return true;
    if (std.mem.eql(u8, requested, tensor_name)) return true;
    if (tensorModulePath(tensor_name)) |module_path| {
        if (std.mem.eql(u8, requested, module_path)) return true;
    }
    var normalized_buf: [256]u8 = undefined;
    if (normalizedTargetModulePathInBuffer(tensor_name, &normalized_buf)) |normalized| {
        if (std.mem.eql(u8, requested, normalized)) return true;
    }
    return switch (kind) {
        .ple_input_gate => std.mem.eql(u8, requested, "inp_gate") or std.mem.eql(u8, requested, "per_layer_input_gate"),
        .ple_projection => std.mem.eql(u8, requested, "per_layer_projection"),
        .ple_model_projection => std.mem.eql(u8, requested, "per_layer_model_proj") or std.mem.eql(u8, requested, "per_layer_model_projection"),
        else => false,
    };
}

fn validateExplicitTargetSelection(
    selection: BootstrapTargetSelection,
    targets: []const LoRATargetTensor,
) !void {
    const requested_modules = switch (selection) {
        .explicit => |modules| modules,
        else => return,
    };
    if (requested_modules.len == 0) return error.NoLoRATargetTensorsResolved;
    for (requested_modules) |requested| {
        var found = false;
        for (targets) |target| {
            const kind = gemma4TextLinearKindForTensor(target.tensor_name) orelse continue;
            if (explicitTargetMatchesTensor(requested, target.tensor_name, kind)) {
                found = true;
                break;
            }
        }
        if (!found) return error.UnknownLoRATargetModule;
    }
}

fn resolvedTargetModulePaths(
    allocator: std.mem.Allocator,
    targets: []const LoRATargetTensor,
) ![][]const u8 {
    const paths = try allocator.alloc([]const u8, targets.len);
    errdefer allocator.free(paths);
    var built: usize = 0;
    errdefer for (paths[0..built]) |path| allocator.free(path);
    for (targets, 0..) |target, idx| {
        paths[idx] = try normalizedTargetModulePath(allocator, target.tensor_name);
        built += 1;
    }
    std.mem.sort([]const u8, paths, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    return paths;
}

fn normalizedTargetModulePath(allocator: std.mem.Allocator, tensor_name: []const u8) ![]const u8 {
    var buf: [256]u8 = undefined;
    const module_path = normalizedTargetModulePathInBuffer(tensor_name, &buf) orelse return error.InvalidLoRATargetTensorName;
    return allocator.dupe(u8, module_path);
}

fn canonicalAdapterBaseTensorName(allocator: std.mem.Allocator, source_tensor_name: []const u8) ![]u8 {
    const module_path = try normalizedTargetModulePath(allocator, source_tensor_name);
    defer allocator.free(module_path);
    return std.fmt.allocPrint(allocator, "{s}.weight", .{module_path});
}

fn sourceTensorNameForCanonicalAdapterBase(
    source_tensor_names: []const []const u8,
    canonical_base_tensor_name: []const u8,
) ?[]const u8 {
    for (source_tensor_names) |source_name| {
        if (std.mem.eql(u8, source_name, canonical_base_tensor_name)) return source_name;
    }

    const canonical_module_path = tensorModulePath(canonical_base_tensor_name) orelse return null;
    for (source_tensor_names) |source_name| {
        var normalized_buf: [256]u8 = undefined;
        const normalized_module_path = normalizedTargetModulePathInBuffer(source_name, &normalized_buf) orelse continue;
        if (std.mem.eql(u8, normalized_module_path, canonical_module_path)) return source_name;
    }
    return null;
}

fn normalizedTargetModulePathInBuffer(tensor_name: []const u8, buf: *[256]u8) ?[]const u8 {
    if (session_factory.canonicalizeGemma4LegacyWeightKey(tensor_name, buf)) |canonical| {
        return tensorModulePath(canonical);
    }

    const kind = gemma4TextLinearKindForTensor(tensor_name);
    if (kind) |linear_kind| {
        if (parseGemma4TextLayerTensorName(tensor_name)) |parsed| {
            if (parsed.is_gguf) {
                const normalized_suffix = switch (linear_kind) {
                    .q_proj => "self_attn.q_proj",
                    .k_proj => "self_attn.k_proj",
                    .v_proj => "self_attn.v_proj",
                    .o_proj => "self_attn.o_proj",
                    .gate_proj => "mlp.gate_proj",
                    .up_proj => "mlp.up_proj",
                    .down_proj => "mlp.down_proj",
                    .ple_input_gate => "per_layer_input.inp_gate",
                    .ple_projection => "per_layer_input.proj",
                    .ple_model_projection => unreachable,
                };
                return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ parsed.layer_index, normalized_suffix }) catch null;
            }
        } else if (linear_kind == .ple_model_projection and std.mem.eql(u8, tensor_name, "per_layer_model_proj.weight")) {
            return "model.per_layer_input.per_layer_model_proj";
        }
    }
    return tensorModulePath(tensor_name);
}

fn tensorModulePath(tensor_name: []const u8) ?[]const u8 {
    if (std.mem.endsWith(u8, tensor_name, ".weight")) return tensor_name[0 .. tensor_name.len - ".weight".len];
    if (std.mem.endsWith(u8, tensor_name, "/weight")) return tensor_name[0 .. tensor_name.len - "/weight".len];
    return null;
}

const LoRAAdapterTensorKind = enum { a, b };

const ParsedLoRAAdapterTensorName = struct {
    base_tensor_base_name: []const u8,
    module_name: []const u8,
    kind: LoRAAdapterTensorKind,
    loop_index: ?usize = null,
};

fn parseLoRAAdapterTensorName(tensor_name: []const u8) ?ParsedLoRAAdapterTensorName {
    if (recursive_lora.parseLoopAdapterTensorName(tensor_name)) |parsed| {
        const module = moduleNameForTensor(parsed.base_tensor_name) orelse return null;
        return .{
            .base_tensor_base_name = parsed.base_tensor_name,
            .module_name = module,
            .kind = if (parsed.kind == .a) .a else .b,
            .loop_index = parsed.loop_index,
        };
    }
    if (std.mem.endsWith(u8, tensor_name, ".lora_A.weight")) {
        const base = tensor_name[0 .. tensor_name.len - ".lora_A.weight".len];
        const module = moduleNameForTensor(base) orelse return null;
        return .{ .base_tensor_base_name = base, .module_name = module, .kind = .a };
    }
    if (std.mem.endsWith(u8, tensor_name, ".lora_B.weight")) {
        const base = tensor_name[0 .. tensor_name.len - ".lora_B.weight".len];
        const module = moduleNameForTensor(base) orelse return null;
        return .{ .base_tensor_base_name = base, .module_name = module, .kind = .b };
    }
    return null;
}

fn parseDoRAMagnitudeTensorName(tensor_name: []const u8) ?[]const u8 {
    const suffix = ".lora_magnitude_vector.weight";
    if (!std.mem.endsWith(u8, tensor_name, suffix)) return null;
    const base = tensor_name[0 .. tensor_name.len - suffix.len];
    if (tensorModulePath(base) == null) return null;
    return base;
}

fn writeBootstrapAdapterCheckpoint(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    base_checkpoint_path: []const u8,
    resolved_tensors: []const LoRATargetTensor,
    rank: usize,
    use_dora: bool,
    init_lora_weights: ?[]const u8,
    eva_stats_path: ?[]const u8,
    lora_ga_stats_path: ?[]const u8,
    recursive_config: recursive_lora.Config,
) !void {
    const init_kind = try parseLoRAInitKind(init_lora_weights);
    const loop_count = if (recursive_config.enabled) recursive_config.loop_count else 1;
    const tensors_per_target: usize = (2 * loop_count) + if (use_dora) @as(usize, 1) else 0;
    var tensors = try allocator.alloc(WriteTensorF32, resolved_tensors.len * tensors_per_target);
    defer allocator.free(tensors);

    var owned_names: std.ArrayListUnmanaged([]const u8) = .empty;
    defer {
        for (owned_names.items) |item| allocator.free(item);
        owned_names.deinit(allocator);
    }
    var owned_shapes: std.ArrayListUnmanaged([]const usize) = .empty;
    defer {
        for (owned_shapes.items) |item| allocator.free(item);
        owned_shapes.deinit(allocator);
    }
    var owned_data: std.ArrayListUnmanaged([]const f32) = .empty;
    defer {
        for (owned_data.items) |item| allocator.free(item);
        owned_data.deinit(allocator);
    }

    const needs_base_tensor = use_dora or init_kind == .pissa or init_kind == .loftq_nf4;
    var base_access: ?tensor_access.TensorAccess = null;
    if (needs_base_tensor) {
        base_access = try openTensorAccessForFile(allocator, base_checkpoint_path);
    }
    defer if (base_access) |*access| access.deinit();

    var eva_stats_access: ?tensor_access.TensorAccess = null;
    if (init_kind == .eva) {
        eva_stats_access = try openTensorAccessForFile(allocator, eva_stats_path orelse return error.MissingInitializerStats);
    }
    defer if (eva_stats_access) |*access| access.deinit();

    var lora_ga_stats_access: ?tensor_access.TensorAccess = null;
    if (init_kind == .lora_ga) {
        lora_ga_stats_access = try openTensorAccessForFile(allocator, lora_ga_stats_path orelse return error.MissingInitializerStats);
    }
    defer if (lora_ga_stats_access) |*access| access.deinit();

    var tensor_idx: usize = 0;
    for (resolved_tensors) |target| {
        const adapter_base_tensor_name = try canonicalAdapterBaseTensorName(allocator, target.tensor_name);
        defer allocator.free(adapter_base_tensor_name);

        var base_tensor: ?Tensor = null;
        if (needs_base_tensor) {
            base_tensor = try loadTensorAsF32(allocator, base_access.?, target.tensor_name);
            if (base_tensor.?.shape.len != 2 or
                base_tensor.?.shape[0] != @as(i64, @intCast(target.output_dim)) or
                base_tensor.?.shape[1] != @as(i64, @intCast(target.input_dim)))
            {
                base_tensor.?.deinit();
                return error.InvalidAdapterTensorShape;
            }
        }
        defer if (base_tensor) |*tensor| tensor.deinit();

        var eva_stats_tensor: ?Tensor = null;
        if (eva_stats_access) |access| {
            eva_stats_tensor = try loadInitializerStatsTensor(allocator, access, target.tensor_name, &.{
                ".eva_activation_covariance",
                ".activation_covariance",
            });
        }
        defer if (eva_stats_tensor) |*tensor| tensor.deinit();

        var lora_ga_stats_tensor: ?Tensor = null;
        if (lora_ga_stats_access) |access| {
            lora_ga_stats_tensor = try loadInitializerStatsTensor(allocator, access, target.tensor_name, &.{
                ".lora_ga_gradient",
                ".weight_gradient",
            });
        }
        defer if (lora_ga_stats_tensor) |*tensor| tensor.deinit();

        const init = try buildInitialLoRAFactors(
            allocator,
            init_kind,
            if (base_tensor) |tensor| tensor.asFloat32() else null,
            if (eva_stats_tensor) |tensor| tensor.asFloat32() else null,
            if (lora_ga_stats_tensor) |tensor| tensor.asFloat32() else null,
            target.output_dim,
            target.input_dim,
            rank,
        );
        const a_data = init.a;
        const b_data = init.b;
        try owned_data.append(allocator, a_data);
        try owned_data.append(allocator, b_data);

        const a_shape = try allocator.dupe(usize, &.{ rank, target.input_dim });
        const b_shape = try allocator.dupe(usize, &.{ target.output_dim, rank });
        try owned_shapes.append(allocator, a_shape);
        try owned_shapes.append(allocator, b_shape);

        for (0..loop_count) |loop_idx| {
            const a_name = if (recursive_config.enabled)
                try recursive_lora.formatLoopAdapterTensorName(allocator, adapter_base_tensor_name, loop_idx, .a)
            else
                try std.fmt.allocPrint(allocator, "{s}.lora_A.weight", .{adapter_base_tensor_name});
            errdefer allocator.free(a_name);
            const b_name = if (recursive_config.enabled)
                try recursive_lora.formatLoopAdapterTensorName(allocator, adapter_base_tensor_name, loop_idx, .b)
            else
                try std.fmt.allocPrint(allocator, "{s}.lora_B.weight", .{adapter_base_tensor_name});
            errdefer allocator.free(b_name);
            try owned_names.append(allocator, a_name);
            try owned_names.append(allocator, b_name);

            tensors[tensor_idx] = .{ .name = a_name, .shape = a_shape, .data = a_data };
            tensor_idx += 1;
            tensors[tensor_idx] = .{ .name = b_name, .shape = b_shape, .data = b_data };
            tensor_idx += 1;
        }

        if (use_dora) {
            const base = base_tensor orelse return error.MissingBaseTensorForAdapter;
            const magnitude_data = try buildDoraMagnitudeFromBaseRowMajor(allocator, base.asFloat32(), target.output_dim, target.input_dim);
            try owned_data.append(allocator, magnitude_data);

            const magnitude_name = try doraMagnitudeTensorName(allocator, adapter_base_tensor_name);
            try owned_names.append(allocator, magnitude_name);

            const magnitude_shape = try allocator.dupe(usize, &.{target.output_dim});
            try owned_shapes.append(allocator, magnitude_shape);

            tensors[tensor_idx] = .{ .name = magnitude_name, .shape = magnitude_shape, .data = magnitude_data };
            tensor_idx += 1;
        }
    }

    try writeHeaderAndTensorsF32(allocator, output_path, tensors[0..tensor_idx]);
}

fn writeBootstrapAdapterCheckpointAtomic(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    base_checkpoint_path: []const u8,
    resolved_tensors: []const LoRATargetTensor,
    rank: usize,
    use_dora: bool,
    init_lora_weights: ?[]const u8,
    eva_stats_path: ?[]const u8,
    lora_ga_stats_path: ?[]const u8,
    recursive_config: recursive_lora.Config,
) !void {
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{output_path});
    defer allocator.free(tmp_path);
    compat.cwd().deleteFile(compat.io(), tmp_path) catch {};
    errdefer compat.cwd().deleteFile(compat.io(), tmp_path) catch {};

    try writeBootstrapAdapterCheckpoint(
        allocator,
        tmp_path,
        base_checkpoint_path,
        resolved_tensors,
        rank,
        use_dora,
        init_lora_weights,
        eva_stats_path,
        lora_ga_stats_path,
        recursive_config,
    );
    try std.Io.Dir.rename(compat.cwd(), tmp_path, compat.cwd(), output_path, compat.io());
}

const InitialLoRAFactors = struct {
    a: []f32,
    b: []f32,
};

fn parseLoRAInitKind(value: ?[]const u8) !LoRAInitKind {
    const text = value orelse return .default;
    if (std.mem.eql(u8, text, "default")) return .default;
    if (std.mem.eql(u8, text, "pissa")) return .pissa;
    if (std.mem.eql(u8, text, "eva")) return .eva;
    if (std.mem.eql(u8, text, "lora-ga") or std.mem.eql(u8, text, "loraga") or std.mem.eql(u8, text, "lora_ga")) return .lora_ga;
    if (std.mem.eql(u8, text, "loftq-nf4")) return .loftq_nf4;
    if (std.mem.eql(u8, text, "loftq")) return .loftq_nf4;
    return error.UnsupportedLoRAInitializer;
}

pub fn validateLoRAInitializerBaseCompatibility(value: ?[]const u8) !void {
    const text = value orelse return;
    if (startsWithIgnoreCase(text, "pissa") or startsWithIgnoreCase(text, "loftq")) {
        return error.LoRAInitializerRequiresAdjustedBase;
    }
    _ = try parseLoRAInitKind(text);
}

fn startsWithIgnoreCase(value: []const u8, prefix: []const u8) bool {
    return value.len >= prefix.len and std.ascii.eqlIgnoreCase(value[0..prefix.len], prefix);
}

fn buildInitialLoRAFactors(
    allocator: std.mem.Allocator,
    init_kind: LoRAInitKind,
    base_weight: ?[]const f32,
    eva_activation_covariance: ?[]const f32,
    lora_ga_gradient: ?[]const f32,
    output_dim: usize,
    input_dim: usize,
    rank: usize,
) !InitialLoRAFactors {
    switch (init_kind) {
        .default => {
            return .{
                .a = try buildDeterministicLoraA(allocator, rank, input_dim),
                .b = try buildZeroF32(allocator, output_dim * rank),
            };
        },
        .pissa => {
            if (rank > @min(output_dim, input_dim)) return error.InvalidLoRARank;
            const base = base_weight orelse return error.MissingBaseTensorForAdapter;
            var result = try lora_init.pissaInit(allocator, base, output_dim, input_dim, rank, 2, 0x9e37_79b9);
            defer result.deinit();
            return .{
                .a = try allocator.dupe(f32, result.a),
                .b = try allocator.dupe(f32, result.b),
            };
        },
        .eva => {
            if (rank > input_dim) return error.InvalidLoRARank;
            const stats = eva_activation_covariance orelse return error.MissingInitializerStats;
            if (stats.len != input_dim * input_dim) return error.InvalidInitializerStatsShape;
            var result = try lora_init.evaInit(allocator, stats, output_dim, input_dim, rank, 4, 0x3e8a_0001);
            defer result.deinit();
            return .{
                .a = try allocator.dupe(f32, result.a),
                .b = try allocator.dupe(f32, result.b),
            };
        },
        .lora_ga => {
            if (rank > @min(output_dim, input_dim)) return error.InvalidLoRARank;
            const stats = lora_ga_gradient orelse return error.MissingInitializerStats;
            if (stats.len != output_dim * input_dim) return error.InvalidInitializerStatsShape;
            var result = try lora_init.loraGaInit(allocator, stats, output_dim, input_dim, rank, 1.0, 4, 0x6a09_e667);
            defer result.deinit();
            return .{
                .a = try allocator.dupe(f32, result.a),
                .b = try allocator.dupe(f32, result.b),
            };
        },
        .loftq_nf4 => {
            if (rank > @min(output_dim, input_dim)) return error.InvalidLoRARank;
            const base = base_weight orelse return error.MissingBaseTensorForAdapter;
            var result = try qlora_nf4.loftqNf4Init(allocator, base, output_dim, input_dim, .{
                .rank = rank,
                .num_iter = 1,
                .power_iters = 2,
                .seed = 0x10f7_0004,
            });
            defer result.deinit();
            return .{
                .a = try allocator.dupe(f32, result.a),
                .b = try allocator.dupe(f32, result.b),
            };
        },
    }
}

fn doraMagnitudeTensorName(allocator: std.mem.Allocator, base_tensor_name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator, "{s}.lora_magnitude_vector.weight", .{base_tensor_name});
}

fn buildDoraMagnitudeFromBaseRowMajor(
    allocator: std.mem.Allocator,
    base_data: []const f32,
    rows: usize,
    cols: usize,
) ![]f32 {
    if (base_data.len != rows * cols) return error.InvalidAdapterTensorShape;
    const magnitude = try allocator.alloc(f32, rows);
    for (0..rows) |row| {
        const values = base_data[row * cols .. (row + 1) * cols];
        var sum: f32 = 0;
        for (values) |value| sum += value * value;
        magnitude[row] = @sqrt(sum + 1e-12);
    }
    return magnitude;
}

fn writeHeaderAndTensorsF32(allocator: std.mem.Allocator, path: []const u8, tensors: []const WriteTensorF32) !void {
    _ = allocator;
    var header_buf: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
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
    for (tensors) |tensor| {
        for (tensor.data) |item| {
            const bits: u32 = @bitCast(item);
            var bits_buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &bits_buf, bits, .little);
            try file.writeStreamingAll(compat.io(), &bits_buf);
        }
    }
}

fn writeHeaderAndRawTensors(allocator: std.mem.Allocator, path: []const u8, tensors: []const WriteTensorRaw) !void {
    var header_buf: std.Io.Writer.Allocating = .init(allocator);
    defer header_buf.deinit();
    const writer = &header_buf.writer;
    try writer.writeByte('{');
    var offset: u64 = 0;
    for (tensors, 0..) |tensor, idx| {
        if (idx != 0) try writer.writeByte(',');
        try writer.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ tensor.name, dtypeName(tensor.dtype) });
        for (tensor.shape, 0..) |dim, dim_idx| {
            if (dim_idx != 0) try writer.writeByte(',');
            try writer.print("{}", .{dim});
        }
        const byte_len: u64 = @intCast(tensor.raw_bytes.len);
        try writer.print("],\"data_offsets\":[{},{}]}}", .{ offset, offset + byte_len });
        offset += byte_len;
    }
    try writer.writeByte('}');

    const io = compat.io();
    var file = try compat.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header_buf.written().len, .little);
    try file.writeStreamingAll(io, &len_buf);
    try file.writeStreamingAll(io, header_buf.written());
    for (tensors) |tensor| try writeFileBytesChunked(io, &file, tensor.raw_bytes);
}

pub const AdapterConfigWriteOptions = struct {
    base_model_name_or_path: []const u8,
    base_model_sha256: ?[]const u8 = null,
    tokenizer_sha256: ?[]const u8 = null,
    chat_template_sha256: ?[]const u8 = null,
    rank: usize,
    alpha: f32,
    target_modules: []const []const u8,
    target_preset: ?[]const u8 = null,
    use_dora: bool = false,
    init_lora_weights: ?[]const u8 = null,
    recursive_lora: recursive_lora.Config = .{},
};

pub fn writeAdapterConfigJson(
    allocator: std.mem.Allocator,
    path: []const u8,
    options: AdapterConfigWriteOptions,
) !void {
    var buffer: std.Io.Writer.Allocating = .init(allocator);
    defer buffer.deinit();
    // Keep adapter_config.json inside PEFT's public schema. Antfly-specific
    // provenance and recursive metadata live in the sidecar manifest below so
    // strict PEFT loaders do not reject unknown constructor arguments.
    try std.json.Stringify.value(.{
        .base_model_name_or_path = options.base_model_name_or_path,
        .bias = "none",
        .fan_in_fan_out = false,
        .inference_mode = false,
        .init_lora_weights = if (options.init_lora_weights) |initializer|
            std.json.Value{ .string = initializer }
        else
            std.json.Value{ .bool = true },
        .lora_alpha = options.alpha,
        .lora_dropout = 0.0,
        .modules_to_save = @as(?[]const []const u8, null),
        .peft_type = "LORA",
        .r = options.rank,
        .target_modules = options.target_modules,
        .task_type = "CAUSAL_LM",
        .use_dora = options.use_dora,
        .use_rslora = false,
    }, .{ .whitespace = .indent_2 }, &buffer.writer);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = path, .data = buffer.written() });
}

pub fn writeAdapterManifestJson(
    allocator: std.mem.Allocator,
    path: []const u8,
    options: AdapterConfigWriteOptions,
) !void {
    const base_model_sha256 = options.base_model_sha256 orelse return error.AdapterProvenanceRequired;
    const tokenizer_sha256 = options.tokenizer_sha256 orelse return error.AdapterProvenanceRequired;
    const chat_template_sha256 = options.chat_template_sha256 orelse return error.AdapterProvenanceRequired;
    const adapter_dir = std.fs.path.dirname(path) orelse ".";
    const adapter_checkpoint_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_checkpoint_file_name });
    defer allocator.free(adapter_checkpoint_path);
    var adapter_checkpoint = try c_file.MmapRegion.init(allocator, adapter_checkpoint_path);
    defer adapter_checkpoint.deinit();
    const adapter_checkpoint_sha256 = try sha256HexAlloc(allocator, adapter_checkpoint.data);
    defer allocator.free(adapter_checkpoint_sha256);
    var buffer: std.Io.Writer.Allocating = .init(allocator);
    defer buffer.deinit();
    try std.json.Stringify.value(AdapterManifest{
        .schema_version = adapter_manifest_schema_v2,
        .status = "complete",
        .artifact_family_version = artifact_family_version,
        .tensor_key_format = adapter_tensor_key_format_v1,
        .adapter_checkpoint_sha256 = adapter_checkpoint_sha256,
        .adapter_checkpoint_size_bytes = @intCast(adapter_checkpoint.data.len),
        .base_model_name_or_path = options.base_model_name_or_path,
        .base_model_sha256 = base_model_sha256,
        .tokenizer_sha256 = tokenizer_sha256,
        .chat_template_sha256 = chat_template_sha256,
        .target_modules = options.target_modules,
        .target_preset = options.target_preset,
        .rank = options.rank,
        .alpha = options.alpha,
        .use_dora = options.use_dora,
        .use_rslora = false,
        .initializer = options.init_lora_weights,
        .recursive_lora = if (options.recursive_lora.enabled) options.recursive_lora else null,
    }, .{ .whitespace = .indent_2 }, &buffer.writer);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = path, .data = buffer.written() });
}

fn targetPresetName(preset: peft.TargetPreset) []const u8 {
    return switch (preset) {
        .all_linear => "all-linear",
        .attention_only => "attention-only",
        .mlp_only => "mlp-only",
        .moe_experts => "moe-experts",
    };
}

fn copySupportingArtifactIfPresent(
    allocator: std.mem.Allocator,
    maybe_src_path: ?[]const u8,
    out_dir: []const u8,
    file_name: []const u8,
) !void {
    const src_path = maybe_src_path orelse return;
    const dst_path = try std.fs.path.join(allocator, &.{ out_dir, file_name });
    defer allocator.free(dst_path);

    const size = try c_file.fileSize(allocator, src_path);
    if (size == 0) {
        try compat.cwd().writeFile(compat.io(), .{ .sub_path = dst_path, .data = "" });
        return;
    }
    if (size <= 100 * 1024 * 1024) {
        const contents = try c_file.readFile(allocator, src_path);
        defer allocator.free(contents);
        try compat.cwd().writeFile(compat.io(), .{ .sub_path = dst_path, .data = contents });
        return;
    }

    var mapped = try c_file.MmapRegion.init(allocator, src_path);
    defer mapped.deinit();
    mapped.adviseSequentialPrefix(mapped.data.len);

    const io = compat.io();
    var file = try compat.cwd().createFile(io, dst_path, .{ .truncate = true });
    defer file.close(io);
    try writeFileBytesChunked(io, &file, mapped.data);
}

fn copyCompressedBaseSupportFiles(allocator: std.mem.Allocator, base_model_dir: []const u8, out_dir: []const u8) !void {
    inline for (.{
        hf_config_file_name,
        tokenizer_config_file_name,
        tokenizer_file_name,
        special_tokens_map_file_name,
        "tokenizer.model",
        "generation_config.json",
        "preprocessor_config.json",
    }) |file_name| {
        const src_path = try std.fs.path.join(allocator, &.{ base_model_dir, file_name });
        defer allocator.free(src_path);
        copySupportingArtifactIfPresent(allocator, src_path, out_dir, file_name) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
    }
}

fn writeRecursiveCompressedBaseMetadata(
    allocator: std.mem.Allocator,
    path: []const u8,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    source_checkpoint_path: []const u8,
    compressed_checkpoint_path: []const u8,
    source_num_layers: usize,
    shared_block_size: usize,
    loop_count: usize,
    tensors_written: usize,
    tensors_skipped: usize,
    source_checkpoint_bytes: u64,
    compressed_checkpoint_bytes: u64,
    compression_ratio: f64,
) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    try std.json.Stringify.value(.{
        .artifact_family_version = artifact_family_version,
        .base_model_dir = base_model_dir,
        .adapter_model_dir = adapter_model_dir,
        .source_checkpoint_path = source_checkpoint_path,
        .compressed_checkpoint_path = compressed_checkpoint_path,
        .source_num_layers = source_num_layers,
        .shared_block_size = shared_block_size,
        .loop_count = loop_count,
        .tensors_written = tensors_written,
        .tensors_skipped = tensors_skipped,
        .source_checkpoint_bytes = source_checkpoint_bytes,
        .compressed_checkpoint_bytes = compressed_checkpoint_bytes,
        .compression_ratio = compression_ratio,
    }, .{ .whitespace = .indent_2 }, &buf.writer);
    try buf.writer.writeByte('\n');
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = path, .data = buf.written() });
}

fn writeFileBytesChunked(io: std.Io, file: *std.Io.File, bytes: []const u8) !void {
    const chunk_size: usize = 8 * 1024 * 1024;
    var offset: usize = 0;
    while (offset < bytes.len) {
        const end = @min(offset + chunk_size, bytes.len);
        try file.writeStreamingAll(io, bytes[offset..end]);
        offset = end;
    }
}

fn loadTensorAsF32(allocator: std.mem.Allocator, access: tensor_access.TensorAccess, name: []const u8) !Tensor {
    var record = try access.getRecord(allocator, name);
    defer record.deinit();
    var tensor = (try record.materializeDense(allocator)) orelse return error.UnsupportedTensorEncoding;
    if (tensor.dtype == .f16 or tensor.dtype == .bf16) {
        const converted = try weight_source.convertToF32(allocator, &tensor);
        tensor.deinit();
        return converted;
    }
    if (tensor.dtype != .f32) {
        tensor.deinit();
        return error.UnsupportedTensorType;
    }
    return tensor;
}

fn loadInitializerStatsTensor(
    allocator: std.mem.Allocator,
    access: tensor_access.TensorAccess,
    base_tensor_name: []const u8,
    suffixes: []const []const u8,
) !Tensor {
    for (suffixes) |suffix| {
        const name = try std.fmt.allocPrint(allocator, "{s}{s}", .{ base_tensor_name, suffix });
        defer allocator.free(name);
        return loadTensorAsF32(allocator, access, name) catch |err| switch (err) {
            error.TensorNotFound => continue,
            else => return err,
        };
    }
    return error.MissingInitializerStats;
}

fn buildMergedSafetensorsFile(
    allocator: std.mem.Allocator,
    base_access: tensor_access.TensorAccess,
    base_names: [][]const u8,
    merged: *const std.StringArrayHashMapUnmanaged(Tensor),
) ![]u8 {
    var ordered_names = try allocator.alloc([]const u8, base_names.len + merged.count());
    defer allocator.free(ordered_names);
    var count: usize = 0;
    for (base_names) |name| {
        ordered_names[count] = name;
        count += 1;
    }
    var it_merged = merged.iterator();
    while (it_merged.next()) |entry| {
        if (!stringSliceContains(base_names, entry.key_ptr.*)) {
            ordered_names[count] = entry.key_ptr.*;
            count += 1;
        }
    }
    std.mem.sort([]const u8, ordered_names[0..count], {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);

    var header_buf: std.Io.Writer.Allocating = .init(allocator);
    defer header_buf.deinit();
    try header_buf.writer.writeByte('{');
    var data_parts = std.ArrayListUnmanaged([]const u8).empty;
    defer data_parts.deinit(allocator);
    var owned_records = std.ArrayListUnmanaged(Tensor).empty;
    defer {
        for (owned_records.items) |*tensor| tensor.deinit();
        owned_records.deinit(allocator);
    }
    var offset: u64 = 0;
    for (ordered_names[0..count], 0..) |name, idx| {
        var tensor: Tensor = undefined;
        if (merged.get(name)) |existing| {
            tensor = existing;
        } else {
            tensor = try loadTensorAsF32(allocator, base_access, name);
            try owned_records.append(allocator, tensor);
        }
        const byte_len = tensor.data.len;
        if (idx != 0) try header_buf.writer.writeByte(',');
        try header_buf.writer.print("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ name, dtypeName(tensor.dtype) });
        for (tensor.shape, 0..) |dim, dim_idx| {
            if (dim_idx != 0) try header_buf.writer.writeByte(',');
            try header_buf.writer.print("{}", .{dim});
        }
        try header_buf.writer.print("],\"data_offsets\":[{},{}]}}", .{ offset, offset + byte_len });
        try data_parts.append(allocator, tensor.data);
        offset += byte_len;
    }
    try header_buf.writer.writeByte('}');

    var file = std.ArrayListUnmanaged(u8).empty;
    errdefer file.deinit(allocator);
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header_buf.written().len, .little);
    try file.appendSlice(allocator, &len_buf);
    try file.appendSlice(allocator, header_buf.written());
    for (data_parts.items) |part| try file.appendSlice(allocator, part);
    return try file.toOwnedSlice(allocator);
}

fn dtypeName(dtype: DType) []const u8 {
    return switch (dtype) {
        .f32 => "F32",
        .f16 => "F16",
        .bf16 => "BF16",
        .f64 => "F64",
        .i8 => "I8",
        .i16 => "I16",
        .i32 => "I32",
        .i64 => "I64",
        .u8 => "U8",
        .bool_ => "BOOL",
    };
}

fn denseRecordDType(encoding: tensor_access.Encoding) ?DType {
    return switch (encoding) {
        .dense => |dtype| dtype,
        .gguf => null,
    };
}

fn openTensorAccessForFile(allocator: std.mem.Allocator, path: []const u8) !tensor_access.TensorAccess {
    if (std.mem.endsWith(u8, path, ".index.json")) {
        try safetensors.validateArtifactSet(allocator, null, path);
        const access = try tensor_access.ShardedSafetensorsAccess.initAbsolute(allocator, path);
        return access.tensorAccess();
    }
    if (std.mem.endsWith(u8, path, ".gguf")) {
        // GGUF access objects mostly serve metadata lookup and lazy record fetch.
        // Prefer libc's allocator when available to avoid Debug allocator overhead
        // while parsing and later deinitializing large headers.
        const gguf_allocator = platform.allocator.processAllocator(allocator);
        const access = try tensor_access.GgufAccess.initAbsolute(gguf_allocator, path);
        return access.tensorAccess();
    }
    const access = try tensor_access.SafetensorsAccess.initAbsolute(allocator, path);
    return access.tensorAccess();
}

fn transpose2DF32(out: []f32, input: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(out.len == rows * cols);
    for (0..rows) |row| {
        for (0..cols) |col| {
            out[col * rows + row] = input[row * cols + col];
        }
    }
}

fn buildDeterministicLoraA(allocator: std.mem.Allocator, rows: usize, cols: usize) ![]f32 {
    const data = try allocator.alloc(f32, rows * cols);
    for (0..rows) |row| {
        for (0..cols) |col| {
            const idx = row * cols + col;
            const angle: f32 = @floatFromInt((row + 1) * (col + 3));
            data[idx] = @sin(angle * 0.013) * 0.01;
        }
    }
    return data;
}

fn buildZeroF32(allocator: std.mem.Allocator, len: usize) ![]f32 {
    const data = try allocator.alloc(f32, len);
    @memset(data, 0.0);
    return data;
}

fn optionalPathInDir(allocator: std.mem.Allocator, dir_path: []const u8, basename: []const u8) !?[]u8 {
    const path = try std.fs.path.join(allocator, &.{ dir_path, basename });
    errdefer allocator.free(path);
    compat.cwd().access(compat.io(), path, .{}) catch {
        allocator.free(path);
        return null;
    };
    return path;
}

fn findDecoderGgufPathInDir(allocator: std.mem.Allocator, dir_path: []const u8) !?[]u8 {
    var dir = compat.cwd().openDir(compat.io(), dir_path, .{ .iterate = true }) catch return null;
    defer dir.close(compat.io());

    var candidates = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (candidates.items) |path| allocator.free(path);
        candidates.deinit(allocator);
    }

    var iter = dir.iterate();
    while (try iter.next(compat.io())) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".gguf")) continue;
        if (isProjectorGgufName(entry.name)) continue;

        const path = try std.fs.path.join(allocator, &.{ dir_path, entry.name });
        errdefer allocator.free(path);
        try candidates.append(allocator, path);
    }

    if (candidates.items.len == 0) return null;
    if (candidates.items.len > 1) {
        std.mem.sort([]u8, candidates.items, {}, struct {
            fn lt(_: void, a: []u8, b: []u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        return error.AmbiguousDecoderGguf;
    }
    const only = candidates.items[0];
    candidates.items.len = 0;
    return only;
}

fn isProjectorGgufName(name: []const u8) bool {
    return std.mem.indexOf(u8, name, "mmproj") != null or
        std.mem.indexOf(u8, name, "projector") != null;
}

fn isRegularFilePath(path: []const u8) bool {
    const stat = compat.cwd().statFile(compat.io(), path, .{}) catch return false;
    return stat.kind == .file;
}

fn freeLoRATargetTensors(allocator: std.mem.Allocator, tensors: []LoRATargetTensor) void {
    for (tensors) |item| {
        allocator.free(item.tensor_name);
        allocator.free(item.module_name);
    }
    allocator.free(tensors);
}

fn freeLoRATensorSummary(allocator: std.mem.Allocator, item: *LoRATensorSummary) void {
    allocator.free(item.base_tensor_name);
    allocator.free(item.adapter_a_tensor_name);
    allocator.free(item.adapter_b_tensor_name);
    if (item.dora_magnitude_tensor_name) |name| allocator.free(name);
    allocator.free(item.module_name);
    item.* = undefined;
}

fn dupeStringSlice(allocator: std.mem.Allocator, value: []const []const u8) ![][]const u8 {
    const out = try allocator.alloc([]const u8, value.len);
    errdefer allocator.free(out);
    var built: usize = 0;
    errdefer for (out[0..built]) |item| allocator.free(item);
    for (value, 0..) |item, idx| {
        out[idx] = try allocator.dupe(u8, item);
        built += 1;
    }
    return out;
}

fn dupeOptionalStringSlice(allocator: std.mem.Allocator, value: ?[]const []const u8) !?[][]const u8 {
    const items = value orelse return null;
    return try dupeStringSlice(allocator, items);
}

fn dupeOptionalString(allocator: std.mem.Allocator, value: ?[]const u8) !?[]const u8 {
    const item = value orelse return null;
    return try allocator.dupe(u8, item);
}

fn adapterInitializerName(value: ?std.json.Value) !?[]const u8 {
    const init = value orelse return null;
    return switch (init) {
        .string => |name| if (name.len > 0) name else error.InvalidLoRAInitializer,
        // PEFT writes `true` for its ordinary Kaiming/zero initialization.
        // The flag only controls creation of new adapter weights; an existing
        // checkpoint already contains those weights, so no named initializer
        // needs to be replayed while loading it.
        .bool, .null => null,
        else => error.InvalidLoRAInitializer,
    };
}

fn stringSliceContains(haystack: []const []const u8, needle: []const u8) bool {
    for (haystack) |item| {
        if (std.mem.eql(u8, item, needle)) return true;
    }
    return false;
}

fn orderedStringSlicesEqual(lhs: []const []const u8, rhs: []const []const u8) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!std.mem.eql(u8, left, right)) return false;
    }
    return true;
}

fn adapterAlphaMatchesManifest(config_alpha: ?f64, manifest_alpha: f32) bool {
    const value = config_alpha orelse return false;
    if (!std.math.isFinite(value) or value <= 0 or value > std.math.floatMax(f32)) return false;
    return @as(f32, @floatCast(value)) == manifest_alpha;
}

fn writeGemma4BootstrapTestConfig(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    num_hidden_layers: usize,
    num_kv_shared_layers: usize,
) !void {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, hf_config_file_name });
    defer allocator.free(config_path);
    const config_json = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "model_type": "gemma4",
        \\  "text_config": {{
        \\    "hidden_size": 3,
        \\    "num_hidden_layers": {d},
        \\    "num_attention_heads": 1,
        \\    "num_key_value_heads": 1,
        \\    "head_dim": 3,
        \\    "intermediate_size": 4,
        \\    "vocab_size": 4,
        \\    "num_kv_shared_layers": {d}
        \\  }}
        \\}}
    , .{ num_hidden_layers, num_kv_shared_layers });
    defer allocator.free(config_json);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = config_path, .data = config_json });
}

test "gemma4 adapter manifest alpha uses the persisted f32 contract" {
    try std.testing.expect(adapterAlphaMatchesManifest(0.1, @as(f32, 0.1)));
    try std.testing.expect(adapterAlphaMatchesManifest(16.0, 16.0));
    try std.testing.expect(!adapterAlphaMatchesManifest(0.2, @as(f32, 0.1)));
    try std.testing.expect(!adapterAlphaMatchesManifest(std.math.nan(f64), 1.0));
}

test "gemma4 adapter target preset cannot mislabel its exact inventory" {
    const base = InspectionSummary{
        .artifact_family_version = artifact_family_version,
        .variant = .adapter_only,
        .model_dir = "adapter",
        .target_preset = "peft-qv",
        .target_modules = &.{"model.layers.0.self_attn.q_proj"},
    };
    try validateAdapterTargetPreset(std.testing.allocator, base);

    var mislabeled = base;
    mislabeled.target_modules = &.{"model.layers.0.mlp.down_proj"};
    try std.testing.expectError(
        error.AdapterTargetPresetMismatch,
        validateAdapterTargetPreset(std.testing.allocator, mislabeled),
    );

    var unknown = base;
    unknown.target_preset = "future-preset";
    try std.testing.expectError(
        error.InvalidAdapterTargetPreset,
        validateAdapterTargetPreset(std.testing.allocator, unknown),
    );
}

test "gemma4 adapter tensor admission rejects non-finite payloads" {
    const shape = [_]i64{ 1, 1 };
    const finite: f32 = 1.0;
    try validateFiniteAdapterTensor(.{
        .descriptor = .{
            .name = "adapter",
            .shape = &shape,
            .encoding = .{ .dense = .f32 },
            .byte_len = @sizeOf(f32),
            .quantized = false,
        },
        .raw_bytes = std.mem.asBytes(&finite),
    });

    const nan: f32 = std.math.nan(f32);
    try std.testing.expectError(error.NonFiniteAdapterTensor, validateFiniteAdapterTensor(.{
        .descriptor = .{
            .name = "adapter",
            .shape = &shape,
            .encoding = .{ .dense = .f32 },
            .byte_len = @sizeOf(f32),
            .quantized = false,
        },
        .raw_bytes = std.mem.asBytes(&nan),
    }));

    const half: f16 = 1.0;
    try std.testing.expectError(error.UnsupportedAdapterTensorEncoding, validateFiniteAdapterTensor(.{
        .descriptor = .{
            .name = "adapter",
            .shape = &shape,
            .encoding = .{ .dense = .f16 },
            .byte_len = @sizeOf(f16),
            .quantized = false,
        },
        .raw_bytes = std.mem.asBytes(&half),
    }));
}

fn optionalStringsEqual(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    if (lhs == null or rhs == null) return lhs == null and rhs == null;
    return std.mem.eql(u8, lhs.?, rhs.?);
}

test "findDecoderGgufPathInDir ignores projector ggufs and returns sole decoder" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "mmproj-gemma.gguf", .data = "projector" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "Gemma Q4 KM.gguf", .data = "decoder" });

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);

    const decoder = try findDecoderGgufPathInDir(allocator, model_dir) orelse return error.TestExpectedDecoderGguf;
    defer allocator.free(decoder);

    try std.testing.expect(std.mem.endsWith(u8, decoder, "Gemma Q4 KM.gguf"));
}

test "findDecoderGgufPathInDir rejects ambiguous decoder ggufs" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "gemma-4-E2B-it-Q4_K_M.gguf", .data = "decoder-a" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "gemma-4-E2B-it-Q5_K_M.gguf", .data = "decoder-b" });

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);

    try std.testing.expectError(error.AmbiguousDecoderGguf, findDecoderGgufPathInDir(allocator, model_dir));
}

test "gemma4 E2B and E4B presets follow shared-KV text schemas" {
    try std.testing.expectEqual(@as(usize, 50), try syntheticGemma4PresetTargetCount(35, 15, .peft_qv));
    try std.testing.expectEqual(@as(usize, 276), try syntheticGemma4PresetTargetCount(35, 15, .text_all_linear));
    try std.testing.expectEqual(@as(usize, 66), try syntheticGemma4PresetTargetCount(42, 24, .peft_qv));
    try std.testing.expectEqual(@as(usize, 343), try syntheticGemma4PresetTargetCount(42, 24, .text_all_linear));

    const all = BootstrapTargetSelection{ .gemma4 = .text_all_linear };
    try std.testing.expect(targetMatchesSelection("model.language_model.layers.3.self_attn.q_proj.weight", all));
    try std.testing.expect(!targetMatchesSelection("model.vision_tower.layers.3.self_attn.q_proj.weight", all));
    try std.testing.expect(!targetMatchesSelection("model.embed_tokens.weight", all));
    try std.testing.expect(!targetMatchesSelection("model.layers.3.input_layernorm.weight", all));
    try std.testing.expect(!targetMatchesSelection("lm_head.weight", all));
}

test "gemma4 bootstrap excludes checkpoint tensors omitted by shared-KV graph" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    try writeGemma4BootstrapTestConfig(allocator, root, 4, 2);

    const checkpoint_path = try std.fs.path.join(allocator, &.{ root, checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    const values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.0.self_attn.v_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.1.self_attn.q_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.1.self_attn.v_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.2.self_attn.q_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.2.self_attn.v_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.3.self_attn.q_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.3.self_attn.v_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
    });

    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    var bootstrap = try bootstrapLoRABundle(allocator, root, adapter_dir, .{
        .rank = 1,
        .gemma4_target_preset = .peft_qv,
    });
    defer freeBootstrapSummary(allocator, &bootstrap);
    try std.testing.expectEqual(@as(usize, 6), bootstrap.resolved_tensors.len);
    try std.testing.expectEqualStrings("model.layers.2.self_attn.q_proj.weight", bootstrap.resolved_tensors[4].tensor_name);
    try std.testing.expectEqualStrings("model.layers.3.self_attn.q_proj.weight", bootstrap.resolved_tensors[5].tensor_name);
    var inspected = try inspectLoRABundle(allocator, root, adapter_dir);
    defer freeLoRABundleInspectionSummary(allocator, &inspected);
    try std.testing.expectEqual(@as(usize, 6), inspected.resolved_tensor_count);

    const invalid_adapter_dir = try std.fs.path.join(allocator, &.{ root, "invalid-adapter" });
    defer allocator.free(invalid_adapter_dir);
    const graph_absent_target = [_][]const u8{"model.layers.3.self_attn.v_proj"};
    try std.testing.expectError(error.UnknownLoRATargetModule, bootstrapLoRABundle(allocator, root, invalid_adapter_dir, .{
        .rank = 1,
        .target_modules = &graph_absent_target,
    }));
}

test "gemma4 bootstrap persists exact targets and rejects unknown explicit modules" {
    const allocator = std.testing.allocator;
    const root = try std.fmt.allocPrint(allocator, "/tmp/termite_gemma4_strict_targets_test_{d}", .{std.posix.system.getpid()});
    defer allocator.free(root);
    compat.cwd().deleteTree(compat.io(), root) catch {};
    try compat.cwd().createDirPath(compat.io(), root);
    defer compat.cwd().deleteTree(compat.io(), root) catch {};
    try writeGemma4BootstrapTestConfig(allocator, root, 2, 0);

    const checkpoint_path = try std.fs.path.join(allocator, &.{ root, checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    const values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.0.self_attn.k_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.0.self_attn.v_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.1.self_attn.q_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.1.self_attn.o_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.1.mlp.gate_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.1.mlp.up_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.1.mlp.down_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.1.per_layer_input.inp_gate.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.1.per_layer_input.proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.per_layer_input.per_layer_model_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.vision_tower.layers.0.self_attn.q_proj.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.embed_tokens.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "model.layers.0.input_layernorm.weight", .shape = &.{ 2, 3 }, .data = &values },
        .{ .name = "lm_head.weight", .shape = &.{ 2, 3 }, .data = &values },
    });

    const out_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(out_dir);
    var summary = try bootstrapLoRABundle(allocator, root, out_dir, .{
        .rank = 1,
        .gemma4_target_preset = .peft_qv,
    });
    defer freeBootstrapSummary(allocator, &summary);

    try std.testing.expectEqual(@as(usize, 3), summary.resolved_tensors.len);
    try std.testing.expectEqual(@as(usize, 3), summary.target_modules.len);
    try std.testing.expectEqualStrings("peft-qv", summary.target_preset.?);
    try std.testing.expectEqualStrings("model.layers.0.self_attn.q_proj", summary.target_modules[0]);
    try std.testing.expectEqualStrings("model.layers.0.self_attn.v_proj", summary.target_modules[1]);
    try std.testing.expectEqualStrings("model.layers.1.self_attn.q_proj", summary.target_modules[2]);

    var inspected = try inspectCheckpoint(allocator, out_dir);
    defer freeInspectionSummary(allocator, &inspected);
    try std.testing.expectEqual(@as(usize, 3), inspected.target_module_count);
    try std.testing.expectEqualStrings(summary.target_modules[2], inspected.target_modules.?[2]);

    const published_before = try c_file.readFile(allocator, summary.adapter_checkpoint_path);
    defer allocator.free(published_before);
    try std.testing.expectError(error.Gemma4RunOutputAlreadyExists, bootstrapLoRABundle(allocator, root, out_dir, .{
        .rank = 1,
        .gemma4_target_preset = .peft_qv,
    }));
    const published_after = try c_file.readFile(allocator, summary.adapter_checkpoint_path);
    defer allocator.free(published_after);
    try std.testing.expectEqualSlices(u8, published_before, published_after);

    const unknown = [_][]const u8{ "q_proj", "lm_head" };
    const unknown_out = try std.fs.path.join(allocator, &.{ root, "unknown" });
    defer allocator.free(unknown_out);
    try std.testing.expectError(error.UnknownLoRATargetModule, bootstrapLoRABundle(allocator, root, unknown_out, .{
        .rank = 1,
        .target_modules = unknown[0..],
    }));

    const empty = [_][]const u8{};
    try std.testing.expectError(error.NoLoRATargetTensorsResolved, bootstrapLoRABundle(allocator, root, unknown_out, .{
        .rank = 1,
        .target_modules = empty[0..],
    }));

    const qv = [_][]const u8{ "q_proj", "v_proj" };
    try std.testing.expectError(error.UnknownLoRATargetModule, bootstrapLoRABundle(allocator, root, unknown_out, .{
        .rank = 1,
        .layer_name = "model.layers.1",
        .target_modules = qv[0..],
    }));
}

test "gemma4 PEFT export preserves payloads and publishes stock tensor keys" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const base_dir = try std.fs.path.join(allocator, &.{ root, "base" });
    defer allocator.free(base_dir);
    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    const peft_dir = try std.fs.path.join(allocator, &.{ root, "peft" });
    defer allocator.free(peft_dir);
    try compat.cwd().createDirPath(compat.io(), base_dir);
    try writeGemma4BootstrapTestConfig(allocator, base_dir, 1, 0);

    const base_checkpoint_path = try std.fs.path.join(allocator, &.{ base_dir, checkpoint_file_name });
    defer allocator.free(base_checkpoint_path);
    try writeHeaderAndTensorsF32(allocator, base_checkpoint_path, &.{
        .{
            .name = "model.layers.0.self_attn.q_proj.weight",
            .shape = &.{ 2, 3 },
            .data = &.{ 1, 2, 3, 4, 5, 6 },
        },
        .{
            .name = "model.layers.0.self_attn.v_proj.weight",
            .shape = &.{ 2, 3 },
            .data = &.{ 6, 5, 4, 3, 2, 1 },
        },
    });

    var bootstrap = try bootstrapLoRABundle(allocator, base_dir, adapter_dir, .{
        .rank = 1,
        .alpha = 2,
        .gemma4_target_preset = .peft_qv,
    });
    defer freeBootstrapSummary(allocator, &bootstrap);

    var exported = try exportPeftAdapter(allocator, base_dir, adapter_dir, peft_dir);
    defer freePeftExportSummary(allocator, &exported);
    try std.testing.expectEqualStrings(stock_peft_tensor_key_format_v1, exported.tensor_key_format);
    try std.testing.expectEqual(@as(usize, 4), exported.tensor_count);
    try std.testing.expect(exported.adapter_checkpoint_size_bytes > 0);
    try validateSha256Hex(exported.adapter_checkpoint_sha256);

    var source = try safetensors.MMapReader.openFileAbsolute(allocator, bootstrap.adapter_checkpoint_path);
    defer source.deinit();
    var destination = try safetensors.MMapReader.openFileAbsolute(allocator, exported.adapter_checkpoint_path);
    defer destination.deinit();
    const source_a_name = "model.layers.0.self_attn.q_proj.weight.lora_A.weight";
    const destination_a_name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
    try std.testing.expect(!destination.header.tensors.contains(source_a_name));
    try std.testing.expect(destination.header.tensors.contains(destination_a_name));
    var source_a = try source.readTensor(source_a_name);
    defer source_a.deinit();
    var destination_a = try destination.readTensor(destination_a_name);
    defer destination_a.deinit();
    try std.testing.expectEqual(source_a.dtype, destination_a.dtype);
    try std.testing.expectEqualSlices(i64, source_a.shape, destination_a.shape);
    try std.testing.expectEqualSlices(u8, source_a.data, destination_a.data);

    const source_config = try c_file.readFile(allocator, bootstrap.adapter_config_path);
    defer allocator.free(source_config);
    const destination_config = try c_file.readFile(allocator, exported.adapter_config_path);
    defer allocator.free(destination_config);
    try std.testing.expectEqualSlices(u8, source_config, destination_config);

    const manifest_bytes = try c_file.readFile(allocator, exported.export_manifest_path);
    defer allocator.free(manifest_bytes);
    var manifest = try std.json.parseFromSlice(PeftExportManifest, allocator, manifest_bytes, .{ .ignore_unknown_fields = false });
    defer manifest.deinit();
    try std.testing.expectEqualStrings(peft_export_manifest_schema_v1, manifest.value.schema_version);
    try std.testing.expectEqualStrings(stock_peft_tensor_key_format_v1, manifest.value.destination_tensor_key_format);
    try std.testing.expectEqualStrings(exported.adapter_checkpoint_sha256, manifest.value.destination_adapter_model_sha256);
    try std.testing.expectEqual(exported.tensor_count, manifest.value.tensor_count);

    try std.testing.expectError(
        error.Gemma4RunOutputAlreadyExists,
        exportPeftAdapter(allocator, base_dir, adapter_dir, peft_dir),
    );
}

test "gemma4 sharded safetensors bootstrap succeeds and legacy materialize fails closed" {
    const allocator = std.testing.allocator;
    const root = try std.fmt.allocPrint(allocator, "/tmp/termite_gemma4_sharded_lifecycle_test_{d}", .{std.posix.system.getpid()});
    defer allocator.free(root);
    compat.cwd().deleteTree(compat.io(), root) catch {};
    try compat.cwd().createDirPath(compat.io(), root);
    defer compat.cwd().deleteTree(compat.io(), root) catch {};
    try writeGemma4BootstrapTestConfig(allocator, root, 1, 0);

    const shard_a_path = try std.fs.path.join(allocator, &.{ root, "model-00001-of-00002.safetensors" });
    defer allocator.free(shard_a_path);
    const shard_b_path = try std.fs.path.join(allocator, &.{ root, "model-00002-of-00002.safetensors" });
    defer allocator.free(shard_b_path);
    const q_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const norm_values = [_]f32{ 7, 8 };
    try writeHeaderAndTensorsF32(allocator, shard_a_path, &.{.{
        .name = "model.layers.0.self_attn.q_proj.weight",
        .shape = &.{ 2, 3 },
        .data = &q_values,
    }});
    try writeHeaderAndTensorsF32(allocator, shard_b_path, &.{.{
        .name = "model.layers.0.input_layernorm.weight",
        .shape = &.{2},
        .data = &norm_values,
    }});
    const index_path = try std.fs.path.join(allocator, &.{ root, "model.safetensors.index.json" });
    defer allocator.free(index_path);
    try compat.cwd().writeFile(compat.io(), .{
        .sub_path = index_path,
        .data =
        \\{"weight_map":{
        \\  "model.layers.0.self_attn.q_proj.weight":"model-00001-of-00002.safetensors",
        \\  "model.layers.0.input_layernorm.weight":"model-00002-of-00002.safetensors"
        \\}}
        ,
    });

    var paths = try resolveArtifactPaths(allocator, root);
    defer paths.deinit();
    try std.testing.expectEqualStrings(index_path, paths.checkpoint_path.?);

    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    const target_modules = [_][]const u8{"q_proj"};
    var bootstrap = try bootstrapLoRABundle(allocator, root, adapter_dir, .{
        .rank = 1,
        .alpha = 1,
        .target_modules = &target_modules,
    });
    defer freeBootstrapSummary(allocator, &bootstrap);
    try std.testing.expectEqual(@as(usize, 1), bootstrap.resolved_tensors.len);
    try std.testing.expectEqualStrings(index_path, bootstrap.checkpoint_path);

    const merged_dir = try std.fs.path.join(allocator, &.{ root, "merged" });
    defer allocator.free(merged_dir);
    try std.testing.expectError(
        error.Gemma4MaterializationRequiresStreaming,
        materializeMergedModel(allocator, root, adapter_dir, merged_dir),
    );
    try std.testing.expectError(error.FileNotFound, compat.cwd().access(compat.io(), merged_dir, .{}));
}

test "gemma4 HF legacy PLE aliases bootstrap and materialize with canonical trainer identities" {
    const allocator = std.testing.allocator;
    const root = try std.fmt.allocPrint(allocator, "/tmp/termite_gemma4_hf_ple_alias_test_{d}", .{std.posix.system.getpid()});
    defer allocator.free(root);
    compat.cwd().deleteTree(compat.io(), root) catch {};
    try compat.cwd().createDirPath(compat.io(), root);
    defer compat.cwd().deleteTree(compat.io(), root) catch {};

    const base_dir = try std.fs.path.join(allocator, &.{ root, "base" });
    defer allocator.free(base_dir);
    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    const merged_dir = try std.fs.path.join(allocator, &.{ root, "merged" });
    defer allocator.free(merged_dir);
    try compat.cwd().createDirPath(compat.io(), base_dir);

    const config_path = try std.fs.path.join(allocator, &.{ base_dir, hf_config_file_name });
    defer allocator.free(config_path);
    try compat.cwd().writeFile(compat.io(), .{
        .sub_path = config_path,
        .data =
        \\{
        \\  "model_type": "gemma4",
        \\  "text_config": {
        \\    "hidden_size": 3,
        \\    "num_hidden_layers": 1,
        \\    "num_attention_heads": 1,
        \\    "num_key_value_heads": 1,
        \\    "head_dim": 3,
        \\    "intermediate_size": 4,
        \\    "vocab_size": 4,
        \\    "hidden_size_per_layer_input": 2
        \\  }
        \\}
        ,
    });

    const aliases = [_]struct {
        legacy: []const u8,
        canonical: []const u8,
    }{
        .{ .legacy = "model.language_model.embed_tokens_per_layer.weight", .canonical = "model.language_model.per_layer_input.per_layer_token_embd.weight" },
        .{ .legacy = "model.language_model.per_layer_model_projection.weight", .canonical = "model.language_model.per_layer_input.per_layer_model_proj.weight" },
        .{ .legacy = "model.language_model.per_layer_projection_norm.weight", .canonical = "model.language_model.per_layer_input.per_layer_proj_norm.weight" },
        .{ .legacy = "model.language_model.layers.0.per_layer_input_gate.weight", .canonical = "model.language_model.layers.0.per_layer_input.inp_gate.weight" },
        .{ .legacy = "model.language_model.layers.0.per_layer_projection.weight", .canonical = "model.language_model.layers.0.per_layer_input.proj.weight" },
        .{ .legacy = "model.language_model.layers.0.post_per_layer_input_norm.weight", .canonical = "model.language_model.layers.0.per_layer_input.post_norm.weight" },
        .{ .legacy = "model.language_model.layers.0.layer_scalar", .canonical = "model.language_model.layers.0.per_layer_input.layer_output_scale.weight" },
    };
    const values_8 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const values_6 = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const values_2 = [_]f32{ 1, 2 };
    const values_3 = [_]f32{ 1, 2, 3 };
    const values_1 = [_]f32{1};
    const checkpoint_path = try std.fs.path.join(allocator, &.{ base_dir, checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{
        .{ .name = aliases[0].legacy, .shape = &.{ 4, 2 }, .data = &values_8 },
        .{ .name = aliases[1].legacy, .shape = &.{ 2, 3 }, .data = &values_6 },
        .{ .name = aliases[2].legacy, .shape = &.{2}, .data = &values_2 },
        .{ .name = aliases[3].legacy, .shape = &.{ 2, 3 }, .data = &values_6 },
        .{ .name = aliases[4].legacy, .shape = &.{ 3, 2 }, .data = &values_6 },
        .{ .name = aliases[5].legacy, .shape = &.{3}, .data = &values_3 },
        .{ .name = aliases[6].legacy, .shape = &.{1}, .data = &values_1 },
    });

    // Trainer backends request canonical graph names. Session construction must
    // make every legacy base tensor available under that canonical identity.
    const session = try session_factory.createNativeSession(allocator, base_dir);
    defer session.close();
    var compute_backend = try session_factory.getComputeBackend(session, allocator);
    defer compute_backend.deinit();
    for (aliases) |alias| {
        const weight = try compute_backend.getWeight(alias.canonical);
        compute_backend.free(weight);
    }

    var bootstrap = try bootstrapLoRABundle(allocator, base_dir, adapter_dir, .{
        .rank = 1,
        .alpha = 1,
        .gemma4_target_preset = .text_all_linear,
    });
    defer freeBootstrapSummary(allocator, &bootstrap);
    try std.testing.expectEqual(@as(usize, 3), bootstrap.resolved_tensors.len);
    try std.testing.expectEqual(@as(usize, 3), bootstrap.target_modules.len);
    try std.testing.expectEqualStrings("model.language_model.layers.0.per_layer_input.inp_gate", bootstrap.target_modules[0]);
    try std.testing.expectEqualStrings("model.language_model.layers.0.per_layer_input.proj", bootstrap.target_modules[1]);
    try std.testing.expectEqualStrings("model.language_model.per_layer_input.per_layer_model_proj", bootstrap.target_modules[2]);

    var adapter_paths = try resolveArtifactPaths(allocator, adapter_dir);
    defer adapter_paths.deinit();
    var adapter_reader = try safetensors.MMapReader.openFileAbsolute(allocator, adapter_paths.adapter_checkpoint_path.?);
    defer adapter_reader.deinit();
    for (bootstrap.target_modules) |module_name| {
        var name_buf: [256]u8 = undefined;
        const trainer_lookup = try std.fmt.bufPrint(&name_buf, "{s}.weight.lora_A.weight", .{module_name});
        try std.testing.expect(adapter_reader.header.tensors.contains(trainer_lookup));
    }

    var materialized = try materializeMergedModel(allocator, base_dir, adapter_dir, merged_dir);
    defer freeMaterializeSummary(allocator, &materialized);
    try std.testing.expectEqual(@as(usize, 3), materialized.merged_lora_tensor_count);
    try std.testing.expectEqual(@as(usize, 4), materialized.copied_base_tensor_count);

    var merged_reader = try safetensors.MMapReader.openFileAbsolute(allocator, materialized.output_checkpoint_path);
    defer merged_reader.deinit();
    var source_access = try openTensorAccessForFile(allocator, checkpoint_path);
    defer source_access.deinit();
    var merged_access = try openTensorAccessForFile(allocator, materialized.output_checkpoint_path);
    defer merged_access.deinit();
    for (aliases) |alias| {
        try std.testing.expect(merged_reader.header.tensors.contains(alias.legacy));
        try std.testing.expect(!merged_reader.header.tensors.contains(alias.canonical));

        var source_tensor = try loadTensorAsF32(allocator, source_access, alias.legacy);
        defer source_tensor.deinit();
        var merged_tensor = try loadTensorAsF32(allocator, merged_access, alias.legacy);
        defer merged_tensor.deinit();
        try std.testing.expectEqualSlices(i64, source_tensor.shape, merged_tensor.shape);
        try std.testing.expectEqualSlices(f32, source_tensor.asFloat32(), merged_tensor.asFloat32());
    }
}

test "gemma4 GGUF bootstrap keys align with graph trainer lookups" {
    const allocator = std.testing.allocator;
    const root = try std.fmt.allocPrint(allocator, "/tmp/termite_gemma4_gguf_adapter_keys_test_{d}", .{std.posix.system.getpid()});
    defer allocator.free(root);
    compat.cwd().deleteTree(compat.io(), root) catch {};
    try compat.cwd().createDirPath(compat.io(), root);
    defer compat.cwd().deleteTree(compat.io(), root) catch {};

    const raw_q_name = "blk.34.attn_q.weight";
    const target = LoRATargetTensor{
        .tensor_name = raw_q_name,
        .module_name = "q_proj",
        .input_dim = 3,
        .output_dim = 2,
    };

    const standard_path = try std.fs.path.join(allocator, &.{ root, "standard.safetensors" });
    defer allocator.free(standard_path);
    try writeBootstrapAdapterCheckpoint(
        allocator,
        standard_path,
        "unused-base.safetensors",
        &.{target},
        1,
        false,
        null,
        null,
        null,
        .{},
    );
    var standard = try safetensors.MMapReader.openFileAbsolute(allocator, standard_path);
    defer standard.deinit();

    // initializeTrainerFromAdapterDir appends `.weight` to these graph slot
    // names. Bootstrap must therefore emit the resulting canonical keys.
    const graph_a_slot = "model.layers.34.self_attn.q_proj.weight.lora_A";
    const graph_b_slot = "model.layers.34.self_attn.q_proj.weight.lora_B";
    const trainer_a_lookup = try std.fmt.allocPrint(allocator, "{s}.weight", .{graph_a_slot});
    defer allocator.free(trainer_a_lookup);
    const trainer_b_lookup = try std.fmt.allocPrint(allocator, "{s}.weight", .{graph_b_slot});
    defer allocator.free(trainer_b_lookup);
    try std.testing.expect(standard.header.tensors.contains(trainer_a_lookup));
    try std.testing.expect(standard.header.tensors.contains(trainer_b_lookup));
    try std.testing.expect(!standard.header.tensors.contains("blk.34.attn_q.weight.lora_A.weight"));

    // Exercise raw-source lookup and canonical recursive/DoRA persistence in
    // one pass: DoRA has to read the base tensor by its original GGUF key.
    const base_path = try std.fs.path.join(allocator, &.{ root, "raw-source.safetensors" });
    defer allocator.free(base_path);
    try writeHeaderAndTensorsF32(allocator, base_path, &.{.{
        .name = raw_q_name,
        .shape = &.{ 2, 3 },
        .data = &.{ 1, 2, 3, 4, 5, 6 },
    }});
    const recursive_dora_path = try std.fs.path.join(allocator, &.{ root, "recursive-dora.safetensors" });
    defer allocator.free(recursive_dora_path);
    try writeBootstrapAdapterCheckpoint(
        allocator,
        recursive_dora_path,
        base_path,
        &.{target},
        1,
        true,
        null,
        null,
        null,
        .{
            .enabled = true,
            .source_num_layers = 70,
            .shared_block_size = 35,
            .loop_count = 2,
        },
    );
    var recursive_dora = try safetensors.MMapReader.openFileAbsolute(allocator, recursive_dora_path);
    defer recursive_dora.deinit();
    try std.testing.expect(recursive_dora.header.tensors.contains("model.layers.34.self_attn.q_proj.weight.loop_0.lora_A.weight"));
    try std.testing.expect(recursive_dora.header.tensors.contains("model.layers.34.self_attn.q_proj.weight.loop_1.lora_B.weight"));
    try std.testing.expect(recursive_dora.header.tensors.contains("model.layers.34.self_attn.q_proj.weight.lora_magnitude_vector.weight"));

    const ple_gate = try canonicalAdapterBaseTensorName(allocator, "blk.34.inp_gate.weight");
    defer allocator.free(ple_gate);
    const ple_projection = try canonicalAdapterBaseTensorName(allocator, "blk.34.proj.weight");
    defer allocator.free(ple_projection);
    const ple_model_projection = try canonicalAdapterBaseTensorName(allocator, "per_layer_model_proj.weight");
    defer allocator.free(ple_model_projection);
    try std.testing.expectEqualStrings("model.layers.34.per_layer_input.inp_gate.weight", ple_gate);
    try std.testing.expectEqualStrings("model.layers.34.per_layer_input.proj.weight", ple_projection);
    try std.testing.expectEqualStrings("model.per_layer_input.per_layer_model_proj.weight", ple_model_projection);

    const base_dir = try std.fs.path.join(allocator, &.{ root, "base" });
    defer allocator.free(base_dir);
    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    const merged_dir = try std.fs.path.join(allocator, &.{ root, "merged" });
    defer allocator.free(merged_dir);
    try compat.cwd().createDirPath(compat.io(), base_dir);
    try compat.cwd().createDirPath(compat.io(), adapter_dir);

    const bundle_base_path = try std.fs.path.join(allocator, &.{ base_dir, checkpoint_file_name });
    defer allocator.free(bundle_base_path);
    try writeHeaderAndTensorsF32(allocator, bundle_base_path, &.{.{
        .name = raw_q_name,
        .shape = &.{ 2, 3 },
        .data = &.{ 1, 2, 3, 4, 5, 6 },
    }});
    const bundle_adapter_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_checkpoint_file_name });
    defer allocator.free(bundle_adapter_path);
    try writeBootstrapAdapterCheckpoint(allocator, bundle_adapter_path, bundle_base_path, &.{target}, 1, false, null, null, null, .{});
    const bundle_config_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_config_file_name });
    defer allocator.free(bundle_config_path);
    try writeAdapterConfigJson(allocator, bundle_config_path, .{
        .base_model_name_or_path = base_dir,
        .rank = 1,
        .alpha = 1,
        .target_modules = &.{"model.layers.34.self_attn.q_proj"},
    });

    var bundle_inspection = try inspectLoRABundle(allocator, base_dir, adapter_dir);
    defer freeLoRABundleInspectionSummary(allocator, &bundle_inspection);
    try std.testing.expectEqual(@as(usize, 1), bundle_inspection.resolved_tensor_count);
    try std.testing.expectEqualStrings("model.layers.34.self_attn.q_proj.weight", bundle_inspection.tensors[0].base_tensor_name);

    var materialized = try materializeMergedModel(allocator, base_dir, adapter_dir, merged_dir);
    defer freeMaterializeSummary(allocator, &materialized);
    var merged_reader = try safetensors.MMapReader.openFileAbsolute(allocator, materialized.output_checkpoint_path);
    defer merged_reader.deinit();
    try std.testing.expect(merged_reader.header.tensors.contains(raw_q_name));
    try std.testing.expect(!merged_reader.header.tensors.contains("model.layers.34.self_attn.q_proj.weight"));
}

fn syntheticGemma4PresetTargetCount(
    layer_count: usize,
    kv_projection_layer_count: usize,
    preset: Gemma4LoRATargetPreset,
) !usize {
    var count: usize = 0;
    const selection = BootstrapTargetSelection{ .gemma4 = preset };
    const common_suffixes = [_][]const u8{
        "attn_q.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "inp_gate.weight",
        "proj.weight",
    };
    for (0..layer_count) |layer| {
        for (common_suffixes) |suffix| {
            var name_buf: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.{s}", .{ layer, suffix });
            if (targetMatchesSelection(name, selection)) count += 1;
        }
        if (layer < kv_projection_layer_count) {
            for ([_][]const u8{ "attn_k.weight", "attn_v.weight" }) |suffix| {
                var name_buf: [96]u8 = undefined;
                const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.{s}", .{ layer, suffix });
                if (targetMatchesSelection(name, selection)) count += 1;
            }
        }
    }
    if (targetMatchesSelection("per_layer_model_proj.weight", selection)) count += 1;
    return count;
}

test "gemma4 initializer frontend rejects residual-adjusted bootstrap and keeps EVA LoRA-GA paths" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidLoRARank, bootstrapLoRABundle(allocator, "missing-base", "missing-out", .{ .rank = 0 }));
    if (comptime @bitSizeOf(usize) > @bitSizeOf(u32)) {
        try std.testing.expectError(error.InvalidLoRARank, bootstrapLoRABundle(allocator, "missing-base", "missing-out", .{ .rank = @as(usize, std.math.maxInt(u32)) + 1 }));
    }
    try std.testing.expectError(error.InvalidLoRAAlpha, bootstrapLoRABundle(allocator, "missing-base", "missing-out", .{ .alpha = 0 }));
    try std.testing.expectError(error.InvalidLoRAAlpha, bootstrapLoRABundle(allocator, "missing-base", "missing-out", .{ .alpha = std.math.nan(f32) }));
    try std.testing.expectError(error.InvalidLoRAAlpha, bootstrapLoRABundle(allocator, "missing-base", "missing-out", .{ .alpha = std.math.inf(f32) }));
    try std.testing.expectError(
        error.LoRAInitializerRequiresAdjustedBase,
        bootstrapLoRABundle(allocator, "missing-base", "missing-out", .{ .init_lora_weights = "pissa" }),
    );
    try std.testing.expectError(
        error.LoRAInitializerRequiresAdjustedBase,
        bootstrapLoRABundle(allocator, "missing-base", "missing-out", .{ .init_lora_weights = "loftq-nf4" }),
    );

    try std.testing.expectEqual(LoRAInitKind.eva, try parseLoRAInitKind("eva"));
    try std.testing.expectEqual(LoRAInitKind.lora_ga, try parseLoRAInitKind("lora-ga"));
    try std.testing.expectEqual(LoRAInitKind.lora_ga, try parseLoRAInitKind("loraga"));
    try std.testing.expectEqual(LoRAInitKind.lora_ga, try parseLoRAInitKind("lora_ga"));

    const eva_cov = [_]f32{
        4.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.5,
    };
    const lora_ga_grad = [_]f32{
        1.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    };
    const eva = try buildInitialLoRAFactors(allocator, .eva, null, &eva_cov, null, 2, 3, 1);
    defer {
        allocator.free(eva.a);
        allocator.free(eva.b);
    }
    const lora_ga = try buildInitialLoRAFactors(allocator, .lora_ga, null, null, &lora_ga_grad, 2, 3, 1);
    defer {
        allocator.free(lora_ga.a);
        allocator.free(lora_ga.b);
    }
    try std.testing.expectEqual(@as(usize, 3), eva.a.len);
    try std.testing.expectEqual(@as(usize, 2), eva.b.len);
    try std.testing.expectEqual(@as(usize, 3), lora_ga.a.len);
    try std.testing.expectEqual(@as(usize, 2), lora_ga.b.len);
    try std.testing.expectError(error.MissingInitializerStats, buildInitialLoRAFactors(allocator, .eva, null, null, null, 2, 3, 1));
}

test "gemma4 adapter inventory is exact paired and DoRA closed" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const adapter_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "adapter" });
    defer allocator.free(adapter_dir);
    try compat.cwd().createDirPath(compat.io(), adapter_dir);
    const checkpoint_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    const config_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_config_file_name });
    defer allocator.free(config_path);

    const module = "model.layers.0.self_attn.q_proj";
    const a_name = module ++ ".weight.lora_A.weight";
    const b_name = module ++ ".weight.lora_B.weight";
    const magnitude_name = module ++ ".weight.lora_magnitude_vector.weight";
    const a_data = [_]f32{ 1, 0 };
    const b_data = [_]f32{ 0, 1 };
    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = "base",
        .rank = 1,
        .alpha = 1,
        .target_modules = &.{module},
    });

    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{.{ .name = a_name, .shape = &.{ 1, 2 }, .data = &a_data }});
    try std.testing.expectError(error.MissingAdapterPair, validateLoRAAdapterInventory(allocator, adapter_dir));

    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{.{ .name = b_name, .shape = &.{ 2, 1 }, .data = &b_data }});
    try std.testing.expectError(error.MissingAdapterPair, validateLoRAAdapterInventory(allocator, adapter_dir));

    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{
        .{ .name = a_name, .shape = &.{ 1, 2 }, .data = &a_data },
        .{ .name = b_name, .shape = &.{ 2, 1 }, .data = &b_data },
    });
    try validateLoRAAdapterInventory(allocator, adapter_dir);

    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = "base",
        .rank = 1,
        .alpha = 1,
        .target_modules = &.{"model.layers.0.self_attn.k_proj"},
    });
    try std.testing.expectError(error.AdapterTargetInventoryMismatch, validateLoRAAdapterInventory(allocator, adapter_dir));

    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = "base",
        .rank = 1,
        .alpha = 1,
        .target_modules = &.{ module, module },
    });
    try std.testing.expectError(error.DuplicateAdapterTargetModule, validateLoRAAdapterInventory(allocator, adapter_dir));

    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = "base",
        .rank = 1,
        .alpha = 1,
        .target_modules = &.{module},
        .use_dora = true,
    });
    try std.testing.expectError(error.AdapterDoRAConfigMismatch, validateLoRAAdapterInventory(allocator, adapter_dir));

    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{
        .{ .name = a_name, .shape = &.{ 1, 2 }, .data = &a_data },
        .{ .name = b_name, .shape = &.{ 2, 1 }, .data = &b_data },
        .{ .name = magnitude_name, .shape = &.{2}, .data = &b_data },
    });
    try validateLoRAAdapterInventory(allocator, adapter_dir);

    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = "base",
        .rank = 1,
        .alpha = 1,
        .target_modules = &.{module},
        .use_dora = false,
    });
    try std.testing.expectError(error.AdapterDoRAConfigMismatch, validateLoRAAdapterInventory(allocator, adapter_dir));

    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{
        .{ .name = a_name, .shape = &.{ 1, 2 }, .data = &a_data },
        .{ .name = b_name, .shape = &.{ 2, 1 }, .data = &b_data },
        .{ .name = "optimizer.moment", .shape = &.{1}, .data = &.{0} },
    });
    try std.testing.expectError(error.UnexpectedAdapterTensor, validateLoRAAdapterInventory(allocator, adapter_dir));
}

test "gemma4 adapter config stays PEFT compatible and sidecar binds provenance" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const adapter_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "adapter" });
    defer allocator.free(adapter_dir);
    try compat.cwd().createDirPath(compat.io(), adapter_dir);
    const checkpoint_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    const config_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_config_file_name });
    defer allocator.free(config_path);
    const manifest_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_manifest_file_name });
    defer allocator.free(manifest_path);

    const module = "model.layers.0.self_attn.q_proj";
    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{
        .{ .name = module ++ ".weight.lora_A.weight", .shape = &.{ 1, 2 }, .data = &.{ 1, 0 } },
        .{ .name = module ++ ".weight.lora_B.weight", .shape = &.{ 2, 1 }, .data = &.{ 0, 1 } },
    });
    const options = AdapterConfigWriteOptions{
        .base_model_name_or_path = "google/gemma-4-e2b-it",
        .base_model_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        .tokenizer_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        .chat_template_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        .rank = 1,
        .alpha = 2,
        .target_modules = &.{module},
        .target_preset = "peft-qv",
    };
    try writeAdapterConfigJson(allocator, config_path, options);
    try writeAdapterManifestJson(allocator, manifest_path, options);

    const config_bytes = try c_file.readFile(allocator, config_path);
    defer allocator.free(config_bytes);
    try std.testing.expect(std.mem.indexOf(u8, config_bytes, "\"init_lora_weights\": true") != null);
    try std.testing.expect(std.mem.indexOf(u8, config_bytes, "antfly_") == null);
    try std.testing.expect(std.mem.indexOf(u8, config_bytes, "target_preset") == null);
    try std.testing.expect(std.mem.indexOf(u8, config_bytes, "recursive_lora") == null);

    var inspected = try inspectCheckpoint(allocator, adapter_dir);
    defer freeInspectionSummary(allocator, &inspected);
    try std.testing.expectEqualStrings(options.base_model_sha256.?, inspected.base_model_sha256.?);
    try std.testing.expectEqualStrings(options.tokenizer_sha256.?, inspected.tokenizer_sha256.?);
    try std.testing.expectEqualStrings(options.chat_template_sha256.?, inspected.chat_template_sha256.?);
    try std.testing.expectEqualStrings("peft-qv", inspected.target_preset.?);
    try std.testing.expect(inspected.init_lora_weights == null);
    try validateLoRAAdapterInventory(allocator, adapter_dir);

    const original_checkpoint = try c_file.readFile(allocator, checkpoint_path);
    defer allocator.free(original_checkpoint);
    const tampered_checkpoint = try allocator.dupe(u8, original_checkpoint);
    defer allocator.free(tampered_checkpoint);
    tampered_checkpoint[tampered_checkpoint.len - 1] ^= 1;
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = checkpoint_path, .data = tampered_checkpoint });
    try std.testing.expectError(error.AdapterCheckpointDigestMismatch, inspectCheckpoint(allocator, adapter_dir));
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = checkpoint_path, .data = original_checkpoint });

    try writeAdapterManifestJson(allocator, manifest_path, .{
        .base_model_name_or_path = options.base_model_name_or_path,
        .base_model_sha256 = options.base_model_sha256,
        .tokenizer_sha256 = options.tokenizer_sha256,
        .chat_template_sha256 = options.chat_template_sha256,
        .rank = options.rank,
        .alpha = options.alpha,
        .target_modules = &.{"model.layers.0.self_attn.k_proj"},
    });
    try std.testing.expectError(error.AdapterManifestConfigMismatch, inspectCheckpoint(allocator, adapter_dir));
}

test "gemma4 legacy materialization admission is bounded and fail closed" {
    try validateLegacyMaterializationSource("model.safetensors", legacy_materialize_max_checkpoint_bytes);
    try std.testing.expectError(
        error.Gemma4MaterializationRequiresStreaming,
        validateLegacyMaterializationSource("model.safetensors", legacy_materialize_max_checkpoint_bytes + 1),
    );
    try std.testing.expectError(
        error.Gemma4MaterializationRequiresStreaming,
        validateLegacyMaterializationSource("model.safetensors.index.json", 1),
    );
    try std.testing.expectError(
        error.Gemma4GgufMaterializationUnsupported,
        validateLegacyMaterializationSource("model.gguf", 1),
    );
    try std.testing.expectError(error.UnsupportedMaterializationSource, validateLegacyMaterializationSource("model.bin", 1));
}

test "gemma4 external adjusted-base initializers cannot load or materialize" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const base_dir = try std.fs.path.join(allocator, &.{ root, "base" });
    defer allocator.free(base_dir);
    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    try compat.cwd().createDirPath(compat.io(), base_dir);
    try compat.cwd().createDirPath(compat.io(), adapter_dir);

    const tensor_name = "model.layers.0.self_attn.q_proj.weight";
    const base_path = try std.fs.path.join(allocator, &.{ base_dir, checkpoint_file_name });
    defer allocator.free(base_path);
    try writeHeaderAndTensorsF32(allocator, base_path, &.{.{
        .name = tensor_name,
        .shape = &.{ 2, 2 },
        .data = &.{ 1, 0, 0, 1 },
    }});
    const adapter_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_checkpoint_file_name });
    defer allocator.free(adapter_path);
    try writeBootstrapAdapterCheckpoint(allocator, adapter_path, base_path, &.{.{
        .tensor_name = tensor_name,
        .module_name = "q_proj",
        .input_dim = 2,
        .output_dim = 2,
    }}, 1, false, null, null, null, .{});
    const config_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_config_file_name });
    defer allocator.free(config_path);

    const cases = [_]struct { initializer: []const u8, out_name: []const u8 }{
        .{ .initializer = "pissa", .out_name = "pissa-out" },
        .{ .initializer = "loftq-nf4", .out_name = "loftq-out" },
    };
    for (cases) |case| {
        try writeAdapterConfigJson(allocator, config_path, .{
            .base_model_name_or_path = base_dir,
            .rank = 1,
            .alpha = 1,
            .target_modules = &.{"model.layers.0.self_attn.q_proj"},
            .init_lora_weights = case.initializer,
        });
        try std.testing.expectError(
            error.LoRAInitializerRequiresAdjustedBase,
            loadLoRABundleScoped(allocator, base_dir, adapter_dir, null),
        );

        const out_dir = try std.fs.path.join(allocator, &.{ root, case.out_name });
        defer allocator.free(out_dir);
        try compat.cwd().createDirPath(compat.io(), out_dir);
        const sentinel_path = try std.fs.path.join(allocator, &.{ out_dir, "sentinel.txt" });
        defer allocator.free(sentinel_path);
        try compat.cwd().writeFile(compat.io(), .{ .sub_path = sentinel_path, .data = "preserve" });
        try std.testing.expectError(
            error.LoRAInitializerRequiresAdjustedBase,
            materializeMergedModel(allocator, base_dir, adapter_dir, out_dir),
        );

        const sentinel = try compat.cwd().readFileAlloc(compat.io(), sentinel_path, allocator, .limited(16));
        defer allocator.free(sentinel);
        try std.testing.expectEqualStrings("preserve", sentinel);
        const output_checkpoint = try std.fs.path.join(allocator, &.{ out_dir, checkpoint_file_name });
        defer allocator.free(output_checkpoint);
        try std.testing.expect(!isRegularFilePath(output_checkpoint));
    }
}

test "gemma4 generic bundle lifecycle rejects inconsistent and recursive adapters before save" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const base_dir = try std.fs.path.join(allocator, &.{ root, "base" });
    defer allocator.free(base_dir);
    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    const out_dir = try std.fs.path.join(allocator, &.{ root, "out" });
    defer allocator.free(out_dir);
    try compat.cwd().createDirPath(compat.io(), base_dir);
    try compat.cwd().createDirPath(compat.io(), adapter_dir);
    try compat.cwd().createDirPath(compat.io(), out_dir);

    const tensor_name = "model.layers.0.self_attn.q_proj.weight";
    const target_modules = [_][]const u8{"model.layers.0.self_attn.q_proj"};
    const target = LoRATargetTensor{
        .tensor_name = tensor_name,
        .module_name = "q_proj",
        .input_dim = 2,
        .output_dim = 2,
    };
    const base_path = try std.fs.path.join(allocator, &.{ base_dir, checkpoint_file_name });
    defer allocator.free(base_path);
    try writeHeaderAndTensorsF32(allocator, base_path, &.{.{
        .name = tensor_name,
        .shape = &.{ 2, 2 },
        .data = &.{ 1, 0, 0, 1 },
    }});
    const adapter_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_checkpoint_file_name });
    defer allocator.free(adapter_path);
    try writeBootstrapAdapterCheckpoint(allocator, adapter_path, base_path, &.{target}, 1, false, null, null, null, .{});
    const config_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_config_file_name });
    defer allocator.free(config_path);

    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = base_dir,
        .rank = 0,
        .alpha = 1,
        .target_modules = &target_modules,
    });
    try std.testing.expectError(error.InvalidLoRARank, loadLoRABundle(allocator, base_dir, adapter_dir));

    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = base_dir,
        .rank = 2,
        .alpha = 1,
        .target_modules = &target_modules,
    });
    try std.testing.expectError(error.AdapterConfigRankMismatch, loadLoRABundle(allocator, base_dir, adapter_dir));

    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = base_dir,
        .rank = 1,
        .alpha = 0,
        .target_modules = &target_modules,
    });
    try std.testing.expectError(error.InvalidLoRAAlpha, loadLoRABundle(allocator, base_dir, adapter_dir));

    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = base_dir,
        .rank = 1,
        .alpha = 1,
        .target_modules = &target_modules,
        .use_dora = true,
    });
    try std.testing.expectError(error.AdapterDoRAConfigMismatch, loadLoRABundle(allocator, base_dir, adapter_dir));

    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = base_dir,
        .rank = 1,
        .alpha = 1,
        .target_modules = &target_modules,
    });
    const manifest_path = try std.fs.path.join(allocator, &.{ adapter_dir, adapter_manifest_file_name });
    defer allocator.free(manifest_path);
    try writeAdapterManifestJson(allocator, manifest_path, .{
        .base_model_name_or_path = base_dir,
        .base_model_sha256 = "0000000000000000000000000000000000000000000000000000000000000000",
        .tokenizer_sha256 = "1111111111111111111111111111111111111111111111111111111111111111",
        .chat_template_sha256 = "2222222222222222222222222222222222222222222222222222222222222222",
        .rank = 1,
        .alpha = 1,
        .target_modules = &target_modules,
        .recursive_lora = .{
            .enabled = true,
            .source_num_layers = 2,
            .shared_block_size = 1,
            .loop_count = 2,
        },
    });
    const sentinel_path = try std.fs.path.join(allocator, &.{ out_dir, "sentinel.txt" });
    defer allocator.free(sentinel_path);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = sentinel_path, .data = "preserve" });
    try std.testing.expectError(
        error.RecursiveLoRANotSupportedByGenericLifecycle,
        materializeMergedModel(allocator, base_dir, adapter_dir, out_dir),
    );
    const sentinel = try compat.cwd().readFileAlloc(compat.io(), sentinel_path, allocator, .limited(16));
    defer allocator.free(sentinel);
    try std.testing.expectEqualStrings("preserve", sentinel);
    try compat.cwd().deleteFile(compat.io(), manifest_path);

    try writeBootstrapAdapterCheckpoint(allocator, adapter_path, base_path, &.{target}, 1, true, null, null, null, .{});
    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = base_dir,
        .rank = 1,
        .alpha = 1,
        .target_modules = &target_modules,
    });
    try std.testing.expectError(error.AdapterDoRAConfigMismatch, loadLoRABundle(allocator, base_dir, adapter_dir));

    try writeAdapterConfigJson(allocator, config_path, .{
        .base_model_name_or_path = base_dir,
        .rank = 1,
        .alpha = 1,
        .target_modules = &target_modules,
        .use_dora = true,
    });
    var bundle = try loadLoRABundle(allocator, base_dir, adapter_dir);
    defer bundle.deinit();

    const published_dir = try std.fs.path.join(allocator, &.{ root, "published" });
    defer allocator.free(published_dir);
    try saveLoRABundle(&bundle, published_dir);
    const published_adapter_path = try std.fs.path.join(allocator, &.{ published_dir, adapter_checkpoint_file_name });
    defer allocator.free(published_adapter_path);
    const published_before = try c_file.readFile(allocator, published_adapter_path);
    defer allocator.free(published_before);
    try std.testing.expectError(error.Gemma4RunOutputAlreadyExists, saveLoRABundle(&bundle, published_dir));
    const published_after = try c_file.readFile(allocator, published_adapter_path);
    defer allocator.free(published_after);
    try std.testing.expectEqualSlices(u8, published_before, published_after);

    bundle.recursive_lora_enabled = true;
    try std.testing.expectError(
        error.RecursiveLoRANotSupportedByGenericLifecycle,
        saveLoRABundle(&bundle, out_dir),
    );
    const saved_adapter_path = try std.fs.path.join(allocator, &.{ out_dir, adapter_checkpoint_file_name });
    defer allocator.free(saved_adapter_path);
    try std.testing.expect(!isRegularFilePath(saved_adapter_path));
}

test "gemma4 bootstrap EVA and LoRA-GA require and consume stats files" {
    const allocator = std.testing.allocator;
    const root = try std.fmt.allocPrint(allocator, "/tmp/termite_gemma4_real_initializer_stats_test_{d}", .{std.posix.system.getpid()});
    defer allocator.free(root);
    compat.cwd().deleteTree(compat.io(), root) catch {};
    try compat.cwd().createDirPath(compat.io(), root);
    defer compat.cwd().deleteTree(compat.io(), root) catch {};
    try writeGemma4BootstrapTestConfig(allocator, root, 1, 0);

    const checkpoint_path = try std.fs.path.join(allocator, &.{ root, checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    const tensor_name = "model.layers.0.self_attn.q_proj.weight";
    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{
        .{ .name = tensor_name, .shape = &.{ 2, 3 }, .data = &.{ 1, 0, 0, 0, 1, 0 } },
    });

    const eva_stats_path = try std.fs.path.join(allocator, &.{ root, "eva_stats.safetensors" });
    defer allocator.free(eva_stats_path);
    const eva_stats_name = try std.fmt.allocPrint(allocator, "{s}.eva_activation_covariance", .{tensor_name});
    defer allocator.free(eva_stats_name);
    try writeHeaderAndTensorsF32(allocator, eva_stats_path, &.{
        .{ .name = eva_stats_name, .shape = &.{ 3, 3 }, .data = &.{ 4, 0, 0, 0, 1, 0, 0, 0, 0.5 } },
    });

    const ga_stats_path = try std.fs.path.join(allocator, &.{ root, "lora_ga_stats.safetensors" });
    defer allocator.free(ga_stats_path);
    const ga_stats_name = try std.fmt.allocPrint(allocator, "{s}.lora_ga_gradient", .{tensor_name});
    defer allocator.free(ga_stats_name);
    try writeHeaderAndTensorsF32(allocator, ga_stats_path, &.{
        .{ .name = ga_stats_name, .shape = &.{ 2, 3 }, .data = &.{ 1, 0, 0, 0, 0, 0 } },
    });

    const targets = [_][]const u8{"q_proj"};
    const missing_out_dir = try std.fs.path.join(allocator, &.{ root, "missing" });
    defer allocator.free(missing_out_dir);
    const eva_out_dir = try std.fs.path.join(allocator, &.{ root, "eva" });
    defer allocator.free(eva_out_dir);
    const ga_out_dir = try std.fs.path.join(allocator, &.{ root, "ga" });
    defer allocator.free(ga_out_dir);
    try std.testing.expectError(error.MissingInitializerStats, bootstrapLoRABundle(allocator, root, missing_out_dir, .{
        .rank = 1,
        .target_modules = targets[0..],
        .init_lora_weights = "eva",
    }));

    var eva_summary = try bootstrapLoRABundle(allocator, root, eva_out_dir, .{
        .rank = 1,
        .target_modules = targets[0..],
        .init_lora_weights = "eva",
        .eva_stats_path = eva_stats_path,
    });
    defer freeBootstrapSummary(allocator, &eva_summary);

    var ga_summary = try bootstrapLoRABundle(allocator, root, ga_out_dir, .{
        .rank = 1,
        .target_modules = targets[0..],
        .init_lora_weights = "lora-ga",
        .lora_ga_stats_path = ga_stats_path,
    });
    defer freeBootstrapSummary(allocator, &ga_summary);

    try std.testing.expectEqualStrings("eva", eva_summary.init_lora_weights.?);
    try std.testing.expectEqualStrings("lora-ga", ga_summary.init_lora_weights.?);

    const adapter_config = try c_file.readFile(allocator, eva_summary.adapter_config_path);
    defer allocator.free(adapter_config);
    try std.testing.expect(std.mem.indexOf(u8, adapter_config, "eva_stats_path") == null);
    try std.testing.expect(std.mem.indexOf(u8, adapter_config, "lora_ga_stats_path") == null);
}

test "gemma4 moe expert preset targets only expert parameter tensors" {
    const allocator = std.testing.allocator;
    const root = try std.fmt.allocPrint(allocator, "/tmp/termite_gemma4_moe_expert_targets_test_{d}", .{std.posix.system.getpid()});
    defer allocator.free(root);
    compat.cwd().deleteTree(compat.io(), root) catch {};
    try compat.cwd().createDirPath(compat.io(), root);
    defer compat.cwd().deleteTree(compat.io(), root) catch {};

    const checkpoint_path = try std.fs.path.join(allocator, &.{ root, checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    const expert_name = "model.layers.0.block_sparse_moe.experts.3.w2.weight";
    const dense_name = "model.layers.0.mlp.down_proj.weight";
    const router_name = "model.layers.0.block_sparse_moe.router.weight";
    try writeHeaderAndTensorsF32(allocator, checkpoint_path, &.{
        .{ .name = expert_name, .shape = &.{ 2, 3 }, .data = &.{ 1, 2, 3, 4, 5, 6 } },
        .{ .name = dense_name, .shape = &.{ 2, 3 }, .data = &.{ 7, 8, 9, 10, 11, 12 } },
        .{ .name = router_name, .shape = &.{ 4, 3 }, .data = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    });

    const targets = try inferLoRATargetTensors(
        checkpoint_path,
        allocator,
        .{ .legacy = .moe_experts },
    );
    defer freeLoRATargetTensors(allocator, targets);

    try std.testing.expectEqual(@as(usize, 1), targets.len);
    try std.testing.expectEqualStrings(expert_name, targets[0].tensor_name);
    try std.testing.expectEqualStrings("moe_expert", targets[0].module_name);
    try std.testing.expectEqual(@as(usize, 2), targets[0].output_dim);
    try std.testing.expectEqual(@as(usize, 3), targets[0].input_dim);
}
