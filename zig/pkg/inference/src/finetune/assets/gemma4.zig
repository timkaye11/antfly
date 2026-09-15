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

//! Offline Gemma 4 checkpoint, adapter, and artifact operations.

const std = @import("std");
const platform = @import("antfly_platform");
const compat = @import("../../io/compat.zig");
const lora = @import("../lora.zig");
const lora_init = @import("../lora_init.zig");
const peft = @import("../peft.zig");
const qlora_nf4 = @import("../qlora_nf4.zig");
const recursive_lora = @import("../recursive_lora.zig");
const artifact_publication = @import("../artifact_publication.zig");
const safetensors = @import("../../models/safetensors.zig");
const tensor_mod = @import("../../backends/tensor.zig");
pub const DType = tensor_mod.DType;
const Tensor = tensor_mod.Tensor;
const tensor_access = @import("../../models/tensor_access.zig");
const weight_source = @import("../../models/weight_source.zig");
const gpt_model = @import("../../models/gpt.zig");
const manifest_mod = @import("../../models/manifest.zig");
const gemma4_mm = @import("../../architectures/gemma4_multimodal.zig");
const session_factory = @import("../../architectures/session_factory.zig");
const c_file = @import("../../util/c_file.zig");

pub const artifact_family_version = "gemma4_lora/v1alpha1";
pub const checkpoint_file_name = "model.safetensors";
pub const adapter_checkpoint_file_name = "adapter_model.safetensors";
pub const hf_config_file_name = "config.json";
pub const adapter_config_file_name = "adapter_config.json";
pub const adapter_manifest_file_name = "antfly_finetune_manifest.json";
pub const adapter_manifest_schema_v2 = "antfly_gemma4_finetune/v2";
pub const adapter_manifest_schema_v3 = "antfly_gemma4_finetune/v3";
pub const adapter_tensor_key_format_v1 = "antfly_gemma4_adapter_keys/v1";
pub const stock_peft_tensor_key_format_v1 = "stock-peft/v1";
pub const peft_export_manifest_file_name = "antfly_peft_export.json";
pub const peft_export_manifest_schema_v1 = "antfly_gemma4_peft_export/v1";
pub const tokenizer_config_file_name = "tokenizer_config.json";
pub const tokenizer_file_name = "tokenizer.json";
pub const special_tokens_map_file_name = "special_tokens_map.json";

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
    /// Seed for the deterministic adapter initializer. Version-2 manifests
    /// predate this field; version-3 manifests must bind it explicitly.
    initialization_seed: ?u64 = null,
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
    initialization_seed: ?u64 = null,
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
    /// Zero preserves the original deterministic initialization exactly.
    /// Non-zero values produce independent, reproducible LoRA-A tensors.
    initialization_seed: u64 = 0,
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
    initialization_seed: u64 = 0,
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

pub const PeftExportManifest = struct {
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
    initialization_seed: ?u64 = null,
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
    initialization_seed: ?u64 = null,
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

pub const LoRAInitKind = enum {
    default,
    pissa,
    loftq_nf4,
    eva,
    lora_ga,
};

const prepared_chat_template_identity = "antfly_gemma_chat/v1";

pub fn sha256HexAlloc(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(bytes);
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
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

pub fn validateSha256Hex(value: []const u8) !void {
    if (value.len != std.crypto.hash.sha2.Sha256.digest_length * 2) return error.InvalidPreparedFingerprint;
    for (value) |byte| if (!std.ascii.isHex(byte)) return error.InvalidPreparedFingerprint;
}

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
        const is_v2 = std.mem.eql(u8, manifest.schema_version, adapter_manifest_schema_v2);
        const is_v3 = std.mem.eql(u8, manifest.schema_version, adapter_manifest_schema_v3);
        if ((!is_v2 and !is_v3) or
            !std.mem.eql(u8, manifest.status, "complete") or
            !std.mem.eql(u8, manifest.artifact_family_version, artifact_family_version) or
            !std.mem.eql(u8, manifest.tensor_key_format, adapter_tensor_key_format_v1))
        {
            return error.InvalidAdapterManifest;
        }
        if ((is_v2 and manifest.initialization_seed != null) or
            (is_v3 and manifest.initialization_seed == null))
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
        .initialization_seed = if (adapter_manifest) |manifest| manifest.initialization_seed else null,
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

    try writeBootstrapAdapterCheckpointAtomic(allocator, staging_adapter_checkpoint_path, checkpoint_path, resolved_tensors, options.rank, options.use_dora, options.init_lora_weights, options.initialization_seed, options.eva_stats_path, options.lora_ga_stats_path, recursive_config);
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
        .initialization_seed = options.initialization_seed,
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
        .initialization_seed = options.initialization_seed,
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

pub fn stockPeftModulePathAlloc(allocator: std.mem.Allocator, module_path: []const u8) ![]u8 {
    // Keep the checkpoint's text-only or multimodal root. Only the PLE
    // module spelling differs between Antfly's graph and stock HF Gemma4.
    const aliases = [_][2][]const u8{
        .{ "per_layer_input.per_layer_model_proj", "per_layer_model_projection" },
        .{ "per_layer_input.inp_gate", "per_layer_input_gate" },
        .{ "per_layer_input.proj", "per_layer_projection" },
    };
    for (aliases) |pair| {
        if (!std.mem.endsWith(u8, module_path, pair[0])) continue;
        const prefix_len = module_path.len - pair[0].len;
        if (prefix_len != 0 and module_path[prefix_len - 1] != '.') continue;
        return std.fmt.allocPrint(allocator, "{s}{s}", .{ module_path[0..prefix_len], pair[1] });
    }
    return allocator.dupe(u8, module_path);
}

pub fn stockPeftAdapterConfigAlloc(allocator: std.mem.Allocator, source: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, source, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidAdapterConfig;
    const targets = parsed.value.object.getPtr("target_modules") orelse return error.MissingAdapterTargets;
    if (targets.* != .array) return error.InvalidAdapterTargets;
    var changed = false;
    var seen = std.StringHashMapUnmanaged(void).empty;
    defer seen.deinit(allocator);
    for (targets.array.items) |*target| {
        if (target.* != .string) return error.InvalidAdapterTargets;
        const translated = try stockPeftModulePathAlloc(parsed.arena.allocator(), target.string);
        changed = changed or !std.mem.eql(u8, target.string, translated);
        target.* = .{ .string = translated };
        const entry = try seen.getOrPut(allocator, translated);
        if (entry.found_existing) return error.PeftExportTargetModuleCollision;
    }
    // Preserve byte-for-byte config compatibility when no alias changes.
    if (!changed) return allocator.dupe(u8, source);
    var buffer: std.Io.Writer.Allocating = .init(allocator);
    defer buffer.deinit();
    try std.json.Stringify.value(parsed.value, .{ .whitespace = .indent_2 }, &buffer.writer);
    try buffer.writer.writeByte('\n');
    return allocator.dupe(u8, buffer.written());
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
        const source_module_path = tensorModulePath(parsed.base_tensor_base_name) orelse return error.InvalidLoRATargetTensorName;
        const module_path = try stockPeftModulePathAlloc(allocator, source_module_path);
        defer allocator.free(module_path);
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
    const destination_config = try stockPeftAdapterConfigAlloc(allocator, source_config);
    defer allocator.free(destination_config);
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
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = staging_config_path, .data = destination_config });

    var destination_checkpoint = try c_file.MmapRegion.init(allocator, staging_checkpoint_path);
    defer destination_checkpoint.deinit();
    const destination_checkpoint_sha256 = try sha256HexAlloc(allocator, destination_checkpoint.data);
    errdefer allocator.free(destination_checkpoint_sha256);
    const adapter_config_sha256 = try sha256HexAlloc(allocator, destination_config);
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
        .initialization_seed = adapter_inspect.initialization_seed,
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
        .initialization_seed = inspected.initialization_seed,
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
        .initialization_seed = bundle.initialization_seed,
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
    try validateMaterializationSource(base_checkpoint_path);

    var inspected = try inspectLoRABundle(allocator, base_model_dir, adapter_model_dir);
    defer freeLoRABundleInspectionSummary(allocator, &inspected);
    try validateGenericLoRABundleInspection(inspected);

    var publication = try artifact_publication.ImmutableDirectoryPublication.init(allocator, compat.io(), out_dir);
    defer publication.deinit();
    var base_access = try openTensorAccessForFile(allocator, base_checkpoint_path);
    defer base_access.deinit();
    var adapter_access = try openTensorAccessForFile(allocator, inspected.adapter_checkpoint_path);
    defer adapter_access.deinit();
    const base_names = try base_access.listNames(allocator);
    defer allocator.free(base_names);

    var materialized_targets = std.StringHashMapUnmanaged(usize).empty;
    defer materialized_targets.deinit(allocator);
    for (inspected.tensors, 0..) |tensor, index| {
        const source_name = sourceTensorNameForCanonicalAdapterBase(base_names, tensor.base_tensor_name) orelse
            return error.MissingBaseTensorForAdapter;
        const result = try materialized_targets.getOrPut(allocator, source_name);
        if (result.found_existing) return error.DuplicateAdapterTargetTensor;
        result.value_ptr.* = index;
    }

    try publication.createStaging();
    const staging_checkpoint_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, checkpoint_file_name });
    defer allocator.free(staging_checkpoint_path);
    try writeStreamingMergedSafetensors(
        allocator,
        base_access,
        adapter_access,
        base_names,
        &materialized_targets,
        inspected,
        staging_checkpoint_path,
    );

    try copySupportingArtifactIfPresent(allocator, base_paths.config_path, publication.staging_dir, hf_config_file_name);
    try copySupportingArtifactIfPresent(allocator, base_paths.tokenizer_config_path, publication.staging_dir, tokenizer_config_file_name);
    try copySupportingArtifactIfPresent(allocator, base_paths.tokenizer_path, publication.staging_dir, tokenizer_file_name);
    try copySupportingArtifactIfPresent(allocator, base_paths.special_tokens_map_path, publication.staging_dir, special_tokens_map_file_name);

    var adapter_paths = try resolveArtifactPaths(allocator, adapter_model_dir);
    defer adapter_paths.deinit();
    try copySupportingArtifactIfPresent(allocator, adapter_paths.tokenizer_config_path, publication.staging_dir, tokenizer_config_file_name);
    try copySupportingArtifactIfPresent(allocator, adapter_paths.tokenizer_path, publication.staging_dir, tokenizer_file_name);
    try copySupportingArtifactIfPresent(allocator, adapter_paths.special_tokens_map_path, publication.staging_dir, special_tokens_map_file_name);

    const summary_artifact_family = try allocator.dupe(u8, artifact_family_version);
    errdefer allocator.free(summary_artifact_family);
    const summary_base_model_dir = try allocator.dupe(u8, inspected.base_model_dir);
    errdefer allocator.free(summary_base_model_dir);
    const summary_adapter_model_dir = try allocator.dupe(u8, inspected.adapter_model_dir);
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
        .merged_lora_tensor_count = inspected.tensors.len,
        .merged_dora_tensor_count = inspected.dora_magnitude_tensor_count,
        .copied_base_tensor_count = base_names.len - inspected.tensors.len,
    };
}

fn writeStreamingMergedSafetensors(
    allocator: std.mem.Allocator,
    base_access: tensor_access.TensorAccess,
    adapter_access: tensor_access.TensorAccess,
    base_names: [][]const u8,
    materialized_targets: *const std.StringHashMapUnmanaged(usize),
    inspected: LoRABundleInspectionSummary,
    output_path: []const u8,
) !void {
    const ordered_names = try allocator.dupe([]const u8, base_names);
    defer allocator.free(ordered_names);
    std.mem.sort([]const u8, ordered_names, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);

    var header: std.Io.Writer.Allocating = .init(allocator);
    defer header.deinit();
    const writer = &header.writer;
    try writer.writeAll("{\"__metadata__\":{\"format\":\"antfly-gemma4-lora-merged\"}");
    var offset: u64 = 0;
    for (ordered_names) |name| {
        var record = try base_access.getRecord(allocator, name);
        defer record.deinit();
        const dtype = switch (record.descriptor.encoding) {
            .dense => |value| value,
            .gguf => return error.Gemma4GgufMaterializationUnsupported,
        };
        if (materialized_targets.contains(name) and
            dtype != .f32 and dtype != .f16 and dtype != .bf16)
        {
            return error.UnsupportedMaterializedTensorType;
        }
        try writer.writeByte(',');
        try writeSafetensorsJsonString(writer, name);
        try writer.print(":{{\"dtype\":\"{s}\",\"shape\":[", .{dtypeName(dtype)});
        for (record.descriptor.shape, 0..) |dim, dim_index| {
            if (dim_index != 0) try writer.writeByte(',');
            try writer.print("{d}", .{dim});
        }
        const end = std.math.add(u64, offset, @intCast(record.descriptor.byte_len)) catch
            return error.MaterializedCheckpointTooLarge;
        try writer.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, end });
        offset = end;
    }
    try writer.writeByte('}');
    while (header.written().len % 8 != 0) try writer.writeByte(' ');

    const io = compat.io();
    var file = try compat.cwd().createFile(io, output_path, .{ .truncate = true });
    defer file.close(io);
    var length_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_bytes, header.written().len, .little);
    try file.writeStreamingAll(io, &length_bytes);
    try file.writeStreamingAll(io, header.written());

    const alpha: f32 = @floatCast(inspected.lora_alpha orelse return error.MissingAdapterConfig);
    for (ordered_names) |name| {
        var record = try base_access.getRecord(allocator, name);
        defer record.deinit();
        if (materialized_targets.get(name)) |target_index| {
            const dtype = switch (record.descriptor.encoding) {
                .dense => |value| value,
                .gguf => return error.Gemma4GgufMaterializationUnsupported,
            };
            try writeMergedLoRATensor(
                allocator,
                io,
                &file,
                base_access,
                adapter_access,
                inspected.tensors[target_index],
                name,
                dtype,
                alpha,
            );
        } else {
            if (record.raw_bytes.len != record.descriptor.byte_len) return error.InvalidTensorByteLength;
            try writeFileBytesChunked(io, &file, record.raw_bytes);
        }
    }
    try validateStreamingMaterializedCheckpoint(
        allocator,
        base_access,
        ordered_names,
        materialized_targets,
        output_path,
    );
}

fn writeMergedLoRATensor(
    allocator: std.mem.Allocator,
    io: std.Io,
    file: *std.Io.File,
    base_access: tensor_access.TensorAccess,
    adapter_access: tensor_access.TensorAccess,
    target: LoRATensorSummary,
    source_base_tensor_name: []const u8,
    output_dtype: DType,
    alpha: f32,
) !void {
    var base_tensor = try loadTensorAsF32(allocator, base_access, source_base_tensor_name);
    defer base_tensor.deinit();
    var a_tensor = try loadTensorAsF32(allocator, adapter_access, target.adapter_a_tensor_name);
    defer a_tensor.deinit();
    var b_tensor = try loadTensorAsF32(allocator, adapter_access, target.adapter_b_tensor_name);
    defer b_tensor.deinit();
    if (base_tensor.shape.len != 2 or
        base_tensor.shape[0] != @as(i64, @intCast(target.output_dim)) or
        base_tensor.shape[1] != @as(i64, @intCast(target.input_dim)))
    {
        return error.InvalidAdapterTensorShape;
    }

    const base_weight = try allocator.alloc(f32, target.input_dim * target.output_dim);
    defer allocator.free(base_weight);
    transpose2DF32(base_weight, base_tensor.asFloat32(), target.output_dim, target.input_dim);
    const adapter_a = try allocator.alloc(f32, target.input_dim * target.rank);
    defer allocator.free(adapter_a);
    transpose2DF32(adapter_a, a_tensor.asFloat32(), target.rank, target.input_dim);
    const adapter_b = try allocator.alloc(f32, target.rank * target.output_dim);
    defer allocator.free(adapter_b);
    transpose2DF32(adapter_b, b_tensor.asFloat32(), target.output_dim, target.rank);

    var magnitude_tensor: ?Tensor = if (target.dora_magnitude_tensor_name) |name|
        try loadTensorAsF32(allocator, adapter_access, name)
    else
        null;
    defer if (magnitude_tensor) |*tensor| tensor.deinit();
    const merged_weight = try allocator.alloc(f32, base_weight.len);
    defer allocator.free(merged_weight);
    const base_matrix = lora.Matrix{ .rows = target.input_dim, .cols = target.output_dim, .data = base_weight };
    const a_matrix = lora.Matrix{ .rows = target.input_dim, .cols = target.rank, .data = adapter_a };
    const b_matrix = lora.Matrix{ .rows = target.rank, .cols = target.output_dim, .data = adapter_b };
    if (magnitude_tensor) |*magnitude| {
        if (magnitude.shape.len != 1 or magnitude.shape[0] != @as(i64, @intCast(target.output_dim))) {
            return error.InvalidAdapterTensorShape;
        }
        lora.doraMergeInto(.{
            .base = base_matrix,
            .adapter_a = a_matrix,
            .adapter_b = b_matrix,
            .magnitude = magnitude.asFloat32(),
            .alpha = alpha,
        }, merged_weight);
    } else {
        lora.mergeInto(base_matrix, a_matrix, b_matrix, alpha, merged_weight);
    }

    const hf_weight = try allocator.alloc(f32, merged_weight.len);
    defer allocator.free(hf_weight);
    transpose2DF32(hf_weight, merged_weight, target.input_dim, target.output_dim);
    try writeF32ValuesAsDType(allocator, io, file, hf_weight, output_dtype);
}

fn writeF32ValuesAsDType(
    allocator: std.mem.Allocator,
    io: std.Io,
    file: *std.Io.File,
    values: []const f32,
    dtype: DType,
) !void {
    if (dtype != .f32 and dtype != .f16 and dtype != .bf16) return error.UnsupportedMaterializedTensorType;
    const element_bytes = dtype.byteSize();
    const chunk_elements = @max(@as(usize, 1), (1024 * 1024) / element_bytes);
    const buffer = try allocator.alloc(u8, chunk_elements * element_bytes);
    defer allocator.free(buffer);
    var start: usize = 0;
    while (start < values.len) {
        const end = @min(start + chunk_elements, values.len);
        for (values[start..end], 0..) |value, index| {
            if (!std.math.isFinite(value)) return error.NonFiniteMaterializedTensor;
            const byte_offset = index * element_bytes;
            switch (dtype) {
                .f32 => std.mem.writeInt(u32, buffer[byte_offset..][0..4], @bitCast(value), .little),
                .f16 => {
                    const half: f16 = @floatCast(value);
                    if (!std.math.isFinite(half)) return error.MaterializedTensorDtypeOverflow;
                    std.mem.writeInt(u16, buffer[byte_offset..][0..2], @bitCast(half), .little);
                },
                .bf16 => std.mem.writeInt(u16, buffer[byte_offset..][0..2], roundF32ToBf16Bits(value), .little),
                else => unreachable,
            }
        }
        try file.writeStreamingAll(io, buffer[0 .. (end - start) * element_bytes]);
        start = end;
    }
}

pub fn roundF32ToBf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    const rounding_bias: u32 = 0x7fff + ((bits >> 16) & 1);
    return @intCast((bits +% rounding_bias) >> 16);
}

fn validateStreamingMaterializedCheckpoint(
    allocator: std.mem.Allocator,
    base_access: tensor_access.TensorAccess,
    ordered_names: []const []const u8,
    materialized_targets: *const std.StringHashMapUnmanaged(usize),
    output_path: []const u8,
) !void {
    var output_access = try openTensorAccessForFile(allocator, output_path);
    defer output_access.deinit();
    const output_names = try output_access.listNames(allocator);
    defer allocator.free(output_names);
    if (output_names.len != ordered_names.len) return error.MaterializedTensorInventoryMismatch;
    for (ordered_names) |name| {
        var source = try base_access.getRecord(allocator, name);
        defer source.deinit();
        var output = output_access.getRecord(allocator, name) catch return error.MaterializedTensorInventoryMismatch;
        defer output.deinit();
        if (!std.mem.eql(i64, source.descriptor.shape, output.descriptor.shape) or
            source.descriptor.byte_len != output.descriptor.byte_len or
            denseRecordDType(source.descriptor.encoding) != denseRecordDType(output.descriptor.encoding))
        {
            return error.MaterializedTensorMetadataMismatch;
        }
        if (materialized_targets.contains(name)) {
            try validateFiniteMaterializedTensor(output);
        } else if (!std.mem.eql(u8, source.raw_bytes, output.raw_bytes)) {
            return error.MaterializedUntouchedTensorMismatch;
        }
    }
}

fn validateFiniteMaterializedTensor(record: tensor_access.Record) !void {
    const dtype = denseRecordDType(record.descriptor.encoding) orelse return error.UnsupportedMaterializedTensorType;
    if (dtype != .f32 and dtype != .f16 and dtype != .bf16) return error.UnsupportedMaterializedTensorType;
    if (record.raw_bytes.len % dtype.byteSize() != 0) return error.InvalidTensorByteLength;
    var offset: usize = 0;
    while (offset < record.raw_bytes.len) : (offset += dtype.byteSize()) {
        const value: f32 = switch (dtype) {
            .f32 => @bitCast(std.mem.readInt(u32, record.raw_bytes[offset..][0..4], .little)),
            .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, record.raw_bytes[offset..][0..2], .little)))),
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, record.raw_bytes[offset..][0..2], .little)) << 16),
            else => unreachable,
        };
        if (!std.math.isFinite(value)) return error.NonFiniteMaterializedTensor;
    }
}

fn writeSafetensorsJsonString(writer: *std.Io.Writer, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |byte| switch (byte) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        else => try writer.writeByte(byte),
    };
    try writer.writeByte('"');
}

pub fn validateMaterializationSource(checkpoint_path: []const u8) !void {
    if (std.mem.endsWith(u8, checkpoint_path, ".gguf")) return error.Gemma4GgufMaterializationUnsupported;
    if (!std.mem.endsWith(u8, checkpoint_path, ".safetensors") and
        !std.mem.endsWith(u8, checkpoint_path, ".safetensors.index.json"))
    {
        return error.UnsupportedMaterializationSource;
    }
}

pub fn freeMaterializeSummary(allocator: std.mem.Allocator, summary: *MaterializeSummary) void {
    allocator.free(summary.artifact_family_version);
    allocator.free(summary.base_model_dir);
    allocator.free(summary.adapter_model_dir);
    allocator.free(summary.output_dir);
    allocator.free(summary.output_checkpoint_path);
    summary.* = undefined;
}

pub fn layerMatchesScope(layer_base_tensor_name: []const u8, layer_name: ?[]const u8) bool {
    const selector = layer_name orelse return true;
    if (parseLayerSelectorIndex(selector)) |want_idx| {
        return parseGemma4LayerIndex(layer_base_tensor_name) == want_idx;
    }
    return std.mem.indexOf(u8, layer_base_tensor_name, selector) != null;
}

pub fn parseGemma4LayerIndex(tensor_name: []const u8) ?usize {
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
// ---------------------------------------------------------------------------
// Safetensors I/O helpers
// ---------------------------------------------------------------------------

pub const WriteTensorF32 = struct {
    name: []const u8,
    shape: []const usize,
    data: []const f32,
};

pub const WriteTensorRaw = struct {
    name: []const u8,
    dtype: tensor_mod.DType,
    shape: []const i64,
    raw_bytes: []const u8,
};

pub const BootstrapTargetSelection = union(enum) {
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

pub fn inferLoRATargetTensors(
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

pub fn targetMatchesSelection(tensor_name: []const u8, selection: BootstrapTargetSelection) bool {
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

pub fn canonicalAdapterBaseTensorName(allocator: std.mem.Allocator, source_tensor_name: []const u8) ![]u8 {
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

pub fn writeBootstrapAdapterCheckpoint(
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
    return writeBootstrapAdapterCheckpointSeeded(
        allocator,
        output_path,
        base_checkpoint_path,
        resolved_tensors,
        rank,
        use_dora,
        init_lora_weights,
        0,
        eva_stats_path,
        lora_ga_stats_path,
        recursive_config,
    );
}

fn writeBootstrapAdapterCheckpointSeeded(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    base_checkpoint_path: []const u8,
    resolved_tensors: []const LoRATargetTensor,
    rank: usize,
    use_dora: bool,
    init_lora_weights: ?[]const u8,
    initialization_seed: u64,
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

        const init = try buildInitialLoRAFactorsSeeded(
            allocator,
            init_kind,
            if (base_tensor) |tensor| tensor.asFloat32() else null,
            if (eva_stats_tensor) |tensor| tensor.asFloat32() else null,
            if (lora_ga_stats_tensor) |tensor| tensor.asFloat32() else null,
            target.output_dim,
            target.input_dim,
            rank,
            deriveLoRAInitializationSeed(initialization_seed, target.tensor_name),
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
    initialization_seed: u64,
    eva_stats_path: ?[]const u8,
    lora_ga_stats_path: ?[]const u8,
    recursive_config: recursive_lora.Config,
) !void {
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{output_path});
    defer allocator.free(tmp_path);
    compat.cwd().deleteFile(compat.io(), tmp_path) catch {};
    errdefer compat.cwd().deleteFile(compat.io(), tmp_path) catch {};

    try writeBootstrapAdapterCheckpointSeeded(
        allocator,
        tmp_path,
        base_checkpoint_path,
        resolved_tensors,
        rank,
        use_dora,
        init_lora_weights,
        initialization_seed,
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

pub fn parseLoRAInitKind(value: ?[]const u8) !LoRAInitKind {
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

pub fn buildInitialLoRAFactors(
    allocator: std.mem.Allocator,
    init_kind: LoRAInitKind,
    base_weight: ?[]const f32,
    eva_activation_covariance: ?[]const f32,
    lora_ga_gradient: ?[]const f32,
    output_dim: usize,
    input_dim: usize,
    rank: usize,
) !InitialLoRAFactors {
    return buildInitialLoRAFactorsSeeded(
        allocator,
        init_kind,
        base_weight,
        eva_activation_covariance,
        lora_ga_gradient,
        output_dim,
        input_dim,
        rank,
        0,
    );
}

fn buildInitialLoRAFactorsSeeded(
    allocator: std.mem.Allocator,
    init_kind: LoRAInitKind,
    base_weight: ?[]const f32,
    eva_activation_covariance: ?[]const f32,
    lora_ga_gradient: ?[]const f32,
    output_dim: usize,
    input_dim: usize,
    rank: usize,
    initialization_seed: u64,
) !InitialLoRAFactors {
    switch (init_kind) {
        .default => {
            return .{
                .a = try buildDeterministicLoraA(allocator, rank, input_dim, initialization_seed),
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

pub fn writeHeaderAndTensorsF32(allocator: std.mem.Allocator, path: []const u8, tensors: []const WriteTensorF32) !void {
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

pub fn writeHeaderAndRawTensors(allocator: std.mem.Allocator, path: []const u8, tensors: []const WriteTensorRaw) !void {
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
    initialization_seed: ?u64 = null,
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
        .schema_version = if (options.initialization_seed == null)
            adapter_manifest_schema_v2
        else
            adapter_manifest_schema_v3,
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
        .initialization_seed = options.initialization_seed,
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

pub fn loadTensorAsF32(allocator: std.mem.Allocator, access: tensor_access.TensorAccess, name: []const u8) !Tensor {
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

pub fn denseRecordDType(encoding: tensor_access.Encoding) ?DType {
    return switch (encoding) {
        .dense => |dtype| dtype,
        .gguf => null,
    };
}

pub fn openTensorAccessForFile(allocator: std.mem.Allocator, path: []const u8) !tensor_access.TensorAccess {
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

pub fn deriveLoRAInitializationSeed(seed: u64, tensor_name: []const u8) u64 {
    if (seed == 0) return 0;
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var seed_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &seed_bytes, seed, .little);
    hasher.update(&seed_bytes);
    hasher.update(tensor_name);
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return std.mem.readInt(u64, digest[0..8], .little);
}

fn splitMix64(value: u64) u64 {
    var z = value +% 0x9e37_79b9_7f4a_7c15;
    z = (z ^ (z >> 30)) *% 0xbf58_476d_1ce4_e5b9;
    z = (z ^ (z >> 27)) *% 0x94d0_49bb_1331_11eb;
    return z ^ (z >> 31);
}

pub fn buildDeterministicLoraA(allocator: std.mem.Allocator, rows: usize, cols: usize, initialization_seed: u64) ![]f32 {
    const data = try allocator.alloc(f32, rows * cols);
    for (0..rows) |row| {
        for (0..cols) |col| {
            const idx = row * cols + col;
            if (initialization_seed == 0) {
                const angle: f32 = @floatFromInt((row + 1) * (col + 3));
                data[idx] = @sin(angle * 0.013) * 0.01;
            } else {
                const mixed = splitMix64(initialization_seed +% @as(u64, @intCast(idx)));
                const unit = @as(f32, @floatFromInt(mixed >> 40)) / 16_777_215.0;
                data[idx] = (unit * 2.0 - 1.0) * 0.01;
            }
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

pub fn findDecoderGgufPathInDir(allocator: std.mem.Allocator, dir_path: []const u8) !?[]u8 {
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

pub fn freeLoRATargetTensors(allocator: std.mem.Allocator, tensors: []LoRATargetTensor) void {
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

pub fn dupeOptionalString(allocator: std.mem.Allocator, value: ?[]const u8) !?[]const u8 {
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

pub fn writeGemma4BootstrapTestConfig(
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

pub fn optionalStringsEqual(lhs: ?[]const u8, rhs: ?[]const u8) bool {
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
