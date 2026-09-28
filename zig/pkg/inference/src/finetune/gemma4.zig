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
// gemma4_train_command.zig.
const std = @import("std");
const assets = @import("assets/gemma4.zig");

const build_options = @import("build_options");
const compat = @import("../io/compat.zig");
const lora = @import("lora.zig");
const peft = @import("peft.zig");
const graph_bridge = @import("graph_bridge.zig");
const manifest_mod = @import("../models/manifest.zig");
const safetensors = @import("../models/safetensors.zig");
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
const gemma4_projector = @import("../architectures/gemma4_projector.zig");
const session_factory = @import("../architectures/session_factory.zig");
const model_manager_mod = @import("../server/model_manager.zig");

// Checkpoint types and operations are owned by the offline asset layer.
pub const artifact_family_version = assets.artifact_family_version;
pub const checkpoint_file_name = assets.checkpoint_file_name;
pub const adapter_checkpoint_file_name = assets.adapter_checkpoint_file_name;
pub const hf_config_file_name = assets.hf_config_file_name;
pub const adapter_config_file_name = assets.adapter_config_file_name;
pub const adapter_manifest_file_name = assets.adapter_manifest_file_name;
pub const adapter_manifest_schema_v2 = assets.adapter_manifest_schema_v2;
pub const adapter_manifest_schema_v3 = assets.adapter_manifest_schema_v3;
pub const adapter_tensor_key_format_v1 = assets.adapter_tensor_key_format_v1;
pub const stock_peft_tensor_key_format_v1 = assets.stock_peft_tensor_key_format_v1;
pub const peft_export_manifest_file_name = assets.peft_export_manifest_file_name;
pub const peft_export_manifest_schema_v1 = assets.peft_export_manifest_schema_v1;
pub const tokenizer_config_file_name = assets.tokenizer_config_file_name;
pub const tokenizer_file_name = assets.tokenizer_file_name;
pub const special_tokens_map_file_name = assets.special_tokens_map_file_name;
pub const default_lora_target_modules = assets.default_lora_target_modules;
pub const Variant = assets.Variant;
pub const Gemma4LoRATargetPreset = assets.Gemma4LoRATargetPreset;
pub const parseGemma4LoRATargetPreset = assets.parseGemma4LoRATargetPreset;
pub const Config = assets.Config;
pub const TextConfig = assets.TextConfig;
pub const AdapterConfig = assets.AdapterConfig;
pub const TokenizerConfig = assets.TokenizerConfig;
pub const InspectionSummary = assets.InspectionSummary;
pub const ArtifactPaths = assets.ArtifactPaths;
pub const LoRATargetTensor = assets.LoRATargetTensor;
pub const BootstrapOptions = assets.BootstrapOptions;
pub const BootstrapSummary = assets.BootstrapSummary;
pub const RecursiveCompressedBaseOptions = assets.RecursiveCompressedBaseOptions;
pub const RecursiveCompressedBaseSummary = assets.RecursiveCompressedBaseSummary;
pub const LoRATensorSummary = assets.LoRATensorSummary;
pub const LoRABundleInspectionSummary = assets.LoRABundleInspectionSummary;
pub const MaterializeSummary = assets.MaterializeSummary;
pub const LoadedLoRALayer = assets.LoadedLoRALayer;
pub const LoadedLoRABundle = assets.LoadedLoRABundle;
pub const AdapterManifest = assets.AdapterManifest;
pub const PeftExportSummary = assets.PeftExportSummary;
pub const ModelProvenance = assets.ModelProvenance;
const LoRAInitKind = assets.LoRAInitKind;
pub const resolveArtifactPaths = assets.resolveArtifactPaths;
pub const inspectCheckpoint = assets.inspectCheckpoint;
pub const freeInspectionSummary = assets.freeInspectionSummary;
pub const bootstrapLoRABundle = assets.bootstrapLoRABundle;
pub const freeBootstrapSummary = assets.freeBootstrapSummary;
pub const exportPeftAdapter = assets.exportPeftAdapter;
pub const freePeftExportSummary = assets.freePeftExportSummary;
pub const materializeRecursiveCompressedBase = assets.materializeRecursiveCompressedBase;
pub const freeRecursiveCompressedBaseSummary = assets.freeRecursiveCompressedBaseSummary;
pub const inspectLoRABundle = assets.inspectLoRABundle;
pub const freeLoRABundleInspectionSummary = assets.freeLoRABundleInspectionSummary;
pub const validateLoRAAdapterInventory = assets.validateLoRAAdapterInventory;
pub const validateLoRABundleInspection = assets.validateLoRABundleInspection;
pub const loadLoRABundle = assets.loadLoRABundle;
pub const loadLoRABundleScoped = assets.loadLoRABundleScoped;
pub const saveLoRABundle = assets.saveLoRABundle;
pub const saveLoRABundleToStaging = assets.saveLoRABundleToStaging;
pub const materializeMergedModel = assets.materializeMergedModel;
pub const freeMaterializeSummary = assets.freeMaterializeSummary;
const layerMatchesScope = assets.layerMatchesScope;
const parseGemma4LayerIndex = assets.parseGemma4LayerIndex;
const inferLoRATargetTensors = assets.inferLoRATargetTensors;
const parseLoRAInitKind = assets.parseLoRAInitKind;
const buildInitialLoRAFactors = assets.buildInitialLoRAFactors;
const writeHeaderAndTensorsF32 = assets.writeHeaderAndTensorsF32;
const findDecoderGgufPathInDir = assets.findDecoderGgufPathInDir;
const freeLoRATargetTensors = assets.freeLoRATargetTensors;
const dupeOptionalString = assets.dupeOptionalString;
const optionalStringsEqual = assets.optionalStringsEqual;
const BootstrapTargetSelection = assets.BootstrapTargetSelection;
const writeGemma4BootstrapTestConfig = assets.writeGemma4BootstrapTestConfig;
const writeBootstrapAdapterCheckpoint = assets.writeBootstrapAdapterCheckpoint;
const buildDeterministicLoraA = assets.buildDeterministicLoraA;
const validateMaterializationSource = assets.validateMaterializationSource;
const stockPeftModulePathAlloc = assets.stockPeftModulePathAlloc;
const stockPeftAdapterConfigAlloc = assets.stockPeftAdapterConfigAlloc;
const targetMatchesSelection = assets.targetMatchesSelection;
const roundF32ToBf16Bits = assets.roundF32ToBf16Bits;
const deriveLoRAInitializationSeed = assets.deriveLoRAInitializationSeed;
const PeftExportManifest = assets.PeftExportManifest;
const writeHeaderAndRawTensors = assets.writeHeaderAndRawTensors;
const openTensorAccessForFile = assets.openTensorAccessForFile;
const canonicalAdapterBaseTensorName = assets.canonicalAdapterBaseTensorName;
const DType = assets.DType;
const loadTensorAsF32 = assets.loadTensorAsF32;
const denseRecordDType = assets.denseRecordDType;
pub const sha256HexAlloc = assets.sha256HexAlloc;
pub const fingerprintGemma4Model = assets.fingerprintGemma4Model;
pub const validateAdapterModelProvenance = assets.validateAdapterModelProvenance;
const validateSha256Hex = assets.validateSha256Hex;
pub const gemma4LoRATargetPresetName = assets.gemma4LoRATargetPresetName;
pub const validateLoRAInitializerBaseCompatibility = assets.validateLoRAInitializerBaseCompatibility;
pub const AdapterConfigWriteOptions = assets.AdapterConfigWriteOptions;
pub const writeAdapterConfigJson = assets.writeAdapterConfigJson;
pub const writeAdapterManifestJson = assets.writeAdapterManifestJson;

pub const prepared_schema_v2 = "gemma4_prepared/v2";
pub const prepared_schema_v3 = "gemma4_prepared/v3";
pub const prepared_schema_v4 = "gemma4_prepared/v4";
pub const prepared_schema_v5 = "gemma4_prepared/v5";
/// Causal chat tokenization: the rendered Gemma transcript owns its literal
/// BOS token, no implicit EOS is appended, and every assistant turn is
/// supervised even when the tokenizer cannot provide byte offsets.
pub const prepared_schema_v6 = "gemma4_prepared/v6";
/// Until the Metal attention training graph is chunked, admitting longer
/// sequences can create an unbounded quadratic allocation.
pub const max_training_seq_len: usize = 2048;
pub const prepared_chat_template_identity = "antfly_gemma_chat/v1";

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

pub const ProjectorFingerprint = struct {
    path: []const u8,
    sha256: []const u8,
    size_bytes: u64,
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

const PreparedInputsSummaryFile = struct {
    summary: PreparedInputsSummary,
};

// ---------------------------------------------------------------------------
// Checkpoint inspection
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Load / save LoRA bundle
// ---------------------------------------------------------------------------

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
    defer native_engine.deinit();
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

fn finishHashAlloc(
    allocator: std.mem.Allocator,
    hasher: *std.crypto.hash.sha2.Sha256,
) ![]const u8 {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
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
            var projected = try gemma4_projector.encodeProjectedImagesFromPath(cb, allocator, gguf_projector_path, &.{bytes});
            defer projected.deinit();
            if (projected.tokens_per_image.len != 1) return error.InvalidPreparedPrompt;
            break :blk projected.tokens_per_image[0];
        },
        .audio => blk: {
            var projected = try gemma4_projector.encodeProjectedAudioFromPath(cb, allocator, gguf_projector_path, &.{bytes});
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

fn isRegularFilePath(path: []const u8) bool {
    const stat = compat.cwd().statFile(compat.io(), path, .{}) catch return false;
    return stat.kind == .file;
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

test "gemma4 sharded safetensors bootstrap and streaming materialization preserve dtypes" {
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
    const q_values = [_]u16{
        roundF32ToBf16Bits(1),
        roundF32ToBf16Bits(2),
        roundF32ToBf16Bits(3),
        roundF32ToBf16Bits(4),
        roundF32ToBf16Bits(5),
        roundF32ToBf16Bits(6),
    };
    const norm_values = [_]f32{ 7, 8 };
    try writeHeaderAndRawTensors(allocator, shard_a_path, &.{.{
        .name = "model.layers.0.self_attn.q_proj.weight",
        .shape = &.{ 2, 3 },
        .dtype = .bf16,
        .raw_bytes = std.mem.asBytes(&q_values),
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

    const trained_adapter_dir = try std.fs.path.join(allocator, &.{ root, "trained-adapter" });
    defer allocator.free(trained_adapter_dir);
    var trained_bundle = try loadLoRABundle(allocator, root, adapter_dir);
    defer trained_bundle.deinit();
    @memset(trained_bundle.layers[0].adapter_a, 1.0);
    @memset(trained_bundle.layers[0].adapter_b, 0.5);
    try saveLoRABundle(&trained_bundle, trained_adapter_dir);
    var trained_inspection = try inspectCheckpoint(allocator, trained_adapter_dir);
    defer freeInspectionSummary(allocator, &trained_inspection);
    try std.testing.expectEqual(@as(?u64, 0), trained_inspection.initialization_seed);

    const merged_dir = try std.fs.path.join(allocator, &.{ root, "merged" });
    defer allocator.free(merged_dir);
    var materialized = try materializeMergedModel(allocator, root, trained_adapter_dir, merged_dir);
    defer freeMaterializeSummary(allocator, &materialized);
    try std.testing.expectEqual(@as(usize, 1), materialized.merged_lora_tensor_count);
    try std.testing.expectEqual(@as(usize, 1), materialized.copied_base_tensor_count);

    var merged_access = try openTensorAccessForFile(allocator, materialized.output_checkpoint_path);
    defer merged_access.deinit();
    var merged_q = try merged_access.getRecord(allocator, "model.layers.0.self_attn.q_proj.weight");
    defer merged_q.deinit();
    try std.testing.expectEqual(DType.bf16, denseRecordDType(merged_q.descriptor.encoding).?);
    const expected_q_values = [_]u16{
        roundF32ToBf16Bits(1.5),
        roundF32ToBf16Bits(2.5),
        roundF32ToBf16Bits(3.5),
        roundF32ToBf16Bits(4.5),
        roundF32ToBf16Bits(5.5),
        roundF32ToBf16Bits(6.5),
    };
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&expected_q_values), merged_q.raw_bytes);
    var merged_norm = try merged_access.getRecord(allocator, "model.layers.0.input_layernorm.weight");
    defer merged_norm.deinit();
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&norm_values), merged_norm.raw_bytes);
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
        .initialization_seed = 17,
    };
    try writeAdapterConfigJson(allocator, config_path, options);
    try writeAdapterManifestJson(allocator, manifest_path, options);

    const config_bytes = try c_file.readFile(allocator, config_path);
    defer allocator.free(config_bytes);
    try std.testing.expect(std.mem.indexOf(u8, config_bytes, "\"init_lora_weights\": true") != null);
    try std.testing.expect(std.mem.indexOf(u8, config_bytes, "antfly_") == null);
    try std.testing.expect(std.mem.indexOf(u8, config_bytes, "target_preset") == null);
    try std.testing.expect(std.mem.indexOf(u8, config_bytes, "recursive_lora") == null);

    var explicit_default = options;
    explicit_default.init_lora_weights = "default";
    try writeAdapterConfigJson(allocator, config_path, explicit_default);
    try writeAdapterManifestJson(allocator, manifest_path, explicit_default);
    const explicit_bytes = try c_file.readFile(allocator, config_path);
    defer allocator.free(explicit_bytes);
    try std.testing.expect(std.mem.indexOf(u8, explicit_bytes, "\"init_lora_weights\": true") != null);

    var inspected = try inspectCheckpoint(allocator, adapter_dir);
    defer freeInspectionSummary(allocator, &inspected);
    try std.testing.expectEqualStrings(options.base_model_sha256.?, inspected.base_model_sha256.?);
    try std.testing.expectEqualStrings(options.tokenizer_sha256.?, inspected.tokenizer_sha256.?);
    try std.testing.expectEqualStrings(options.chat_template_sha256.?, inspected.chat_template_sha256.?);
    try std.testing.expectEqualStrings("peft-qv", inspected.target_preset.?);
    try std.testing.expectEqual(@as(?u64, 17), inspected.initialization_seed);
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

test "gemma4 deterministic initializer supports independent seeded adapters" {
    const allocator = std.testing.allocator;
    const legacy = try buildDeterministicLoraA(allocator, 2, 3, 0);
    defer allocator.free(legacy);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 3) * 0.013) * 0.01, legacy[0], 1e-8);

    const first = try buildDeterministicLoraA(allocator, 2, 3, 17);
    defer allocator.free(first);
    const repeated = try buildDeterministicLoraA(allocator, 2, 3, 17);
    defer allocator.free(repeated);
    const other = try buildDeterministicLoraA(allocator, 2, 3, 42);
    defer allocator.free(other);
    try std.testing.expectEqualSlices(f32, first, repeated);
    try std.testing.expect(!std.mem.eql(f32, first, other));
    try std.testing.expect(deriveLoRAInitializationSeed(17, "layer.0.weight") !=
        deriveLoRAInitializationSeed(17, "layer.1.weight"));
}

test "gemma4 streaming materialization admits single and sharded safetensors" {
    try validateMaterializationSource("model.safetensors");
    try validateMaterializationSource("model.safetensors.index.json");
    try std.testing.expectError(
        error.Gemma4GgufMaterializationUnsupported,
        validateMaterializationSource("model.gguf"),
    );
    try std.testing.expectError(error.UnsupportedMaterializationSource, validateMaterializationSource("model.bin"));
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

test "gemma4 PEFT export restores HF PLE names in tensors and target configuration" {
    const allocator = std.testing.allocator;
    const cases = [_][2][]const u8{
        .{ "model.language_model.layers.3.per_layer_input.inp_gate", "model.language_model.layers.3.per_layer_input_gate" },
        .{ "model.language_model.layers.3.per_layer_input.proj", "model.language_model.layers.3.per_layer_projection" },
        .{ "model.language_model.per_layer_input.per_layer_model_proj", "model.language_model.per_layer_model_projection" },
        .{ "model.per_layer_input.per_layer_model_proj", "model.per_layer_model_projection" },
        .{ "model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.3.self_attn.q_proj" },
    };
    for (cases) |pair| {
        const actual = try stockPeftModulePathAlloc(allocator, pair[0]);
        defer allocator.free(actual);
        try std.testing.expectEqualStrings(pair[1], actual);
    }
    const source =
        \\{"r":16,"lora_alpha":32,"custom":{"keep":true},"target_modules":["model.language_model.layers.3.per_layer_input.inp_gate","model.language_model.per_layer_input.per_layer_model_proj"]}
    ;
    const config = try stockPeftAdapterConfigAlloc(allocator, source);
    defer allocator.free(config);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, config, .{});
    defer parsed.deinit();
    const targets = parsed.value.object.get("target_modules").?.array.items;
    try std.testing.expectEqualStrings(cases[0][1], targets[0].string);
    try std.testing.expectEqualStrings(cases[2][1], targets[1].string);
    try std.testing.expectEqual(@as(i64, 16), parsed.value.object.get("r").?.integer);
    try std.testing.expect(parsed.value.object.get("custom").?.object.get("keep").?.bool);
    try std.testing.expectError(error.PeftExportTargetModuleCollision, stockPeftAdapterConfigAlloc(allocator,
        \\{"target_modules":["model.per_layer_input.per_layer_model_proj","model.per_layer_model_projection"]}
    ));
}
