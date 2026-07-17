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

// Model manager: lazy-loads models and caches ready-to-use pipelines.
//
// Given a model directory path, loads the manifest, creates a tokenizer
// and backend session, and returns a pipeline ready for inference.

const std = @import("std");
const build_options = @import("build_options");
const platform = @import("antfly_platform");

const backends = @import("../backends/backends.zig");
const model_caps = @import("../models/capabilities.zig");
const manifest_mod = @import("../models/manifest.zig");
const c_file = @import("../util/c_file.zig");
const gguf_format = @import("../gguf/format.zig");
const gguf_metadata = @import("../gguf/metadata.zig");
const gguf_tensor_types = @import("../gguf/tensor_types.zig");
const gguf_writer = @import("../gguf/writer.zig");
const hf_tokenizer = @import("inference_hf_tokenizer");
const sentencepiece = @import("inference_tokenizer").sentencepiece;
const tokenizer_mod = @import("inference_tokenizer");
const embedding_mod = @import("../pipelines/embedding.zig");
const EmbeddingPipeline = embedding_mod.EmbeddingPipeline;
const EmbeddingConfig = embedding_mod.EmbeddingConfig;
const PoolingStrategy = embedding_mod.PoolingStrategy;
const RerankingPipeline = @import("../pipelines/reranking.zig").RerankingPipeline;
const RerankingConfig = @import("../pipelines/reranking.zig").RerankingConfig;
const ScoringMode = @import("../pipelines/reranking.zig").ScoringMode;
const ClassificationPipeline = @import("../pipelines/classification.zig").ClassificationPipeline;
const ClassificationConfig = @import("../pipelines/classification.zig").ClassificationConfig;
const cleanup_model_mod = @import("../finetune/entity_cleanup_model.zig");
const NerPipeline = @import("../pipelines/ner.zig").NerPipeline;
const NerConfig = @import("../pipelines/ner.zig").NerConfig;
const GlinerPipeline = @import("../pipelines/gliner.zig").GlinerPipeline;
const GlinerConfig = @import("../pipelines/gliner.zig").GlinerConfig;
const generation = @import("../pipelines/generation.zig");
const ChatTemplate = generation.ChatTemplate;
const session_factory = @import("../architectures/session_factory.zig");
const graph_mod = @import("../graph/root.zig");
const kernel_jit_profile_output = @import("../kernel_jit_profile_output.zig");
const runtime = @import("../runtime/root.zig");

fn shouldPreferNativeSession(man: manifest_mod.ModelManifest) bool {
    // GLiNER has a native DeBERTa + span-head path. When native weights are
    // present, prefer the directory-backed session so the model does not get
    // pinned to ONNX just because an export also exists.
    if (!manifestHasNativeAssets(man)) return false;
    if (man.model_type == .embedder and
        man.visual_model_path == null and
        man.audio_model_path == null and
        man.text_projection_path == null and
        man.visual_projection_path == null and
        man.audio_projection_path == null)
    {
        return true;
    }
    if (man.gliner_model_type.len > 0) return true;
    switch (man.model_type) {
        .classifier, .recognizer => return true,
        else => {},
    }
    return switch (man.native_arch_hint) {
        .clip, .whisper, .florence, .layoutlmv3 => true,
        .clap, .none => false,
    };
}

fn nativeBackendsAvailable() bool {
    return build_options.enable_native or build_options.enable_metal or build_options.enable_cuda;
}

fn manifestHasNativeAssets(man: manifest_mod.ModelManifest) bool {
    return man.gguf_path != null or man.safetensors_path != null or man.safetensors_index_path != null;
}

fn shouldUseMetalWholeModelExecutor(session: backends.Session) bool {
    return session.backend() == .metal;
}

fn spinLock(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) {
        std.atomic.spinLoopHint();
    }
}

pub fn shouldPreferSentencePieceOverride(man: manifest_mod.ModelManifest, model_dir: []const u8, allocator: std.mem.Allocator) bool {
    if (!c_file.fileExistsInDir(allocator, model_dir, "tokenizer.model")) return false;
    return manifestLooksLikeGemma(man, model_dir, allocator);
}

pub fn shouldEnableGemmaSentencePieceCompat(man: manifest_mod.ModelManifest, model_dir: []const u8, allocator: std.mem.Allocator) bool {
    return manifestLooksLikeGemma(man, model_dir, allocator);
}

pub fn loadSentencePieceAddedTokens(model_dir: []const u8, allocator: std.mem.Allocator, sp: *sentencepiece.Processor) !void {
    const added_tokens_path = std.fmt.allocPrint(allocator, "{s}/added_tokens.json", .{model_dir}) catch return;
    defer allocator.free(added_tokens_path);
    const added_tokens_bytes = c_file.readFile(allocator, added_tokens_path) catch return;
    defer allocator.free(added_tokens_bytes);
    try loadSentencePieceAddedTokenMap(allocator, added_tokens_bytes, sp);

    const tokenizer_json_path = std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_dir}) catch return;
    defer allocator.free(tokenizer_json_path);
    const tokenizer_json_bytes = c_file.readFile(allocator, tokenizer_json_path) catch return;
    defer allocator.free(tokenizer_json_bytes);
    try loadSentencePieceAddedTokenArray(allocator, tokenizer_json_bytes, sp);
}

fn loadSentencePieceAddedTokenMap(
    allocator: std.mem.Allocator,
    json_bytes: []const u8,
    sp: *sentencepiece.Processor,
) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return;
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* != .integer) continue;
        try sp.addExternalSpecialToken(entry.key_ptr.*, @intCast(entry.value_ptr.integer));
    }
}

fn loadSentencePieceAddedTokenArray(
    allocator: std.mem.Allocator,
    json_bytes: []const u8,
    sp: *sentencepiece.Processor,
) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return;
    const added_tokens = parsed.value.object.get("added_tokens") orelse return;
    if (added_tokens != .array) return;
    for (added_tokens.array.items) |item| {
        if (item != .object) continue;
        const content = item.object.get("content") orelse continue;
        const id = item.object.get("id") orelse continue;
        if (content != .string or id != .integer) continue;
        try sp.addExternalSpecialToken(content.string, @intCast(id.integer));
    }
}

fn manifestLooksLikeGemma(man: manifest_mod.ModelManifest, model_dir: []const u8, allocator: std.mem.Allocator) bool {
    _ = man;
    if (std.mem.indexOf(u8, model_dir, "gemma") != null) return true;

    const cfg_path = std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir}) catch return false;
    defer allocator.free(cfg_path);
    const cfg_bytes = c_file.readFile(allocator, cfg_path) catch return false;
    defer allocator.free(cfg_bytes);

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, cfg_bytes, .{}) catch return false;
    defer parsed.deinit();
    const obj = parsed.value.object;
    const model_type = obj.get("model_type") orelse return false;
    if (model_type != .string) return false;
    return std.mem.startsWith(u8, model_type.string, "gemma");
}

fn appendJsonString(buf: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try buf.append(allocator, '"');
    for (value) |c| switch (c) {
        '"' => try buf.appendSlice(allocator, "\\\""),
        '\\' => try buf.appendSlice(allocator, "\\\\"),
        '\n' => try buf.appendSlice(allocator, "\\n"),
        '\r' => try buf.appendSlice(allocator, "\\r"),
        '\t' => try buf.appendSlice(allocator, "\\t"),
        else => {
            if (c < 0x20) {
                const escaped = try std.fmt.allocPrint(allocator, "\\u{X:0>4}", .{@as(u8, c)});
                defer allocator.free(escaped);
                try buf.appendSlice(allocator, escaped);
            } else {
                try buf.append(allocator, c);
            }
        },
    };
    try buf.append(allocator, '"');
}

const LegacyWordPieceMeta = struct {
    do_lower_case: bool = false,
    unk_token: []const u8 = "[UNK]",
    pad_token: []const u8 = "[PAD]",
    cls_token: []const u8 = "[CLS]",
    sep_token: []const u8 = "[SEP]",
    mask_token: []const u8 = "[MASK]",
    unk_token_owned: ?[]u8 = null,
    pad_token_owned: ?[]u8 = null,
    cls_token_owned: ?[]u8 = null,
    sep_token_owned: ?[]u8 = null,
    mask_token_owned: ?[]u8 = null,

    fn deinit(self: *LegacyWordPieceMeta, allocator: std.mem.Allocator) void {
        if (self.unk_token_owned) |buf| allocator.free(buf);
        if (self.pad_token_owned) |buf| allocator.free(buf);
        if (self.cls_token_owned) |buf| allocator.free(buf);
        if (self.sep_token_owned) |buf| allocator.free(buf);
        if (self.mask_token_owned) |buf| allocator.free(buf);
    }
};

fn replaceLegacyToken(allocator: std.mem.Allocator, slot: *[]const u8, owned_slot: *?[]u8, value: []const u8) !void {
    const duped = try allocator.dupe(u8, value);
    if (owned_slot.*) |buf| allocator.free(buf);
    owned_slot.* = duped;
    slot.* = duped;
}

fn extractLegacyTokenString(val: std.json.Value) ?[]const u8 {
    return switch (val) {
        .string => |s| s,
        .object => |obj| blk: {
            if (obj.get("content")) |content| {
                if (content == .string) break :blk content.string;
            }
            break :blk null;
        },
        else => null,
    };
}

fn applyLegacyTokenizerJson(meta: *LegacyWordPieceMeta, json_bytes: []const u8, allocator: std.mem.Allocator) void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .object) return;
    const obj = parsed.value.object;

    if (obj.get("do_lower_case")) |v| {
        if (v == .bool) meta.do_lower_case = v.bool;
    }
    if (obj.get("unk_token")) |v| {
        if (extractLegacyTokenString(v)) |s| replaceLegacyToken(allocator, &meta.unk_token, &meta.unk_token_owned, s) catch {};
    }
    if (obj.get("pad_token")) |v| {
        if (extractLegacyTokenString(v)) |s| replaceLegacyToken(allocator, &meta.pad_token, &meta.pad_token_owned, s) catch {};
    }
    if (obj.get("cls_token")) |v| {
        if (extractLegacyTokenString(v)) |s| replaceLegacyToken(allocator, &meta.cls_token, &meta.cls_token_owned, s) catch {};
    }
    if (obj.get("sep_token")) |v| {
        if (extractLegacyTokenString(v)) |s| replaceLegacyToken(allocator, &meta.sep_token, &meta.sep_token_owned, s) catch {};
    }
    if (obj.get("mask_token")) |v| {
        if (extractLegacyTokenString(v)) |s| replaceLegacyToken(allocator, &meta.mask_token, &meta.mask_token_owned, s) catch {};
    }
}

fn appendAddedToken(
    buf: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    first: *bool,
    token: []const u8,
    id: i64,
) !void {
    if (!first.*) try buf.append(allocator, ',');
    first.* = false;
    try buf.appendSlice(allocator, "{\"id\":");
    const id_bytes = try std.fmt.allocPrint(allocator, "{d}", .{id});
    defer allocator.free(id_bytes);
    try buf.appendSlice(allocator, id_bytes);
    try buf.appendSlice(allocator, ",\"content\":");
    try appendJsonString(buf, allocator, token);
    try buf.appendSlice(allocator, ",\"special\":true}");
}

fn loadLegacyWordPieceTokenizerFromDir(allocator: std.mem.Allocator, model_dir: []const u8) !*hf_tokenizer.HfTokenizer {
    const vocab_path = try std.fmt.allocPrint(allocator, "{s}/vocab.txt", .{model_dir});
    defer allocator.free(vocab_path);
    const vocab_bytes = try c_file.readFile(allocator, vocab_path);
    defer allocator.free(vocab_bytes);

    var meta = LegacyWordPieceMeta{};
    defer meta.deinit(allocator);
    var tokenizer_config_bytes_opt: ?[]u8 = null;
    defer if (tokenizer_config_bytes_opt) |bytes| allocator.free(bytes);
    var special_tokens_map_bytes_opt: ?[]u8 = null;
    defer if (special_tokens_map_bytes_opt) |bytes| allocator.free(bytes);

    const tokenizer_config_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_config.json", .{model_dir});
    defer allocator.free(tokenizer_config_path);
    if (c_file.readFile(allocator, tokenizer_config_path)) |tokenizer_config_bytes| {
        tokenizer_config_bytes_opt = tokenizer_config_bytes;
        applyLegacyTokenizerJson(&meta, tokenizer_config_bytes, allocator);
    } else |_| {}

    const special_tokens_map_path = try std.fmt.allocPrint(allocator, "{s}/special_tokens_map.json", .{model_dir});
    defer allocator.free(special_tokens_map_path);
    if (c_file.readFile(allocator, special_tokens_map_path)) |special_tokens_map_bytes| {
        special_tokens_map_bytes_opt = special_tokens_map_bytes;
        applyLegacyTokenizerJson(&meta, special_tokens_map_bytes, allocator);
    } else |_| {}

    var vocab_entries = std.ArrayListUnmanaged([]const u8).empty;
    defer vocab_entries.deinit(allocator);

    var line_it = std.mem.tokenizeScalar(u8, vocab_bytes, '\n');
    while (line_it.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, "\r");
        if (line.len == 0) continue;
        try vocab_entries.append(allocator, line);
    }

    var unk_id: i64 = -1;
    var pad_id: i64 = -1;
    var cls_id: i64 = -1;
    var sep_id: i64 = -1;
    var mask_id: i64 = -1;
    for (vocab_entries.items, 0..) |token, idx| {
        const id: i64 = @intCast(idx);
        if (std.mem.eql(u8, token, meta.unk_token)) unk_id = id;
        if (std.mem.eql(u8, token, meta.pad_token)) pad_id = id;
        if (std.mem.eql(u8, token, meta.cls_token)) cls_id = id;
        if (std.mem.eql(u8, token, meta.sep_token)) sep_id = id;
        if (std.mem.eql(u8, token, meta.mask_token)) mask_id = id;
    }

    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator, "{\"model\":{\"type\":\"WordPiece\",\"unk_token\":");
    try appendJsonString(&buf, allocator, meta.unk_token);
    try buf.appendSlice(allocator, ",\"continuing_subword_prefix\":\"##\",\"max_input_chars_per_word\":100,\"vocab\":{");
    for (vocab_entries.items, 0..) |token, idx| {
        if (idx > 0) try buf.append(allocator, ',');
        try appendJsonString(&buf, allocator, token);
        try buf.append(allocator, ':');
        const id_bytes = try std.fmt.allocPrint(allocator, "{d}", .{idx});
        defer allocator.free(id_bytes);
        try buf.appendSlice(allocator, id_bytes);
    }
    try buf.appendSlice(allocator, "}},\"normalizer\":{\"type\":\"BertNormalizer\",\"lowercase\":");
    try buf.appendSlice(allocator, if (meta.do_lower_case) "true" else "false");
    try buf.appendSlice(allocator, "},\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},\"added_tokens\":[");

    var first_added = true;
    if (pad_id >= 0) try appendAddedToken(&buf, allocator, &first_added, meta.pad_token, pad_id);
    if (unk_id >= 0) try appendAddedToken(&buf, allocator, &first_added, meta.unk_token, unk_id);
    if (cls_id >= 0) try appendAddedToken(&buf, allocator, &first_added, meta.cls_token, cls_id);
    if (sep_id >= 0) try appendAddedToken(&buf, allocator, &first_added, meta.sep_token, sep_id);
    if (mask_id >= 0) try appendAddedToken(&buf, allocator, &first_added, meta.mask_token, mask_id);
    try buf.appendSlice(allocator, "]");

    if (cls_id >= 0 and sep_id >= 0) {
        try buf.appendSlice(allocator, ",\"post_processor\":{\"type\":\"BertProcessing\",\"cls\":[");
        try appendJsonString(&buf, allocator, meta.cls_token);
        const cls_id_bytes = try std.fmt.allocPrint(allocator, ",{d}],\"sep\":[", .{cls_id});
        defer allocator.free(cls_id_bytes);
        try buf.appendSlice(allocator, cls_id_bytes);
        try appendJsonString(&buf, allocator, meta.sep_token);
        const sep_id_bytes = try std.fmt.allocPrint(allocator, ",{d}]", .{sep_id});
        defer allocator.free(sep_id_bytes);
        try buf.appendSlice(allocator, sep_id_bytes);
        try buf.appendSlice(allocator, "}");
    }

    try buf.append(allocator, '}');
    const tokenizer_json = try buf.toOwnedSlice(allocator);
    defer allocator.free(tokenizer_json);
    return hf_tokenizer.HfTokenizer.loadFromBytes(allocator, tokenizer_json);
}

pub fn loadHuggingFaceTokenizerFromDir(allocator: std.mem.Allocator, model_dir: []const u8) !*hf_tokenizer.HfTokenizer {
    return loadHuggingFaceTokenizerFromDirOrGguf(allocator, model_dir, null);
}

pub fn loadHuggingFaceTokenizerFromDirOrGguf(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    gguf_path: ?[]const u8,
) !*hf_tokenizer.HfTokenizer {
    const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_dir});
    defer allocator.free(tok_path);
    if (c_file.readFile(allocator, tok_path)) |tok_bytes| {
        defer allocator.free(tok_bytes);
        return hf_tokenizer.HfTokenizer.loadFromBytes(allocator, tok_bytes);
    } else |_| {}

    if (c_file.fileExistsInDir(allocator, model_dir, "vocab.txt")) {
        return loadLegacyWordPieceTokenizerFromDir(allocator, model_dir);
    }

    if (gguf_path) |path| {
        return loadHuggingFaceTokenizerFromGguf(allocator, path);
    }

    return error.NoTokenizerFound;
}

fn loadHuggingFaceTokenizerFromGguf(allocator: std.mem.Allocator, gguf_path: []const u8) !*hf_tokenizer.HfTokenizer {
    var region = try c_file.MmapRegion.init(allocator, gguf_path);
    defer region.deinit();

    const parse_allocator = platform.allocator.processAllocator(allocator);
    var parsed = try gguf_format.parse(parse_allocator, region.data);
    defer parsed.deinit(parse_allocator);

    const view = gguf_metadata.View.init(&parsed);
    const model_name = view.getString("tokenizer.ggml.model") orelse return error.NoTokenizerFound;

    const flavor: GgufBpeTokenizerFlavor = if (std.mem.eql(u8, model_name, "gpt2"))
        .byte_level
    else if (std.mem.eql(u8, model_name, "gemma4"))
        .gemma4
    else
        return error.NoTokenizerFound;

    const tokenizer_bytes = try bpeTokenizerJsonFromGguf(allocator, &parsed, flavor);
    defer allocator.free(tokenizer_bytes);

    const tok = try hf_tokenizer.HfTokenizer.loadFromBytes(allocator, tokenizer_bytes);
    tok.applySpecialTokenIds(
        metadataTokenId(&parsed, "tokenizer.ggml.bos_token_id"),
        metadataTokenId(&parsed, "tokenizer.ggml.eos_token_id"),
        metadataTokenId(&parsed, "tokenizer.ggml.padding_token_id"),
        metadataTokenId(&parsed, "tokenizer.ggml.unknown_token_id"),
    );
    return tok;
}

const GgufBpeTokenizerFlavor = enum {
    byte_level,
    gemma4,
};

fn bpeTokenizerJsonFromGguf(
    allocator: std.mem.Allocator,
    parsed: *const gguf_format.File,
    flavor: GgufBpeTokenizerFlavor,
) ![]u8 {
    const tokens = try getRequiredMetadataArray(parsed, "tokenizer.ggml.tokens", .string);
    const merges = try getRequiredMetadataArray(parsed, "tokenizer.ggml.merges", .string);
    const token_types = if (findMetadataEntry(parsed, "tokenizer.ggml.token_type") != null)
        try getRequiredMetadataArray(parsed, "tokenizer.ggml.token_type", null)
    else
        null;

    var tokenizer_json = std.ArrayListUnmanaged(u8).empty;
    defer tokenizer_json.deinit(allocator);

    switch (flavor) {
        .byte_level => {
            try tokenizer_json.appendSlice(allocator, "{\"model\":{\"type\":\"BPE\",\"byte_fallback\":false,\"vocab\":{");
        },
        .gemma4 => {
            try tokenizer_json.appendSlice(
                allocator,
                "{\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"String\":\" \"},\"behavior\":\"MergedWithPrevious\",\"invert\":false},\"model\":{\"type\":\"BPE\",\"fuse_unk\":true,\"byte_fallback\":true,\"vocab\":{",
            );
        },
    }
    for (tokens.values, 0..) |token_value, idx| {
        const token = switch (token_value) {
            .string => |value| value,
            else => return error.InvalidTokenizerMetadata,
        };
        if (idx > 0) try tokenizer_json.append(allocator, ',');
        try appendJsonString(&tokenizer_json, allocator, token);
        try tokenizer_json.append(allocator, ':');
        const id_bytes = try std.fmt.allocPrint(allocator, "{d}", .{idx});
        defer allocator.free(id_bytes);
        try tokenizer_json.appendSlice(allocator, id_bytes);
    }
    try tokenizer_json.appendSlice(allocator, "},\"merges\":[");
    for (merges.values, 0..) |merge_value, idx| {
        const merge = switch (merge_value) {
            .string => |value| value,
            else => return error.InvalidTokenizerMetadata,
        };
        if (idx > 0) try tokenizer_json.append(allocator, ',');
        try appendJsonString(&tokenizer_json, allocator, merge);
    }
    switch (flavor) {
        .byte_level => try tokenizer_json.appendSlice(allocator, "]},\"pre_tokenizer\":{\"type\":\"ByteLevel\"},\"added_tokens\":["),
        .gemma4 => try tokenizer_json.appendSlice(allocator, "]},\"added_tokens\":["),
    }

    try appendSpecialTokensFromMetadata(&tokenizer_json, allocator, parsed, tokens, token_types);
    try tokenizer_json.appendSlice(allocator, "]}");

    return tokenizer_json.toOwnedSlice(allocator);
}

fn appendSpecialTokensFromMetadata(
    buf: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    parsed: *const gguf_format.File,
    tokens: gguf_format.MetadataArray,
    token_types: ?gguf_format.MetadataArray,
) !void {
    var first_added = true;
    var seen = std.AutoHashMapUnmanaged(i64, void){};
    defer seen.deinit(allocator);

    const special_id_keys = [_][]const u8{
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.unknown_token_id",
    };
    for (special_id_keys) |key| {
        const token_id = metadataTokenId(parsed, key) orelse continue;
        const token = metadataTokenStringById(tokens, token_id) orelse continue;
        if (seen.contains(token_id)) continue;
        try seen.put(allocator, token_id, {});
        try appendAddedToken(buf, allocator, &first_added, token, token_id);
    }

    if (token_types) |types| {
        for (tokens.values, 0..) |token_value, idx| {
            const token = switch (token_value) {
                .string => |value| value,
                else => return error.InvalidTokenizerMetadata,
            };
            const token_type = try metadataI64At(types, idx);
            if (token_type == 1 or token_type == 6) continue;
            const token_id: i64 = @intCast(idx);
            if (seen.contains(token_id)) continue;
            try seen.put(allocator, token_id, {});
            try appendAddedToken(buf, allocator, &first_added, token, token_id);
        }
    }
}

fn metadataTokenId(parsed: *const gguf_format.File, key: []const u8) ?i32 {
    const view = gguf_metadata.View.init(parsed);
    const raw_id = view.getU64(key) orelse return null;
    return std.math.cast(i32, raw_id);
}

fn metadataTokenStringById(tokens: gguf_format.MetadataArray, token_id: i32) ?[]const u8 {
    if (token_id < 0) return null;
    const token_index: usize = @intCast(token_id);
    if (token_index >= tokens.values.len) return null;
    return switch (tokens.values[token_index]) {
        .string => |value| value,
        else => null,
    };
}

fn metadataI64At(arr: gguf_format.MetadataArray, index: usize) !i64 {
    if (index >= arr.values.len) return error.InvalidTokenizerMetadata;
    return switch (arr.values[index]) {
        .i32 => |value| value,
        .i64 => |value| value,
        .u32 => |value| value,
        .u64 => |value| std.math.cast(i64, value) orelse return error.InvalidTokenizerMetadata,
        else => return error.InvalidTokenizerMetadata,
    };
}

fn findMetadataEntry(parsed: *const gguf_format.File, key: []const u8) ?*const gguf_format.MetadataEntry {
    for (parsed.metadata) |*entry| {
        if (std.mem.eql(u8, entry.key, key)) return entry;
    }
    return null;
}

pub fn loadSentencePieceTokenizerFromDirOrGguf(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    gguf_path: ?[]const u8,
) !*sentencepiece.Processor {
    const sp = try allocator.create(sentencepiece.Processor);
    errdefer allocator.destroy(sp);

    if (c_file.fileExistsInDir(allocator, model_dir, "tokenizer.model")) {
        const sp_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.model", .{model_dir});
        defer allocator.free(sp_path);
        sp.* = try sentencepiece.Processor.initFromPath(allocator, sp_path);
        return sp;
    }

    const resolved_gguf_path = gguf_path orelse return error.NoTokenizerFound;
    sp.* = try loadSentencePieceTokenizerFromGguf(allocator, resolved_gguf_path);
    return sp;
}

fn adoptAndConfigureSentencePieceTokenizer(
    owned: *?*sentencepiece.Processor,
    sp: *sentencepiece.Processor,
    man: manifest_mod.ModelManifest,
    model_dir: []const u8,
    allocator: std.mem.Allocator,
) !void {
    std.debug.assert(owned.* == null);
    owned.* = sp;
    if (shouldEnableGemmaSentencePieceCompat(man, model_dir, allocator)) {
        sp.setPreserveInlineSpecialsAfterLiteralBos(true);
    }
    try loadSentencePieceAddedTokens(model_dir, allocator, sp);
}

fn loadSentencePieceTokenizerFromGguf(allocator: std.mem.Allocator, gguf_path: []const u8) !sentencepiece.Processor {
    var region = try c_file.MmapRegion.init(allocator, gguf_path);
    defer region.deinit();

    const parse_allocator = platform.allocator.processAllocator(allocator);
    var parsed = try gguf_format.parse(parse_allocator, region.data);
    defer parsed.deinit(parse_allocator);

    const view = gguf_metadata.View.init(&parsed);
    const model_name = view.getString("tokenizer.ggml.model") orelse return error.NoTokenizerFound;
    if (!(std.mem.eql(u8, model_name, "llama") or
        std.mem.eql(u8, model_name, "bert") or
        std.mem.eql(u8, model_name, "t5") or
        std.mem.startsWith(u8, model_name, "gemma")))
    {
        return error.NoTokenizerFound;
    }

    const tokens = try getRequiredMetadataArray(&parsed, "tokenizer.ggml.tokens", .string);
    const scores = try getRequiredMetadataArray(&parsed, "tokenizer.ggml.scores", null);
    const token_types = try getRequiredMetadataArray(&parsed, "tokenizer.ggml.token_type", null);
    if (tokens.values.len != scores.values.len or tokens.values.len != token_types.values.len) {
        return error.InvalidTokenizerMetadata;
    }

    const unknown_token_index = view.getU64("tokenizer.ggml.unknown_token_id");
    const pieces = try allocator.alloc(sentencepiece.PieceInit, tokens.values.len);
    defer allocator.free(pieces);

    var saw_byte_piece = false;
    var saw_unknown_piece = false;
    for (tokens.values, 0..) |token_value, idx| {
        const token_text = switch (token_value) {
            .string => |value| value,
            else => return error.InvalidTokenizerMetadata,
        };
        const score = switch (scores.values[idx]) {
            .f32 => |value| value,
            .f64 => |value| @as(f32, @floatCast(value)),
            else => return error.InvalidTokenizerMetadata,
        };
        const token_type_i64 = switch (token_types.values[idx]) {
            .i32 => |value| value,
            .i64 => |value| value,
            .u32 => |value| @as(i64, value),
            .u64 => |value| std.math.cast(i64, value) orelse return error.InvalidTokenizerMetadata,
            else => return error.InvalidTokenizerMetadata,
        };
        if (token_type_i64 < 0 or token_type_i64 > std.math.maxInt(u8)) {
            return error.InvalidTokenizerMetadata;
        }
        var token_type: u8 = @intCast(token_type_i64);
        if (unknown_token_index) |unknown_id| {
            if (unknown_id == idx) token_type = 2;
        }
        if (token_type == 6) saw_byte_piece = true;
        if (token_type == 2) saw_unknown_piece = true;
        pieces[idx] = .{
            .text = token_text,
            .score = score,
            .piece_type = token_type,
        };
    }
    if (!saw_unknown_piece) return error.InvalidTokenizerMetadata;

    const add_dummy_prefix = view.getBool("tokenizer.ggml.add_space_prefix") orelse true;
    const remove_extra_whitespaces = view.getBool("tokenizer.ggml.remove_extra_whitespaces") orelse true;
    const unk_surface = blk: {
        const unk_id = unknown_token_index orelse break :blk " \xe2\x81\x87 ";
        if (unk_id >= tokens.values.len) break :blk " \xe2\x81\x87 ";
        break :blk switch (tokens.values[@intCast(unk_id)]) {
            .string => |value| value,
            else => " \xe2\x81\x87 ",
        };
    };

    return sentencepiece.Processor.initFromPieces(allocator, pieces, .{
        .byte_fallback = saw_byte_piece,
        .unk_surface = unk_surface,
        .add_dummy_prefix = add_dummy_prefix,
        .remove_extra_whitespaces = remove_extra_whitespaces,
    });
}

fn getRequiredMetadataArray(
    parsed: *const gguf_format.File,
    key: []const u8,
    expected_element_type: ?gguf_format.MetadataValueType,
) !gguf_format.MetadataArray {
    for (parsed.metadata) |entry| {
        if (!std.mem.eql(u8, entry.key, key)) continue;
        const arr = switch (entry.value) {
            .array => |value| value,
            else => return error.InvalidTokenizerMetadata,
        };
        if (expected_element_type) |elem_type| {
            if (arr.element_type != elem_type) return error.InvalidTokenizerMetadata;
        }
        return arr;
    }
    return error.InvalidTokenizerMetadata;
}

pub fn isModelDirPotentiallyLoadableInCurrentBuild(allocator: std.mem.Allocator, model_dir: []const u8) bool {
    var man = manifest_mod.loadFromDir(allocator, model_dir) catch return false;
    defer man.deinit();
    return isManifestPotentiallyLoadableInCurrentBuild(man);
}

pub fn isManifestPotentiallyLoadableInCurrentBuild(man: manifest_mod.ModelManifest) bool {
    if (man.hasIncompleteGlinerBundle()) return false;
    if (man.hasIncompleteColqwenBundle()) return false;
    if (man.hasIncompleteClipclapGgufBundle()) return false;
    if (man.hasIncompleteFlorence2GgufBundle()) return false;
    if (man.onnx_path != null or
        man.visual_model_path != null or
        man.audio_model_path != null or
        man.text_projection_path != null or
        man.visual_projection_path != null or
        man.audio_projection_path != null)
    {
        return true;
    }
    if (nativeBackendsAvailable() and manifestHasNativeAssets(man)) {
        return true;
    }
    return false;
}

const DeclaredOptionalSessionKind = enum {
    vision,
    audio,
    text_projection,
    visual_projection,
    audio_projection,
};

const DeclaredOptionalSession = struct {
    kind: DeclaredOptionalSessionKind,
    path: ?[]const u8,
};

const declared_optional_session_count = @typeInfo(DeclaredOptionalSessionKind).@"enum".fields.len;

fn declaredOptionalSessions(manifest: *const manifest_mod.ModelManifest) [declared_optional_session_count]DeclaredOptionalSession {
    return .{
        .{ .kind = .vision, .path = manifest.visual_model_path },
        .{ .kind = .audio, .path = manifest.audio_model_path },
        .{ .kind = .text_projection, .path = manifest.text_projection_path },
        .{ .kind = .visual_projection, .path = manifest.visual_projection_path },
        .{ .kind = .audio_projection, .path = manifest.audio_projection_path },
    };
}

fn declaredOptionalSessionsComplete(
    manifest: *const manifest_mod.ModelManifest,
    loaded: [declared_optional_session_count]bool,
) bool {
    for (declaredOptionalSessions(manifest), loaded) |declared, is_loaded| {
        if (declared.path != null and !is_loaded) return false;
    }
    return true;
}

fn bindOptionalSessionProfile(
    session_manager: *backends.SessionManager,
    kind: DeclaredOptionalSessionKind,
    profile_bundle: ?graph_mod.kernel_jit.QualifiedProfileBundle,
) !void {
    if (profile_bundle) |bundle| {
        const component_path = switch (kind) {
            .vision => bundle.vision,
            .audio => bundle.audio,
            .text_projection => bundle.text_projection,
            .visual_projection => bundle.visual_projection,
            .audio_projection => bundle.audio_projection,
        };
        if (component_path) |path| {
            session_manager.kernel_jit.qualified_profile_path = path;
        } else {
            if (session_manager.kernel_jit.mode.failClosed()) {
                return error.MissingKernelJitProfileBundleComponent;
            }
            std.log.warn(
                "kernel JIT profile bundle has no {s} member; using bundled kernels",
                .{@tagName(kind)},
            );
            session_manager.kernel_jit.qualified_profile_path = null;
            session_manager.kernel_jit.mode = .off;
        }
    } else if (session_manager.kernel_jit.qualified_profile_path != null) {
        if (session_manager.kernel_jit.mode.failClosed()) {
            return error.KernelJitQualifiedProfileOptionalSessionUnsupported;
        }
        // One exact profile is bound to one model fingerprint. Never apply
        // the primary model's profile to an optional submodel.
        session_manager.kernel_jit.qualified_profile_path = null;
        session_manager.kernel_jit.mode = .off;
    }
}

fn ownSelectedFirstBackendPreference(
    allocator: std.mem.Allocator,
    preferred: []const backends.BackendType,
    selected: backends.BackendType,
) ![]backends.BackendType {
    var count: usize = 1;
    for (preferred) |backend| {
        if (backend != selected) count += 1;
    }
    const result = try allocator.alloc(backends.BackendType, count);
    result[0] = selected;
    var index: usize = 1;
    for (preferred) |backend| {
        if (backend == selected) continue;
        result[index] = backend;
        index += 1;
    }
    return result;
}

pub const LoadedModel = struct {
    manifest: manifest_mod.ModelManifest,
    hf_tok: ?*hf_tokenizer.HfTokenizer,
    sp_tok: ?*sentencepiece.Processor,
    session: backends.Session,
    session_manager: *backends.SessionManager,
    /// Backend order captured when the primary session was loaded, with the
    /// selected backend first. Optional vision/audio/projection sessions use
    /// this owned order while inheriting the manager's current JIT load
    /// context, so an explicit startup preload backend cannot silently drift.
    optional_session_preferred_backends: []backends.BackendType,
    model_dir: []const u8,
    allocator: std.mem.Allocator,
    chat_tmpl: ?*ChatTemplate = null,
    shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache = null,
    shared_prefetch: ?*runtime.tier.shared.SharedPrefetchState = null,
    prompt_prefix_cache: runtime.kv.prompt_cache.PromptPrefixCache,
    native_generate_coordinator: ?*runtime.scheduler.native_generate.NativeGenerateCoordinator = null,
    native_generation_graph_cache: graph_mod.cache.GraphCache,
    // ponytail: model-wide safety lock; replace with per-request backend state only when continuous batching is proven safe.
    native_generate_lock: std.atomic.Mutex = .unlocked,
    // Multimodal sessions (CLIP/CLAP/CLIPCLAP)
    embedding_session_lock: std.atomic.Mutex = .unlocked,
    reranking_session_lock: std.atomic.Mutex = .unlocked,
    // Recognizer pipelines reuse the primary backend session. CUDA and Metal
    // sessions own mutable streams, scratch arenas, plan caches, and counters,
    // so one LoadedModel must not execute overlapping recognition requests.
    recognizer_session_lock: std.atomic.Mutex = .unlocked,
    cleanup_head_lock: std.atomic.Mutex = .unlocked,
    vision_session: ?backends.Session = null,
    audio_session: ?backends.Session = null,
    text_projection: ?backends.Session = null,
    visual_projection: ?backends.Session = null,
    audio_projection: ?backends.Session = null,
    kernel_jit_profile_bundle: ?kernel_jit_profile_output.LoadedProfileBundle = null,
    resident_projection_stats: embedding_mod.AtomicResidentProjectionStats = .{},
    cleanup_head: ?*cleanup_model_mod.CleanupHead = null,
    cleanup_head_loaded: bool = false,

    pub fn getTokenizer(self: *LoadedModel) tokenizer_mod.Tokenizer {
        if (self.hf_tok) |ht| return ht.tokenizer();
        if (self.sp_tok) |sp| return sp.tokenizer();
        unreachable;
    }

    pub fn attachIo(self: *LoadedModel, io: std.Io) void {
        session_factory.attachIo(self.session, io);
        if (self.vision_session) |session| session_factory.attachIo(session, io);
        if (self.audio_session) |session| session_factory.attachIo(session, io);
        if (self.text_projection) |session| session_factory.attachIo(session, io);
        if (self.visual_projection) |session| session_factory.attachIo(session, io);
        if (self.audio_projection) |session| session_factory.attachIo(session, io);
    }

    /// Snapshot all model-component exact-JIT counters while optional-session
    /// publication is stable. Metrics must not read those lazy pointers
    /// directly while an embedding request can materialize them.
    pub fn metalExactJitDispatchStats(self: *LoadedModel) session_factory.MetalExactJitDispatchStats {
        spinLock(&self.embedding_session_lock);
        defer self.embedding_session_lock.unlock();
        var total = session_factory.MetalExactJitDispatchStats{};
        const sessions = [_]?backends.Session{
            self.session,
            self.vision_session,
            self.audio_session,
            self.text_projection,
            self.visual_projection,
            self.audio_projection,
        };
        for (sessions) |maybe_session| {
            const session = maybe_session orelse continue;
            if (session_factory.getMetalExactJitDispatchStats(session)) |stats| total.add(stats);
        }
        return total;
    }

    pub fn lockNativeGeneration(self: *LoadedModel, io: std.Io) void {
        platform.sync.lockYieldingIo(&self.native_generate_lock, io);
    }

    pub fn unlockNativeGeneration(self: *LoadedModel) void {
        self.native_generate_lock.unlock();
    }

    pub fn nativeGenerationMutex(self: *LoadedModel) *std.atomic.Mutex {
        return &self.native_generate_lock;
    }

    pub fn lockRecognizerSession(self: *LoadedModel, io: std.Io) void {
        platform.sync.lockYieldingIo(&self.recognizer_session_lock, io);
    }

    pub fn unlockRecognizerSession(self: *LoadedModel) void {
        self.recognizer_session_lock.unlock();
    }

    pub fn wholeModelExecutor(self: *LoadedModel, allocator: std.mem.Allocator, kv_dtype: ?runtime.kv.pool.KvDType) !?graph_mod.model_runtime.ModelExecutor {
        const gpt_config = session_factory.getGptConfig(self.session) orelse return null;
        if (build_options.enable_metal and shouldUseMetalWholeModelExecutor(self.session) and graph_mod.metal_executor.supportsSession(self.session)) {
            return try graph_mod.metal_executor.createModelExecutor(
                allocator,
                self.session,
                gpt_config,
                kv_dtype,
                self.shared_moe_cache,
            );
        }
        if (!graph_mod.live_model_executor.supportsSession(self.session)) return null;
        return try graph_mod.live_model_executor.createModelExecutor(
            allocator,
            self.session,
            gpt_config,
            kv_dtype,
            self.shared_moe_cache,
        );
    }

    fn ensureOptionalSession(
        self: *LoadedModel,
        kind: DeclaredOptionalSessionKind,
        slot: *?backends.Session,
        path: ?[]const u8,
    ) !void {
        if (slot.* != null) return;
        const session_path = path orelse return;
        const shared_ctx = backends.imported_onnx_session.sharedBackendContext(self.session);
        var session_manager = self.session_manager.*.withPreferredBackends(
            self.allocator,
            self.optional_session_preferred_backends,
        );
        const profile_bundle = if (self.kernel_jit_profile_bundle) |*bundle| blk: {
            const mapped = bundle.kernelJitBundleForMode(session_manager.kernel_jit.mode);
            session_manager.kernel_jit.qualified_profile_path = mapped.primary;
            break :blk mapped;
        } else null;
        try bindOptionalSessionProfile(&session_manager, kind, profile_bundle);
        slot.* = try session_manager.loadModelWithImportedOnnxContext(session_path, shared_ctx);
    }

    /// Load every optional model/projection declared by the manifest without
    /// running media inference. Startup preload calls this while the node owns
    /// the exclusive JIT qualification phase, preventing a first media request
    /// from triggering dynamic compiler work after publication.
    pub fn materializeDeclaredOptionalSessions(self: *LoadedModel) !void {
        spinLock(&self.embedding_session_lock);
        defer self.embedding_session_lock.unlock();

        for (declaredOptionalSessions(&self.manifest)) |declared| {
            switch (declared.kind) {
                .vision => try self.ensureOptionalSession(.vision, &self.vision_session, declared.path),
                .audio => try self.ensureOptionalSession(.audio, &self.audio_session, declared.path),
                .text_projection => try self.ensureOptionalSession(.text_projection, &self.text_projection, declared.path),
                .visual_projection => try self.ensureOptionalSession(.visual_projection, &self.visual_projection, declared.path),
                .audio_projection => try self.ensureOptionalSession(.audio_projection, &self.audio_projection, declared.path),
            }
        }
        if (!self.declaredOptionalSessionsMaterializedUnlocked()) {
            return error.OptionalSessionMaterializationIncomplete;
        }
    }

    pub fn declaredOptionalSessionsMaterialized(self: *LoadedModel) bool {
        spinLock(&self.embedding_session_lock);
        defer self.embedding_session_lock.unlock();
        return self.declaredOptionalSessionsMaterializedUnlocked();
    }

    fn declaredOptionalSessionsMaterializedUnlocked(self: *const LoadedModel) bool {
        return declaredOptionalSessionsComplete(&self.manifest, .{
            self.vision_session != null,
            self.audio_session != null,
            self.text_projection != null,
            self.visual_projection != null,
            self.audio_projection != null,
        });
    }

    pub fn ensureVisionSession(self: *LoadedModel) !void {
        spinLock(&self.embedding_session_lock);
        defer self.embedding_session_lock.unlock();
        try self.ensureOptionalSession(.vision, &self.vision_session, self.manifest.visual_model_path);
    }

    pub fn ensureEmbeddingAssets(self: *LoadedModel, include_text: bool, include_image: bool, include_audio: bool) !void {
        spinLock(&self.embedding_session_lock);
        defer self.embedding_session_lock.unlock();

        if (include_text) {
            try self.ensureOptionalSession(.text_projection, &self.text_projection, self.manifest.text_projection_path);
        }
        if (include_image) {
            try self.ensureOptionalSession(.vision, &self.vision_session, self.manifest.visual_model_path);
            try self.ensureOptionalSession(.visual_projection, &self.visual_projection, self.manifest.visual_projection_path);
        }
        if (include_audio) {
            try self.ensureOptionalSession(.audio, &self.audio_session, self.manifest.audio_model_path);
            try self.ensureOptionalSession(.audio_projection, &self.audio_projection, self.manifest.audio_projection_path);
        }
    }

    pub fn embeddingPipeline(self: *LoadedModel, allocator: std.mem.Allocator) EmbeddingPipeline {
        const tok = self.getTokenizer();
        const resident_text_encoder = self.session.backend() == .cuda and blk: {
            const arch = session_factory.getGenericEncoderArchConfig(self.session) catch break :blk false;
            break :blk switch (arch) {
                .bert => true,
                .deberta => false,
            };
        };
        var pipeline = EmbeddingPipeline.init(allocator, self.session, tok, .{
            .max_length = self.manifest.max_position_embeddings,
            .normalize = self.manifest.normalize,
            .pooling = switch (self.manifest.pooling) {
                .mean => .mean,
                .cls => .cls,
                .max => .max,
                .last => .last,
            },
            .text_prefix = self.manifest.embedding_text_prefix,
            // BERT/XLM-R GGUFs declare their maximum context (BGE-M3: 8192)
            // while accepting dynamic sequence lengths. Padding every request
            // to that context would turn a 256-token encoder benchmark into an
            // unrelated 8192-token workload.
            .trim_padding_to_batch_max = isJinaStyleEmbeddingManifest(&self.manifest) or resident_text_encoder,
            .resident_qwen3_embedding = isJinaStyleEmbeddingManifest(&self.manifest),
            .resident_text_encoder = resident_text_encoder,
        });
        if (usesClipImagePreprocessProfile(&self.manifest)) {
            pipeline.config.image_preprocess_profile = .clip;
        }
        if (session_factory.getClipConfig(self.session)) |cfg| {
            pipeline.config.image_size = cfg.image_size;
            if (cfg.family == .clip) pipeline.config.image_preprocess_profile = .clip;
        } else if (self.vision_session) |vs| {
            if (session_factory.getClipConfig(vs)) |cfg| {
                pipeline.config.image_size = cfg.image_size;
                if (cfg.family == .clip) pipeline.config.image_preprocess_profile = .clip;
            }
        }
        pipeline.vision_session = self.vision_session;
        pipeline.audio_session = self.audio_session;
        pipeline.text_projection = self.text_projection;
        pipeline.visual_projection = self.visual_projection;
        pipeline.audio_projection = self.audio_projection;
        pipeline.resident_projection_stats = &self.resident_projection_stats;
        return pipeline;
    }

    pub fn rerankingPipeline(self: *LoadedModel, allocator: std.mem.Allocator) RerankingPipeline {
        const tok = self.getTokenizer();
        return RerankingPipeline.init(allocator, self.session, tok, .{
            .max_length = self.manifest.max_position_embeddings,
            .mode = if (self.manifest.hasCapability("late_interaction") or
                self.manifest.hasCapability("colbert") or
                self.manifest.hasCapability("colqwen") or
                self.manifest.hasCapability("multimodal_late_interaction"))
                ScoringMode.late_interaction
            else
                ScoringMode.cross_encoder,
            .single_text_encoding = if (self.manifest.prefersGenerationEncodingForLateInteraction()) .generation else .encoder,
            .add_bos_token = self.manifest.add_bos_token,
            .distributed = runtime.distributed.configFromEnv(),
        });
    }

    pub fn lockRerankingSession(self: *LoadedModel) void {
        spinLock(&self.reranking_session_lock);
    }

    pub fn unlockRerankingSession(self: *LoadedModel) void {
        self.reranking_session_lock.unlock();
    }

    pub fn classificationPipeline(self: *LoadedModel, allocator: std.mem.Allocator, config: ClassificationConfig) ClassificationPipeline {
        const tok = self.getTokenizer();
        var effective = config;
        effective.distributed = runtime.distributed.configFromEnv();
        return ClassificationPipeline.init(allocator, self.session, tok, effective);
    }

    pub fn nerPipeline(self: *LoadedModel, allocator: std.mem.Allocator) NerPipeline {
        const tok = self.getTokenizer();
        // Cast id2label from ?[][]const u8 to ?[]const []const u8
        const id2label: ?[]const []const u8 = if (self.manifest.id2label) |labels| labels else null;
        return NerPipeline.init(allocator, self.session, tok, .{
            .max_length = self.manifest.max_position_embeddings,
            .id2label = id2label,
            .distributed = runtime.distributed.configFromEnv(),
        });
    }

    pub fn isGlinerModel(self: *LoadedModel) bool {
        return self.manifest.gliner_model_type.len > 0;
    }

    pub fn supportsClassification(self: *LoadedModel) bool {
        return model_caps.modelSupportsCapability(
            @tagName(self.manifest.model_type),
            self.manifest.gliner_model_type,
            self.manifest.capabilities,
            "classification",
        );
    }

    pub fn supportsExtraction(self: *LoadedModel) bool {
        return model_caps.modelSupportsCapability(
            @tagName(self.manifest.model_type),
            self.manifest.gliner_model_type,
            self.manifest.capabilities,
            "extraction",
        );
    }

    pub fn supportsRelationExtraction(self: *LoadedModel) bool {
        return model_caps.modelSupportsCapability(
            @tagName(self.manifest.model_type),
            self.manifest.gliner_model_type,
            self.manifest.capabilities,
            "relations",
        );
    }

    pub fn glinerPipeline(self: *LoadedModel, allocator: std.mem.Allocator) GlinerPipeline {
        const tok = self.getTokenizer();
        return .{
            .allocator = allocator,
            .session = self.session,
            .tok = tok,
            .config = .{
                .max_width = self.manifest.gliner_max_width,
                .max_length = self.manifest.max_position_embeddings,
                .threshold = self.manifest.gliner_threshold,
                .flat_ner = self.manifest.gliner_flat_ner,
                .default_labels = self.manifest.gliner_default_labels,
                .relation_labels = self.manifest.gliner_relation_labels,
                .relation_threshold = self.manifest.gliner_relation_threshold,
                .model_type = self.manifest.gliner_model_type,
                .capabilities = self.manifest.capabilities,
                .token_p = self.manifest.gliner_token_p,
                .token_c = self.manifest.gliner_token_c,
                .token_e = self.manifest.gliner_token_e,
                .token_r = self.manifest.gliner_token_r,
                .token_sep_text = self.manifest.gliner_token_sep_text,
                .distributed = runtime.distributed.configFromEnv(),
            },
        };
    }

    pub fn getCleanupHead(self: *LoadedModel, io: std.Io) !?*const cleanup_model_mod.CleanupHead {
        platform.sync.lockYieldingIo(&self.cleanup_head_lock, io);
        defer self.cleanup_head_lock.unlock();
        if (self.cleanup_head_loaded) return self.cleanup_head;

        const loaded = (try cleanup_model_mod.loadHeadIfPresent(self.allocator, self.model_dir)) orelse {
            self.cleanup_head_loaded = true;
            return null;
        };
        const head = try self.allocator.create(cleanup_model_mod.CleanupHead);
        head.* = loaded;
        self.cleanup_head = head;
        self.cleanup_head_loaded = true;
        return head;
    }

    pub fn deinit(self: *LoadedModel) void {
        self.native_generation_graph_cache.deinit();
        self.prompt_prefix_cache.deinit();
        self.session.close();
        if (self.vision_session) |vs| vs.close();
        if (self.audio_session) |as_| as_.close();
        if (self.text_projection) |tp| tp.close();
        if (self.visual_projection) |vp| vp.close();
        if (self.audio_projection) |ap| ap.close();
        if (self.kernel_jit_profile_bundle) |*bundle| bundle.deinit();
        if (self.hf_tok) |ht| ht.deinitSelf();
        if (self.sp_tok) |sp| {
            sp.deinit();
            self.allocator.destroy(sp);
        }
        self.allocator.free(self.optional_session_preferred_backends);
        if (self.chat_tmpl) |ct| {
            var ct_mut = @constCast(ct);
            ct_mut.deinit();
            self.allocator.destroy(ct_mut);
        }
        if (self.shared_moe_cache) |cache| {
            cache.deinit();
            self.allocator.destroy(cache);
        }
        if (self.shared_prefetch) |state| {
            state.deinit();
            self.allocator.destroy(state);
        }
        if (self.native_generate_coordinator) |coordinator| {
            coordinator.deinit();
            self.allocator.destroy(coordinator);
        }
        if (self.cleanup_head) |head| {
            head.deinit();
            self.allocator.destroy(head);
        }
        self.manifest.deinit();
        self.allocator.free(self.model_dir);
    }
};

fn isJinaStyleEmbeddingManifest(manifest: *const manifest_mod.ModelManifest) bool {
    return std.mem.eql(u8, manifest.config_model_arch, "jina_embeddings_v5") or
        (manifest.pooling == .last and std.mem.eql(u8, manifest.embedding_text_prefix, "Document: "));
}

fn usesClipImagePreprocessProfile(manifest: *const manifest_mod.ModelManifest) bool {
    return std.mem.eql(u8, manifest.config_model_arch, "clip") or
        std.mem.eql(u8, manifest.config_model_arch, "clipclap") or
        manifest.isClipclapGgufBundle();
}

pub const ModelManager = struct {
    const LoadedModelMap = std.StringHashMapUnmanaged(*LoadedModel);

    pub const LoadedModelsGuard = struct {
        manager: *ModelManager,

        pub fn models(self: *const @This()) *const LoadedModelMap {
            return &self.manager.loaded;
        }

        pub fn deinit(self: *@This()) void {
            self.manager.state_mutex.unlock();
            self.* = undefined;
        }
    };

    allocator: std.mem.Allocator,
    session_manager: backends.SessionManager,
    loaded: LoadedModelMap,
    loaded_aliases: LoadedModelMap,
    /// Serializes every loaded-map read, mutation, and iteration, plus prompt
    /// cache ownership. Cold loading happens outside this lock so operational
    /// endpoints can still inspect the registry while a model starts.
    state_mutex: std.atomic.Mutex = .unlocked,
    /// Suppresses duplicate cold loads while keeping the state lock brief.
    // ponytail: serialize cold loads until measured startup contention justifies
    // a per-model singleflight table and its extra ownership states.
    load_mutex: std.atomic.Mutex = .unlocked,
    prompt_cache_owner: ?*LoadedModel = null,

    pub fn init(allocator: std.mem.Allocator, session_manager: backends.SessionManager) ModelManager {
        return .{
            .allocator = allocator,
            .session_manager = session_manager,
            .loaded = LoadedModelMap{},
            .loaded_aliases = LoadedModelMap{},
        };
    }

    fn lockMutex(mutex: *std.atomic.Mutex, io: ?std.Io) void {
        if (io) |active_io| {
            platform.sync.lockYieldingIo(mutex, active_io);
        } else {
            platform.sync.lockYielding(mutex);
        }
    }

    fn lockState(self: *ModelManager, io: ?std.Io) void {
        lockMutex(&self.state_mutex, io);
    }

    fn lockLoad(self: *ModelManager, io: ?std.Io) void {
        lockMutex(&self.load_mutex, io);
    }

    pub fn lockLoadedModels(self: *ModelManager, io: std.Io) LoadedModelsGuard {
        self.lockState(io);
        return .{ .manager = self };
    }

    pub fn deinit(self: *ModelManager) void {
        self.lockLoad(null);
        defer self.load_mutex.unlock();
        self.lockState(null);
        defer self.state_mutex.unlock();
        var it = self.loaded.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.loaded.deinit(self.allocator);
        var alias_it = self.loaded_aliases.iterator();
        while (alias_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.loaded_aliases.deinit(self.allocator);
        self.prompt_cache_owner = null;
    }

    pub fn attachIo(self: *ModelManager, io: std.Io) void {
        self.lockLoad(io);
        defer self.load_mutex.unlock();
        self.lockState(io);
        defer self.state_mutex.unlock();
        self.session_manager.io = io;
        var it = self.loaded.iterator();
        while (it.next()) |entry| entry.value_ptr.*.attachIo(io);
    }

    /// Claim the node-wide prompt cache for one process-stable loaded model.
    /// Different models fail closed instead of mutating an active model's KV
    /// manager from outside that model's generation lock.
    pub fn tryActivatePromptCache(
        self: *ModelManager,
        io: std.Io,
        model: *LoadedModel,
        node_config: runtime.kv.prompt_cache.Config,
    ) bool {
        self.lockState(io);
        defer self.state_mutex.unlock();
        if (!node_config.enabled) return false;
        if (self.prompt_cache_owner) |owner| {
            if (owner != model) return false;
        } else {
            self.prompt_cache_owner = model;
        }
        model.prompt_prefix_cache.configure(node_config);
        return true;
    }

    pub fn detachPromptCacheResourceUsageObserver(self: *ModelManager) void {
        self.lockState(null);
        defer self.state_mutex.unlock();
        var it = self.loaded.valueIterator();
        while (it.next()) |model| model.*.prompt_prefix_cache.detachResourceUsageObserver();
    }

    /// Load a model from a directory path. Returns a cached model if already loaded.
    pub fn loadFromDir(self: *ModelManager, model_dir: []const u8) !*LoadedModel {
        return self.loadFromDirImpl(model_dir, self.session_manager.preferred_backends, true, true);
    }

    pub fn loadFromDirWithPreferredBackends(
        self: *ModelManager,
        model_dir: []const u8,
        preferred_backends: []const backends.BackendType,
        cache_default_alias: bool,
    ) !*LoadedModel {
        return self.loadFromDirImpl(model_dir, preferred_backends, cache_default_alias, false);
    }

    fn loadFromDirImpl(
        self: *ModelManager,
        model_dir: []const u8,
        preferred_backends: []const backends.BackendType,
        cache_default_alias: bool,
        use_default_alias: bool,
    ) !*LoadedModel {
        if (try self.findLoadedModel(model_dir, preferred_backends, use_default_alias)) |model| return model;

        self.lockLoad(null);
        defer self.load_mutex.unlock();

        // A different caller may have completed the same load while this one
        // waited for duplicate-load suppression.
        if (try self.findLoadedModel(model_dir, preferred_backends, use_default_alias)) |model| return model;

        var session_manager = self.session_manager.withPreferredBackends(self.allocator, preferred_backends);
        const model = try self.loadFromDirUncached(model_dir, &session_manager);
        errdefer destroyLoadedModel(self, model);

        const published = try self.publishLoadedModel(model_dir, model, cache_default_alias);
        if (published != model) destroyLoadedModel(self, model);
        return published;
    }

    fn findLoadedModel(
        self: *ModelManager,
        model_dir: []const u8,
        preferred_backends: []const backends.BackendType,
        use_default_alias: bool,
    ) !?*LoadedModel {
        self.lockState(null);
        defer self.state_mutex.unlock();

        if (use_default_alias) {
            if (self.loaded.get(model_dir)) |model| return model;
            if (self.loaded_aliases.get(model_dir)) |model| return model;
        }
        for (preferred_backends) |backend| {
            if (!backend.supportsDirectSessionLoad()) continue;
            const variant_key = try backendVariantCacheKey(self.allocator, model_dir, backend);
            defer self.allocator.free(variant_key);
            if (self.loaded.get(variant_key)) |model| return model;
            if (self.loaded_aliases.get(variant_key)) |model| return model;
        }
        return null;
    }

    fn loadFromDirUncached(
        self: *ModelManager,
        model_dir: []const u8,
        sm: *backends.SessionManager,
    ) !*LoadedModel {
        // Load manifest
        var man = try manifest_mod.loadFromDir(self.allocator, model_dir);
        errdefer man.deinit();
        if (man.hasIncompleteGlinerBundle()) return error.IncompleteGlinerBundle;
        if (man.hasIncompleteColqwenBundle()) return error.IncompleteColqwenBundle;
        if (man.hasIncompleteClipclapGgufBundle()) return error.IncompleteClipclapGgufBundle;
        if (man.hasIncompleteFlorence2GgufBundle()) return error.IncompleteFlorence2Bundle;

        var qualified_profile_bundle: ?kernel_jit_profile_output.LoadedProfileBundle =
            if (sm.kernel_jit.qualified_profile_path) |path|
                try kernel_jit_profile_output.loadQualifiedProfileBundleIfPresent(
                    self.allocator,
                    sm.io orelse std.Options.debug_io,
                    path,
                )
            else
                null;
        errdefer if (qualified_profile_bundle) |*bundle| bundle.deinit();
        if (qualified_profile_bundle) |*bundle| {
            sm.kernel_jit.qualified_profile_path = bundle.kernelJitBundle().primary;
            if (!sm.kernel_jit.mode.failClosed() and !bundle.hasQualifiedKernels(.primary)) {
                std.log.warn("kernel JIT profile bundle primary has no qualified winner; using bundled kernels", .{});
                sm.kernel_jit.qualified_profile_path = null;
                sm.kernel_jit.mode = .off;
            }
        }

        // Load tokenizer
        var hf_tok: ?*hf_tokenizer.HfTokenizer = null;
        errdefer if (hf_tok) |ht| ht.deinitSelf();
        var sp_tok: ?*sentencepiece.Processor = null;
        errdefer if (sp_tok) |sp| {
            sp.deinit();
            self.allocator.destroy(sp);
        };

        const tokenizer_type = blk: {
            if (shouldPreferSentencePieceOverride(man, model_dir, self.allocator)) {
                break :blk manifest_mod.TokenizerType.sentencepiece;
            }
            break :blk man.tokenizer_type orelse return error.NoTokenizerFound;
        };

        switch (tokenizer_type) {
            .huggingface => {
                hf_tok = try loadHuggingFaceTokenizerFromDirOrGguf(self.allocator, model_dir, man.gguf_path);
            },
            .sentencepiece => {
                const sp = try loadSentencePieceTokenizerFromDirOrGguf(self.allocator, model_dir, man.gguf_path);
                try adoptAndConfigureSentencePieceTokenizer(&sp_tok, sp, man, model_dir, self.allocator);
            },
        }

        // Load session.
        const session = try loadSessionForPreferredBackends(self.allocator, sm.preferred_backends, model_dir, man, sm);
        errdefer session.close();
        const optional_session_preferred_backends = try ownSelectedFirstBackendPreference(
            self.allocator,
            sm.preferred_backends,
            session.backend(),
        );
        errdefer self.allocator.free(optional_session_preferred_backends);

        // Load chat template if available (for generator models)
        const chat_tmpl: ?*ChatTemplate = if (man.chat_template) |ct_source| blk2: {
            const ct = self.allocator.create(ChatTemplate) catch break :blk2 null;
            ct.* = ChatTemplate.init(
                self.allocator,
                ct_source,
                man.bos_token,
                man.eos_token,
                man.unk_token,
                man.pad_token,
            ) catch |err| {
                std.log.warn("chat template init failed for {s}: {s}", .{ model_dir, @errorName(err) });
                self.allocator.destroy(ct);
                break :blk2 null;
            };
            break :blk2 ct;
        } else null;
        errdefer if (chat_tmpl) |ct| {
            var ct_mut = @constCast(ct);
            ct_mut.deinit();
            self.allocator.destroy(ct_mut);
        };

        // Create loaded model
        const shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache = blk: {
            if (session_factory.getGptConfig(session)) |cfg| {
                if (cfg.usesMoe()) {
                    const cache = try self.allocator.create(runtime.moe.shared.SharedExpertCache);
                    cache.* = runtime.moe.shared.SharedExpertCache.init(self.allocator);
                    break :blk cache;
                }
            }
            break :blk null;
        };
        errdefer if (shared_moe_cache) |cache| {
            cache.deinit();
            self.allocator.destroy(cache);
        };
        const shared_prefetch: ?*runtime.tier.shared.SharedPrefetchState = if (session_factory.getGptConfig(session)) |_| blk: {
            const state = try self.allocator.create(runtime.tier.shared.SharedPrefetchState);
            state.* = runtime.tier.shared.SharedPrefetchState.init(self.allocator);
            errdefer {
                state.deinit();
                self.allocator.destroy(state);
            }
            try session_factory.attachSharedPrefetchState(session, state);
            break :blk state;
        } else null;
        errdefer if (shared_prefetch) |state| {
            state.deinit();
            self.allocator.destroy(state);
        };
        const native_generate_coordinator: ?*runtime.scheduler.native_generate.NativeGenerateCoordinator = if (session_factory.getGptConfig(session)) |_| blk: {
            const coordinator = try self.allocator.create(runtime.scheduler.native_generate.NativeGenerateCoordinator);
            coordinator.* = runtime.scheduler.native_generate.NativeGenerateCoordinator.init(self.allocator);
            break :blk coordinator;
        } else null;
        errdefer if (native_generate_coordinator) |coordinator| self.allocator.destroy(coordinator);
        const owned_model_dir = try self.allocator.dupe(u8, model_dir);
        errdefer self.allocator.free(owned_model_dir);
        const model = try self.allocator.create(LoadedModel);
        errdefer self.allocator.destroy(model);
        model.* = .{
            .manifest = man,
            .hf_tok = hf_tok,
            .sp_tok = sp_tok,
            .session = session,
            .session_manager = &self.session_manager,
            .optional_session_preferred_backends = optional_session_preferred_backends,
            .model_dir = owned_model_dir,
            .allocator = self.allocator,
            .chat_tmpl = chat_tmpl,
            .shared_moe_cache = shared_moe_cache,
            .shared_prefetch = shared_prefetch,
            .prompt_prefix_cache = runtime.kv.prompt_cache.PromptPrefixCache.init(self.allocator),
            .native_generate_coordinator = native_generate_coordinator,
            .native_generation_graph_cache = graph_mod.cache.GraphCache.init(self.allocator),
            .vision_session = null,
            .audio_session = null,
            .text_projection = null,
            .visual_projection = null,
            .audio_projection = null,
            .kernel_jit_profile_bundle = null,
        };

        if (build_options.enable_metal and shouldUseMetalWholeModelExecutor(session)) {
            if (session_factory.getGptConfig(session)) |gpt_config| {
                if (graph_mod.metal_executor.supportsSession(session)) {
                    _ = graph_mod.metal_executor.prewarmSharedDecoderRuntime(self.allocator, session, gpt_config) catch |err| {
                        std.log.warn("metal decoder-runtime prewarm failed for {s}: {s}", .{ model_dir, @errorName(err) });
                    };
                }
            }
        }

        model.kernel_jit_profile_bundle = qualified_profile_bundle;
        qualified_profile_bundle = null;

        return model;
    }

    fn publishLoadedModel(
        self: *ModelManager,
        model_dir: []const u8,
        model: *LoadedModel,
        cache_default_alias: bool,
    ) !*LoadedModel {
        const variant_key = try backendVariantCacheKey(self.allocator, model_dir, model.session.backend());
        var variant_key_owned = true;
        defer if (variant_key_owned) self.allocator.free(variant_key);

        const maybe_alias_key = if (cache_default_alias) try self.allocator.dupe(u8, model_dir) else null;
        var alias_key_owned = maybe_alias_key != null;
        defer if (alias_key_owned) self.allocator.free(maybe_alias_key.?);

        self.lockState(null);
        defer self.state_mutex.unlock();

        var inserted_variant = false;
        const published = self.loaded.get(variant_key) orelse blk: {
            try self.loaded.put(self.allocator, variant_key, model);
            variant_key_owned = false;
            inserted_variant = true;
            break :blk model;
        };
        errdefer if (inserted_variant) {
            if (self.loaded.fetchRemove(variant_key)) |removed| self.allocator.free(removed.key);
        };

        if (maybe_alias_key) |alias_key| {
            if (self.loaded.get(model_dir) == null and self.loaded_aliases.get(model_dir) == null) {
                try self.loaded_aliases.put(self.allocator, alias_key, published);
                alias_key_owned = false;
            }
        }
        return published;
    }

    fn destroyLoadedModel(self: *ModelManager, model: *LoadedModel) void {
        model.deinit();
        self.allocator.destroy(model);
    }
};

fn backendVariantCacheKey(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    backend: backends.BackendType,
) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}\nbackend={s}", .{ model_dir, @tagName(backend) });
}

fn preferredModelPathForBackend(
    model_dir: []const u8,
    man: manifest_mod.ModelManifest,
    backend: backends.BackendType,
) ?[]const u8 {
    return switch (backend) {
        .onnx => man.onnx_path orelse model_dir,
        .native, .metal, .cuda, .wasm => if (!manifestHasNativeAssets(man) and man.onnx_path != null)
            man.onnx_path.?
        else
            model_dir,
        .pjrt => null,
    };
}

fn effectiveLoadBackends(
    scratch: *[7]backends.BackendType,
    preferred_backends: []const backends.BackendType,
    man: manifest_mod.ModelManifest,
) []const backends.BackendType {
    if (!shouldPreferNativeSession(man)) return preferred_backends;

    var idx: usize = 0;
    for (preferred_backends) |backend| {
        if (backend == .onnx) continue;
        scratch[idx] = backend;
        idx += 1;
    }
    for (preferred_backends) |backend| {
        if (backend == .onnx) {
            scratch[idx] = backend;
            idx += 1;
        }
    }
    return scratch[0..idx];
}

fn loadSessionForPreferredBackends(
    allocator: std.mem.Allocator,
    preferred_backends: []const backends.BackendType,
    model_dir: []const u8,
    man: manifest_mod.ModelManifest,
    source_session_manager: *const backends.SessionManager,
) !backends.Session {
    var effective_scratch: [7]backends.BackendType = undefined;
    const effective_backends = effectiveLoadBackends(&effective_scratch, preferred_backends, man);
    for (effective_backends) |backend| {
        if (!backend.supportsDirectSessionLoad()) continue;
        if (source_session_manager.kernel_jit.mode.failClosed() and
            !backend.supportsKernelJitSession()) continue;
        const candidate_path = preferredModelPathForBackend(model_dir, man, backend) orelse continue;
        if (source_session_manager.kernel_jit.mode.failClosed() and
            std.mem.endsWith(u8, candidate_path, ".onnx")) continue;
        var single_backend = [_]backends.BackendType{backend};
        var backend_session_manager = source_session_manager.*.withPreferredBackends(allocator, single_backend[0..]);
        if (backend_session_manager.loadModel(candidate_path)) |session| {
            return session;
        } else |err| {
            if (source_session_manager.kernel_jit.qualified_profile_path != null and backend == .metal) return err;
            if (graph_mod.kernel_jit.isRequiredFailure(source_session_manager.kernel_jit.mode, err)) return err;
        }
    }

    std.log.err("loadModel({s}) failed: no backend accepted model", .{model_dir});
    std.log.err("manifest paths onnx={?s} visual={?s} audio={?s} text_projection={?s} visual_projection={?s} audio_projection={?s}", .{
        man.onnx_path,
        man.visual_model_path,
        man.audio_model_path,
        man.text_projection_path,
        man.visual_projection_path,
        man.audio_projection_path,
    });
    if (source_session_manager.kernel_jit.qualified_profile_path != null) {
        return error.KernelJitProfileRequiresMetalBackend;
    }
    if (source_session_manager.kernel_jit.mode.failClosed()) {
        return error.KernelJitRequiredBackendUnavailable;
    }
    return error.NoModelFileFound;
}

test "shouldPreferNativeSession prefers native GLiNER weights" {
    const allocator = std.testing.allocator;
    var man = manifest_mod.ModelManifest{ .allocator = allocator };
    defer man.deinit();

    try std.testing.expect(!shouldPreferNativeSession(man));

    man.gliner_model_type = try allocator.dupe(u8, "gliner2");
    try std.testing.expect(!shouldPreferNativeSession(man));

    man.safetensors_path = try allocator.dupe(u8, "model.safetensors");
    try std.testing.expect(shouldPreferNativeSession(man));
}

test "optional sessions retain the selected backend before fallbacks" {
    const preferred = [_]backends.BackendType{ .metal, .native };
    const owned = try ownSelectedFirstBackendPreference(
        std.testing.allocator,
        &preferred,
        .native,
    );
    defer std.testing.allocator.free(owned);
    try std.testing.expectEqualSlices(backends.BackendType, &.{ .native, .metal }, owned);

    const added = try ownSelectedFirstBackendPreference(
        std.testing.allocator,
        &preferred,
        .cuda,
    );
    defer std.testing.allocator.free(added);
    try std.testing.expectEqualSlices(backends.BackendType, &.{ .cuda, .metal, .native }, added);
}

test "required preload completeness includes every declared optional session" {
    const manifest = manifest_mod.ModelManifest{
        .allocator = std.testing.allocator,
        .visual_model_path = "vision.gguf",
        .audio_model_path = "audio.gguf",
        .text_projection_path = "text-projection.onnx",
        .visual_projection_path = "visual-projection.onnx",
        .audio_projection_path = "audio-projection.onnx",
    };
    const declared = declaredOptionalSessions(&manifest);
    try std.testing.expectEqual(@as(usize, 5), declared.len);
    try std.testing.expect(!declaredOptionalSessionsComplete(
        &manifest,
        .{ true, true, true, true, false },
    ));
    try std.testing.expect(declaredOptionalSessionsComplete(
        &manifest,
        .{ true, true, true, true, true },
    ));

    const text_only = manifest_mod.ModelManifest{ .allocator = std.testing.allocator };
    try std.testing.expect(declaredOptionalSessionsComplete(
        &text_only,
        .{ false, false, false, false, false },
    ));
}

test "optional sessions bind only their component-qualified profile" {
    var manager = backends.SessionManager.init(std.testing.allocator);
    manager.kernel_jit = .{ .mode = .on, .qualified_profile_path = "primary.json" };
    const bundle: graph_mod.kernel_jit.QualifiedProfileBundle = .{
        .primary = "primary.json",
        .vision = "vision.json",
        .audio = "audio.json",
    };

    try bindOptionalSessionProfile(&manager, .audio, bundle);
    try std.testing.expectEqualStrings("audio.json", manager.kernel_jit.qualified_profile_path.?);
    try bindOptionalSessionProfile(&manager, .text_projection, bundle);
    try std.testing.expect(manager.kernel_jit.qualified_profile_path == null);
    try std.testing.expectEqual(graph_mod.kernel_jit.Mode.off, manager.kernel_jit.mode);

    manager.kernel_jit = .{ .mode = .required, .qualified_profile_path = "primary.json" };
    try std.testing.expectError(
        error.MissingKernelJitProfileBundleComponent,
        bindOptionalSessionProfile(&manager, .text_projection, bundle),
    );

    manager.kernel_jit = .{ .mode = .on, .qualified_profile_path = "primary.json" };
    try bindOptionalSessionProfile(&manager, .vision, null);
    try std.testing.expect(manager.kernel_jit.qualified_profile_path == null);
    try std.testing.expectEqual(graph_mod.kernel_jit.Mode.off, manager.kernel_jit.mode);
}

test "preferredModelPathForBackend keeps metal/native on model directory when native assets exist" {
    const allocator = std.testing.allocator;
    var man = manifest_mod.ModelManifest{ .allocator = allocator };
    defer man.deinit();

    man.onnx_path = try allocator.dupe(u8, "/tmp/model.onnx");
    man.safetensors_path = try allocator.dupe(u8, "/tmp/model.safetensors");

    try std.testing.expectEqualStrings("/tmp/model.onnx", preferredModelPathForBackend("/tmp/model", man, .onnx).?);
    try std.testing.expectEqualStrings("/tmp/model", preferredModelPathForBackend("/tmp/model", man, .metal).?);
    try std.testing.expectEqualStrings("/tmp/model", preferredModelPathForBackend("/tmp/model", man, .native).?);
    try std.testing.expectEqualStrings("/tmp/model", preferredModelPathForBackend("/tmp/model", man, .metal).?);
}

test "preferredModelPathForBackend routes direct compute backends to onnx path for onnx-only bundle" {
    const allocator = std.testing.allocator;
    var man = manifest_mod.ModelManifest{ .allocator = allocator };
    defer man.deinit();

    man.onnx_path = try allocator.dupe(u8, "/tmp/text_model.onnx");
    man.visual_model_path = try allocator.dupe(u8, "/tmp/visual_model.onnx");
    man.audio_model_path = try allocator.dupe(u8, "/tmp/audio_model.onnx");

    try std.testing.expectEqualStrings("/tmp/text_model.onnx", preferredModelPathForBackend("/tmp/model", man, .onnx).?);
    try std.testing.expectEqualStrings("/tmp/text_model.onnx", preferredModelPathForBackend("/tmp/model", man, .native).?);
    try std.testing.expectEqualStrings("/tmp/text_model.onnx", preferredModelPathForBackend("/tmp/model", man, .metal).?);
    try std.testing.expectEqualStrings("/tmp/text_model.onnx", preferredModelPathForBackend("/tmp/model", man, .metal).?);
}

test "shouldPreferNativeSession prefers split GLiNER gguf bundle" {
    const allocator = std.testing.allocator;
    var man = manifest_mod.ModelManifest{ .allocator = allocator };
    defer man.deinit();

    man.gliner_model_type = try allocator.dupe(u8, "gliner2");
    man.gguf_path = try allocator.dupe(u8, "encoder.gguf");
    man.gliner_head_gguf_path = try allocator.dupe(u8, "gliner_head.gguf");
    try std.testing.expect(shouldPreferNativeSession(man));
}

test "isManifestPotentiallyLoadableInCurrentBuild rejects incomplete GLiNER bundle" {
    const allocator = std.testing.allocator;
    var man = manifest_mod.ModelManifest{ .allocator = allocator };
    defer man.deinit();

    man.gliner_model_type = try allocator.dupe(u8, "gliner2");
    man.gguf_path = try allocator.dupe(u8, "encoder.gguf");
    try std.testing.expect(!isManifestPotentiallyLoadableInCurrentBuild(man));
}

test "shouldPreferNativeSession prefers native CLIP, Whisper, and Florence weights" {
    const allocator = std.testing.allocator;

    var clip = manifest_mod.ModelManifest{ .allocator = allocator, .native_arch_hint = .clip };
    defer clip.deinit();
    try std.testing.expect(!shouldPreferNativeSession(clip));
    clip.safetensors_path = try allocator.dupe(u8, "model.safetensors");
    try std.testing.expect(shouldPreferNativeSession(clip));

    var whisper = manifest_mod.ModelManifest{ .allocator = allocator, .native_arch_hint = .whisper };
    defer whisper.deinit();
    try std.testing.expect(!shouldPreferNativeSession(whisper));
    whisper.safetensors_path = try allocator.dupe(u8, "model.safetensors");
    try std.testing.expect(shouldPreferNativeSession(whisper));

    var florence = manifest_mod.ModelManifest{ .allocator = allocator, .native_arch_hint = .florence };
    defer florence.deinit();
    try std.testing.expect(!shouldPreferNativeSession(florence));
    florence.safetensors_path = try allocator.dupe(u8, "model.safetensors");
    try std.testing.expect(shouldPreferNativeSession(florence));
}

test "shouldPreferNativeSession prefers native classifier and recognizer weights" {
    const allocator = std.testing.allocator;

    var classifier = manifest_mod.ModelManifest{ .allocator = allocator, .model_type = .classifier };
    defer classifier.deinit();
    try std.testing.expect(!shouldPreferNativeSession(classifier));
    classifier.safetensors_path = try allocator.dupe(u8, "model.safetensors");
    try std.testing.expect(shouldPreferNativeSession(classifier));

    var recognizer = manifest_mod.ModelManifest{ .allocator = allocator, .model_type = .recognizer };
    defer recognizer.deinit();
    try std.testing.expect(!shouldPreferNativeSession(recognizer));
    recognizer.safetensors_path = try allocator.dupe(u8, "model.safetensors");
    try std.testing.expect(shouldPreferNativeSession(recognizer));
}

test "effectiveLoadBackends keeps gpu native backends ahead of cpu native before onnx" {
    const allocator = std.testing.allocator;
    const preferred = [_]backends.BackendType{ .onnx, .metal, .native };
    var scratch: [7]backends.BackendType = undefined;

    var classifier = manifest_mod.ModelManifest{ .allocator = allocator, .model_type = .classifier };
    defer classifier.deinit();
    classifier.safetensors_path = try allocator.dupe(u8, "model.safetensors");

    const effective = effectiveLoadBackends(&scratch, &preferred, classifier);
    try std.testing.expectEqualSlices(backends.BackendType, &.{ .metal, .native, .onnx }, effective);
}

test "effectiveLoadBackends preserves explicit onnx-only classifier preference" {
    const allocator = std.testing.allocator;
    const preferred = [_]backends.BackendType{.onnx};
    var scratch: [7]backends.BackendType = undefined;

    var classifier = manifest_mod.ModelManifest{ .allocator = allocator, .model_type = .classifier };
    defer classifier.deinit();
    classifier.safetensors_path = try allocator.dupe(u8, "model.safetensors");

    const effective = effectiveLoadBackends(&scratch, &preferred, classifier);
    try std.testing.expectEqualSlices(backends.BackendType, &preferred, effective);
}

test "isManifestPotentiallyLoadableInCurrentBuild accepts onnx-only models when onnx model support is enabled" {
    const allocator = std.testing.allocator;

    var onnx_only = manifest_mod.ModelManifest{ .allocator = allocator };
    defer onnx_only.deinit();
    onnx_only.onnx_path = try allocator.dupe(u8, "model.onnx");

    try std.testing.expect(isManifestPotentiallyLoadableInCurrentBuild(onnx_only));

    var native_model = manifest_mod.ModelManifest{ .allocator = allocator };
    defer native_model.deinit();
    native_model.safetensors_path = try allocator.dupe(u8, "model.safetensors");
    try std.testing.expect(isManifestPotentiallyLoadableInCurrentBuild(native_model));
}

test "isManifestPotentiallyLoadableInCurrentBuild hides incomplete colqwen bundles" {
    const allocator = std.testing.allocator;
    var colqwen = manifest_mod.ModelManifest{ .allocator = allocator };
    defer colqwen.deinit();
    colqwen.inference_bundle_family = try allocator.dupe(u8, "colqwen2_gguf_bundle/v1");
    colqwen.config_model_arch = try allocator.dupe(u8, "qwen2");
    colqwen.gguf_path = try allocator.dupe(u8, "model.gguf");
    colqwen.config_path = try allocator.dupe(u8, "config.json");
    colqwen.model_manifest_path = try allocator.dupe(u8, "model_manifest.json");
    colqwen.tokenizer_json_path = try allocator.dupe(u8, "tokenizer.json");
    colqwen.tokenizer_config_path = try allocator.dupe(u8, "tokenizer_config.json");
    colqwen.preprocessor_config_path = try allocator.dupe(u8, "preprocessor_config.json");
    try std.testing.expect(!isManifestPotentiallyLoadableInCurrentBuild(colqwen));

    colqwen.processor_config_path = try allocator.dupe(u8, "processor_config.json");
    try std.testing.expect(isManifestPotentiallyLoadableInCurrentBuild(colqwen));
}

test "ClipClap manifest selects CLIP image preprocessing profile" {
    const allocator = std.testing.allocator;
    var clipclap = manifest_mod.ModelManifest{ .allocator = allocator };
    defer clipclap.deinit();

    clipclap.config_model_arch = try allocator.dupe(u8, "clipclap");
    try std.testing.expect(usesClipImagePreprocessProfile(&clipclap));

    var siglip = manifest_mod.ModelManifest{ .allocator = allocator };
    defer siglip.deinit();
    siglip.config_model_arch = try allocator.dupe(u8, "siglip");
    try std.testing.expect(!usesClipImagePreprocessProfile(&siglip));
}

test "ModelManager loads split gliner bundle and exposes runtime pipeline" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "config.json",
        .data =
        \\{"model_type":"recognizer","hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":8,"vocab_size":16,"max_position_embeddings":16,"position_buckets":16}
        ,
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "gliner_config.json",
        .data = "{\"model_type\":\"gliner2\",\"max_width\":4,\"capabilities\":[\"extraction\"]}",
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "model_manifest.json",
        .data = "{\"type\":\"recognizer\",\"capabilities\":[\"extraction\"]}",
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "antfly_inference_bundle.json",
        .data = "{\"family\":\"gliner2_split_bundle/v1\",\"wrapper\":\"gliner2\",\"encoder\":\"encoder.gguf\",\"head\":\"gliner_head.safetensors\"}",
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "tokenizer.json",
        .data =
        \\{
        \\  "version":"1.0",
        \\  "normalizer":{"type":"BertNormalizer","lowercase":true},
        \\  "pre_tokenizer":{"type":"BertPreTokenizer"},
        \\  "post_processor":{"type":"BertProcessing","sep":["[SEP]",3],"cls":["[CLS]",2]},
        \\  "added_tokens":[
        \\    {"id":0,"content":"[PAD]"},
        \\    {"id":1,"content":"[UNK]"},
        \\    {"id":2,"content":"[CLS]"},
        \\    {"id":3,"content":"[SEP]"}
        \\  ],
        \\  "model":{
        \\    "type":"WordPiece",
        \\    "unk_token":"[UNK]",
        \\    "continuing_subword_prefix":"##",
        \\    "max_input_chars_per_word":100,
        \\    "vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"hello":4,"person":5}
        \\  }
        \\}
        ,
    });
    try writeTinyDebertaEncoderGgufForModelManagerTest(tmp.dir, allocator, "encoder.gguf");
    try writeTinyHeadSafetensorsForModelManagerTest(tmp.dir, allocator, "gliner_head.safetensors");

    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(dir_path);

    var manager = ModelManager.init(allocator, .{
        .allocator = allocator,
        .preferred_backends = &.{.native},
    });
    defer manager.deinit();

    const model = try manager.loadFromDir(dir_path);
    try std.testing.expect(model.isGlinerModel());
    try std.testing.expect(model.supportsExtraction());
    try std.testing.expectEqualStrings("gliner2_split_bundle/v1", model.manifest.inference_bundle_family);

    var pipeline = model.glinerPipeline(allocator);
    try std.testing.expectEqualStrings("gliner2", pipeline.config.model_type);
    try std.testing.expectError(error.MissingSpecialTokenIds, pipeline.recognizeBatch(&.{"hello"}, &.{"person"}));
}

test "ModelManager loads split gliner gguf-head bundle and exposes runtime pipeline" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "config.json",
        .data =
        \\{"model_type":"recognizer","hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":8,"vocab_size":16,"max_position_embeddings":16,"position_buckets":16}
        ,
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "gliner_config.json",
        .data = "{\"model_type\":\"gliner2\",\"max_width\":4,\"capabilities\":[\"extraction\"]}",
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "model_manifest.json",
        .data = "{\"type\":\"recognizer\",\"capabilities\":[\"extraction\"]}",
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "antfly_inference_bundle.json",
        .data = "{\"family\":\"gliner2_split_bundle/v1\",\"wrapper\":\"gliner2\",\"encoder\":\"encoder.gguf\",\"head\":\"gliner_head.gguf\"}",
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "tokenizer.json",
        .data =
        \\{
        \\  "version":"1.0",
        \\  "normalizer":{"type":"BertNormalizer","lowercase":true},
        \\  "pre_tokenizer":{"type":"BertPreTokenizer"},
        \\  "post_processor":{"type":"BertProcessing","sep":["[SEP]",3],"cls":["[CLS]",2]},
        \\  "added_tokens":[
        \\    {"id":0,"content":"[PAD]"},
        \\    {"id":1,"content":"[UNK]"},
        \\    {"id":2,"content":"[CLS]"},
        \\    {"id":3,"content":"[SEP]"}
        \\  ],
        \\  "model":{
        \\    "type":"WordPiece",
        \\    "unk_token":"[UNK]",
        \\    "continuing_subword_prefix":"##",
        \\    "max_input_chars_per_word":100,
        \\    "vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"hello":4,"person":5}
        \\  }
        \\}
        ,
    });
    try writeTinyDebertaEncoderGgufForModelManagerTest(tmp.dir, allocator, "encoder.gguf");
    try writeTinyHeadGgufForModelManagerTest(tmp.dir, allocator, "gliner_head.gguf");

    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(dir_path);

    var manager = ModelManager.init(allocator, .{
        .allocator = allocator,
        .preferred_backends = &.{.native},
    });
    defer manager.deinit();

    const model = try manager.loadFromDir(dir_path);
    try std.testing.expect(model.isGlinerModel());
    try std.testing.expect(model.supportsExtraction());
    try std.testing.expectEqualStrings("gliner2_split_bundle/v1", model.manifest.inference_bundle_family);

    var pipeline = model.glinerPipeline(allocator);
    try std.testing.expectEqualStrings("gliner2", pipeline.config.model_type);
    try std.testing.expectError(error.MissingSpecialTokenIds, pipeline.recognizeBatch(&.{"hello"}, &.{"person"}));
}

test "shouldPreferSentencePieceOverride still prefers sentencepiece for multimodal gemma" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tokenizer.model", .data = "fake-spm" });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "added_tokens.json",
        .data = "{\n  \"<image_soft_token>\": 262144\n}\n",
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "config.json",
        .data = "{\n  \"model_type\": \"gemma3\"\n}\n",
    });

    var man = manifest_mod.ModelManifest{ .allocator = allocator };
    defer man.deinit();

    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(dir_path);

    try std.testing.expect(shouldPreferSentencePieceOverride(man, dir_path, allocator));
}

test "shouldEnableGemmaSentencePieceCompat applies to gguf-only gemma dirs" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var man = manifest_mod.ModelManifest{ .allocator = allocator };
    defer man.deinit();
    man.tokenizer_type = .sentencepiece;

    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "gemma-4-e2b-it-gguf" });
    defer allocator.free(dir_path);

    try std.testing.expect(shouldEnableGemmaSentencePieceCompat(man, dir_path, allocator));
    try std.testing.expect(!shouldPreferSentencePieceOverride(man, dir_path, allocator));
}

test "sentencepiece tokenizer is owned before added-token failure" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "added_tokens.json",
        .data = "]",
    });

    const dir_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(dir_path);
    const man = manifest_mod.ModelManifest{ .allocator = allocator };

    const Harness = struct {
        fn run(a: std.mem.Allocator, model_dir: []const u8, manifest: manifest_mod.ModelManifest) !void {
            var owned: ?*sentencepiece.Processor = null;
            errdefer if (owned) |sp| {
                sp.deinit();
                a.destroy(sp);
            };

            const sp = try a.create(sentencepiece.Processor);
            errdefer if (owned == null) a.destroy(sp);
            sp.* = try sentencepiece.Processor.initFromPieces(a, &.{
                .{ .text = "<unk>", .score = 0, .piece_type = 2 },
                .{ .text = "token", .score = -1, .piece_type = 1 },
            }, .{});

            try adoptAndConfigureSentencePieceTokenizer(&owned, sp, manifest, model_dir, a);

            // Keep an unexpected success leak-free so expectError reports only
            // the missing post-load failure.
            sp.deinit();
            a.destroy(sp);
            owned = null;
        }
    };

    try std.testing.expectError(error.SyntaxError, Harness.run(allocator, dir_path, man));
}

test "loadSentencePieceAddedTokens overlays gemma special tokens from tokenizer json" {
    const allocator = std.testing.allocator;
    const model_dir = "models/google/gemma-3-4b-it";
    if (!c_file.fileExistsInDir(allocator, model_dir, "tokenizer.model")) return error.SkipZigTest;
    if (!c_file.fileExistsInDir(allocator, model_dir, "tokenizer.json")) return error.SkipZigTest;

    const sp_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.model", .{model_dir});
    defer allocator.free(sp_path);
    var sp = try sentencepiece.Processor.initFromPath(allocator, sp_path);
    defer sp.deinit();

    try loadSentencePieceAddedTokens(model_dir, allocator, &sp);
    try std.testing.expectEqual(@as(?i32, 105), sp.piece_map.get("<start_of_turn>"));
    try std.testing.expectEqual(@as(?i32, 262144), sp.extra_reserved_map.get("<image_soft_token>"));
    try std.testing.expectEqual("<start_of_turn>".len, sp.special_matcher.findPrefixLen("<start_of_turn>"));

    const encoded = try sp.tokenizer().encodeForGenerationConfigured(allocator, "<start_of_turn>", 16, false);
    defer {
        var encoded_mut = encoded;
        encoded_mut.deinit();
    }
    var found = false;
    for (encoded.ids[0..encoded.attention_mask.len], 0..) |id, idx| {
        if (encoded.attention_mask[idx] == 0) break;
        if (id == 105) {
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "gemma sentencepiece prompt parity against hf tokenizer" {
    const allocator = std.testing.allocator;
    const model_dir = "models/google/gemma-3-4b-it";
    if (!c_file.fileExistsInDir(allocator, model_dir, "tokenizer.model")) return error.SkipZigTest;
    if (!c_file.fileExistsInDir(allocator, model_dir, "tokenizer.json")) return error.SkipZigTest;

    const prompt =
        "<bos><start_of_turn>user\n" ++
        "<start_of_image>Describe this image.<end_of_turn>\n" ++
        "<start_of_turn>model\n";

    const sp_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.model", .{model_dir});
    defer allocator.free(sp_path);
    var sp = try sentencepiece.Processor.initFromPath(allocator, sp_path);
    defer sp.deinit();
    try loadSentencePieceAddedTokens(model_dir, allocator, &sp);

    const tokenizer_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_dir});
    defer allocator.free(tokenizer_path);
    const tokenizer_bytes = try c_file.readFile(allocator, tokenizer_path);
    defer allocator.free(tokenizer_bytes);
    var hf = try hf_tokenizer.HfTokenizer.loadFromBytes(allocator, tokenizer_bytes);
    defer hf.deinitSelf();

    var sp_encoded = try sp.tokenizer().encodeForGenerationConfigured(allocator, prompt, 512, false);
    defer sp_encoded.deinit();
    var hf_encoded = try hf.tokenizer().encodeForGenerationConfigured(allocator, prompt, 512, false);
    defer hf_encoded.deinit();

    var sp_count: usize = 0;
    while (sp_count < sp_encoded.attention_mask.len and sp_encoded.attention_mask[sp_count] != 0) : (sp_count += 1) {}
    var hf_count: usize = 0;
    while (hf_count < hf_encoded.attention_mask.len and hf_encoded.attention_mask[hf_count] != 0) : (hf_count += 1) {}
    try std.testing.expectEqual(sp_count, hf_count);
    try std.testing.expectEqualSlices(i32, sp_encoded.ids[0..sp_count], hf_encoded.ids[0..hf_count]);
}

fn writeTinyDebertaEncoderGgufForModelManagerTest(
    dir: anytype,
    allocator: std.mem.Allocator,
    sub_path: []const u8,
) !void {
    const metadata = [_]gguf_format.MetadataEntry{
        .{ .key = "general.architecture", .value = .{ .string = "deberta" } },
        .{ .key = "general.alignment", .value = .{ .u32 = @intCast(gguf_format.default_alignment) } },
        .{ .key = "deberta.vocab_size", .value = .{ .u32 = 16 } },
        .{ .key = "deberta.embedding_length", .value = .{ .u32 = 4 } },
        .{ .key = "deberta.block_count", .value = .{ .u32 = 1 } },
        .{ .key = "deberta.attention.head_count", .value = .{ .u32 = 2 } },
        .{ .key = "deberta.feed_forward_length", .value = .{ .u32 = 8 } },
        .{ .key = "deberta.context_length", .value = .{ .u32 = 16 } },
        .{ .key = "deberta.position_buckets", .value = .{ .u32 = 16 } },
        .{ .key = "deberta.label_count", .value = .{ .u32 = 1 } },
    };
    const dims_vocab = [_]u64{ 4, 16 };
    const dims_hidden = [_]u64{4};
    const dims_rel = [_]u64{ 4, 16 };
    const dims_dense = [_]u64{ 4, 4 };
    const dims_intermediate = [_]u64{ 4, 8 };
    const dims_output = [_]u64{ 8, 4 };
    const dims_intermediate_bias = [_]u64{8};
    const tensors = [_]gguf_writer.TensorSpec{
        .{ .name = "embeddings.word_embeddings.weight", .dimensions = &dims_vocab, .tensor_type = .{ .known = .F32 } },
        .{ .name = "embeddings.LayerNorm.weight", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "embeddings.LayerNorm.bias", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.rel_embeddings.weight", .dimensions = &dims_rel, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.LayerNorm.weight", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.LayerNorm.bias", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.attention.self.query_proj.weight", .dimensions = &dims_dense, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.attention.self.query_proj.bias", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.attention.self.key_proj.weight", .dimensions = &dims_dense, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.attention.self.key_proj.bias", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.attention.self.value_proj.weight", .dimensions = &dims_dense, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.attention.self.value_proj.bias", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.attention.output.dense.weight", .dimensions = &dims_dense, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.attention.output.dense.bias", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.attention.output.LayerNorm.weight", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.attention.output.LayerNorm.bias", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.intermediate.dense.weight", .dimensions = &dims_intermediate, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.intermediate.dense.bias", .dimensions = &dims_intermediate_bias, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.output.dense.weight", .dimensions = &dims_output, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.output.dense.bias", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.output.LayerNorm.weight", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
        .{ .name = "encoder.layer.0.output.LayerNorm.bias", .dimensions = &dims_hidden, .tensor_type = .{ .known = .F32 } },
    };

    var layout = try gguf_writer.buildLayout(allocator, &metadata, &tensors);
    defer layout.deinit(allocator);

    var data = std.ArrayListUnmanaged(u8).empty;
    defer data.deinit(allocator);
    try data.appendSlice(allocator, layout.header_bytes);
    const data_region_offset = std.mem.alignForward(usize, layout.header_bytes.len, @intCast(layout.alignment));
    try data.appendNTimes(allocator, 0, data_region_offset - layout.header_bytes.len);

    var written_offset: u64 = 0;
    for (tensors, layout.offsets) |tensor, offset| {
        if (offset > written_offset) {
            try data.appendNTimes(allocator, 0, @intCast(offset - written_offset));
            written_offset = offset;
        }
        const byte_len = gguf_tensor_types.byteLen(tensor.tensor_type, tensor.dimensions) orelse return error.UnsupportedTensorType;
        try data.appendNTimes(allocator, 0, byte_len);
        written_offset += @intCast(byte_len);
    }

    try dir.writeFile(std.testing.io, .{ .sub_path = sub_path, .data = data.items });
}

fn writeTinyHeadSafetensorsForModelManagerTest(
    dir: anytype,
    allocator: std.mem.Allocator,
    sub_path: []const u8,
) !void {
    const json =
        \\{"span_rep.test":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}
    ;
    var data = std.ArrayListUnmanaged(u8).empty;
    defer data.deinit(allocator);
    try appendLeModelManagerTest(u64, allocator, &data, json.len);
    try data.appendSlice(allocator, json);
    try data.appendSlice(allocator, std.mem.asBytes(&[_]f32{ 0.0, 0.0 }));
    try dir.writeFile(std.testing.io, .{ .sub_path = sub_path, .data = data.items });
}

fn writeTinyHeadGgufForModelManagerTest(
    dir: anytype,
    allocator: std.mem.Allocator,
    sub_path: []const u8,
) !void {
    const metadata = [_]gguf_format.MetadataEntry{
        .{ .key = "general.architecture", .value = .{ .string = "antfly-gliner-head" } },
        .{ .key = "general.alignment", .value = .{ .u32 = @intCast(gguf_format.default_alignment) } },
    };
    const dims = [_]u64{2};
    const tensors = [_]gguf_writer.TensorSpec{
        .{ .name = "span_rep.test", .dimensions = &dims, .tensor_type = .{ .known = .F32 } },
    };

    var layout = try gguf_writer.buildLayout(allocator, &metadata, &tensors);
    defer layout.deinit(allocator);

    var data = std.ArrayListUnmanaged(u8).empty;
    defer data.deinit(allocator);
    try data.appendSlice(allocator, layout.header_bytes);
    const data_region_offset = std.mem.alignForward(usize, layout.header_bytes.len, @intCast(layout.alignment));
    try data.appendNTimes(allocator, 0, data_region_offset - layout.header_bytes.len);
    try data.appendSlice(allocator, std.mem.asBytes(&[_]f32{ 0.0, 0.0 }));

    try dir.writeFile(std.testing.io, .{ .sub_path = sub_path, .data = data.items });
}

fn appendLeModelManagerTest(
    comptime T: type,
    allocator: std.mem.Allocator,
    data: *std.ArrayListUnmanaged(u8),
    value: T,
) !void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, value, .little);
    try data.appendSlice(allocator, &buf);
}

test "load huggingface tokenizer from gguf gpt2 metadata" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const gguf_bytes = try buildTestGgufWithGpt2Tokenizer(allocator);
    defer allocator.free(gguf_bytes);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "ggml-model-i2_s.gguf", .data = gguf_bytes });

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);
    const gguf_path = try std.fs.path.join(allocator, &.{ model_dir, "ggml-model-i2_s.gguf" });
    defer allocator.free(gguf_path);

    var tok = try loadHuggingFaceTokenizerFromDirOrGguf(allocator, model_dir, gguf_path);
    defer tok.deinitSelf();

    var encoded = try tok.tokenizer().encodeForGenerationConfigured(allocator, "hello", 8, true);
    defer encoded.deinit();

    try std.testing.expectEqual(@as(i32, 0), encoded.ids[0]);
    try std.testing.expectEqual(@as(i32, 1), encoded.ids[1]);
    try std.testing.expectEqual(@as(i32, 1), encoded.attention_mask[0]);
    try std.testing.expectEqual(@as(i32, 1), encoded.attention_mask[1]);
}

test "load huggingface tokenizer from gguf gemma4 bpe metadata" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const gguf_bytes = try buildTestGgufWithGemma4Tokenizer(allocator);
    defer allocator.free(gguf_bytes);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "gemma4-q4_0.gguf", .data = gguf_bytes });

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);
    const gguf_path = try std.fs.path.join(allocator, &.{ model_dir, "gemma4-q4_0.gguf" });
    defer allocator.free(gguf_path);

    var tok = try loadHuggingFaceTokenizerFromDirOrGguf(allocator, model_dir, gguf_path);
    defer tok.deinitSelf();

    var encoded = try tok.tokenizer().encodeForGenerationConfigured(allocator, "hello world", 8, true);
    defer encoded.deinit();

    try std.testing.expectEqual(@as(i32, 2), encoded.ids[0]);
    try std.testing.expectEqual(@as(i32, 4), encoded.ids[1]);
    try std.testing.expectEqual(@as(i32, 5), encoded.ids[2]);
    try std.testing.expectEqual(@as(i32, 1), encoded.attention_mask[0]);
    try std.testing.expectEqual(@as(i32, 1), encoded.attention_mask[1]);
    try std.testing.expectEqual(@as(i32, 1), encoded.attention_mask[2]);

    const special_ids = try tok.tokenizer().encode(allocator, "<|turn>hello");
    defer allocator.free(special_ids);
    try std.testing.expectEqualSlices(i32, &.{ 6, 4 }, special_ids);

    const decoded = try tok.tokenizer().decode(allocator, &.{ 4, 5 });
    defer allocator.free(decoded);
    try std.testing.expectEqualStrings("hello world", decoded);
}

fn buildTestGgufWithGpt2Tokenizer(allocator: std.mem.Allocator) ![]u8 {
    var data = std.ArrayListUnmanaged(u8).empty;
    defer data.deinit(allocator);

    try data.appendSlice(allocator, gguf_format.magic);
    try appendTestLe(u32, allocator, &data, 3);
    try appendTestLe(u64, allocator, &data, 0);
    try appendTestLe(u64, allocator, &data, 7);

    try appendTestMetadataString(allocator, &data, "general.architecture", "bitnet-b1.58");
    try appendTestMetadataString(allocator, &data, "tokenizer.ggml.model", "gpt2");
    try appendTestMetadataStringArray(allocator, &data, "tokenizer.ggml.tokens", &.{
        "<|begin_of_text|>",
        "hello",
        "<|end_of_text|>",
    });
    try appendTestMetadataStringArray(allocator, &data, "tokenizer.ggml.merges", &.{});
    try appendTestMetadataU32(allocator, &data, "tokenizer.ggml.bos_token_id", 0);
    try appendTestMetadataU32(allocator, &data, "tokenizer.ggml.eos_token_id", 2);
    try appendTestMetadataBool(allocator, &data, "tokenizer.ggml.add_bos_token", true);

    return data.toOwnedSlice(allocator);
}

fn buildTestGgufWithGemma4Tokenizer(allocator: std.mem.Allocator) ![]u8 {
    var data = std.ArrayListUnmanaged(u8).empty;
    defer data.deinit(allocator);

    try data.appendSlice(allocator, gguf_format.magic);
    try appendTestLe(u32, allocator, &data, 3);
    try appendTestLe(u64, allocator, &data, 0);
    try appendTestLe(u64, allocator, &data, 10);

    try appendTestMetadataString(allocator, &data, "general.architecture", "gemma4");
    try appendTestMetadataString(allocator, &data, "tokenizer.ggml.model", "gemma4");
    try appendTestMetadataStringArray(allocator, &data, "tokenizer.ggml.tokens", &.{
        "<pad>",
        "<eos>",
        "<bos>",
        "<unk>",
        "hello",
        "▁world",
        "<|turn>",
    });
    try appendTestMetadataStringArray(allocator, &data, "tokenizer.ggml.merges", &.{});
    try appendTestMetadataI32Array(allocator, &data, "tokenizer.ggml.token_type", &.{ 3, 3, 3, 2, 1, 1, 3 });
    try appendTestMetadataU32(allocator, &data, "tokenizer.ggml.bos_token_id", 2);
    try appendTestMetadataU32(allocator, &data, "tokenizer.ggml.eos_token_id", 1);
    try appendTestMetadataU32(allocator, &data, "tokenizer.ggml.padding_token_id", 0);
    try appendTestMetadataU32(allocator, &data, "tokenizer.ggml.unknown_token_id", 3);
    try appendTestMetadataBool(allocator, &data, "tokenizer.ggml.add_bos_token", true);

    return data.toOwnedSlice(allocator);
}

fn appendTestLe(comptime T: type, allocator: std.mem.Allocator, data: *std.ArrayListUnmanaged(u8), value: T) !void {
    const bytes = std.mem.asBytes(&std.mem.nativeToLittle(T, value));
    try data.appendSlice(allocator, bytes);
}

fn appendTestString(allocator: std.mem.Allocator, data: *std.ArrayListUnmanaged(u8), value: []const u8) !void {
    try appendTestLe(u64, allocator, data, value.len);
    try data.appendSlice(allocator, value);
}

fn appendTestMetadataString(allocator: std.mem.Allocator, data: *std.ArrayListUnmanaged(u8), key: []const u8, value: []const u8) !void {
    try appendTestString(allocator, data, key);
    try appendTestLe(u32, allocator, data, @intFromEnum(gguf_format.MetadataValueType.string));
    try appendTestString(allocator, data, value);
}

fn appendTestMetadataU32(allocator: std.mem.Allocator, data: *std.ArrayListUnmanaged(u8), key: []const u8, value: u32) !void {
    try appendTestString(allocator, data, key);
    try appendTestLe(u32, allocator, data, @intFromEnum(gguf_format.MetadataValueType.u32));
    try appendTestLe(u32, allocator, data, value);
}

fn appendTestMetadataBool(allocator: std.mem.Allocator, data: *std.ArrayListUnmanaged(u8), key: []const u8, value: bool) !void {
    try appendTestString(allocator, data, key);
    try appendTestLe(u32, allocator, data, @intFromEnum(gguf_format.MetadataValueType.bool_));
    try appendTestLe(u8, allocator, data, @intFromBool(value));
}

fn appendTestMetadataStringArray(allocator: std.mem.Allocator, data: *std.ArrayListUnmanaged(u8), key: []const u8, values: []const []const u8) !void {
    try appendTestString(allocator, data, key);
    try appendTestLe(u32, allocator, data, @intFromEnum(gguf_format.MetadataValueType.array));
    try appendTestLe(u32, allocator, data, @intFromEnum(gguf_format.MetadataValueType.string));
    try appendTestLe(u64, allocator, data, values.len);
    for (values) |value| try appendTestString(allocator, data, value);
}

fn appendTestMetadataI32Array(allocator: std.mem.Allocator, data: *std.ArrayListUnmanaged(u8), key: []const u8, values: []const i32) !void {
    try appendTestString(allocator, data, key);
    try appendTestLe(u32, allocator, data, @intFromEnum(gguf_format.MetadataValueType.array));
    try appendTestLe(u32, allocator, data, @intFromEnum(gguf_format.MetadataValueType.i32));
    try appendTestLe(u64, allocator, data, values.len);
    for (values) |value| try appendTestLe(i32, allocator, data, value);
}
