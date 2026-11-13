// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
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

//! Test-binary-only real HTTP ingress for portable Metal KV qualification.
//! The companion process drives the ordinary generated checkpoint endpoints;
//! this module adds no request field, environment-controlled production gate,
//! or legacy context-policy/profile authority.
const std = @import("std");

const ResidentCandidateV5 = struct {
    schema: []const u8,
    training_policy: []const u8,
    implementation_semantic_sha256: []const u8,
    profile_path: []const u8,
    profile_sha256: []const u8,
    profile_semantic_sha256: []const u8,
    resident_hybrid_configuration_path: []const u8,
    resident_hybrid_configuration_sha256: []const u8,
    resident_hybrid_shape_authority_sha256: []const u8,
    resident_hybrid_policy_module_sha256: []const u8,
};

fn loadResidentQualityAuthority(task_allocator: std.mem.Allocator, task_io: std.Io, path: []const u8, expected_candidate_sha256: [32]u8, profile_path: []const u8, profile: @import("runtime/kv/compaction_profile.zig").Profile, source_freeze_sha256: [32]u8) !@import("pipelines/generation.zig").ResidentHybridAuthority {
    try requireAbsolute(path);
    const candidate_bytes = try std.Io.Dir.cwd().readFileAlloc(task_io, path, task_allocator, .limited(128 * 1024));
    defer task_allocator.free(candidate_bytes);
    var actual_candidate_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(candidate_bytes, &actual_candidate_sha256, .{});
    if (!std.mem.eql(u8, &actual_candidate_sha256, &expected_candidate_sha256)) return error.InvalidMetalCheckpointQualificationInput;
    var candidate = std.json.parseFromSlice(ResidentCandidateV5, task_allocator, candidate_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch return error.InvalidMetalCheckpointQualificationInput;
    defer candidate.deinit();
    const c = candidate.value;
    const profile_sha = try @import("runtime/kv/compaction_profile.zig").profileArtifactDigest(profile);
    const semantic_sha = profile.semanticDigest();
    if (!std.mem.eql(u8, c.profile_path, profile_path) or
        !std.mem.eql(u8, &profile_sha, &try @import("context_native_checkpoint_quality.zig").parseDigest(c.profile_sha256)) or
        !std.mem.eql(u8, &semantic_sha, &try @import("context_native_checkpoint_quality.zig").parseDigest(c.profile_semantic_sha256)))
        return error.InvalidMetalCheckpointQualificationInput;
    if ((!std.mem.eql(u8, c.schema, "antfly.native-kv-operator-candidate.v5") and
        !std.mem.eql(u8, c.schema, "antfly.native-kv-operator-candidate.v6") and
        !std.mem.eql(u8, c.schema, "antfly.native-kv-operator-candidate.v7")) or
        !std.mem.eql(u8, c.training_policy, "private_train_challenge_hybrid_resident_v1") or
        !std.fs.path.isAbsolute(c.resident_hybrid_configuration_path))
        return error.InvalidMetalCheckpointQualificationInput;
    const configuration_bytes = try std.Io.Dir.cwd().readFileAlloc(task_io, c.resident_hybrid_configuration_path, task_allocator, .limited(128 * 1024));
    defer task_allocator.free(configuration_bytes);
    const authority = try @import("context_resident_hybrid_staged_configuration.zig").parse(task_allocator, configuration_bytes, .{
        .configuration_sha256 = try @import("context_native_checkpoint_quality.zig").parseDigest(c.resident_hybrid_configuration_sha256),
        .implementation_semantic_sha256 = try @import("context_native_checkpoint_quality.zig").parseDigest(c.implementation_semantic_sha256),
        .shape_authority_sha256 = try @import("context_native_checkpoint_quality.zig").parseDigest(c.resident_hybrid_shape_authority_sha256),
        .policy_module_sha256 = try @import("context_native_checkpoint_quality.zig").parseDigest(c.resident_hybrid_policy_module_sha256),
        .profile_semantic_sha256 = semantic_sha,
    });
    if (!std.mem.eql(u8, &authority.source_freeze_sha256, &source_freeze_sha256)) return error.InvalidMetalCheckpointQualificationInput;
    return .{ .configuration_sha256 = authority.configuration_sha256, .source_freeze_sha256 = authority.source_freeze_sha256, .shape_authority_sha256 = authority.shape_authority_sha256, .policy_module_sha256 = authority.policy_module_sha256, .e2_guarded_development_trial = authority.guarded_development_trial, .cohort = switch (authority.cohort) {
        .e4_8128_v1 => .e4_8128_v1,
        .e2_8128_v1 => .e2_8128_v1,
    } };
}
const builtin = @import("builtin");
const build_options = @import("build_options");
const httpx = @import("httpx");
const platform = @import("antfly_platform");
const backends = @import("backends/backends.zig");
const server_mod = @import("server/server.zig");
const quality_checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");

const allocator = std.testing.allocator;
const io = std.testing.io;
const default_model_name = "google/gemma-4-E4B-it-qat-q4_0-gguf";

fn validateQualificationModel(name: []const u8) ![]const u8 {
    for ([_][]const u8{
        "google/gemma-4-E2B-it-qat-q4_0-gguf",
        default_model_name,
    }) |allowed| if (std.mem.eql(u8, name, allowed)) return name;
    return error.UnsupportedMetalCheckpointQualificationModel;
}

fn qualificationModel() ![]const u8 {
    return validateQualificationModel(platform.env.getenv("ANTFLY_METAL_CHECKPOINT_MODEL") orelse default_model_name);
}

test "context Metal checkpoint qualification model selection is explicit" {
    try std.testing.expectEqualStrings(default_model_name, try validateQualificationModel(default_model_name));
    const e2b = "google/gemma-4-E2B-it-qat-q4_0-gguf";
    try std.testing.expectEqualStrings(e2b, try validateQualificationModel(e2b));
    try std.testing.expectError(error.UnsupportedMetalCheckpointQualificationModel, validateQualificationModel(""));
    try std.testing.expectError(error.UnsupportedMetalCheckpointQualificationModel, validateQualificationModel("google/gemma-4-26B-A4B-it-qat-q4_0-gguf"));
}
const default_max_lifetime_ns: u64 = 12 * std.time.ns_per_hour;
const maximum_lifetime_seconds: u64 = 48 * 60 * 60 + 5 * 60;
const metal_backends = [_]backends.BackendType{.metal};

/// Admit conservative KV, fitting and mapped-weight reservations together.
/// The collector separately enforces the unchanged 8 GiB process RSS cap;
/// these reservation envelopes do not raise it or the 512 MiB solver bound.
fn qualificationGenerationBudgetOverrides() server_mod.BudgetOverrides {
    return .{
        .host_limit_bytes = 8 * 1024 * 1024 * 1024,
        .backend_limit_bytes = 12 * 1024 * 1024 * 1024,
        .combined_limit_bytes = 16 * 1024 * 1024 * 1024,
        .kv_limit_bytes = 6 * 1024 * 1024 * 1024,
        .scratch_limit_bytes = 6 * 1024 * 1024 * 1024,
    };
}

test "context Metal qualification budgets admit both models at 8K and 32K" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    const memory = @import("runtime/tier/memory.zig");
    const test_generation = @import("pipelines/generation.zig");
    const GptConfig = @import("models/gpt.zig").Config;
    const Encoding = @import("runtime/kv/attention_match_checkpoint.zig").Encoding;
    const cases = [_]struct { config: GptConfig, weight_bytes: usize }{
        .{ .config = .{
            .family = .gemma,
            .gemma4_channel_protocol = true,
            .hidden_size = 1536,
            .num_hidden_layers = 35,
            .num_attention_heads = 8,
            .num_key_value_heads = 1,
            .attention_head_dim = 256,
            .global_head_dim = 512,
            .sliding_window = 512,
            .sliding_window_pattern = 5,
            .num_kv_shared_layers = 20,
            .ple_hidden_size = 256,
            .vocab_size = 262144,
            .max_position_embeddings = 131072,
        }, .weight_bytes = 3_349_516_256 },
        .{ .config = .{
            .family = .gemma,
            .gemma4_channel_protocol = true,
            .hidden_size = 2560,
            .num_hidden_layers = 42,
            .num_attention_heads = 8,
            .num_key_value_heads = 2,
            .attention_head_dim = 256,
            .global_head_dim = 512,
            .sliding_window = 512,
            .num_kv_shared_layers = 18,
            .ple_hidden_size = 256,
            .vocab_size = 262144,
            .max_position_embeddings = 131072,
        }, .weight_bytes = 5_154_941_280 },
    };
    for (cases) |case| {
        for ([_]usize{ 8192, 32768 }) |context_tokens| {
            const output_tokens: usize = if (context_tokens == 8192) 64 else 256;
            const prompt_tokens = context_tokens - output_tokens;
            for ([_]Encoding{ .f16, .fp8, .rotated4 }) |encoding| {
                const plan = try test_generation.NativeCheckpointPrepare.hostPlan(case.config, prompt_tokens, encoding, true);
                var budget = memory.RunBudget.init(qualificationGenerationBudgetOverrides().apply(.{}));
                try budget.reserveEstimate(.{
                    .prompt_tokens = prompt_tokens,
                    .retained_tokens = prompt_tokens,
                    .kv_bytes = 0,
                    .kv_tier = .backend,
                    .scratch_bytes = plan.reservation_bytes,
                    .scratch_tier = .host,
                });
                const components = [_]memory.GptGenerationBudgetComponent{.{
                    .backend = .metal,
                    .kv_dtype = .f16,
                    .config = case.config,
                }};
                const admitted = try memory.reserveGptGenerationPrefill(&budget, &components, prompt_tokens, output_tokens, 1024);
                // Model admission also accounts for mapped weights. Reserve the
                // larger observed host-weight footprint for either model.
                _ = try budget.tryReserveWeight(.host, 233_979_810);
                _ = try budget.tryReserveWeight(.backend, case.weight_bytes);
                try std.testing.expectEqual(@as(usize, 1024), admitted.max_chunk_rows);
                try std.testing.expect(plan.fit_bytes <= test_generation.NativeCheckpointPrepare.max_transient_host_bytes);
            }
        }
    }
}

test "compiled packed census derives real E2 and E4 GPT layouts" {
    const GptConfig = @import("models/gpt.zig").Config;
    const context_policy = @import("runtime/context_policy.zig");
    const census_mod = @import("context_compiled_packed_layout_census.zig");
    const configs = [_]GptConfig{
        .{ .family = .gemma, .gemma4_channel_protocol = true, .hidden_size = 1536, .num_hidden_layers = 35, .num_attention_heads = 8, .num_key_value_heads = 1, .attention_head_dim = 256, .global_head_dim = 512, .sliding_window = 512, .sliding_window_pattern = 5, .num_kv_shared_layers = 20, .ple_hidden_size = 256, .vocab_size = 262144, .max_position_embeddings = 131072 },
        .{ .family = .gemma, .gemma4_channel_protocol = true, .hidden_size = 2560, .num_hidden_layers = 42, .num_attention_heads = 8, .num_key_value_heads = 2, .attention_head_dim = 256, .global_head_dim = 512, .sliding_window = 512, .num_kv_shared_layers = 18, .ple_hidden_size = 256, .vocab_size = 262144, .max_position_embeddings = 131072 },
    };
    const expected = [_]u32{ 15, 24 };
    for (configs, expected) |config, want| {
        const layout = try context_policy.Layout.init(config);
        var projected: [42]census_mod.Layer = undefined;
        for (layout.layers[0..layout.len], 0..) |layer, i| projected[i] = .{ .layer = @intCast(i), .owner = layer.owner };
        const census = try census_mod.derive(projected[0..layout.len], try context_policy.layoutDigest(layout));
        try std.testing.expectEqual(@as(u32, @intCast(layout.len)), census.attention_layers);
        try std.testing.expectEqual(want, census.owner_count);
    }
}

fn requiredEnv(comptime name: [:0]const u8) ![]const u8 {
    const value = platform.env.getenv(name) orelse return error.MissingMetalCheckpointQualificationInput;
    if (value.len == 0) return error.InvalidMetalCheckpointQualificationInput;
    return value;
}

fn requireAbsolute(path: []const u8) !void {
    if (!std.fs.path.isAbsolute(path)) return error.InvalidMetalCheckpointQualificationInput;
}

fn installQualificationBinding(node: *server_mod.Node, path: []const u8, expected_hex: []const u8) !void {
    try requireAbsolute(path);
    const expected = @import("context_native_checkpoint_quality.zig").parseDigest(expected_hex) catch
        return error.InvalidMetalCheckpointQualificationInput;
    try @import("context_native_kv_operator_candidate.zig").installFromConfigFileWithDigest(allocator, io, node, path, expected);
}

fn qualificationLifetimeNs() !u64 {
    const raw = platform.env.getenv("ANTFLY_METAL_CHECKPOINT_MAX_LIFETIME_SECONDS") orelse
        return default_max_lifetime_ns;
    const seconds = std.fmt.parseInt(u64, raw, 10) catch
        return error.InvalidMetalCheckpointQualificationInput;
    if (seconds == 0 or seconds > maximum_lifetime_seconds)
        return error.InvalidMetalCheckpointQualificationInput;
    return std.math.mul(u64, seconds, std.time.ns_per_s) catch
        error.InvalidMetalCheckpointQualificationInput;
}

fn qualificationMaxConcurrentRequests() !usize {
    const raw = platform.env.getenv("ANTFLY_METAL_CHECKPOINT_MAX_CONCURRENT_REQUESTS") orelse
        return 1;
    const value = std.fmt.parseInt(usize, raw, 10) catch
        return error.InvalidMetalCheckpointQualificationInput;
    if (value == 0 or value > 64) return error.InvalidMetalCheckpointQualificationInput;
    return value;
}

fn requireAbsent(path: []const u8) !void {
    _ = std.Io.Dir.cwd().statFile(io, path, .{ .follow_symlinks = false }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    return error.MetalCheckpointQualificationControlExists;
}

fn stopRequested(path: []const u8) !bool {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{ .follow_symlinks = false }) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    if (stat.kind != .file) return error.InvalidMetalCheckpointQualificationStop;
    return true;
}

fn writeReady(
    path: []const u8,
    port: u16,
    load_generation: u64,
    model_name: []const u8,
    identity: @import("runtime/kv/attention_match_checkpoint.zig").Identity,
) !void {
    if (load_generation == 0) return error.InvalidMetalCheckpointQualificationIdentity;
    try requireAbsent(path);
    const pid = std.posix.system.getpid();
    const temporary = try std.fmt.allocPrint(allocator, "{s}.part-{d}", .{ path, pid });
    defer allocator.free(temporary);
    try requireAbsent(temporary);
    defer std.Io.Dir.cwd().deleteFile(io, temporary) catch {};
    const weights = std.fmt.bytesToHex(identity.weights, .lower);
    const tokenizer = std.fmt.bytesToHex(identity.tokenizer, .lower);
    const template = std.fmt.bytesToHex(identity.template, .lower);
    const layout = std.fmt.bytesToHex(identity.layout, .lower);
    const body = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.metal-checkpoint-test-server.v1",
        .backend = "metal",
        .metal_enabled = build_options.enable_metal,
        .cuda_enabled = build_options.enable_cuda,
        .pid = pid,
        .port = port,
        .model = model_name,
        .load_generation = load_generation,
        .loaded_identity = .{
            .weights = weights[0..],
            .tokenizer = tokenizer[0..],
            .template = template[0..],
            .layout = layout[0..],
        },
    }, .{});
    defer allocator.free(body);
    var file = try std.Io.Dir.cwd().createFile(io, temporary, .{ .exclusive = true, .truncate = false });
    var file_open = true;
    defer if (file_open) file.close(io);
    try file.writeStreamingAll(io, body);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
    file.close(io);
    file_open = false;
    // The rename is the publication point. Recheck so an existing caller file
    // is never intentionally replaced; the qualification directory is private
    // to its orchestrator and the temporary name is PID-scoped.
    try requireAbsent(path);
    try std.Io.Dir.rename(std.Io.Dir.cwd(), temporary, std.Io.Dir.cwd(), path, io);
}

fn performancePromptCount(model: *server_mod.model_manager.LoadedModel, text: []const u8) !usize {
    const template = model.chat_tmpl orelse return error.MissingChatTemplate;
    const rendered = try template.applyWithOptions(allocator, &.{.{ .role = "user", .content = text }}, .{ .enable_thinking = false });
    defer allocator.free(rendered);
    var encoded = try @import("pipelines/generation.zig").encodeNativeGenerationPrompt(
        model.getTokenizer(),
        allocator,
        rendered,
        131072,
        model.manifest.add_bos_token,
        model.manifest.bos_token,
    );
    defer encoded.deinit();
    var count: usize = 0;
    for (encoded.attention_mask) |mask| {
        if (mask == 0) break;
        count += 1;
    }
    return count;
}

fn sizedPerformancePrompt(model: *server_mod.model_manager.LoadedModel, target: usize) ![]u8 {
    const prefix = "This is a synthetic repeated-token capacity and performance payload. Reference text:";
    const suffix =
        "\nRespond only with the decimal integers from 1 through 10000 in ascending order, separated by single spaces. " ++
        "Continue the sequence without commentary and do not stop early.";
    var low: usize = 0;
    var high = target;
    while (low <= high) {
        const repeats = low + (high - low) / 2;
        const text = try allocator.alloc(u8, prefix.len + repeats * 2 + suffix.len);
        @memcpy(text[0..prefix.len], prefix);
        for (0..repeats) |i| @memcpy(text[prefix.len + i * 2 ..][0..2], " a");
        @memcpy(text[prefix.len + repeats * 2 ..], suffix);
        const count = performancePromptCount(model, text) catch |err| {
            allocator.free(text);
            return err;
        };
        if (count == target) return text;
        allocator.free(text);
        if (count < target) {
            low = repeats + 1;
        } else {
            if (repeats == 0) break;
            high = repeats - 1;
        }
    }
    return error.UnrepresentableMetalCheckpointPerformancePrompt;
}

fn writeExclusive(path: []const u8, data: []const u8) ![32]u8 {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, data);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(data);
    hash.update("\n");
    return hash.finalResult();
}

fn writeBinaryExclusive(path: []const u8, data: []const u8) ![32]u8 {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, data);
    try file.sync(io);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    return digest;
}

const CompiledSingleDecodeRawCapture = struct {
    path: []const u8,

    fn write(context: *anyopaque, logits: []const f32) !void {
        const self: *@This() = @ptrCast(@alignCast(context));
        _ = try writeBinaryExclusive(self.path, std.mem.sliceAsBytes(logits));
    }
};

fn hashFile(path: []const u8, maximum: u64) ![32]u8 {
    try requireAbsolute(path);
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file or stat.size == 0 or stat.size > maximum) return error.InvalidMetalCheckpointQualificationInput;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    while (offset < stat.size) {
        const count: usize = @intCast(@min(stat.size - offset, buffer.len));
        const read = try file.readPositionalAll(io, buffer[0..count], offset);
        if (read == 0) return error.InvalidMetalCheckpointQualificationInput;
        hash.update(buffer[0..read]);
        offset += read;
    }
    return hash.finalResult();
}

fn selfExecutableDigest() ![32]u8 {
    const path = try std.process.executablePathAlloc(io, allocator);
    defer allocator.free(path);
    return hashFile(path, std.math.maxInt(u64));
}

fn qualificationHttpConfig(base: httpx.ServerConfig) httpx.ServerConfig {
    var config = base;
    // The qualification driver deliberately sends one request at a time and
    // uses the stop file, not an idle pooled socket, as its lifetime control.
    // Closing each response connection makes that ownership explicit: no
    // client connection may outlive the final response and race teardown of
    // the test allocator, router, or Node. This is private harness transport
    // policy; the production Node configuration is left unchanged.
    config.keep_alive = false;
    config.max_requests_per_connection = 1;
    return config;
}

fn logModelAdmissionDiagnostic(node: *server_mod.Node, event: []const u8) u64 {
    const diagnostic = node.model_manager.memoryAdmissionDiagnostic();
    const current = node.model_manager.currentAdmissionAmounts();
    std.debug.print(
        "metal_checkpoint_admission_diagnostic: {{\"schema\":\"antfly.metal-checkpoint-admission-diagnostic.v1\",\"event\":\"{s}\",\"pressure\":\"{s}\",\"denied_count\":{d},\"last_denied_ns\":{d},\"request_count\":{d},\"requested_host_bytes\":{d},\"requested_backend_bytes\":{d},\"requested_kv_bytes\":{d},\"requested_scratch_bytes\":{d},\"admitted_host_bytes\":{d},\"admitted_backend_bytes\":{d},\"host_limit_bytes\":{d},\"backend_limit_bytes\":{d},\"combined_limit_bytes\":{d},\"available_memory_bytes\":{d},\"current\":{{\"host_weight_bytes\":{d},\"backend_weight_bytes\":{d},\"host_kv_bytes\":{d},\"backend_kv_bytes\":{d},\"host_scratch_bytes\":{d},\"backend_scratch_bytes\":{d},\"host_total_bytes\":{d},\"backend_total_bytes\":{d},\"kv_total_bytes\":{d},\"scratch_total_bytes\":{d}}}}}\n",
        .{
            event,
            diagnostic.pressure,
            diagnostic.denied_count,
            diagnostic.last_denied_ns,
            diagnostic.request_count,
            diagnostic.requested_host_bytes,
            diagnostic.requested_backend_bytes,
            diagnostic.requested_kv_bytes,
            diagnostic.requested_scratch_bytes,
            diagnostic.admitted_host_bytes,
            diagnostic.admitted_backend_bytes,
            diagnostic.host_limit_bytes,
            diagnostic.backend_limit_bytes,
            diagnostic.combined_limit_bytes,
            diagnostic.available_memory_bytes,
            current.host_weight_bytes,
            current.backend_weight_bytes,
            current.host_kv_bytes,
            current.backend_kv_bytes,
            current.host_scratch_bytes,
            current.backend_scratch_bytes,
            current.hostTotalBytes(),
            current.backendTotalBytes(),
            current.kvTotalBytes(),
            current.scratchTotalBytes(),
        },
    );
    return diagnostic.denied_count;
}

test "context Metal qualification HTTP transport closes each request" {
    const base: httpx.ServerConfig = .{
        .host = "127.0.0.1",
        .port = 17291,
        .max_connections = 4,
        .max_request_tasks = 2,
        .keep_alive = true,
        .max_requests_per_connection = 17,
    };
    const config = qualificationHttpConfig(base);
    try std.testing.expect(!config.keep_alive);
    try std.testing.expectEqual(@as(u32, 1), config.max_requests_per_connection);
    try std.testing.expectEqual(base.port, config.port);
    try std.testing.expectEqual(base.max_connections, config.max_connections);
    try std.testing.expectEqual(base.max_request_tasks, config.max_request_tasks);
    // The helper receives a value; the caller's production-derived policy is
    // not mutated while making the private listener single-request-owned.
    try std.testing.expect(base.keep_alive);
    try std.testing.expectEqual(@as(u32, 17), base.max_requests_per_connection);
}

test "context Metal native checkpoint qualification server" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    if (comptime build_options.enable_cuda) return error.InvalidMetalCheckpointQualificationBuild;

    const models_root = platform.env.getenv("ANTFLY_METAL_CHECKPOINT_MODELS_ROOT") orelse return error.SkipZigTest;
    if (models_root.len == 0) return error.InvalidMetalCheckpointQualificationInput;
    const model_name = try qualificationModel();
    const port_text = try requiredEnv("ANTFLY_METAL_CHECKPOINT_PORT");
    const ready_path = try requiredEnv("ANTFLY_METAL_CHECKPOINT_READY_FILE");
    const stop_path = try requiredEnv("ANTFLY_METAL_CHECKPOINT_STOP_FILE");
    const max_lifetime_ns = try qualificationLifetimeNs();
    try requireAbsolute(models_root);
    try requireAbsolute(ready_path);
    try requireAbsolute(stop_path);
    if (std.mem.eql(u8, ready_path, stop_path)) return error.InvalidMetalCheckpointQualificationInput;
    try requireAbsent(ready_path);
    try requireAbsent(stop_path);
    const parsed_port = try std.fmt.parseInt(u16, port_text, 10);
    if (parsed_port == 0) return error.InvalidMetalCheckpointQualificationInput;

    var node = try server_mod.Node.init(allocator, .{
        .models_dir = models_root,
        .max_loaded_models = 1,
        .max_concurrent_requests = try qualificationMaxConcurrentRequests(),
        .http_max_connections = 4,
        .http_max_request_tasks = 2,
        .http_max_body_bytes = 2 * 1024 * 1024,
        .keep_alive_ms = 12 * 60 * 60 * 1000,
        .process_termination_available = true,
        .generation_budget_overrides = qualificationGenerationBudgetOverrides(),
    });
    defer node.deinit();
    node.session_manager.preferred_backends = &metal_backends;
    node.session_manager.required_backend = .metal;
    node.model_manager.session_manager.preferred_backends = &metal_backends;
    node.model_manager.session_manager.required_backend = .metal;
    node.test_allow_metal_native_checkpoint = true;
    try node.attachIo(io);

    const qualification_binding = platform.env.getenv("ANTFLY_METAL_CHECKPOINT_QUALIFICATION_BINDING");
    const qualification_binding_sha256 = platform.env.getenv("ANTFLY_METAL_CHECKPOINT_QUALIFICATION_BINDING_SHA256");
    if (qualification_binding != null or qualification_binding_sha256 != null) {
        try installQualificationBinding(&node, qualification_binding orelse return error.InvalidMetalCheckpointQualificationInput, qualification_binding_sha256 orelse return error.InvalidMetalCheckpointQualificationInput);
    }

    const model_dir = try node.resolveModelPath(io, model_name, "generators");
    defer allocator.free(model_dir);
    var model = try node.model_manager.acquireFromDirWithPreferredBackends(model_dir, &metal_backends, true);
    defer model.release();
    const loaded = model.get();
    if (loaded.session.backend() != .metal) return error.MetalCheckpointQualificationBackendMismatch;
    const identity = try loaded.nativeKvIdentity(.f16, null);
    if (loaded.load_generation == 0) return error.InvalidMetalCheckpointQualificationIdentity;

    var server_io = std.Io.Threaded.init(allocator, .{ .concurrent_limit = .limited(8) });
    defer server_io.deinit();
    var http_server = httpx.Server.initWithConfig(
        allocator,
        server_io.io(),
        qualificationHttpConfig(node.httpServerConfig("127.0.0.1", parsed_port)),
    );
    defer http_server.deinit();
    try node.registerHttpRoutes(&http_server);
    var listener = httpx.Server.ListenerTask.init(&http_server);
    defer {
        listener.requestStop();
        listener.join() catch {};
    }
    try http_server.bind();
    const address = http_server.boundAddress() orelse return error.MissingMetalCheckpointQualificationAddress;
    if (address != .ip4 or !std.mem.eql(u8, &address.ip4.bytes, &.{ 127, 0, 0, 1 }) or address.ip4.port != parsed_port)
        return error.MetalCheckpointQualificationAddressMismatch;
    try listener.start();
    const startup_deadline = platform.time.monotonicNs() + 30 * std.time.ns_per_s;
    while (!http_server.listen_started.load(.acquire)) {
        if (listener.runtimeFailure()) |err| return err;
        if (platform.time.monotonicNs() >= startup_deadline) return error.MetalCheckpointQualificationStartTimeout;
        try io.sleep(.fromMilliseconds(1), .awake);
    }
    try writeReady(ready_path, parsed_port, loaded.load_generation, model_name, identity);

    const lifetime_deadline = platform.time.monotonicNs() + max_lifetime_ns;
    var observed_denied_count = logModelAdmissionDiagnostic(&node, "startup");
    while (true) {
        const denied_count = node.model_manager.memoryAdmissionDiagnostic().denied_count;
        if (denied_count != observed_denied_count)
            observed_denied_count = logModelAdmissionDiagnostic(&node, "denied_count_changed");
        if (try stopRequested(stop_path)) break;
        if (listener.runtimeFailure()) |err| return err;
        if (platform.time.monotonicNs() >= lifetime_deadline) return error.MetalCheckpointQualificationLifetimeExceeded;
        try io.sleep(.fromMilliseconds(25), .awake);
    }
    listener.shutdown(25);
    try listener.join();
    try std.testing.expectEqual(httpx.Server.ListenerTask.RuntimeState.stopped, listener.runtimeState());
    const stats = http_server.runtimeStats();
    try std.testing.expectEqual(@as(usize, 0), stats.active_connections);
    try std.testing.expectEqual(@as(usize, 0), stats.active_requests);
    try std.testing.expectEqual(@as(usize, 0), stats.body_buffer_in_use_bytes);
    const runtime_stats = http_server.httpRuntimeStats();
    try std.testing.expectEqual(@as(usize, 0), runtime_stats.active_listener_leases);
    try std.testing.expectEqual(@as(usize, 0), runtime_stats.reserved_connection_capacity);
    try std.testing.expectEqual(@as(usize, 0), runtime_stats.reserved_request_capacity);
    try std.testing.expectEqual(@as(usize, 0), runtime_stats.reserved_h1_request_capacity);
    try std.testing.expectEqual(@as(usize, 0), runtime_stats.active_h1_cancellation_observers);
}

test "context Metal checkpoint successive installed owner model qualification" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    if (comptime build_options.enable_cuda) return error.InvalidMetalCheckpointQualificationBuild;
    const input_path = platform.env.getenv("ANTFLY_METAL_CHECKPOINT_CYCLE_INPUT") orelse return error.SkipZigTest;
    var phase: []const u8 = "input";
    errdefer std.log.err("metal_checkpoint_cycle_driver_rejected phase={s}", .{phase});
    const output_path = try requiredEnv("ANTFLY_METAL_CHECKPOINT_CYCLE_OUTPUT");
    const models_root = try requiredEnv("ANTFLY_METAL_CHECKPOINT_MODELS_ROOT");
    const model_name = try qualificationModel();
    try requireAbsolute(input_path);
    try requireAbsolute(output_path);
    try requireAbsolute(models_root);
    try requireAbsent(output_path);
    const generation = @import("pipelines/generation.zig");
    const cycles = @import("runtime/kv/metal_checkpoint_cycles.zig");
    const Input = struct {
        schema: []const u8,
        model: []const u8,
        expected_prompt_tokens: usize,
        messages: []const generation.Message,
        request: cycles.Request,
    };
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, input_path, allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(raw);
    var parsed = try std.json.parseFromSlice(Input, allocator, raw, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const input = parsed.value;
    const legacy_cycles = std.mem.eql(u8, input.schema, "antfly.metal-checkpoint-cycles.input.v1");
    const candidate_cycles = std.mem.eql(u8, input.schema, "antfly.metal-checkpoint-cycles.input.v2");
    const legacy_policy = @import("runtime/kv/attention_match_fit_policy.zig").Policy.legacy_fixed;
    const candidate_policy = @import("runtime/kv/attention_match_fit_policy.zig").Policy.cumulative256_exact_cycle_v1;
    if ((!legacy_cycles and !candidate_cycles) or
        (legacy_cycles and !std.meta.eql(input.request.fit_policy, legacy_policy)) or
        (candidate_cycles and !std.meta.eql(input.request.fit_policy, candidate_policy)) or
        !std.mem.eql(u8, input.model, model_name) or input.messages.len == 0)
        return error.InvalidMetalCheckpointQualificationInput;
    _ = try input.request.capacity(input.expected_prompt_tokens);
    phase = "model_load";
    var node = try server_mod.Node.init(allocator, .{
        .models_dir = models_root,
        .max_loaded_models = 1,
        .max_concurrent_requests = 1,
        .process_termination_available = true,
        .generation_budget_overrides = .{
            .host_limit_bytes = 8 * 1024 * 1024 * 1024,
            .backend_limit_bytes = 8 * 1024 * 1024 * 1024,
            .combined_limit_bytes = 10 * 1024 * 1024 * 1024,
            .kv_limit_bytes = 4 * 1024 * 1024 * 1024,
            .scratch_limit_bytes = 6 * 1024 * 1024 * 1024,
        },
    });
    defer node.deinit();
    node.session_manager.preferred_backends = &metal_backends;
    node.session_manager.required_backend = .metal;
    node.model_manager.session_manager.preferred_backends = &metal_backends;
    node.model_manager.session_manager.required_backend = .metal;
    try node.attachIo(io);
    const model_dir = try node.resolveModelPath(io, model_name, "generators");
    defer allocator.free(model_dir);
    var handle = try node.model_manager.acquireFromDirWithPreferredBackends(model_dir, &metal_backends, true);
    defer handle.release();
    const model = handle.get();
    if (model.session.backend() != .metal) return error.MetalCheckpointQualificationBackendMismatch;
    phase = "identity";
    const identity = try model.nativeKvIdentity(.f16, null);
    phase = "tokenization";
    const template = model.chat_tmpl orelse return error.MissingChatTemplate;
    const prompt = try template.applyWithOptions(allocator, input.messages, .{ .enable_thinking = false });
    defer allocator.free(prompt);
    var encoded = try generation.encodeNativeGenerationPrompt(model.getTokenizer(), allocator, prompt, @import("runtime/kv/attention_match_checkpoint.zig").max_context_tokens, model.manifest.add_bos_token, model.manifest.bos_token);
    defer encoded.deinit();
    var count: usize = 0;
    while (count < encoded.attention_mask.len and encoded.attention_mask[count] != 0) : (count += 1) {}
    if (count != input.expected_prompt_tokens) return error.InvalidMetalCheckpointQualificationInput;
    const tokens = try allocator.alloc(i64, count);
    defer allocator.free(tokens);
    for (tokens, encoded.ids[0..count]) |*target, token| target.* = token;
    const factory = @import("architectures/session_factory.zig");
    var budget = @import("runtime/tier/memory.zig").RunBudget.init(.{
        .host_limit_bytes = 8 * 1024 * 1024 * 1024,
        .backend_limit_bytes = 8 * 1024 * 1024 * 1024,
        .combined_limit_bytes = 10 * 1024 * 1024 * 1024,
        .kv_limit_bytes = 4 * 1024 * 1024 * 1024,
        .scratch_limit_bytes = 6 * 1024 * 1024 * 1024,
    });
    phase = "backend";
    var cb = try factory.getComputeBackendWithBudget(model.session, allocator, &budget);
    defer cb.deinit();
    var pipeline: generation.NativeGenerationPipeline = .{
        .allocator = allocator,
        .io = io,
        .cb = cb,
        .session = model.session,
        .load_generation = model.load_generation,
        .gpt_config = factory.getGptConfig(model.session) orelse return error.InvalidModelForGeneration,
        .kv_dtype = .f16,
        .attention_match_identity = identity,
        .tokenizer = model.getTokenizer(),
        .chat_template = model.chat_tmpl,
        .add_bos_token = model.manifest.add_bos_token,
        .bos_token = model.manifest.bos_token,
        .execution_lock = model.nativeGenerationMutex(),
    };
    defer pipeline.deinit();
    phase = "native_cycles";
    const report = try pipeline.qualifyMetalCheckpointCycles(tokens, input.request);
    var input_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &input_digest, .{});
    var prompt_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(prompt, &prompt_digest, .{});
    const input_hex = std.fmt.bytesToHex(input_digest, .lower);
    const prompt_hex = std.fmt.bytesToHex(prompt_digest, .lower);
    const token_hex = std.fmt.bytesToHex(try @import("backends/context_teacher_forced.zig").promptTokenDigest(tokens), .lower);
    const implementation = std.fmt.bytesToHex(@import("compaction_identity").digest, .lower);
    const body = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.metal-checkpoint-cycles.report.v3",
        .model_directory = model.model_dir,
        .model = model_name,
        .encoding = @tagName(input.request.encoding),
        .load_generation = model.load_generation,
        .model_identity = identity,
        .input_sha256 = input_hex[0..],
        .rendered_prompt_sha256 = prompt_hex[0..],
        .prompt_tokens_sha256 = token_hex[0..],
        .implementation_sha256 = implementation[0..],
        .report = report,
    }, .{ .whitespace = .indent_2 });
    defer allocator.free(body);
    phase = "report_write";
    var output = try std.Io.Dir.cwd().createFile(io, output_path, .{ .exclusive = true, .truncate = false });
    defer output.close(io);
    try output.writeStreamingAll(io, body);
    try output.writeStreamingAll(io, "\n");
    try output.sync(io);
}

test "context Metal checkpoint performance requests materializer" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    if (comptime build_options.enable_cuda) return error.InvalidMetalCheckpointQualificationBuild;
    const models_root = platform.env.getenv("ANTFLY_METAL_CHECKPOINT_MODELS_ROOT") orelse return error.SkipZigTest;
    const output_dir = platform.env.getenv("ANTFLY_METAL_CHECKPOINT_PERF_OUTPUT_DIR") orelse return error.SkipZigTest;
    const model_name = try qualificationModel();
    if (models_root.len == 0 or output_dir.len == 0) return error.InvalidMetalCheckpointQualificationInput;
    try requireAbsolute(models_root);
    try requireAbsolute(output_dir);
    try requireAbsent(output_dir);
    try std.Io.Dir.createDirAbsolute(io, output_dir, std.Io.File.Permissions.fromMode(0o700));

    var node = try server_mod.Node.init(allocator, .{
        .models_dir = models_root,
        .max_loaded_models = 1,
        .max_concurrent_requests = try qualificationMaxConcurrentRequests(),
        .process_termination_available = true,
        .generation_budget_overrides = .{
            .host_limit_bytes = 8 * 1024 * 1024 * 1024,
            .backend_limit_bytes = 8 * 1024 * 1024 * 1024,
            .combined_limit_bytes = 10 * 1024 * 1024 * 1024,
            .kv_limit_bytes = 4 * 1024 * 1024 * 1024,
            .scratch_limit_bytes = 6 * 1024 * 1024 * 1024,
        },
    });
    defer node.deinit();
    node.session_manager.preferred_backends = &metal_backends;
    node.session_manager.required_backend = .metal;
    node.model_manager.session_manager.preferred_backends = &metal_backends;
    node.model_manager.session_manager.required_backend = .metal;
    try node.attachIo(io);
    const model_dir = try node.resolveModelPath(io, model_name, "generators");
    defer allocator.free(model_dir);
    var handle = try node.model_manager.acquireFromDirWithPreferredBackends(model_dir, &metal_backends, true);
    defer handle.release();
    const model = handle.get();
    if (model.session.backend() != .metal) return error.MetalCheckpointQualificationBackendMismatch;
    const identity = try model.nativeKvIdentity(.f16, null);
    if (model.load_generation == 0) return error.InvalidMetalCheckpointQualificationIdentity;

    const Spec = struct { name: []const u8, prompt_tokens: usize, max_tokens: usize };
    const specs = [_]Spec{
        .{ .name = "request-8k.json", .prompt_tokens = 8128, .max_tokens = 64 },
        .{ .name = "request-32k.json", .prompt_tokens = 32512, .max_tokens = 256 },
    };
    var request_sha256: [specs.len][32]u8 = undefined;
    for (specs, 0..) |spec, index| {
        const prompt = try sizedPerformancePrompt(model, spec.prompt_tokens);
        defer allocator.free(prompt);
        if (try performancePromptCount(model, prompt) != spec.prompt_tokens)
            return error.MetalCheckpointPerformancePromptLengthMismatch;
        const body = try std.json.Stringify.valueAlloc(allocator, .{
            .model = model_name,
            .messages = &.{.{ .role = "user", .content = prompt }},
            .mode = "eager",
            .cache_dtype = "f16",
            .max_tokens = spec.max_tokens,
            .temperature = 0,
            .prompt_cache = false,
            .stream = true,
            .stream_options = .{ .include_usage = true },
            .chat_template_kwargs = .{ .enable_thinking = false },
        }, .{});
        defer allocator.free(body);
        const path = try std.fs.path.join(allocator, &.{ output_dir, spec.name });
        defer allocator.free(path);
        request_sha256[index] = try writeExclusive(path, body);
    }
    const weights = std.fmt.bytesToHex(identity.weights, .lower);
    const tokenizer = std.fmt.bytesToHex(identity.tokenizer, .lower);
    const template = std.fmt.bytesToHex(identity.template, .lower);
    const layout = std.fmt.bytesToHex(identity.layout, .lower);
    const request_8k = std.fmt.bytesToHex(request_sha256[0], .lower);
    const request_32k = std.fmt.bytesToHex(request_sha256[1], .lower);
    const metadata = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.metal-checkpoint-performance-requests.v1",
        .backend = "metal",
        .model = model_name,
        .load_generation = model.load_generation,
        .synthetic_payload = true,
        .qualification_label = "synthetic_repeated_token_capacity_and_performance_only",
        .loaded_identity = .{
            .weights = weights[0..],
            .tokenizer = tokenizer[0..],
            .template = template[0..],
            .layout = layout[0..],
        },
        .requests = .{
            .request_8k = .{ .file = specs[0].name, .prompt_tokens = specs[0].prompt_tokens, .max_tokens = specs[0].max_tokens, .sha256 = request_8k[0..] },
            .request_32k = .{ .file = specs[1].name, .prompt_tokens = specs[1].prompt_tokens, .max_tokens = specs[1].max_tokens, .sha256 = request_32k[0..] },
        },
    }, .{});
    defer allocator.free(metadata);
    const metadata_path = try std.fs.path.join(allocator, &.{ output_dir, "metadata.json" });
    defer allocator.free(metadata_path);
    _ = try writeExclusive(metadata_path, metadata);
}
fn validateAcceptedSnapshotCapture(enabled: bool, case_count: usize, supplied: bool, encoding: quality_checkpoint.Encoding) !void {
    if (enabled and (case_count != 1 or supplied or encoding != .f16)) return error.InvalidMetalCheckpointQualificationInput;
}

fn runMetalNativeCheckpointQuality(supplied_plan_path: ?[]const u8, prepared_plan_path: ?[]const u8, focused_future_probe: bool) !void {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    if (comptime build_options.enable_cuda) return error.InvalidMetalCheckpointQualificationBuild;
    const input_path = platform.env.getenv("ANTFLY_METAL_NATIVE_QUALITY_INPUT") orelse return error.SkipZigTest;
    const input_root = try requiredEnv("ANTFLY_METAL_NATIVE_QUALITY_INPUT_ROOT");
    const output_root = try requiredEnv("ANTFLY_METAL_NATIVE_QUALITY_OUTPUT_ROOT");
    const models_root = try requiredEnv("ANTFLY_METAL_CHECKPOINT_MODELS_ROOT");
    const model_name = try qualificationModel();
    inline for (.{ input_path, input_root, output_root, models_root }) |path| try requireAbsolute(path);
    const driver = @import("context_native_checkpoint_quality.zig");
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, input_path, allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(raw);
    var parsed = try std.json.parseFromSlice(driver.Input, allocator, raw, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const input = parsed.value;
    if (!std.mem.eql(u8, input.schema, "antfly.native-checkpoint-quality.input.v1") or
        !std.mem.eql(u8, input.model, model_name) or input.cases.len == 0 or input.cases.len > 100)
        return error.InvalidMetalCheckpointQualificationInput;
    if (focused_future_probe and
        (input.cases.len != 1 or input.cases[0].prompt_count != 7936 or
            input.cases[0].reference_count != 20 or input.cases[0].request.output_tokens != 256 or
            input.cases[0].request.encoding != .f16 or supplied_plan_path != null or prepared_plan_path != null))
        return error.InvalidFocusedFutureProbeInput;
    const capture_accepted_snapshot = capture: {
        const raw_capture = platform.env.getenv("ANTFLY_METAL_NATIVE_QUALITY_CAPTURE_ACCEPTED_SNAPSHOT") orelse break :capture false;
        if (!std.mem.eql(u8, raw_capture, "1")) return error.InvalidMetalCheckpointQualificationInput;
        break :capture true;
    };
    if (supplied_plan_path != null and prepared_plan_path != null) return error.InvalidMetalCheckpointQualificationInput;
    try validateAcceptedSnapshotCapture(capture_accepted_snapshot, input.cases.len, supplied_plan_path != null or prepared_plan_path != null, input.cases[0].request.encoding);
    const expected_main = try driver.parseDigest(input.main_gguf_sha256);
    var inputs = try std.Io.Dir.cwd().openDir(io, input_root, .{ .follow_symlinks = false });
    defer inputs.close(io);
    var outputs = try std.Io.Dir.cwd().openDir(io, output_root, .{ .follow_symlinks = false });
    defer outputs.close(io);
    var node = try server_mod.Node.init(allocator, .{
        .models_dir = models_root,
        .max_loaded_models = 1,
        .max_concurrent_requests = 1,
        .process_termination_available = true,
        .generation_budget_overrides = .{
            .host_limit_bytes = 8 * 1024 * 1024 * 1024,
            .backend_limit_bytes = 8 * 1024 * 1024 * 1024,
            .combined_limit_bytes = 10 * 1024 * 1024 * 1024,
            .kv_limit_bytes = 4 * 1024 * 1024 * 1024,
            .scratch_limit_bytes = 6 * 1024 * 1024 * 1024,
        },
    });
    defer node.deinit();
    node.session_manager.preferred_backends = &metal_backends;
    node.session_manager.required_backend = .metal;
    node.model_manager.session_manager.preferred_backends = &metal_backends;
    node.model_manager.session_manager.required_backend = .metal;
    try node.attachIo(io);
    const private_profile_path = platform.env.getenv("ANTFLY_METAL_QUALITY_PRIVATE_PROFILE");
    const worker_receipt_path = platform.env.getenv("ANTFLY_METAL_QUALITY_WORKER_RECEIPT");
    const full_cli_receipt_path = platform.env.getenv("ANTFLY_METAL_QUALITY_FULL_CLI_RECEIPT");
    const staged_private = private_profile_path != null;
    var staged_future_training = false;
    var staged_resident_hybrid = false;
    var resident_hybrid_authority: ?@import("pipelines/generation.zig").ResidentHybridAuthority = null;
    const resident_candidate_path = platform.env.getenv("ANTFLY_METAL_QUALITY_RESIDENT_CANDIDATE_V5");
    const resident_candidate_sha256 = platform.env.getenv("ANTFLY_METAL_QUALITY_RESIDENT_CANDIDATE_V5_SHA256");
    if ((resident_candidate_path == null) != (resident_candidate_sha256 == null)) return error.InvalidMetalCheckpointQualificationInput;
    if ((private_profile_path != null) != (worker_receipt_path != null) or
        (!focused_future_probe and (private_profile_path != null) != (full_cli_receipt_path != null)) or
        (focused_future_probe and full_cli_receipt_path != null) or
        (resident_candidate_path != null and (private_profile_path == null or focused_future_probe)))
        return error.InvalidMetalCheckpointQualificationInput;
    var prepared_producer: ?@import("server/native_kv_verified_sidecar.zig").ExpectedProducer = null;
    var prepared_execution: ?@import("runtime/kv/attention_match_release.zig").Execution = null;
    if (private_profile_path) |profile_path| {
        try requireAbsolute(profile_path);
        try requireAbsolute(worker_receipt_path.?);
        if (full_cli_receipt_path) |path| try requireAbsolute(path);
        const loader = @import("context_serving_entrypoint.zig");
        const resident_profile = @import("context_resident_hybrid_compatibility_profile.zig");
        var resident_candidate: ?resident_profile.Loaded = if (resident_candidate_path != null) load: {
            const profile_bytes = try std.Io.Dir.cwd().readFileAlloc(io, profile_path, allocator, .limited(128 * 1024));
            defer allocator.free(profile_bytes);
            break :load try resident_profile.parseBytes(allocator, profile_bytes);
        } else null;
        defer if (resident_candidate) |*loaded| loaded.deinit();
        var candidate: ?loader.Candidate = if (resident_candidate != null) null else try loader.loadCandidate(allocator, io, profile_path);
        defer if (candidate) |*loaded| loaded.deinit(allocator);
        if (resident_candidate == null and !candidate.?.isPrivateDense()) return error.InvalidMetalCheckpointQualificationInput;
        const private_profile = if (resident_candidate) |loaded| loaded.profile else candidate.?.profile().*;
        const worker_receipt = try std.Io.Dir.cwd().readFileAlloc(io, worker_receipt_path.?, allocator, .limited(4 * 1024 * 1024));
        defer allocator.free(worker_receipt);
        const training_policy: @import("pipelines/generation.zig").NativeCheckpointTrainingPolicy = if (resident_candidate_path != null)
            .private_train_challenge_hybrid_resident_v1
        else switch (candidate.?) {
            .private_dense => |loaded| if (std.mem.eql(u8, loaded.parsed.value.training_policy, "private_future_decode_mix_v1"))
                .private_future_decode_mix_v1
            else
                .private_dense512_reference_v1,
            else => unreachable,
        };
        staged_future_training = training_policy == .private_future_decode_mix_v1;
        staged_resident_hybrid = training_policy == .private_train_challenge_hybrid_resident_v1;
        if (staged_resident_hybrid) {
            resident_hybrid_authority = try loadResidentQualityAuthority(
                allocator,
                io,
                resident_candidate_path orelse return error.InvalidMetalCheckpointQualificationInput,
                try @import("context_native_checkpoint_quality.zig").parseDigest(resident_candidate_sha256 orelse return error.InvalidMetalCheckpointQualificationInput),
                profile_path,
                private_profile,
                resident_candidate.?.source_freeze_sha256,
            );
        } else if (resident_candidate_path != null) return error.InvalidMetalCheckpointQualificationInput;
        const staged = @import("context_quality_worker_staged_candidate.zig");
        const evidence = if (focused_future_probe)
            try staged.installFocusedFutureDiagnostic(allocator, io, &node, private_profile, worker_receipt)
        else install: {
            const full_cli_receipt = try std.Io.Dir.cwd().readFileAlloc(io, full_cli_receipt_path.?, allocator, .limited(4 * 1024 * 1024));
            defer allocator.free(full_cli_receipt);
            break :install try staged.installWithTraining(allocator, io, &node, private_profile, worker_receipt, full_cli_receipt, training_policy, resident_hybrid_authority);
        };
        if (evidence.cross_binary_source_join_verified == focused_future_probe) return error.InvalidQualityWorkerAuthority;
        const private_policy = @import("runtime/kv/attention_match_fit_policy.zig").Policy.cumulative256_exact_cycle_v1;
        prepared_producer = .{
            .profile_sha256 = evidence.profile_sha256,
            .profile_semantic_sha256 = evidence.profile_semantic_sha256,
            // AFATT002 producer source_sha256 is the whole compaction
            // implementation authority, not the native profile runtime hash.
            .implementation_semantic_sha256 = evidence.authority_implementation_sha256,
            .binary_sha256 = evidence.full_cli_binary_sha256,
            .policy_sha256 = private_policy.digest(),
            .policy_name = "cumulative256_exact_cycle_v1",
        };
        prepared_execution = evidence.execution;
        std.debug.print("native_quality_staged_private worker={x} full_cli={x} profile={x} semantic={x} profile_implementation={x} authority_implementation={x} cross_binary_source_join_verified={}\n", .{
            evidence.worker_binary_sha256,              evidence.full_cli_binary_sha256,         evidence.profile_sha256,
            evidence.profile_semantic_sha256,           evidence.implementation_semantic_sha256, evidence.authority_implementation_sha256,
            evidence.cross_binary_source_join_verified,
        });
    }
    // This diagnostic mode intentionally enters the same staged-candidate and
    // quality caller as the formal worker.  It only narrows the retained case;
    // it does not provide qualification credit or a public serving surface.
    if (focused_future_probe and (!staged_private or !staged_future_training))
        return error.InvalidFocusedFutureProbeAuthority;
    const model_dir = try node.resolveModelPath(io, model_name, "generators");
    defer allocator.free(model_dir);
    var handle = try node.model_manager.acquireFromDirWithPreferredBackends(model_dir, &metal_backends, true);
    defer handle.release();
    const model = handle.get();
    if (model.session.backend() != .metal) return error.MetalCheckpointQualificationBackendMismatch;
    const native = try model.nativeKvIdentity(.f16, null);
    const factory = @import("architectures/session_factory.zig");
    const actual_main = factory.getLoadedGgufArtifactDigest(model.session) orelse return error.UnsupportedCompactionArtifact;
    if (!native.eql(input.native_identity) or !std.mem.eql(u8, &expected_main, &actual_main))
        return error.ContextQualificationLoadedIdentityMismatch;
    const generation = @import("pipelines/generation.zig");
    var budget = @import("runtime/tier/memory.zig").RunBudget.init(.{
        .host_limit_bytes = 8 * 1024 * 1024 * 1024,
        .backend_limit_bytes = 8 * 1024 * 1024 * 1024,
        .combined_limit_bytes = 10 * 1024 * 1024 * 1024,
        .kv_limit_bytes = 4 * 1024 * 1024 * 1024,
        .scratch_limit_bytes = 6 * 1024 * 1024 * 1024,
    });
    var cb = try factory.getComputeBackendWithBudget(model.session, allocator, &budget);
    defer cb.deinit();
    var pipeline: generation.NativeGenerationPipeline = .{
        .allocator = allocator,
        .io = io,
        .cb = cb,
        .session = model.session,
        .load_generation = model.load_generation,
        .gpt_config = factory.getGptConfig(model.session) orelse return error.InvalidModelForGeneration,
        .kv_dtype = .f16,
        .attention_match_identity = native,
        .tokenizer = model.getTokenizer(),
        .chat_template = model.chat_tmpl,
        .add_bos_token = model.manifest.add_bos_token,
        .bos_token = model.manifest.bos_token,
        .execution_lock = model.nativeGenerationMutex(),
    };
    pipeline.prepared_quality_execution = prepared_execution;
    defer pipeline.deinit();
    var manifest_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &manifest_digest, .{});
    if (focused_future_probe) {
        const retained_case_sha256 = try driver.parseDigest("46fe0e6474d497e29674778ebe3c8a0ff70cf0702837ffb2cfda5da22ca647ed");
        if (!std.mem.eql(u8, &manifest_digest, &retained_case_sha256))
            return error.InvalidFocusedFutureProbeInput;
    }
    var session: driver.Session = .{
        .allocator = allocator,
        .io = io,
        .inputs = inputs,
        .outputs = outputs,
        .model = model,
        .pipeline = &pipeline,
        .manifest_sha256 = manifest_digest,
        .capture_accepted_snapshot = capture_accepted_snapshot,
        .staged_private_policy = staged_private,
        .staged_future_training_policy = staged_future_training,
        .staged_resident_hybrid_policy = staged_resident_hybrid,
        .resident_hybrid_authority = resident_hybrid_authority,
    };
    const supplied_mod = @import("context_supplied_checkpoint_quality.zig");
    var supplied_plan_raw: ?[]u8 = null;
    defer if (supplied_plan_raw) |bytes| allocator.free(bytes);
    var supplied_plan: ?std.json.Parsed(supplied_mod.Input) = null;
    defer if (supplied_plan) |*parsed_supplied| parsed_supplied.deinit();
    var supplied_root: ?std.Io.Dir = null;
    defer if (supplied_root) |*dir| dir.close(io);
    if (supplied_plan_path) |plan_path| {
        try requireAbsolute(plan_path);
        const root_path = try requiredEnv("ANTFLY_METAL_SUPPLIED_CHECKPOINT_QUALITY_ROOT");
        try requireAbsolute(root_path);
        supplied_plan_raw = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(4 * 1024 * 1024));
        supplied_plan = try std.json.parseFromSlice(supplied_mod.Input, allocator, supplied_plan_raw.?, .{ .allocate = .alloc_always });
        if (input.cases.len != 1 or !std.mem.eql(u8, supplied_plan.?.value.model, input.model) or
            !std.mem.eql(u8, supplied_plan.?.value.case_id, input.cases[0].id) or
            supplied_plan.?.value.prompt_count != input.cases[0].prompt_count or
            !std.mem.eql(u8, &try supplied_mod.parseDigest(supplied_plan.?.value.case_manifest_sha256), &manifest_digest))
            return error.InvalidSuppliedCheckpointQualityInput;
        supplied_root = try std.Io.Dir.cwd().openDir(io, root_path, .{ .follow_symlinks = false });
        session.supplied_input = supplied_plan.?.value;
        session.supplied_root = supplied_root.?;
    }
    const prepared_mod = @import("context_prepared_serving_quality.zig");
    var prepared_plan_raw: ?[]u8 = null;
    defer if (prepared_plan_raw) |bytes| allocator.free(bytes);
    var prepared_plan: ?std.json.Parsed(prepared_mod.Input) = null;
    defer if (prepared_plan) |*parsed_prepared| parsed_prepared.deinit();
    var prepared_root: ?std.Io.Dir = null;
    defer if (prepared_root) |*dir| dir.close(io);
    if (prepared_plan_path) |plan_path| {
        if (!staged_private or prepared_producer == null) return error.InvalidPreparedServingQualityInput;
        try requireAbsolute(plan_path);
        const root_path = try requiredEnv("ANTFLY_METAL_PREPARED_SERVING_QUALITY_ROOT");
        try requireAbsolute(root_path);
        prepared_plan_raw = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(4 * 1024 * 1024));
        prepared_plan = try std.json.parseFromSlice(prepared_mod.Input, allocator, prepared_plan_raw.?, .{ .allocate = .alloc_always });
        if (input.cases.len != 1 or !std.mem.eql(u8, prepared_plan.?.value.model, input.model) or
            !std.mem.eql(u8, prepared_plan.?.value.case_id, input.cases[0].id) or
            prepared_plan.?.value.prompt_count != input.cases[0].prompt_count or
            !std.mem.eql(u8, &try driver.parseDigest(prepared_plan.?.value.case_manifest_sha256), &manifest_digest))
            return error.InvalidPreparedServingQualityInput;
        prepared_root = try std.Io.Dir.cwd().openDir(io, root_path, .{ .follow_symlinks = false });
        session.prepared_input = prepared_plan.?.value;
        session.prepared_root = prepared_root.?;
        session.prepared_producer = prepared_producer.?;
    }
    const route_parity_diagnostic = platform.env.getenvBool("ANTFLY_METAL_QUALITY_ROUTE_PARITY_DIAGNOSTIC");
    if (route_parity_diagnostic and prepared_plan_path == null)
        return error.InvalidMetalCheckpointQualificationInput;
    session.route_parity_diagnostic = route_parity_diagnostic;
    for (input.cases, 0..) |case, index| {
        for (input.cases[0..index]) |previous|
            if (std.mem.eql(u8, previous.id, case.id) and previous.request.encoding == case.request.encoding)
                return error.DuplicateNativeQualityCase;
        session.execute(case, index) catch |err| {
            const failure = try std.json.Stringify.valueAlloc(allocator, .{
                .schema = "antfly.native-checkpoint-quality.failure.v1",
                .release_eligible = false,
                .case_index = index,
                .case_id = case.id,
                .encoding = @tagName(case.request.encoding),
                .@"error" = @errorName(err),
                .manifest_sha256 = manifest_digest,
            }, .{ .whitespace = .indent_2 });
            defer allocator.free(failure);
            var file = try outputs.createFile(io, "failure.json", .{ .exclusive = true, .truncate = false });
            defer file.close(io);
            try file.writeStreamingAll(io, failure);
            try file.sync(io);
            return err;
        };
        if (!native.eql(try model.nativeKvIdentity(.f16, null)) or model.load_generation != pipeline.load_generation)
            return error.StaleContextView;
    }
}

test "native accepted snapshot capture is one fitted case only" {
    try validateAcceptedSnapshotCapture(false, 100, true, .f16);
    try validateAcceptedSnapshotCapture(false, 1, false, .fp8);
    try validateAcceptedSnapshotCapture(false, 1, false, .rotated4);
    try validateAcceptedSnapshotCapture(true, 1, false, .f16);
    try std.testing.expectError(error.InvalidMetalCheckpointQualificationInput, validateAcceptedSnapshotCapture(true, 2, false, .f16));
    try std.testing.expectError(error.InvalidMetalCheckpointQualificationInput, validateAcceptedSnapshotCapture(true, 1, true, .f16));
    try std.testing.expectError(error.InvalidMetalCheckpointQualificationInput, validateAcceptedSnapshotCapture(true, 1, false, .fp8));
    try std.testing.expectError(error.InvalidMetalCheckpointQualificationInput, validateAcceptedSnapshotCapture(true, 1, false, .rotated4));
}

test "context Metal native checkpoint paired quality qualification" {
    try runMetalNativeCheckpointQuality(null, null, false);
}

test "context Metal focused retained E4 future probe diagnostic" {
    if (!platform.env.getenvBool("ANTFLY_METAL_FOCUSED_FUTURE_PROBE_DIAGNOSTIC")) return error.SkipZigTest;
    try runMetalNativeCheckpointQuality(null, null, true);
}

test "context Metal supplied checkpoint paired quality qualification" {
    const plan = platform.env.getenv("ANTFLY_METAL_SUPPLIED_CHECKPOINT_QUALITY_INPUT") orelse return error.SkipZigTest;
    try runMetalNativeCheckpointQuality(plan, null, false);
}

test "context Metal prepared serving checkpoint paired quality qualification" {
    const plan = platform.env.getenv("ANTFLY_METAL_PREPARED_SERVING_QUALITY_INPUT") orelse return error.SkipZigTest;
    try runMetalNativeCheckpointQuality(null, plan, false);
}

test "context Metal native quality tokenizer materializer" {
    if (!builtin.is_test or !build_options.enable_metal or
        !platform.env.getenvBool("ANTFLY_METAL_QUALITY_MATERIALIZER")) return error.SkipZigTest;
    const model_name = try qualificationModel();
    const models_root = try requiredEnv("ANTFLY_METAL_CHECKPOINT_MODELS_DIR");
    const campaign_root = try requiredEnv("ANTFLY_METAL_QUALITY_CAMPAIGN_DIR");
    try requireAbsolute(models_root);
    try requireAbsolute(campaign_root);
    const session = try requiredEnv("ANTFLY_METAL_QUALITY_SESSION_ID");
    if (session.len != 64) return error.InvalidMetalCheckpointQualificationInput;
    var session_bytes: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&session_bytes, session);
    const maximum = try std.fmt.parseInt(u32, try requiredEnv("ANTFLY_METAL_QUALITY_PREPARATION_CONTEXT"), 10);
    if (maximum != 16384 and maximum != 65536) return error.InvalidMetalCheckpointQualificationInput;
    var campaign = try std.Io.Dir.cwd().openDir(io, campaign_root, .{ .follow_symlinks = false });
    defer campaign.close(io);
    var node = try server_mod.Node.init(allocator, .{
        .models_dir = models_root,
        .max_loaded_models = 1,
        .max_concurrent_requests = 1,
        .process_termination_available = true,
        .generation_budget_overrides = .{
            .host_limit_bytes = 8 * 1024 * 1024 * 1024,
            .backend_limit_bytes = 8 * 1024 * 1024 * 1024,
            .combined_limit_bytes = 10 * 1024 * 1024 * 1024,
            .kv_limit_bytes = 4 * 1024 * 1024 * 1024,
            .scratch_limit_bytes = 6 * 1024 * 1024 * 1024,
        },
    });
    defer node.deinit();
    node.session_manager.preferred_backends = &metal_backends;
    node.session_manager.required_backend = .metal;
    node.model_manager.session_manager.preferred_backends = &metal_backends;
    node.model_manager.session_manager.required_backend = .metal;
    try node.attachIo(io);
    const model_dir = try node.resolveModelPath(io, model_name, "generators");
    defer allocator.free(model_dir);
    var handle = try node.model_manager.acquireFromDirWithPreferredBackends(model_dir, &metal_backends, true);
    defer handle.release();
    const model = handle.get();
    if (model.session.backend() != .metal) return error.MetalCheckpointQualificationBackendMismatch;
    const native = try model.nativeKvIdentity(.f16, null);
    // Hash the actual loaded main GGUF through its retained session, rather
    // than treating a caller's expected file digest as an observation.
    const factory = @import("architectures/session_factory.zig");
    const artifact = factory.getLoadedGgufArtifactDigest(model.session) orelse return error.UnsupportedCompactionArtifact;
    const identity: server_mod.ContextServingLoadedIdentity = .{
        .load_generation = model.load_generation,
        .artifact_sha256 = artifact,
        .tokenizer_sha256 = native.tokenizer,
        .template_sha256 = native.template,
        .layout_sha256 = native.layout,
        .profile_sha256 = @splat(0),
        .implementation_sha256 = @import("compaction_identity").digest,
    };
    const main_hex = std.fmt.bytesToHex(artifact, .lower);
    const implementation_hex = std.fmt.bytesToHex(@import("compaction_identity").digest, .lower);
    const weights_hex = std.fmt.bytesToHex(native.weights, .lower);
    const tokenizer_hex = std.fmt.bytesToHex(native.tokenizer, .lower);
    const template_hex = std.fmt.bytesToHex(native.template, .lower);
    const layout_hex = std.fmt.bytesToHex(native.layout, .lower);
    const native_head_policy = platform.env.getenvBool("ANTFLY_METAL_QUALITY_NATIVE_HEAD_POLICY");
    const fitting = @import("pipelines/generation.zig").NativeCheckpointPrepare.fittingOptions();
    const protected_prefix: u32 = if (native_head_policy) @intCast(fitting.prefix_rows) else 512;
    const protected_recent: u32 = if (native_head_policy) @intCast(fitting.recent_rows) else 512;
    const ready = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.native-quality-materializer.ready.v1",
        .preparation_only = true,
        .model_outputs_observed = false,
        .model_forward_steps = @as(u32, 0),
        .model = model_name,
        .model_directory = model.model_dir,
        .load_generation = model.load_generation,
        .native_identity = .{ .weights = weights_hex[0..], .tokenizer = tokenizer_hex[0..], .template = template_hex[0..], .layout = layout_hex[0..] },
        .main_gguf_sha256 = main_hex[0..],
        .implementation_sha256 = implementation_hex[0..],
        .native_head_policy = native_head_policy,
        .protected_prefix_rows = protected_prefix,
        .protected_recent_rows = protected_recent,
    }, .{});
    defer allocator.free(ready);
    var ready_file = try campaign.createFile(io, "native-materializer-ready.json", .{ .exclusive = true, .truncate = false });
    defer ready_file.close(io);
    try ready_file.writeStreamingAll(io, ready);
    try ready_file.writeStreamingAll(io, "\n");
    try ready_file.sync(io);
    var materializer: @import("context_native_quality_materializer.zig").Session = .{
        .allocator = allocator,
        .io = io,
        .campaign_dir = campaign,
        .model = model_name,
        .loaded_model = model,
        .loaded_identity = identity,
        .expected_session_id = session[0..64].*,
        .max_context_tokens = maximum,
        .max_kv_tokens = maximum / 2,
        .keep_prefix_tokens = protected_prefix,
        .keep_recent_tokens = protected_recent,
    };
    try materializer.run();
    if (model.load_generation != identity.load_generation or
        !std.meta.eql(native, try model.nativeKvIdentity(.f16, null)))
        return error.StaleContextView;
}

test "context Metal checkpoint persistent installed owner soak qualification" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    if (comptime build_options.enable_cuda) return error.InvalidMetalCheckpointQualificationBuild;
    const input_path = platform.env.getenv("ANTFLY_METAL_CHECKPOINT_SOAK_INPUT") orelse return error.SkipZigTest;
    const output_path = try requiredEnv("ANTFLY_METAL_CHECKPOINT_SOAK_OUTPUT");
    const journal_path = try requiredEnv("ANTFLY_METAL_CHECKPOINT_SOAK_JOURNAL");
    const models_root = try requiredEnv("ANTFLY_METAL_CHECKPOINT_MODELS_ROOT");
    const model_name = try qualificationModel();
    inline for (.{ input_path, output_path, journal_path, models_root }) |path| try requireAbsolute(path);
    try requireAbsent(output_path);
    try requireAbsent(journal_path);
    const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
    const soak = @import("runtime/kv/metal_checkpoint_soak.zig");
    const driver = @import("context_native_checkpoint_quality.zig");
    const CaptureInput = struct {
        source_freeze: driver.File,
        compatibility_profile: driver.File,
        development_policy: driver.File,
        heldout_bundle: ?driver.File = null,
        heldout_authority_sha256: ?[]const u8 = null,
        future_hho_equivalence_output_path: ?[]const u8 = null,
        future_train_p64_output_dir: ?[]const u8 = null,
        future_train_phase_p64_output_dir: ?[]const u8 = null,
        future_train_p64_continuation_origin_sha256: ?[]const u8 = null,
        binary_sha256: []const u8,
    };
    const Input = struct {
        schema: []const u8,
        model: []const u8,
        main_gguf_sha256: []const u8,
        native_identity: checkpoint.Identity,
        checkpoints: [2]driver.File,
        checkpoint_producer_metadata: ?[2]driver.File = null,
        recent_prefill_capture: ?CaptureInput = null,
        compiled_observation_source_freeze: ?driver.File = null,
        request: soak.Request,
    };
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, input_path, allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(raw);
    var parsed = try std.json.parseFromSlice(Input, allocator, raw, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const input = parsed.value;
    if (!std.mem.eql(u8, input.schema, "antfly.native-checkpoint-soak.input.v2") or
        !std.mem.eql(u8, input.model, model_name)) return error.InvalidNativeSoakRequest;
    try input.request.validate();
    const candidate_policy = !std.meta.eql(input.request.producer_policy, @import("runtime/kv/attention_match_fit_policy.zig").Policy.legacy_fixed);
    if (candidate_policy != (input.checkpoint_producer_metadata != null)) return error.InvalidNativeSoakRequest;
    const expected_main = try driver.parseDigest(input.main_gguf_sha256);
    var node = try server_mod.Node.init(allocator, .{
        .models_dir = models_root,
        .max_loaded_models = 1,
        .max_concurrent_requests = 1,
        .process_termination_available = true,
        .generation_budget_overrides = .{
            .host_limit_bytes = 8 * 1024 * 1024 * 1024,
            .backend_limit_bytes = 8 * 1024 * 1024 * 1024,
            .combined_limit_bytes = 10 * 1024 * 1024 * 1024,
            .kv_limit_bytes = 4 * 1024 * 1024 * 1024,
            .scratch_limit_bytes = 6 * 1024 * 1024 * 1024,
        },
    });
    defer node.deinit();
    node.session_manager.preferred_backends = &metal_backends;
    node.session_manager.required_backend = .metal;
    node.model_manager.session_manager.preferred_backends = &metal_backends;
    node.model_manager.session_manager.required_backend = .metal;
    try node.attachIo(io);
    const model_dir = try node.resolveModelPath(io, model_name, "generators");
    defer allocator.free(model_dir);
    var handle = try node.model_manager.acquireFromDirWithPreferredBackends(model_dir, &metal_backends, true);
    defer handle.release();
    const model = handle.get();
    if (model.session.backend() != .metal) return error.MetalCheckpointQualificationBackendMismatch;
    const native = try model.nativeKvIdentity(.f16, null);
    const factory = @import("architectures/session_factory.zig");
    const actual_main = factory.getLoadedGgufArtifactDigest(model.session) orelse return error.UnsupportedCompactionArtifact;
    if (!native.eql(input.native_identity) or !std.mem.eql(u8, &expected_main, &actual_main))
        return error.ContextQualificationLoadedIdentityMismatch;
    const generation = @import("pipelines/generation.zig");
    var budget = @import("runtime/tier/memory.zig").RunBudget.init(.{
        .host_limit_bytes = 8 * 1024 * 1024 * 1024,
        .backend_limit_bytes = 8 * 1024 * 1024 * 1024,
        .combined_limit_bytes = 10 * 1024 * 1024 * 1024,
        .kv_limit_bytes = 4 * 1024 * 1024 * 1024,
        .scratch_limit_bytes = 6 * 1024 * 1024 * 1024,
    });
    var cb = try factory.getComputeBackendWithBudget(model.session, allocator, &budget);
    defer cb.deinit();
    var pipeline: generation.NativeGenerationPipeline = .{
        .allocator = allocator,
        .io = io,
        .cb = cb,
        .session = model.session,
        .load_generation = model.load_generation,
        .gpt_config = factory.getGptConfig(model.session) orelse return error.InvalidModelForGeneration,
        .kv_dtype = .f16,
        .attention_match_identity = native,
        .tokenizer = model.getTokenizer(),
        .chat_template = model.chat_tmpl,
        .add_bos_token = model.manifest.add_bos_token,
        .bos_token = model.manifest.bos_token,
        .execution_lock = model.nativeGenerationMutex(),
    };
    defer pipeline.deinit();
    var saved: [2]checkpoint.Owned = undefined;
    var saved_policy_sha256: [2][32]u8 = undefined;
    var saved_count: usize = 0;
    defer for (saved[0..saved_count]) |*value| value.deinit();
    for (input.checkpoints, &saved, &saved_policy_sha256, 0..) |file_ref, *value, *saved_policy, index| {
        try requireAbsolute(file_ref.path);
        var producer_checkpoint_sha256: ?[32]u8 = null;
        if (input.checkpoint_producer_metadata) |producer_refs| {
            const producer_ref = producer_refs[index];
            try requireAbsolute(producer_ref.path);
            const producer_raw = try std.Io.Dir.cwd().readFileAlloc(io, producer_ref.path, allocator, .limited(16 * 1024));
            defer allocator.free(producer_raw);
            var producer_digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(producer_raw, &producer_digest, .{});
            if (!std.mem.eql(u8, &producer_digest, &try driver.parseDigest(producer_ref.sha256)))
                return error.NativeSoakCheckpointChanged;
            const ProducerBinding = struct {
                schema: []const u8,
                afatt002_sha256_hex: []const u8,
                checkpoint_sha256_hex: []const u8,
                policy_sha256_hex: []const u8,
            };
            var producer = try std.json.parseFromSlice(ProducerBinding, allocator, producer_raw, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
            defer producer.deinit();
            if (!std.mem.eql(u8, producer.value.schema, "antfly.native-kv-prepared-sidecar.v1"))
                return error.InvalidNativeSoakRequest;
            _ = try driver.parseDigest(producer.value.afatt002_sha256_hex);
            saved_policy.* = try driver.parseDigest(producer.value.policy_sha256_hex);
            producer_checkpoint_sha256 = try driver.parseDigest(producer.value.checkpoint_sha256_hex);
        } else saved_policy.* = @import("runtime/kv/attention_match_fit_policy.zig").Policy.legacy_fixed.digest();
        var file = try std.Io.Dir.cwd().openFile(io, file_ref.path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
        defer file.close(io);
        const stat = try file.stat(io);
        if (stat.kind != .file or stat.size == 0 or stat.size > checkpoint.serving_limits.max_file_bytes)
            return error.InvalidNativeSoakRequest;
        const wire = try allocator.alloc(u8, @intCast(stat.size));
        defer allocator.free(wire);
        if (try file.readPositionalAll(io, wire, 0) != wire.len) return error.InvalidNativeSoakRequest;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(wire, &digest, .{});
        if (!std.mem.eql(u8, &digest, &try driver.parseDigest(file_ref.sha256)) or
            (producer_checkpoint_sha256 != null and !std.mem.eql(u8, &digest, &producer_checkpoint_sha256.?)))
            return error.NativeSoakCheckpointChanged;
        value.* = try checkpoint.decodeWithControl(allocator, wire, native, checkpoint.serving_limits, .{});
        saved_count += 1;
    }
    if (std.c.getenv("ANTFLY_METAL_COMPILED_SINGLE_DECODE_DIAGNOSTIC")) |raw_mode_c| {
        const raw_mode = std.mem.span(raw_mode_c);
        const prefix_tokens: usize = if (std.mem.eql(u8, raw_mode, "8127-full") or std.mem.eql(u8, raw_mode, "8127-greedy")) 8127 else if (std.mem.eql(u8, raw_mode, "8128-full") or std.mem.eql(u8, raw_mode, "8128-greedy")) 8128 else return error.InvalidNativeSoakRequest;
        const mode: @import("pipelines/generation.zig").NativeGenerationPipeline.CompiledSingleDecodeMode = if (std.mem.eql(u8, raw_mode, "8127-full") or std.mem.eql(u8, raw_mode, "8128-full")) .full_host_logits else .terminal_greedy;
        const legacy_full_logits_route = if (std.c.getenv("ANTFLY_TEST_LEGACY_FULL_LOGITS_DECODE")) |legacy_raw|
            std.mem.eql(u8, std.mem.span(legacy_raw), "1")
        else
            false;
        if (legacy_full_logits_route and mode != .full_host_logits) return error.InvalidNativeSoakRequest;
        var raw_capture: ?CompiledSingleDecodeRawCapture = null;
        if (std.c.getenv("ANTFLY_METAL_COMPILED_SINGLE_DECODE_RAW_F32_PATH")) |raw_path_c| {
            const raw_path = std.mem.span(raw_path_c);
            if (mode != .full_host_logits or !std.fs.path.isAbsolute(raw_path)) return error.InvalidNativeSoakRequest;
            raw_capture = .{ .path = raw_path };
        }
        const capture: ?@import("pipelines/generation.zig").NativeGenerationPipeline.CompiledSingleDecodeFullLogitsCapture = if (raw_capture) |*value| .{
            .context = value,
            .write_fn = CompiledSingleDecodeRawCapture.write,
        } else null;
        const exact_source_sha256 = try driver.parseDigest(input.checkpoints[0].sha256);
        const required_source_sha256 = try driver.parseDigest("a275fd233cadeac3a070b522babed949e0dbff91e56d968b6eeafdb0a02f6c37");
        if (!std.mem.eql(u8, &exact_source_sha256, &required_source_sha256) or saved[0].state.logical_length != 8128) return error.NativeSoakCheckpointChanged;
        if (pipeline.gpt_config.num_hidden_layers != 42 or pipeline.gpt_config.num_attention_heads != 8 or pipeline.gpt_config.global_head_dim != 512) return error.InvalidModelForGeneration;
        if (!@import("graph/metal_executor.zig").supportsSession(model.session)) return error.MissingCompiledModelRuntime;
        pipeline.graph_cache = &model.native_generation_graph_cache;
        pipeline.compiled_partition_backend = .metal;
        pipeline.compiled_attachment_target = .whole_model;
        const memory = @import("runtime/tier/memory.zig");
        const components = [_]memory.GptGenerationBudgetComponent{.{ .backend = .metal, .kv_dtype = .f16, .config = pipeline.gpt_config }};
        const admission = try memory.reserveGptGenerationPrefill(&budget, &components, prefix_tokens, 1, 512);
        if (admission.max_chunk_rows != 512 or admission.max_scratch_bytes > 3840 * 1024 * 1024) return error.CompactionWorkspaceLimit;
        var prefill_plan = try @import("runtime/scheduler/native_generate.zig").PrefillChunkPlan.init(allocator, prefix_tokens, admission.max_chunk_rows, admission.max_scratch_bytes, .sole_idle, .fixed_ceiling);
        defer prefill_plan.deinit(allocator);
        const result = try pipeline.diagnoseCompiledSingleDecode(saved[0].state, .{ .max_tokens = 1, .prefill_chunk_size = admission.max_chunk_rows, .prefill_chunk_plan = prefill_plan }, prefix_tokens, mode, capture, legacy_full_logits_route);
        if (!native.eql(try model.nativeKvIdentity(.f16, null)) or model.load_generation != pipeline.load_generation) return error.StaleContextView;
        const body = try std.json.Stringify.valueAlloc(allocator, result, .{ .whitespace = .indent_2 });
        defer allocator.free(body);
        _ = try writeExclusive(output_path, body);
        return;
    }
    if (std.c.getenv("ANTFLY_METAL_COMPILED_AFKV_EXACT_KV_DIAGNOSTIC")) |raw_mode| {
        if (!std.mem.eql(u8, std.mem.span(raw_mode), "1")) return error.InvalidNativeSoakRequest;
        const exact_source_sha256 = try driver.parseDigest(input.checkpoints[0].sha256);
        const required_source_sha256 = try driver.parseDigest("a275fd233cadeac3a070b522babed949e0dbff91e56d968b6eeafdb0a02f6c37");
        const current_cohort = if (std.c.getenv("ANTFLY_METAL_COMPILED_AFKV_CURRENT_COHORT")) |cohort_mode_raw| std.mem.eql(u8, std.mem.span(cohort_mode_raw), "1") else false;
        const exact_source_rows: usize = input.request.exact_source_rows orelse 8128;
        if ((!current_cohort and !std.mem.eql(u8, &exact_source_sha256, &required_source_sha256)) or saved[0].state.logical_length != exact_source_rows) return error.NativeSoakCheckpointChanged;
        if ((!current_cohort and pipeline.gpt_config.num_hidden_layers != 42) or
            (current_cohort and pipeline.gpt_config.num_hidden_layers != 35) or
            pipeline.gpt_config.num_attention_heads != 8 or pipeline.gpt_config.global_head_dim != 512) return error.InvalidModelForGeneration;
        if (!@import("graph/metal_executor.zig").supportsSession(model.session)) return error.MissingCompiledModelRuntime;
        pipeline.graph_cache = &model.native_generation_graph_cache;
        pipeline.compiled_partition_backend = .metal;
        pipeline.compiled_attachment_target = .whole_model;
        const memory = @import("runtime/tier/memory.zig");
        const components = [_]memory.GptGenerationBudgetComponent{.{ .backend = .metal, .kv_dtype = .f16, .config = pipeline.gpt_config }};
        const admission = try memory.reserveGptGenerationPrefill(&budget, &components, exact_source_rows, 1, 512);
        if (admission.max_chunk_rows != 512 or admission.max_scratch_bytes > 3840 * 1024 * 1024) return error.CompactionWorkspaceLimit;
        var prefill_plan = try @import("runtime/scheduler/native_generate.zig").PrefillChunkPlan.init(allocator, exact_source_rows, admission.max_chunk_rows, admission.max_scratch_bytes, .sole_idle, .fixed_ceiling);
        defer prefill_plan.deinit(allocator);
        const context_policy = @import("runtime/context_policy.zig");
        const maximum_context: usize = if (exact_source_rows <= 8128) 8192 else 32768;
        const packed_plan = try (context_policy.Policy{}).admit(maximum_context, null, exact_source_rows, 1, 0);
        const packed_reservation = try budget.tryReserveKv(.backend, 2 * 1024 * 1024 * 1024);
        defer budget.release(packed_reservation);
        const result = try pipeline.diagnoseCompiledAfkvExactKv(saved[0].state, .{ .max_tokens = 1, .prefill_chunk_size = admission.max_chunk_rows, .prefill_chunk_plan = prefill_plan }, .{ .plan = packed_plan, .layout = try context_policy.Layout.init(pipeline.gpt_config), .max_query_rows = 1, .physical_budget_bytes = 2 * 1024 * 1024 * 1024, .optimize_decode = true, .optimize_prefill = true });
        if (!native.eql(try model.nativeKvIdentity(.f16, null)) or model.load_generation != pipeline.load_generation) return error.StaleContextView;
        const freeze = input.compiled_observation_source_freeze orelse return error.InvalidNativeSoakRequest;
        const source_freeze_sha256 = try hashFile(freeze.path, 16 * 1024 * 1024);
        if (!std.mem.eql(u8, &source_freeze_sha256, &try driver.parseDigest(freeze.sha256))) return error.InvalidNativeSoakRequest;
        const observed_host_staging = result.peak_host_staging_bytes orelse return error.InvalidCompiledExactImportReceipt;
        const observation = @import("context_compiled_packed_observation.zig");
        const census_mod = @import("context_compiled_packed_layout_census.zig");
        const active_layers = result.layout_witness.layer_count;
        if (active_layers == 0 or active_layers > result.layout_witness.layers.len) return error.InvalidContextLayout;
        var census_layers: [42]census_mod.Layer = undefined;
        for (result.layout_witness.layers[0..active_layers], 0..) |layer, i| census_layers[i] = .{ .layer = @intCast(i), .owner = layer.owner };
        const census = try census_mod.derive(census_layers[0..active_layers], result.layout_witness.context_layout_sha256);
        const document: observation.Document = .{
            .schema = observation.schema_name,
            .diagnostic_only = true,
            .qualification_credit = false,
            .serving_activation = false,
            .compiled_prefill_completed = true,
            .exact_export_completed = result.source_unchanged and result.imported_owners == census.owner_count and result.comparison.compared_owners == census.owner_count and result.comparison.total_key_mismatches == 0 and result.comparison.total_value_mismatches == 0 and result.comparison.all_f16_identical and result.copied_bytes > 0 and result.release.released and (!result.release.physical_available or result.release.post_release_live_bytes == 0),
            .exact_export_abi = observation.export_abi_name,
            .exact_export_peak_host_staging_bytes = observed_host_staging,
            .source_checkpoint_sha256 = exact_source_sha256,
            .source_freeze_sha256 = source_freeze_sha256,
            .test_binary_sha256 = try selfExecutableDigest(),
            .implementation_semantic_sha256 = @import("compaction_identity").digest,
            .model_identity = native,
            .process_rss_limit_bytes = 8 * 1024 * 1024 * 1024,
            .layout_census = .{ .attention_layers = census.attention_layers, .owner_count = census.owner_count, .context_layout_sha256 = census.layout_sha256, .owner_projection_sha256 = census.owner_projection_sha256 },
            .result = .{
                .source_authority = .{
                    .model_weights_sha256 = result.source_authority.model_weights_sha256,
                    .model_identity_layout_sha256 = result.source_authority.model_identity_layout_sha256,
                    .context_layout_sha256 = result.source_authority.context_layout_sha256,
                    .token_stream_sha256 = result.source_authority.token_stream_sha256,
                    .source_revision = result.source_authority.source_revision,
                    .source_sequence = result.source_authority.source_sequence,
                    .logical_tokens = result.source_authority.logical_tokens,
                },
                .copied_bytes = result.copied_bytes,
                .imported_owners = result.imported_owners,
                .source_unchanged = result.source_unchanged,
                .release_complete = result.release.released and (!result.release.physical_available or result.release.post_release_live_bytes == 0),
                .compared_owners = result.comparison.compared_owners,
                .total_key_mismatches = result.comparison.total_key_mismatches,
                .total_value_mismatches = result.comparison.total_value_mismatches,
                .all_f16_identical = result.comparison.all_f16_identical,
            },
        };
        const body = try std.json.Stringify.valueAlloc(allocator, document, .{ .whitespace = .indent_2 });
        defer allocator.free(body);
        _ = try writeExclusive(output_path, body);
        return;
    }
    const packed_serving_greedy64_raw = std.c.getenv("ANTFLY_METAL_COMPILED_PACKED_SERVING_GREEDY64_DIAGNOSTIC");
    const packed_strict_greedy64_raw = std.c.getenv("ANTFLY_METAL_COMPILED_PACKED_GREEDY64_DIAGNOSTIC");
    if (packed_serving_greedy64_raw != null and packed_strict_greedy64_raw != null) return error.InvalidNativeSoakRequest;
    if (packed_serving_greedy64_raw orelse packed_strict_greedy64_raw) |raw_mode| {
        const serving_route = packed_serving_greedy64_raw != null;
        if (!std.mem.eql(u8, std.mem.span(raw_mode), "1")) return error.InvalidNativeSoakRequest;
        const exact_source_sha256 = try driver.parseDigest(input.checkpoints[0].sha256);
        const required_source_sha256 = try driver.parseDigest("a275fd233cadeac3a070b522babed949e0dbff91e56d968b6eeafdb0a02f6c37");
        if (!std.mem.eql(u8, &exact_source_sha256, &required_source_sha256) or saved[0].state.logical_length != 8128) return error.NativeSoakCheckpointChanged;
        if (pipeline.gpt_config.num_hidden_layers != 42 or pipeline.gpt_config.num_attention_heads != 8 or pipeline.gpt_config.global_head_dim != 512) return error.InvalidModelForGeneration;
        if (!@import("graph/metal_executor.zig").supportsSession(model.session)) return error.MissingCompiledModelRuntime;
        pipeline.graph_cache = &model.native_generation_graph_cache;
        pipeline.compiled_partition_backend = .metal;
        pipeline.compiled_attachment_target = .whole_model;
        const memory = @import("runtime/tier/memory.zig");
        const components = [_]memory.GptGenerationBudgetComponent{.{ .backend = .metal, .kv_dtype = .f16, .config = pipeline.gpt_config }};
        const admission = try memory.reserveGptGenerationPrefill(&budget, &components, 8128, 64, 512);
        if (admission.max_chunk_rows != 512 or admission.max_scratch_bytes > 3840 * 1024 * 1024) return error.CompactionWorkspaceLimit;
        var prefill_plan = try @import("runtime/scheduler/native_generate.zig").PrefillChunkPlan.init(allocator, 8128, admission.max_chunk_rows, admission.max_scratch_bytes, .sole_idle, .fixed_ceiling);
        defer prefill_plan.deinit(allocator);
        const context_policy = @import("runtime/context_policy.zig");
        const packed_plan = try (context_policy.Policy{}).admit(8192, null, 8128, 64, 0);
        const packed_reservation = try budget.tryReserveKv(.backend, 2 * 1024 * 1024 * 1024);
        defer budget.release(packed_reservation);
        const result = if (serving_route)
            try pipeline.diagnoseCompiledPackedServingGreedy64(saved[0].state, .{ .max_tokens = 64, .prefill_chunk_size = admission.max_chunk_rows, .prefill_chunk_plan = prefill_plan }, .{ .plan = packed_plan, .layout = try context_policy.Layout.init(pipeline.gpt_config), .max_query_rows = 512, .physical_budget_bytes = 2 * 1024 * 1024 * 1024, .optimize_decode = true, .optimize_prefill = true })
        else
            try pipeline.diagnoseCompiledPackedGreedy64(saved[0].state, .{ .max_tokens = 64, .prefill_chunk_size = admission.max_chunk_rows, .prefill_chunk_plan = prefill_plan }, .{ .plan = packed_plan, .layout = try context_policy.Layout.init(pipeline.gpt_config), .max_query_rows = 512, .physical_budget_bytes = 2 * 1024 * 1024 * 1024, .optimize_decode = true, .optimize_prefill = true });
        if (!native.eql(try model.nativeKvIdentity(.f16, null)) or model.load_generation != pipeline.load_generation) return error.StaleContextView;
        const body = try std.json.Stringify.valueAlloc(allocator, .{ .schema = if (serving_route) "antfly.compiled-packed-serving-greedy64-diagnostic.v1" else "antfly.compiled-packed-greedy64-token-parity-diagnostic.v1", .diagnostic_only = true, .qualification_credit = false, .serving_activation = false, .source_checkpoint_sha256 = exact_source_sha256, .binary_sha256 = try selfExecutableDigest(), .model_identity = native, .process_rss_limit_bytes = 8 * 1024 * 1024 * 1024, .result = result }, .{ .whitespace = .indent_2 });
        defer allocator.free(body);
        _ = try writeExclusive(output_path, body);
        return;
    }
    if (std.c.getenv("ANTFLY_METAL_COMPILED_PACKED_DIFF64_DIAGNOSTIC")) |raw_mode| {
        if (!std.mem.eql(u8, std.mem.span(raw_mode), "1")) return error.InvalidNativeSoakRequest;
        const exact_source_sha256 = try driver.parseDigest(input.checkpoints[0].sha256);
        const required_source_sha256 = try driver.parseDigest("a275fd233cadeac3a070b522babed949e0dbff91e56d968b6eeafdb0a02f6c37");
        if (!std.mem.eql(u8, &exact_source_sha256, &required_source_sha256) or saved[0].state.logical_length != 8128) return error.NativeSoakCheckpointChanged;
        if (pipeline.gpt_config.num_hidden_layers != 42 or pipeline.gpt_config.num_attention_heads != 8 or pipeline.gpt_config.global_head_dim != 512) return error.InvalidModelForGeneration;
        if (!@import("graph/metal_executor.zig").supportsSession(model.session)) return error.MissingCompiledModelRuntime;
        pipeline.graph_cache = &model.native_generation_graph_cache;
        pipeline.compiled_partition_backend = .metal;
        pipeline.compiled_attachment_target = .whole_model;
        const memory = @import("runtime/tier/memory.zig");
        const components = [_]memory.GptGenerationBudgetComponent{.{ .backend = .metal, .kv_dtype = .f16, .config = pipeline.gpt_config }};
        const admission = try memory.reserveGptGenerationPrefill(&budget, &components, 8128, 64, 512);
        if (admission.max_chunk_rows != 512 or admission.max_scratch_bytes > 3840 * 1024 * 1024) return error.CompactionWorkspaceLimit;
        var prefill_plan = try @import("runtime/scheduler/native_generate.zig").PrefillChunkPlan.init(allocator, 8128, admission.max_chunk_rows, admission.max_scratch_bytes, .sole_idle, .fixed_ceiling);
        defer prefill_plan.deinit(allocator);
        const context_policy = @import("runtime/context_policy.zig");
        const packed_plan = try (context_policy.Policy{}).admit(8192, null, 8128, 64, 0);
        const packed_reservation = try budget.tryReserveKv(.backend, 2 * 1024 * 1024 * 1024);
        defer budget.release(packed_reservation);
        const result = try pipeline.diagnoseCompiledPackedDiff64(saved[0].state, .{ .max_tokens = 64, .prefill_chunk_size = admission.max_chunk_rows, .prefill_chunk_plan = prefill_plan }, .{ .plan = packed_plan, .layout = try context_policy.Layout.init(pipeline.gpt_config), .max_query_rows = 512, .physical_budget_bytes = 2 * 1024 * 1024 * 1024, .optimize_decode = true, .optimize_prefill = true });
        if (!native.eql(try model.nativeKvIdentity(.f16, null)) or model.load_generation != pipeline.load_generation) return error.StaleContextView;
        const body = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.compiled-packed-diff64-diagnostic.v1", .diagnostic_only = true, .qualification_credit = false, .serving_activation = false, .source_checkpoint_sha256 = exact_source_sha256, .binary_sha256 = try selfExecutableDigest(), .model_identity = native, .process_rss_limit_bytes = 8 * 1024 * 1024 * 1024, .result = result }, .{ .whitespace = .indent_2 });
        defer allocator.free(body);
        _ = try writeExclusive(output_path, body);
        return;
    }
    if (std.c.getenv("ANTFLY_METAL_COMPILED_PACKED_IMPORT_DIAGNOSTIC")) |raw_mode| {
        if (!std.mem.eql(u8, std.mem.span(raw_mode), "1")) return error.InvalidNativeSoakRequest;
        const exact_source_sha256 = try driver.parseDigest(input.checkpoints[0].sha256);
        const required_source_sha256 = try driver.parseDigest("a275fd233cadeac3a070b522babed949e0dbff91e56d968b6eeafdb0a02f6c37");
        if (!std.mem.eql(u8, &exact_source_sha256, &required_source_sha256) or saved[0].state.logical_length != 8128)
            return error.NativeSoakCheckpointChanged;
        if (pipeline.gpt_config.num_hidden_layers != 42 or pipeline.gpt_config.num_attention_heads != 8 or pipeline.gpt_config.global_head_dim != 512)
            return error.InvalidModelForGeneration;
        if (!@import("graph/metal_executor.zig").supportsSession(model.session)) return error.MissingCompiledModelRuntime;
        pipeline.graph_cache = &model.native_generation_graph_cache;
        pipeline.compiled_partition_backend = .metal;
        pipeline.compiled_attachment_target = .whole_model;
        const memory = @import("runtime/tier/memory.zig");
        const components = [_]memory.GptGenerationBudgetComponent{.{ .backend = .metal, .kv_dtype = .f16, .config = pipeline.gpt_config }};
        const admission = try memory.reserveGptGenerationPrefill(&budget, &components, 8128, 64, 512);
        if (admission.max_chunk_rows != 512) return error.CompactionWorkspaceLimit;
        var prefill_plan = try @import("runtime/scheduler/native_generate.zig").PrefillChunkPlan.init(allocator, 8128, admission.max_chunk_rows, admission.max_scratch_bytes, .sole_idle, .fixed_ceiling);
        defer prefill_plan.deinit(allocator);
        const context_policy = @import("runtime/context_policy.zig");
        const packed_plan = try (context_policy.Policy{}).admit(8192, null, 8128, 64, 0);
        const packed_reservation = try budget.tryReserveKv(.backend, 2 * 1024 * 1024 * 1024);
        defer budget.release(packed_reservation);
        const result = try pipeline.diagnoseCompiledPackedImportOneStep(
            saved[0].state,
            .{ .max_tokens = 64, .prefill_chunk_size = admission.max_chunk_rows, .prefill_chunk_plan = prefill_plan },
            .{
                .plan = packed_plan,
                .layout = try context_policy.Layout.init(pipeline.gpt_config),
                .max_query_rows = 512,
                .physical_budget_bytes = 2 * 1024 * 1024 * 1024,
                .optimize_decode = true,
                .optimize_prefill = true,
            },
        );
        if (!native.eql(try model.nativeKvIdentity(.f16, null)) or model.load_generation != pipeline.load_generation)
            return error.StaleContextView;
        const body = try std.json.Stringify.valueAlloc(allocator, .{
            .schema = "antfly.compiled-packed-import-one-step-diagnostic.v1",
            .diagnostic_only = true,
            .qualification_credit = false,
            .serving_activation = false,
            .source_checkpoint_sha256 = exact_source_sha256,
            .binary_sha256 = try selfExecutableDigest(),
            .model_identity = native,
            .process_rss_limit_bytes = 8 * 1024 * 1024 * 1024,
            .result = result,
        }, .{ .whitespace = .indent_2 });
        defer allocator.free(body);
        _ = try writeExclusive(output_path, body);
        return;
    }
    if (std.c.getenv("ANTFLY_METAL_RECENT_PREFILL_CAPTURE_DIAGNOSTIC")) |raw_mode| {
        if (!std.mem.eql(u8, std.mem.span(raw_mode), "1")) return error.InvalidNativeSoakRequest;
        const capture = input.recent_prefill_capture orelse return error.InvalidNativeSoakRequest;
        const Direct = @import("runtime/kv/attention_match_recent64_direct_diagnostic.zig");
        const direct_fit_mode: ?Direct.Mode = if (std.c.getenv("ANTFLY_METAL_RECENT64_DIRECT_FIT_DIAGNOSTIC")) |raw_direct| blk: {
            const value = std.mem.span(raw_direct);
            if (std.mem.eql(u8, value, "1") or std.mem.eql(u8, value, "A")) break :blk .established_a;
            if (std.mem.eql(u8, value, "B")) break :blk .incremental_geometry_b;
            if (std.mem.eql(u8, value, "C")) break :blk .incremental_reduction_c;
            return error.InvalidNativeSoakRequest;
        } else null;
        const direct_fit_requested = direct_fit_mode != null;
        const capture_output_dir: ?[]const u8 = if (std.c.getenv("ANTFLY_METAL_RECENT64_CAPTURE_OUTPUT_DIRECTORY")) |raw_capture_dir| std.mem.span(raw_capture_dir) else null;
        if (capture_output_dir) |path| {
            try requireAbsolute(path);
            try std.Io.Dir.createDirAbsolute(io, path, std.Io.File.Permissions.fromMode(0o700));
        }
        const candidate_output_path: ?[]const u8 = if (direct_fit_requested) try requiredEnv("ANTFLY_METAL_RECENT64_CANDIDATE_OUTPUT") else null;
        if (candidate_output_path) |path| {
            try requireAbsolute(path);
            try requireAbsent(path);
        }
        const exact_source_sha256 = try driver.parseDigest(input.checkpoints[0].sha256);
        const required_source_sha256 = try driver.parseDigest("a275fd233cadeac3a070b522babed949e0dbff91e56d968b6eeafdb0a02f6c37");
        if (!std.mem.eql(u8, &exact_source_sha256, &required_source_sha256)) return error.NativeSoakCheckpointChanged;
        const binary_sha256 = try selfExecutableDigest();
        if (!std.mem.eql(u8, &binary_sha256, &try driver.parseDigest(capture.binary_sha256)))
            return error.ContextQualificationLoadedIdentityMismatch;
        const source_freeze_sha256 = try hashFile(capture.source_freeze.path, 16 * 1024 * 1024);
        const profile_sha256 = try hashFile(capture.compatibility_profile.path, 1024 * 1024);
        const policy_sha256 = try hashFile(capture.development_policy.path, 1024 * 1024);
        if (!std.mem.eql(u8, &source_freeze_sha256, &try driver.parseDigest(capture.source_freeze.sha256)) or
            !std.mem.eql(u8, &profile_sha256, &try driver.parseDigest(capture.compatibility_profile.sha256)) or
            !std.mem.eql(u8, &policy_sha256, &try driver.parseDigest(capture.development_policy.sha256)))
            return error.InvalidNativeSoakRequest;
        var heldout: ?@import("runtime/kv/attention_match_prepared_heldout_bundle.zig").Owned = null;
        defer if (heldout) |*owned| owned.deinit();
        const hho_equivalence_requested = capture.future_hho_equivalence_output_path != null;
        if (direct_fit_requested or hho_equivalence_requested) {
            const heldout_ref = capture.heldout_bundle orelse return error.InvalidNativeSoakRequest;
            const heldout_authority = capture.heldout_authority_sha256 orelse return error.InvalidNativeSoakRequest;
            const heldout_file_sha256 = try hashFile(heldout_ref.path, 16 * 1024 * 1024);
            if (!std.mem.eql(u8, &heldout_file_sha256, &try driver.parseDigest(heldout_ref.sha256)))
                return error.InvalidNativeSoakRequest;
            if (hho_equivalence_requested and !std.mem.eql(u8, &heldout_file_sha256, &try driver.parseDigest("c8bbe569c24678eeaca54498aec59b4eda2f7510fdfad003aae0e7e99dcc49a7")))
                return error.InvalidNativeSoakRequest;
            const heldout_wire = try std.Io.Dir.cwd().readFileAlloc(io, heldout_ref.path, allocator, .limited(16 * 1024 * 1024));
            defer allocator.free(heldout_wire);
            heldout = try @import("runtime/kv/attention_match_prepared_heldout_bundle.zig").decodeOwned(
                allocator,
                heldout_wire,
                try driver.parseDigest(heldout_authority),
            );
        }
        if (pipeline.gpt_config.num_hidden_layers != 42 or pipeline.gpt_config.num_attention_heads != 8 or
            pipeline.gpt_config.global_head_dim != 512 or pipeline.gpt_config.num_kv_shared_layers != 18)
            return error.InvalidModelForGeneration;
        if (!@import("graph/metal_executor.zig").supportsSession(model.session)) return error.MissingCompiledModelRuntime;
        pipeline.graph_cache = &model.native_generation_graph_cache;
        pipeline.compiled_partition_backend = .metal;
        pipeline.compiled_attachment_target = .whole_model;

        const memory = @import("runtime/tier/memory.zig");
        const components = [_]memory.GptGenerationBudgetComponent{.{
            .backend = .metal,
            .kv_dtype = .f16,
            .config = pipeline.gpt_config,
        }};
        const prefill_chunk_ceiling: usize = 512;
        const admission = try memory.reserveGptGenerationPrefill(&budget, &components, 8128, 64, prefill_chunk_ceiling);
        var prefill_plan = try @import("runtime/scheduler/native_generate.zig").PrefillChunkPlan.init(
            allocator,
            8128,
            admission.max_chunk_rows,
            admission.max_scratch_bytes,
            .sole_idle,
            .fixed_ceiling,
        );
        defer prefill_plan.deinit(allocator);
        const compute: *@import("ops/metal_compute.zig").MetalCompute = @ptrCast(@alignCast(cb.ptr));
        const observed = try compute.provider_impl.nativeCheckpointObserved();
        try observed.destination.validate();
        if (std.mem.allEqual(u8, &observed.backend_environment_sha256, 0)) return error.InvalidNativeSoakRequest;
        const execution = compute.attentionMatchExecution();
        const execution_sha256 = @import("runtime/kv/attention_match_release.zig").configurationFingerprint(execution, observed.backend_environment_sha256);
        var input_receipt_sha256: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(raw, &input_receipt_sha256, .{});
        const Fixture = @import("runtime/kv/attention_match_recent64_captured_fixture.zig");
        const Publisher = struct {
            allocator: std.mem.Allocator,
            output_dir: []const u8,
            provenance: Fixture.Provenance,

            fn publish(raw_context: *anyopaque, training: @import("runtime/kv/attention_match_prefill_recent64_capture.zig").Training, topology: [@import("runtime/kv/attention_match_prefill_recent64_capture.zig").consumer_count]@import("runtime/kv/attention_match_prefill_recent64_capture.zig").Consumer) !void {
                const self: *@This() = @ptrCast(@alignCast(raw_context));
                const values_path = try std.fs.path.join(self.allocator, &.{ self.output_dir, Fixture.values_name });
                defer self.allocator.free(values_path);
                const positions_path = try std.fs.path.join(self.allocator, &.{ self.output_dir, Fixture.positions_name });
                defer self.allocator.free(positions_path);
                const manifest_path = try std.fs.path.join(self.allocator, &.{ self.output_dir, Fixture.manifest_name });
                defer self.allocator.free(manifest_path);
                const manifest = try Fixture.buildManifest(training, topology, self.provenance);
                const values_sha = try writeBinaryExclusive(values_path, std.mem.sliceAsBytes(training.values));
                if (!std.mem.eql(u8, &values_sha, &manifest.values.sha256)) return error.Recent64CapturedFixtureChanged;
                const positions_sha = try writeBinaryExclusive(positions_path, std.mem.sliceAsBytes(training.positions));
                if (!std.mem.eql(u8, &positions_sha, &manifest.positions.sha256)) return error.Recent64CapturedFixtureChanged;
                const body = try std.json.Stringify.valueAlloc(self.allocator, manifest, .{});
                defer self.allocator.free(body);
                _ = try writeExclusive(manifest_path, body); // Commit marker: always last.
            }
        };
        var publisher_context: Publisher = undefined;
        var capture_serialization_reservation: ?@import("runtime/tier/memory.zig").Reservation = null;
        defer if (capture_serialization_reservation) |reservation| budget.release(reservation);
        const training_publisher: ?@import("pipelines/generation.zig").NativeGenerationPipeline.RecentPrefillTrainingPublisher = if (capture_output_dir) |path| blk: {
            capture_serialization_reservation = try budget.tryReserveScratch(.host, Fixture.serialization_scratch_bytes);
            publisher_context = .{ .allocator = allocator, .output_dir = path, .provenance = .{
                .source_checkpoint_sha256 = exact_source_sha256,
                .source_freeze_sha256 = source_freeze_sha256,
                .binary_sha256 = binary_sha256,
                .implementation_semantic_sha256 = @import("compaction_identity").digest,
                .execution_sha256 = execution_sha256,
                .profile_sha256 = profile_sha256,
                .development_policy_sha256 = policy_sha256,
                .identity = .{ .weights = input.native_identity.weights, .tokenizer = input.native_identity.tokenizer, .template = input.native_identity.template, .layout = input.native_identity.layout },
            } };
            break :blk .{ .context = &publisher_context, .publish_fn = Publisher.publish };
        } else null;
        var diagnostic = try pipeline.diagnoseRecentPrefillQueryCapture(
            saved[0].state,
            .{ .max_tokens = 64, .prefill_chunk_size = admission.max_chunk_rows, .prefill_chunk_plan = prefill_plan },
            .{
                .source_freeze_sha256 = source_freeze_sha256,
                .implementation_semantic_sha256 = @import("compaction_identity").digest,
                .execution_sha256 = execution_sha256,
                .source_checkpoint_sha256 = exact_source_sha256,
                .profile_sha256 = profile_sha256,
                .policy_sha256 = policy_sha256,
                .device_registry_id = observed.destination.device_registry_id,
                .device_name_sha256 = observed.destination.device_name_sha256,
                .compiler_identity_sha256 = observed.destination.compiler_identity_sha256,
                .backend_environment_sha256 = observed.backend_environment_sha256,
            },
            if (direct_fit_requested) (heldout orelse return error.InvalidNativeSoakRequest) else null,
            &budget,
            direct_fit_mode,
            training_publisher,
        );
        defer diagnostic.deinit();
        if (capture.future_hho_equivalence_output_path) |hho_output_path| {
            try requireAbsolute(hho_output_path);
            try requireAbsent(hho_output_path);
            const retained_hho = if (heldout) |*owned| owned else return error.InvalidNativeSoakRequest;
            var hho = try pipeline.diagnoseFutureHhoEquivalence(
                saved[0].state,
                .{ .max_tokens = 512, .prefill_chunk_size = 512 },
                .{
                    .source_freeze_sha256 = source_freeze_sha256,
                    .implementation_semantic_sha256 = @import("compaction_identity").digest,
                    .execution_sha256 = execution_sha256,
                    .source_checkpoint_sha256 = exact_source_sha256,
                    .profile_sha256 = profile_sha256,
                    .policy_sha256 = policy_sha256,
                    .device_registry_id = observed.destination.device_registry_id,
                    .device_name_sha256 = observed.destination.device_name_sha256,
                    .compiler_identity_sha256 = observed.destination.compiler_identity_sha256,
                    .backend_environment_sha256 = observed.backend_environment_sha256,
                },
                retained_hho,
                &budget,
            );
            defer hho.deinit();
            const hho_report = try std.json.Stringify.valueAlloc(allocator, .{
                .schema = "antfly.future-hho-equivalence-diagnostic.v1",
                .role = "HHO_EQ",
                .diagnostic_only = true,
                .qualification_credit = false,
                .serving_activation = false,
                .source_checkpoint_sha256 = exact_source_sha256,
                .source_freeze_sha256 = source_freeze_sha256,
                .binary_sha256 = binary_sha256,
                .diagnostic_completed = true,
                .equivalent = hho.all_f32_bit_identical,
                .source_authority = hho.source_authority,
                .branch_authority = hho.branch_authority,
                .continuation_ids = hho.continuation_ids,
                .continuation_ids_sha256 = hho.continuation_ids_sha256,
                .captured_authority_sha256 = hho.captured_authority_sha256,
                .retained_authority_sha256 = hho.retained_authority_sha256,
                .tuples = hho.tuples,
                .total_mismatches = hho.total_mismatches,
                .compared_values = hho.compared_values,
                .maximum_absolute_delta = hho.maximum_absolute_delta,
                .rmse = hho.rmse,
                .all_f32_bit_identical = hho.all_f32_bit_identical,
                .native_plan_calls = hho.native_plan_calls,
                .native_commit_calls = hho.native_commit_calls,
                .native_copy_bytes = hho.native_copy_bytes,
                .capture_elapsed_ns = hho.capture_elapsed_ns,
            }, .{ .whitespace = .indent_2 });
            defer allocator.free(hho_report);
            _ = try writeExclusive(hho_output_path, hho_report);
        }
        const future_requested = capture.future_train_p64_output_dir != null;
        const future_phase_requested = capture.future_train_phase_p64_output_dir != null;
        if ((future_requested or future_phase_requested) != (capture.future_train_p64_continuation_origin_sha256 != null)) return error.InvalidNativeSoakRequest;
        if (capture.future_train_p64_output_dir) |future_dir| {
            try requireAbsolute(future_dir);
            const origin = try driver.parseDigest(capture.future_train_p64_continuation_origin_sha256.?);
            var future = try pipeline.diagnoseTeacherForcedFutureTrainP64(
                saved[0].state,
                .{ .max_tokens = 512, .prefill_chunk_size = 512 },
                .{
                    .source_freeze_sha256 = source_freeze_sha256,
                    .implementation_semantic_sha256 = @import("compaction_identity").digest,
                    .execution_sha256 = execution_sha256,
                    .source_checkpoint_sha256 = exact_source_sha256,
                    .profile_sha256 = profile_sha256,
                    .policy_sha256 = policy_sha256,
                    .device_registry_id = observed.destination.device_registry_id,
                    .device_name_sha256 = observed.destination.device_name_sha256,
                    .compiler_identity_sha256 = observed.destination.compiler_identity_sha256,
                    .backend_environment_sha256 = observed.backend_environment_sha256,
                },
                origin,
                &budget,
            );
            defer future.deinit();
            const values_path = try std.fs.path.join(allocator, &.{ future_dir, "future-train-p64-values.f32" });
            defer allocator.free(values_path);
            const positions_path = try std.fs.path.join(allocator, &.{ future_dir, "future-train-p64-positions.u64" });
            defer allocator.free(positions_path);
            const report_path = try std.fs.path.join(allocator, &.{ future_dir, "future-train-p64-report.json" });
            defer allocator.free(report_path);
            const values_sha = try writeBinaryExclusive(values_path, std.mem.sliceAsBytes(future.training.values));
            const positions_sha = try writeBinaryExclusive(positions_path, std.mem.sliceAsBytes(future.training.positions));
            const report = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.teacher-forced-future-train-p64-diagnostic.v3", .role = "TRAIN_P64", .diagnostic_only = true, .qualification_credit = false, .serving_activation = false, .source_checkpoint_sha256 = exact_source_sha256, .source_freeze_sha256 = source_freeze_sha256, .binary_sha256 = binary_sha256, .continuation_ids_sha256 = future.continuation_ids_sha256, .continuation_origin_sha256 = future.continuation_origin_sha256, .producer_receipt_sha256 = future.producer_receipt_sha256, .training_authority_sha256 = future.training.authority_sha256, .payload_sha256 = future.training.payload_sha256, .values_sha256 = values_sha, .positions_sha256 = positions_sha, .source_authority = future.source_authority, .branch_authority = future.branch_authority, .batch_count = future.training.batches.len, .heldout_rows = 0, .train_rows = 64, .first_train_position = 8132, .last_train_position = 8636, .native_plan_calls = future.native_plan_calls, .native_commit_calls = future.native_commit_calls, .native_copy_bytes = future.native_copy_bytes, .capture_elapsed_ns = future.capture_elapsed_ns, .host_scratch_envelope_bytes = @TypeOf(future).host_scratch_envelope_bytes, .backend_scratch_envelope_bytes = @TypeOf(future).backend_scratch_envelope_bytes, .budget_host_scratch_before = future.budget_host_scratch_before, .budget_backend_scratch_before = future.budget_backend_scratch_before, .budget_host_scratch_after_reserve = future.budget_host_scratch_after_reserve, .budget_backend_scratch_after_reserve = future.budget_backend_scratch_after_reserve }, .{});
            defer allocator.free(report);
            _ = try writeExclusive(report_path, report);
        }
        if (capture.future_train_phase_p64_output_dir) |future_dir| {
            try requireAbsolute(future_dir);
            const origin = try driver.parseDigest(capture.future_train_p64_continuation_origin_sha256.?);
            var future = try pipeline.diagnoseTeacherForcedFutureTrainPhaseP64(
                saved[0].state,
                .{ .max_tokens = 512, .prefill_chunk_size = 512 },
                .{
                    .source_freeze_sha256 = source_freeze_sha256,
                    .implementation_semantic_sha256 = @import("compaction_identity").digest,
                    .execution_sha256 = execution_sha256,
                    .source_checkpoint_sha256 = exact_source_sha256,
                    .profile_sha256 = profile_sha256,
                    .policy_sha256 = policy_sha256,
                    .device_registry_id = observed.destination.device_registry_id,
                    .device_name_sha256 = observed.destination.device_name_sha256,
                    .compiler_identity_sha256 = observed.destination.compiler_identity_sha256,
                    .backend_environment_sha256 = observed.backend_environment_sha256,
                },
                origin,
                &budget,
            );
            defer future.deinit();
            const values_path = try std.fs.path.join(allocator, &.{ future_dir, "future-train-phase-p64-values.f32" });
            defer allocator.free(values_path);
            const positions_path = try std.fs.path.join(allocator, &.{ future_dir, "future-train-phase-p64-positions.u64" });
            defer allocator.free(positions_path);
            const report_path = try std.fs.path.join(allocator, &.{ future_dir, "future-train-phase-p64-report.json" });
            defer allocator.free(report_path);
            const values_sha = try writeBinaryExclusive(values_path, std.mem.sliceAsBytes(future.training.values));
            const positions_sha = try writeBinaryExclusive(positions_path, std.mem.sliceAsBytes(future.training.positions));
            const report = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.teacher-forced-future-train-phase-p64-diagnostic.v1", .role = "TRAIN_P64_PHASE_BALANCED_V1", .diagnostic_only = true, .qualification_credit = false, .serving_activation = false, .source_checkpoint_sha256 = exact_source_sha256, .source_freeze_sha256 = source_freeze_sha256, .binary_sha256 = binary_sha256, .continuation_ids_sha256 = future.continuation_ids_sha256, .continuation_origin_sha256 = future.continuation_origin_sha256, .producer_receipt_sha256 = future.producer_receipt_sha256, .training_authority_sha256 = future.training.authority_sha256, .payload_sha256 = future.training.payload_sha256, .values_sha256 = values_sha, .positions_sha256 = positions_sha, .source_authority = future.source_authority, .branch_authority = future.branch_authority, .batch_count = future.training.batches.len, .heldout_rows = 0, .train_rows = 64, .first_train_position = 8129, .last_train_position = 8633, .phase_count = 7, .phase_counts = [_]usize{ 10, 9, 9, 9, 9, 9, 9 }, .native_plan_calls = future.native_plan_calls, .native_commit_calls = future.native_commit_calls, .native_copy_bytes = future.native_copy_bytes, .capture_elapsed_ns = future.capture_elapsed_ns, .host_scratch_envelope_bytes = @TypeOf(future).host_scratch_envelope_bytes, .backend_scratch_envelope_bytes = @TypeOf(future).backend_scratch_envelope_bytes, .budget_host_scratch_before = future.budget_host_scratch_before, .budget_backend_scratch_before = future.budget_backend_scratch_before, .budget_host_scratch_after_reserve = future.budget_host_scratch_after_reserve, .budget_backend_scratch_after_reserve = future.budget_backend_scratch_after_reserve }, .{});
            defer allocator.free(report);
            _ = try writeExclusive(report_path, report);
        }
        if (direct_fit_requested != (diagnostic.direct_fit != null)) return error.InvalidNativeSoakProgress;
        const DirectFitReport = struct {
            policy: []const u8,
            capture_elapsed_ns: u64,
            fit_elapsed_ns: u64,
            capture_and_fit_elapsed_ns: u64,
            compiled_runtime_reset_before_fit: bool,
            prefill_chunk_rows: usize = 0,
            train_rows_per_consumer: usize,
            heldout_rows_per_consumer: usize,
            authenticated_global_heads: usize,
            v18_eligible_heads: usize,
            v18_attempted_heads: usize,
            v18_accepted_heads: usize,
            v18_preserved_heads: usize,
            legacy_validated_heads: usize,
            maximum_head_mass_error: f64,
            maximum_head_output_rmse: f64,
            source_global_bytes: usize,
            candidate_global_bytes: usize,
            actual_global_savings_ratio: f64,
            source_physical_bytes: usize,
            candidate_physical_bytes: usize,
            physical_savings_ratio: f64,
            fit_host_cap_bytes: usize,
            fit_backend_cap_bytes: usize,
            fit_combined_cap_bytes: usize,
            candidate_path: []const u8,
            candidate_sha256: [32]u8,
        };
        var direct_report: ?DirectFitReport = null;
        if (diagnostic.direct_fit) |direct_fit| {
            const candidate_wire = try checkpoint.encodeWithEncoding(allocator, direct_fit.candidate.state, checkpoint.serving_limits, .f16, .{});
            defer allocator.free(candidate_wire);
            const candidate_sha256 = try writeBinaryExclusive(candidate_output_path.?, candidate_wire);
            direct_report = .{ .policy = Direct.policy(direct_fit_mode.?), .capture_elapsed_ns = diagnostic.capture_elapsed_ns, .fit_elapsed_ns = direct_fit.elapsed_ns, .capture_and_fit_elapsed_ns = try std.math.add(u64, diagnostic.capture_elapsed_ns, direct_fit.elapsed_ns), .compiled_runtime_reset_before_fit = diagnostic.compiled_runtime_reset_before_fit, .train_rows_per_consumer = direct_fit.train_rows_per_consumer, .heldout_rows_per_consumer = direct_fit.heldout_rows_per_consumer, .authenticated_global_heads = direct_fit.authenticated_global_heads, .v18_eligible_heads = direct_fit.stats.v18_eligible_heads, .v18_attempted_heads = direct_fit.stats.v18_attempted_heads, .v18_accepted_heads = direct_fit.stats.v18_accepted_heads, .v18_preserved_heads = direct_fit.stats.v18_preserved_heads, .legacy_validated_heads = direct_fit.stats.validated_heads, .maximum_head_mass_error = direct_fit.stats.max_head_mass_error, .maximum_head_output_rmse = direct_fit.stats.max_head_output_rmse, .source_global_bytes = direct_fit.stats.source_global_bytes, .candidate_global_bytes = direct_fit.stats.candidate_global_bytes, .actual_global_savings_ratio = direct_fit.stats.actual_global_savings_ratio, .source_physical_bytes = direct_fit.source_physical_bytes, .candidate_physical_bytes = direct_fit.candidate_physical_bytes, .physical_savings_ratio = direct_fit.physical_savings_ratio, .fit_host_cap_bytes = Direct.host_fit_cap_bytes, .fit_backend_cap_bytes = Direct.head_workspace_bytes, .fit_combined_cap_bytes = Direct.combined_cap_bytes, .candidate_path = candidate_output_path.?, .candidate_sha256 = candidate_sha256 };
            direct_report.?.prefill_chunk_rows = admission.max_chunk_rows;
        }
        const body = try std.json.Stringify.valueAlloc(allocator, .{
            .schema = "antfly.recent-prefill-query-capture-diagnostic.v1",
            .diagnostic_only = true,
            .qualification_credit = false,
            .serving_activation = false,
            .source_checkpoint_sha256 = exact_source_sha256,
            .source_freeze_sha256 = source_freeze_sha256,
            .binary_sha256 = binary_sha256,
            .input_receipt_sha256 = input_receipt_sha256,
            .implementation_semantic_sha256 = @import("compaction_identity").digest,
            .execution_sha256 = execution_sha256,
            .profile_sha256 = profile_sha256,
            .development_policy_sha256 = policy_sha256,
            .device = observed.destination,
            .backend_environment_sha256 = observed.backend_environment_sha256,
            .geometry = .{ .logical_tokens = 8128, .fitted_middle_end_exclusive = 6080, .horizon = 64, .consumers = 7, .query_heads = 8, .head_dim = 512 },
            .capture_epoch = diagnostic.capture_epoch,
            .prefill_chunk_rows = admission.max_chunk_rows,
            .capture_elapsed_ns = diagnostic.capture_elapsed_ns,
            .capture_path = .{
                .native_plan_calls = diagnostic.native_plan_calls,
                .native_commit_calls = diagnostic.native_commit_calls,
                .native_copy_bytes = diagnostic.native_copy_bytes,
                .canonical_copy_calls = diagnostic.canonical_copy_calls,
                .training_values_sha256 = diagnostic.training_values_sha256,
                .training_positions_sha256 = diagnostic.training_positions_sha256,
            },
            .source_authority = diagnostic.source_authority,
            .training = .{ .batches = diagnostic.training.batches.len, .values = diagnostic.training.values.len, .positions = diagnostic.training.positions.len, .authority_sha256 = diagnostic.training.authority_sha256, .portfolio_sha256 = diagnostic.training.portfolio_sha256 },
            .direct_fit = direct_report,
            .admission = .{ .host_physical_bytes = diagnostic.admitted_host_physical_bytes, .backend_bytes = diagnostic.admitted_backend_bytes, .combined_physical_bytes = diagnostic.admitted_combined_physical_bytes, .process_rss_limit_bytes = 8 * 1024 * 1024 * 1024 },
        }, .{ .whitespace = .indent_2 });
        defer allocator.free(body);
        _ = try writeExclusive(output_path, body);
        return;
    }
    if (std.c.getenv("ANTFLY_METAL_SAVED_AFKV_DECODE_DIAGNOSTIC")) |raw_mode| {
        if (!std.mem.eql(u8, std.mem.span(raw_mode), "1")) return error.InvalidNativeSoakRequest;
        const result = try pipeline.diagnoseSavedAfkvDecode(saved[0].state, &budget);
        if (!native.eql(try model.nativeKvIdentity(.f16, null)) or model.load_generation != pipeline.load_generation) return error.StaleContextView;
        const body = try std.json.Stringify.valueAlloc(allocator, .{
            .schema = "antfly.saved-afkv-decode-diagnostic.v1",
            .diagnostic_only = true,
            .qualification_credit = false,
            .producer_metadata_used = false,
            .producer_policy_validated = false,
            .production_budget_admission = false,
            .local_capacity_policy = "legacy_spare512_v1",
            .checkpoint_sha256 = input.checkpoints[0].sha256,
            .model_identity = native,
            .main_gguf_sha256 = actual_main,
            .load_generation = model.load_generation,
            .result = result,
        }, .{ .whitespace = .indent_2 });
        defer allocator.free(body);
        if (!result.owner_released or result.post_release_live_bytes != 0 or result.invalid or
            result.decode_steps != 63 or result.attention_calls != 42 * 63 or
            result.donor_appends != 24 * 63 or result.direct_ffn_successes != 42 * 63 or
            result.direct_ffn_fallbacks != 0 or result.canonical_last_logits_calls != 63 or
            result.prepared_greedy_successes != 0)
            return error.InvalidNativeSoakProgress;
        _ = try writeExclusive(output_path, body);
        return;
    }
    var journal = try std.Io.Dir.cwd().createFile(io, journal_path, .{ .exclusive = true, .truncate = false });
    defer journal.close(io);
    const report = try pipeline.qualifyMetalCheckpointSoak(.{ saved[0].state, saved[1].state }, saved_policy_sha256, input.request, &journal);
    if (!native.eql(try model.nativeKvIdentity(.f16, null)) or model.load_generation != pipeline.load_generation)
        return error.StaleContextView;
    var input_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &input_digest, .{});
    const observed = try @as(*@import("ops/metal_compute.zig").MetalCompute, @ptrCast(@alignCast(cb.ptr))).provider_impl.nativeCheckpointObserved();
    const execution = @as(*@import("ops/metal_compute.zig").MetalCompute, @ptrCast(@alignCast(cb.ptr))).attentionMatchExecution();
    const body = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.native-checkpoint-soak.report.v2",
        .model = model_name,
        .model_directory = model.model_dir,
        .model_identity = native,
        .main_gguf_sha256 = actual_main,
        .load_generation = model.load_generation,
        .input_sha256 = input_digest,
        .implementation_sha256 = @import("compaction_identity").digest,
        .observed = observed,
        .execution = execution,
        .configuration_sha256 = @import("runtime/kv/attention_match_release.zig").configurationFingerprint(execution, observed.backend_environment_sha256),
        .encoding = @tagName(input.request.encoding),
        .report = report,
    }, .{ .whitespace = .indent_2 });
    defer allocator.free(body);
    _ = try writeExclusive(output_path, body);
}

test "context recent64 captured TRAIN replay diagnostic" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const manifest_path = platform.env.getenv("ANTFLY_METAL_RECENT64_REPLAY_MANIFEST") orelse return error.SkipZigTest;
    const manifest_sha = try requiredEnv("ANTFLY_METAL_RECENT64_REPLAY_MANIFEST_SHA256");
    const source_path = try requiredEnv("ANTFLY_METAL_RECENT64_REPLAY_SOURCE_AFKV");
    const heldout_path = try requiredEnv("ANTFLY_METAL_RECENT64_REPLAY_HELDOUT_BUNDLE");
    const heldout_authority = try requiredEnv("ANTFLY_METAL_RECENT64_REPLAY_HELDOUT_AUTHORITY_SHA256");
    const output_path = try requiredEnv("ANTFLY_METAL_RECENT64_REPLAY_OUTPUT");
    const candidate_path = try requiredEnv("ANTFLY_METAL_RECENT64_REPLAY_CANDIDATE_OUTPUT");
    inline for (.{ manifest_path, source_path, heldout_path, output_path, candidate_path }) |path| try requireAbsolute(path);
    try requireAbsent(output_path);
    try requireAbsent(candidate_path);
    const mode_raw = try requiredEnv("ANTFLY_METAL_RECENT64_REPLAY_MODE");
    const Direct = @import("runtime/kv/attention_match_recent64_direct_diagnostic.zig");
    const mode: Direct.Mode = if (std.mem.eql(u8, mode_raw, "A")) .established_a else if (std.mem.eql(u8, mode_raw, "B")) .incremental_geometry_b else if (std.mem.eql(u8, mode_raw, "C")) .incremental_reduction_c else return error.InvalidNativeSoakRequest;
    const Replay = @import("runtime/kv/attention_match_recent64_captured_replay.zig");
    var result = try Replay.run(allocator, io, .{
        .manifest_path = manifest_path,
        .expected_manifest_sha256 = try @import("context_native_checkpoint_quality.zig").parseDigest(manifest_sha),
        .expected_binary_sha256 = try selfExecutableDigest(),
        .source_afkv_path = source_path,
        .heldout_path = heldout_path,
        .heldout_authority_sha256 = try @import("context_native_checkpoint_quality.zig").parseDigest(heldout_authority),
        .mode = mode,
    }, .{});
    defer result.deinit();
    const candidate_wire = try @import("runtime/kv/attention_match_checkpoint.zig").encodeWithEncoding(allocator, result.candidate.state, @import("runtime/kv/attention_match_checkpoint.zig").serving_limits, .f16, .{});
    defer allocator.free(candidate_wire);
    const candidate_sha256 = try writeBinaryExclusive(candidate_path, candidate_wire);
    const body = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.recent64-captured-replay-result.v1",
        .mode = @tagName(mode),
        .policy = Direct.policy(mode),
        .candidate_sha256 = candidate_sha256,
        .stats = result.stats,
        .source_physical_bytes = result.source_physical_bytes,
        .candidate_physical_bytes = result.candidate_physical_bytes,
        .physical_savings_ratio = result.physical_savings_ratio,
        .diagnostic_only = true,
        .qualification_credit = false,
        .production_qualified = false,
    }, .{});
    defer allocator.free(body);
    _ = try writeExclusive(output_path, body);
}

fn parseRecentFutureReplayMode(raw: []const u8) !@import("runtime/kv/attention_match_recent64_future_primary_direct.zig").Mode {
    return if (std.mem.eql(u8, raw, "A")) .established_a else if (std.mem.eql(u8, raw, "C")) .incremental_reduction_c else if (std.mem.eql(u8, raw, "D")) .optimized_gram_advisory_d else if (std.mem.eql(u8, raw, "E")) .full128_incremental_reduction_e else if (std.mem.eql(u8, raw, "F")) .full128_budget_91_95_3_f else if (std.mem.eql(u8, raw, "G")) .full128_phase_balanced_g else error.InvalidNativeSoakRequest;
}

test "recent future replay mode F is explicit and does not reinterpret E" {
    try std.testing.expectEqual(@import("runtime/kv/attention_match_recent64_future_primary_direct.zig").Mode.full128_incremental_reduction_e, try parseRecentFutureReplayMode("E"));
    try std.testing.expectEqual(@import("runtime/kv/attention_match_recent64_future_primary_direct.zig").Mode.full128_budget_91_95_3_f, try parseRecentFutureReplayMode("F"));
    try std.testing.expectError(error.InvalidNativeSoakRequest, parseRecentFutureReplayMode("f"));
}

test "recent future replay mode G is explicit and phase balanced only" {
    try std.testing.expectEqual(@import("runtime/kv/attention_match_recent64_future_primary_direct.zig").Mode.full128_phase_balanced_g, try parseRecentFutureReplayMode("G"));
    try std.testing.expectError(error.InvalidNativeSoakRequest, parseRecentFutureReplayMode("g"));
}

test "context recent32 future primary32 captured replay diagnostic" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const recent_manifest = platform.env.getenv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_RECENT_MANIFEST") orelse return error.SkipZigTest;
    const future_report = try requiredEnv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_FUTURE_REPORT");
    const source_path = try requiredEnv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_SOURCE_AFKV");
    const heldout_path = try requiredEnv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_HELDOUT_BUNDLE");
    const output_path = try requiredEnv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_OUTPUT");
    const candidate_path = try requiredEnv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_CANDIDATE_OUTPUT");
    inline for (.{ recent_manifest, future_report, source_path, heldout_path, output_path, candidate_path }) |path| try requireAbsolute(path);
    try requireAbsent(output_path);
    try requireAbsent(candidate_path);
    const Direct = @import("runtime/kv/attention_match_recent64_future_primary_direct.zig");
    const mode_raw = try requiredEnv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_MODE");
    const mode: Direct.Mode = try parseRecentFutureReplayMode(mode_raw);
    const parseDigest = @import("context_native_checkpoint_quality.zig").parseDigest;
    const Replay = @import("runtime/kv/attention_match_recent64_future_primary_replay.zig");
    const consumer_binary = try selfExecutableDigest();
    const consumer_freeze = try parseDigest(try requiredEnv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_SOURCE_FREEZE_SHA256"));
    const producer_binary_raw = platform.env.getenv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_PRODUCER_BINARY_SHA256");
    const producer_freeze_raw = platform.env.getenv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_PRODUCER_SOURCE_FREEZE_SHA256");
    const rejected_fixture_dir = platform.env.getenv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_REJECTED_FIXTURE_DIR");
    const rejected_fixture_config_raw = platform.env.getenv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_REJECTED_FIXTURE_CONFIG_SHA256");
    if ((rejected_fixture_dir == null) != (rejected_fixture_config_raw == null)) return error.InvalidNativeSoakRequest;
    if (rejected_fixture_dir) |path| {
        try requireAbsolute(path);
        try requireAbsent(path);
        if (mode != .full128_incremental_reduction_e) return error.InvalidNativeSoakRequest;
    }
    if ((producer_binary_raw == null) != (producer_freeze_raw == null)) return error.InvalidNativeSoakRequest;
    const producer_binary = if (producer_binary_raw) |raw| try parseDigest(raw) else consumer_binary;
    const producer_freeze = if (producer_freeze_raw) |raw| try parseDigest(raw) else consumer_freeze;
    const same_producer = std.mem.eql(u8, &producer_binary, &consumer_binary) and std.mem.eql(u8, &producer_freeze, &consumer_freeze);
    const recent_manifest_sha256 = try parseDigest(try requiredEnv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_RECENT_MANIFEST_SHA256"));
    const future_report_sha256 = try parseDigest(try requiredEnv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_FUTURE_REPORT_SHA256"));
    var result = try Replay.run(allocator, io, .{
        .recent_manifest_path = recent_manifest,
        .recent_manifest_sha256 = recent_manifest_sha256,
        .future_report_path = future_report,
        .future_report_sha256 = future_report_sha256,
        .expected_binary_sha256 = consumer_binary,
        .expected_source_freeze_sha256 = consumer_freeze,
        .producer_binary_sha256 = if (producer_binary_raw != null) producer_binary else null,
        .producer_source_freeze_sha256 = if (producer_freeze_raw != null) producer_freeze else null,
        .allow_cross_producer_builtin_test = !same_producer,
        .source_afkv_path = source_path,
        .heldout_path = heldout_path,
        .heldout_authority_sha256 = try parseDigest(try requiredEnv("ANTFLY_METAL_RECENT_FUTURE_REPLAY_HELDOUT_AUTHORITY_SHA256")),
        .mode = mode,
        .rejected_fixture_output_dir = rejected_fixture_dir,
        .rejected_fixture_config_sha256 = if (rejected_fixture_config_raw) |raw| try parseDigest(raw) else null,
    }, .{});
    defer result.deinit();
    const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
    const candidate_wire = try checkpoint.encodeWithEncoding(allocator, result.candidate.state, checkpoint.serving_limits, .f16, .{});
    defer allocator.free(candidate_wire);
    const candidate_sha256 = try writeBinaryExclusive(candidate_path, candidate_wire);
    const body = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.recent32-future-primary32-replay-result.v1",
        .mode = @tagName(mode),
        .policy = Direct.policy(mode),
        .policy_sha256 = Direct.policyDigest(mode),
        .hybrid_digest = result.hybrid_digest,
        .capture_producer_binary_sha256 = producer_binary,
        .capture_producer_source_freeze_sha256 = producer_freeze,
        .consumer_binary_sha256 = consumer_binary,
        .consumer_source_freeze_sha256 = consumer_freeze,
        .recent_manifest_sha256 = recent_manifest_sha256,
        .future_report_sha256 = future_report_sha256,
        .same_producer = same_producer,
        .training_geometry = if (mode == .full128_phase_balanced_g) "full_recent64_future_phase_balanced64" else if (mode == .full128_incremental_reduction_e or mode == .full128_budget_91_95_3_f) "full_recent64_future64" else "fixed_recent32_future32",
        .training_rows_per_consumer = @as(usize, if (mode == .full128_incremental_reduction_e or mode == .full128_budget_91_95_3_f or mode == .full128_phase_balanced_g) 128 else 64),
        .independent_train_split = @as(usize, if (mode == .full128_incremental_reduction_e or mode == .full128_budget_91_95_3_f or mode == .full128_phase_balanced_g) 64 else 32),
        .candidate_sha256 = candidate_sha256,
        .stats = result.stats,
        .source_physical_bytes = result.source_physical_bytes,
        .candidate_physical_bytes = result.candidate_physical_bytes,
        .physical_savings_ratio = result.physical_savings_ratio,
        .diagnostic_only = true,
        .qualification_credit = false,
        .production_qualified = false,
    }, .{});
    defer allocator.free(body);
    _ = try writeExclusive(output_path, body);
}
