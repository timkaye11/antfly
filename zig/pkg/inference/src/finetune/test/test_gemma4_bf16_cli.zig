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
const ml = @import("ml");
const platform = @import("antfly_platform");
const build_options = @import("build_options");
const train_command = @import("../gemma4_train_command.zig");
const finetune = @import("../gemma4.zig");
const gemma_chat_data = @import("../gemma_chat_data.zig");
const gemma4_real = @import("../gemma4_real_autodiff.zig");
const real_autodiff = @import("../real_autodiff_trainer.zig");
const metal_runtime = @import("../../backends/metal_runtime.zig");
const safetensors = @import("../../models/safetensors.zig");
const compat = @import("../../io/compat.zig");

const FixtureTensor = struct {
    name: []const u8,
    shape: []const usize,
};

const tiny_config =
    \\{
    \\  "model_type": "gemma4_text",
    \\  "dtype": "bfloat16",
    \\  "hidden_size": 8,
    \\  "num_hidden_layers": 1,
    \\  "num_attention_heads": 2,
    \\  "num_key_value_heads": 1,
    \\  "head_dim": 4,
    \\  "intermediate_size": 16,
    \\  "vocab_size": 16,
    \\  "max_position_embeddings": 16,
    \\  "sliding_window": 0,
    \\  "num_kv_shared_layers": 0,
    \\  "hidden_size_per_layer_input": 0,
    \\  "hidden_activation": "gelu_pytorch_tanh",
    \\  "rms_norm_eps": 0.000001,
    \\  "rope_theta": 10000.0,
    \\  "tie_word_embeddings": true,
    \\  "bos_token_id": 2,
    \\  "eos_token_id": 1,
    \\  "pad_token_id": 0
    \\}
;

const tiny_tensors = [_]FixtureTensor{
    .{ .name = "model.embed_tokens.weight", .shape = &.{ 16, 8 } },
    .{ .name = "model.layers.0.input_layernorm.weight", .shape = &.{8} },
    .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &.{ 8, 8 } },
    .{ .name = "model.layers.0.self_attn.k_proj.weight", .shape = &.{ 4, 8 } },
    .{ .name = "model.layers.0.self_attn.v_proj.weight", .shape = &.{ 4, 8 } },
    .{ .name = "model.layers.0.self_attn.q_norm.weight", .shape = &.{4} },
    .{ .name = "model.layers.0.self_attn.k_norm.weight", .shape = &.{4} },
    .{ .name = "model.layers.0.self_attn.o_proj.weight", .shape = &.{ 8, 8 } },
    .{ .name = "model.layers.0.post_attention_layernorm.weight", .shape = &.{8} },
    .{ .name = "model.layers.0.pre_feedforward_layernorm.weight", .shape = &.{8} },
    .{ .name = "model.layers.0.mlp.gate_proj.weight", .shape = &.{ 16, 8 } },
    .{ .name = "model.layers.0.mlp.up_proj.weight", .shape = &.{ 16, 8 } },
    .{ .name = "model.layers.0.mlp.down_proj.weight", .shape = &.{ 8, 16 } },
    .{ .name = "model.layers.0.post_feedforward_layernorm.weight", .shape = &.{8} },
    .{ .name = "model.norm.weight", .shape = &.{8} },
};

const EnvironmentOverride = struct {
    const c = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };

    allocator: std.mem.Allocator,
    name: [:0]const u8,
    previous: ?[:0]u8,

    fn set(allocator: std.mem.Allocator, name: [:0]const u8, value: [:0]const u8) !EnvironmentOverride {
        const previous = if (platform.env.getenv(name)) |old| try allocator.dupeZ(u8, old) else null;
        errdefer if (previous) |old| allocator.free(old);
        if (c.setenv(name.ptr, value.ptr, 1) != 0) return error.SkipZigTest;
        return .{ .allocator = allocator, .name = name, .previous = previous };
    }

    fn unset(allocator: std.mem.Allocator, name: [:0]const u8) !EnvironmentOverride {
        const previous = if (platform.env.getenv(name)) |old| try allocator.dupeZ(u8, old) else null;
        errdefer if (previous) |old| allocator.free(old);
        if (c.unsetenv(name.ptr) != 0) return error.SkipZigTest;
        return .{ .allocator = allocator, .name = name, .previous = previous };
    }

    fn deinit(self: *EnvironmentOverride) void {
        if (self.previous) |old| {
            _ = c.setenv(self.name.ptr, old.ptr, 1);
            self.allocator.free(old);
        } else {
            _ = c.unsetenv(self.name.ptr);
        }
        self.* = undefined;
    }
};

fn elementCount(shape: []const usize) !usize {
    var count: usize = 1;
    for (shape) |dim| count = try std.math.mul(usize, count, dim);
    return count;
}

fn fixtureValue(tensor: FixtureTensor, tensor_index: usize, element_index: usize) f32 {
    if (std.mem.indexOf(u8, tensor.name, "norm.weight") != null or
        std.mem.indexOf(u8, tensor.name, "layernorm.weight") != null)
    {
        return 1.0;
    }
    const bucket: i32 = @intCast((element_index * 7 + tensor_index * 3) % 17);
    return @as(f32, @floatFromInt(bucket - 8)) / 64.0;
}

fn f32ToBf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    const rounded = bits +% (0x7fff + ((bits >> 16) & 1));
    return @intCast(rounded >> 16);
}

fn writeTinyBf16Checkpoint(allocator: std.mem.Allocator, path: []const u8) !void {
    var offsets: [tiny_tensors.len + 1]u64 = undefined;
    offsets[0] = 0;
    for (tiny_tensors, 0..) |tensor, index| {
        offsets[index + 1] = offsets[index] + @as(u64, try elementCount(tensor.shape)) * 2;
    }

    var header: std.Io.Writer.Allocating = .init(allocator);
    defer header.deinit();
    const writer = &header.writer;
    try writer.writeAll("{\"__metadata__\":{\"format\":\"pt\"}");
    for (tiny_tensors, 0..) |tensor, index| {
        try writer.print(",\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{tensor.name});
        for (tensor.shape, 0..) |dim, dim_index| {
            if (dim_index > 0) try writer.writeByte(',');
            try writer.print("{d}", .{dim});
        }
        try writer.print("],\"data_offsets\":[{d},{d}]}}", .{ offsets[index], offsets[index + 1] });
    }
    try writer.writeByte('}');

    const json = header.written();
    const padding = (8 - json.len % 8) % 8;
    const header_size: u64 = json.len + padding;
    var file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    defer file.close(compat.io());
    var file_buffer: [16 * 1024]u8 = undefined;
    var file_writer = file.writer(compat.io(), &file_buffer);
    defer file_writer.end() catch {};
    const out = &file_writer.interface;
    var size_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &size_bytes, header_size, .little);
    try out.writeAll(&size_bytes);
    try out.writeAll(json);
    for (0..padding) |_| try out.writeByte(' ');
    for (tiny_tensors, 0..) |tensor, tensor_index| {
        for (0..try elementCount(tensor.shape)) |element_index| {
            var raw: [2]u8 = undefined;
            std.mem.writeInt(u16, &raw, f32ToBf16Bits(fixtureValue(tensor, tensor_index, element_index)), .little);
            try out.writeAll(&raw);
        }
    }
    try out.flush();
    try file.sync(compat.io());
}

fn expectPublishedStepReport(allocator: std.mem.Allocator, report_path: []const u8) !void {
    const bytes = try compat.cwd().readFileAlloc(compat.io(), report_path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    const Report = struct {
        report: struct {
            epoch_history: []const struct {
                examples_seen: usize,
                supervised_tokens_seen: usize,
                optimizer_steps: usize,
                graph_executor_steps: u64,
                graph_executor_fallback_steps: u64,
                graph_executor_partitions: u64,
                graph_executor_command_dispatches: u64,
                graph_executor_native_partitions: u64,
                graph_executor_unsupported_ops: u64,
                graph_executor_interpreter_fallbacks: u64,
                graph_executor_true_host_outputs: u64,
                metal_optimizer_steps: u64,
            },
        },
    };
    const parsed = try std.json.parseFromSlice(Report, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 1), parsed.value.report.epoch_history.len);
    const epoch = parsed.value.report.epoch_history[0];
    try std.testing.expectEqual(@as(usize, 1), epoch.examples_seen);
    try std.testing.expectEqual(@as(usize, 1), epoch.supervised_tokens_seen);
    try std.testing.expectEqual(@as(usize, 1), epoch.optimizer_steps);
    try std.testing.expectEqual(@as(u64, 1), epoch.metal_optimizer_steps);
    try std.testing.expect(epoch.graph_executor_steps > 0);
    try std.testing.expect(epoch.graph_executor_partitions > 0);
    try std.testing.expect(epoch.graph_executor_command_dispatches > 0);
    try std.testing.expectEqual(@as(u64, 0), epoch.graph_executor_fallback_steps);
    try std.testing.expectEqual(@as(u64, 0), epoch.graph_executor_native_partitions);
    try std.testing.expectEqual(@as(u64, 0), epoch.graph_executor_unsupported_ops);
    try std.testing.expectEqual(@as(u64, 0), epoch.graph_executor_interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), epoch.graph_executor_true_host_outputs);
}

fn expectSavedLoraUpdate(allocator: std.mem.Allocator, checkpoint_path: []const u8) !void {
    var reader = try safetensors.MMapReader.openFileAbsolute(allocator, checkpoint_path);
    defer reader.deinit();
    var tensor = try reader.readTensor("model.layers.0.self_attn.q_proj.weight.lora_B.weight");
    defer tensor.deinit();
    try std.testing.expect(tensor.dtype == .f32);
    try std.testing.expectEqual(@as(usize, 0), tensor.data.len % @sizeOf(f32));
    var updated = false;
    var offset: usize = 0;
    while (offset < tensor.data.len) : (offset += @sizeOf(f32)) {
        const bits = std.mem.readInt(u32, tensor.data[offset..][0..@sizeOf(f32)], .little);
        const value: f32 = @bitCast(bits);
        try std.testing.expect(std.math.isFinite(value));
        if (value != 0.0) updated = true;
    }
    try std.testing.expect(updated);
}

test "gemma4 BF16 strict Metal CLI self-enables executor and publishes one real optimizer step" {
    const required_metal = platform.env.getenvBoolDefault("TERMITE_REQUIRE_METAL_TESTS", false);
    if (comptime !build_options.enable_metal) {
        if (required_metal)
            return error.RequiredMetalBuildDisabled;
        return error.SkipZigTest;
    }
    if (!metal_runtime.metalDeviceAvailable()) {
        if (required_metal)
            return error.RequiredMetalDeviceUnavailable;
        return error.SkipZigTest;
    }
    if (required_metal and build_options.benchmark_source_revision.len != 40) {
        return error.RequiredOracleSourceRevisionNotEmbedded;
    }

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var enabled = try EnvironmentOverride.unset(allocator, "TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR");
    defer enabled.deinit();
    var disabled = try EnvironmentOverride.unset(allocator, "TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR");
    defer disabled.deinit();
    var parity_nodes = try EnvironmentOverride.unset(allocator, "TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_NODE_IDS");
    defer parity_nodes.deinit();
    var parity = try EnvironmentOverride.unset(allocator, "TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK");
    defer parity.deinit();
    var grad_readback = try EnvironmentOverride.unset(allocator, "TERMITE_DEBUG_DEVICE_GRAD_NORM");
    defer grad_readback.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const base_dir = try std.fs.path.join(allocator, &.{ root, "base" });
    defer allocator.free(base_dir);
    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    const prepared_path = try std.fs.path.join(allocator, &.{ root, "prepared.json" });
    defer allocator.free(prepared_path);
    const eval_prepared_path = try std.fs.path.join(allocator, &.{ root, "prepared-eval.json" });
    defer allocator.free(eval_prepared_path);
    const train_source_path = try std.fs.path.join(allocator, &.{ root, "train-source.jsonl" });
    defer allocator.free(train_source_path);
    const eval_source_path = try std.fs.path.join(allocator, &.{ root, "eval-source.jsonl" });
    defer allocator.free(eval_source_path);
    const out_dir = try std.fs.path.join(allocator, &.{ root, "out" });
    defer allocator.free(out_dir);
    try compat.cwd().createDirPath(io, base_dir);

    const config_path = try std.fs.path.join(allocator, &.{ base_dir, "config.json" });
    defer allocator.free(config_path);
    try compat.cwd().writeFile(io, .{ .sub_path = config_path, .data = tiny_config });
    const checkpoint_path = try std.fs.path.join(allocator, &.{ base_dir, finetune.checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    try writeTinyBf16Checkpoint(allocator, checkpoint_path);
    const train_source_json =
        \\{"schema":"gemma_chat/v1","id":"train-row-1","split":"train","messages":[{"role":"user","content":"train prompt"},{"role":"assistant","content":"train answer"}],"metadata":{"source":"bf16-cli-train","group_id":"train-group-1"}}
        \\
    ;
    const eval_source_json =
        \\{"schema":"gemma_chat/v1","id":"eval-row-1","split":"validation","messages":[{"role":"user","content":"eval prompt"},{"role":"assistant","content":"eval answer"}],"metadata":{"source":"bf16-cli-eval","group_id":"eval-group-1"}}
        \\
    ;
    try compat.cwd().writeFile(io, .{ .sub_path = train_source_path, .data = train_source_json });
    try compat.cwd().writeFile(io, .{ .sub_path = eval_source_path, .data = eval_source_json });
    var train_source = try gemma_chat_data.loadExamples(allocator, train_source_path, "train");
    defer train_source.deinit();
    var eval_source = try gemma_chat_data.loadExamples(allocator, eval_source_path, "validation");
    defer eval_source.deinit();
    try std.testing.expectEqual(@as(usize, 1), train_source.examples.len);
    try std.testing.expectEqual(@as(usize, 1), eval_source.examples.len);
    const train_record_digest = try finetune.fingerprintGemmaChatSourceRecordAlloc(allocator, train_source.examples[0]);
    defer allocator.free(train_record_digest);
    const eval_record_digest = try finetune.fingerprintGemmaChatSourceRecordAlloc(allocator, eval_source.examples[0]);
    defer allocator.free(eval_record_digest);

    var bootstrap = try finetune.bootstrapLoRABundle(allocator, base_dir, adapter_dir, .{
        .rank = 2,
        .alpha = 2,
        .gemma4_target_preset = .peft_qv,
    });
    defer finetune.freeBootstrapSummary(allocator, &bootstrap);

    var prompt_ids = [_]i32{ 2, 3, 4 };
    var response_ids = [_]i32{5};
    var input_ids = [_]i32{ 2, 3, 4, 5 };
    var labels = [_]i32{ -100, -100, -100, 5 };
    var examples = [_]finetune.PreparedExampleInput{.{
        .mode = .instruction,
        .prompt_input_ids = &prompt_ids,
        .response_input_ids = &response_ids,
        .num_prompt_tokens = prompt_ids.len,
        .num_response_tokens = response_ids.len,
        .input_ids = &input_ids,
        .labels = &labels,
        .num_input_tokens = input_ids.len,
        .num_supervised_tokens = 1,
        .source_id = train_source.examples[0].id,
        .source_group_id = train_source.examples[0].metadata.group_id,
        .source_name = train_source.examples[0].metadata.source,
        .source_record_sha256 = train_record_digest,
        .rendered_chat_sha256 = "2222222222222222222222222222222222222222222222222222222222222222",
    }};
    var eval_prompt_ids = [_]i32{ 6, 7, 8 };
    var eval_response_ids = [_]i32{9};
    var eval_input_ids = [_]i32{ 6, 7, 8, 9 };
    var eval_labels = [_]i32{ -100, -100, -100, 9 };
    var eval_examples = [_]finetune.PreparedExampleInput{.{
        .mode = .instruction,
        .prompt_input_ids = &eval_prompt_ids,
        .response_input_ids = &eval_response_ids,
        .num_prompt_tokens = eval_prompt_ids.len,
        .num_response_tokens = eval_response_ids.len,
        .input_ids = &eval_input_ids,
        .labels = &eval_labels,
        .num_input_tokens = eval_input_ids.len,
        .num_supervised_tokens = 1,
        .source_id = eval_source.examples[0].id,
        .source_group_id = eval_source.examples[0].metadata.group_id,
        .source_name = eval_source.examples[0].metadata.source,
        .source_record_sha256 = eval_record_digest,
        .rendered_chat_sha256 = "4444444444444444444444444444444444444444444444444444444444444444",
    }};
    var provenance = try finetune.fingerprintGemma4Model(allocator, base_dir);
    defer provenance.deinit(allocator);
    const train_digest = try finetune.fingerprintPreparedExamplesV3Alloc(allocator, &examples);
    defer allocator.free(train_digest);
    const eval_digest = try finetune.fingerprintPreparedExamplesV3Alloc(allocator, &eval_examples);
    defer allocator.free(eval_digest);
    const train_source_digest = try finetune.fingerprintGemmaDatasetSourceAlloc(allocator, train_source_path, "train");
    defer allocator.free(train_source_digest);
    const eval_source_digest = try finetune.fingerprintGemmaDatasetSourceAlloc(allocator, eval_source_path, "validation");
    defer allocator.free(eval_source_digest);
    try finetune.savePreparedInputsSummary(allocator, prepared_path, .{
        .artifact_family_version = finetune.artifact_family_version,
        .model_dir = base_dir,
        .schema_version = finetune.prepared_schema_v6,
        .base_model_sha256 = provenance.base_model_sha256,
        .tokenizer_sha256 = provenance.tokenizer_sha256,
        .chat_template_sha256 = provenance.chat_template_sha256,
        .prepared_examples_sha256 = train_digest,
        .source_dataset_path = train_source_path,
        .source_dataset_sha256 = train_source_digest,
        .source_split = "train",
        .source_revision = train_source_digest,
        .max_examples = 1,
        .examples_seen = 1,
        .max_seq_len = input_ids.len,
        .max_prompt_tokens = prompt_ids.len,
        .max_response_tokens = response_ids.len,
        .max_input_tokens = input_ids.len,
        .max_supervised_tokens = 1,
        .examples = &examples,
    });
    try finetune.savePreparedInputsSummary(allocator, eval_prepared_path, .{
        .artifact_family_version = finetune.artifact_family_version,
        .model_dir = base_dir,
        .schema_version = finetune.prepared_schema_v6,
        .base_model_sha256 = provenance.base_model_sha256,
        .tokenizer_sha256 = provenance.tokenizer_sha256,
        .chat_template_sha256 = provenance.chat_template_sha256,
        .prepared_examples_sha256 = eval_digest,
        .source_dataset_path = eval_source_path,
        .source_dataset_sha256 = eval_source_digest,
        .source_split = "validation",
        .source_revision = eval_source_digest,
        .max_examples = 1,
        .examples_seen = 1,
        .max_seq_len = eval_input_ids.len,
        .max_prompt_tokens = eval_prompt_ids.len,
        .max_response_tokens = eval_response_ids.len,
        .max_input_tokens = eval_input_ids.len,
        .max_supervised_tokens = 1,
        .examples = &eval_examples,
    });

    const args = [_][]const u8{
        base_dir,
        adapter_dir,
        prepared_path,
        out_dir,
        "--trainer",
        "autodiff",
        "--backend",
        "metal",
        "--lr",
        "0.001",
        "--max-examples",
        "1",
        "--eval-prepared",
        eval_prepared_path,
        "--eval-max-examples",
        "1",
        "--epochs",
        "1",
        "--grad-accum",
        "1",
    };
    try train_command.runFromArgsWithoutSummary(allocator, io, &args);

    const report_path = try std.fs.path.join(allocator, &.{ out_dir, "training_report.json" });
    defer allocator.free(report_path);
    try expectPublishedStepReport(allocator, report_path);
    const run_manifest_path = try std.fs.path.join(allocator, &.{ out_dir, "run_manifest.json" });
    defer allocator.free(run_manifest_path);
    try compat.cwd().access(io, run_manifest_path, .{});
    const output_adapter_path = try std.fs.path.join(allocator, &.{ out_dir, finetune.adapter_checkpoint_file_name });
    defer allocator.free(output_adapter_path);
    try expectSavedLoraUpdate(allocator, output_adapter_path);

    const published_before = try compat.cwd().readFileAlloc(io, output_adapter_path, allocator, .limited(1024 * 1024));
    defer allocator.free(published_before);
    try std.testing.expectError(error.Gemma4RunOutputAlreadyExists, train_command.runFromArgsWithoutSummary(allocator, io, &args));
    const published_after = try compat.cwd().readFileAlloc(io, output_adapter_path, allocator, .limited(1024 * 1024));
    defer allocator.free(published_after);
    try std.testing.expectEqualSlices(u8, published_before, published_after);

    // Exercise the private request/capture boundary as an actual compiled
    // Metal step. The external Python packager independently validates and
    // converts these files into gemma4_oracle_trace/v1.
    if (build_options.benchmark_source_revision.len == 40) {
        const oracle_request_path = try std.fs.path.join(allocator, &.{ root, "oracle-request.json" });
        defer allocator.free(oracle_request_path);
        const oracle_out_dir = try std.fs.path.join(allocator, &.{ root, "oracle-out" });
        defer allocator.free(oracle_out_dir);
        const oracle_capture_dir = try std.fs.path.join(allocator, &.{ root, "oracle-capture" });
        defer allocator.free(oracle_capture_dir);
        const executable_path = try std.process.executablePathAlloc(io, allocator);
        defer allocator.free(executable_path);
        var executable_fingerprint = try finetune.fingerprintProjectorFile(allocator, executable_path);
        defer finetune.freeProjectorFingerprint(allocator, &executable_fingerprint);
        var prepared_fingerprint = try finetune.fingerprintProjectorFile(allocator, prepared_path);
        defer finetune.freeProjectorFingerprint(allocator, &prepared_fingerprint);
        const initial_adapter_path = try std.fs.path.join(allocator, &.{ adapter_dir, finetune.adapter_checkpoint_file_name });
        defer allocator.free(initial_adapter_path);
        var initial_adapter_fingerprint = try finetune.fingerprintProjectorFile(allocator, initial_adapter_path);
        defer finetune.freeProjectorFingerprint(allocator, &initial_adapter_fingerprint);
        const executable_sha256 = try std.fmt.allocPrint(allocator, "sha256:{s}", .{executable_fingerprint.sha256});
        defer allocator.free(executable_sha256);
        const prepared_sha256 = try std.fmt.allocPrint(allocator, "sha256:{s}", .{prepared_fingerprint.sha256});
        defer allocator.free(prepared_sha256);
        const adapter_sha256 = try std.fmt.allocPrint(allocator, "sha256:{s}", .{initial_adapter_fingerprint.sha256});
        defer allocator.free(adapter_sha256);
        const request_payload = .{
            .schema_version = "antfly_gemma4_lora_zig_oracle_request/v1",
            .implementation = .{
                .version = build_options.inference_version,
                .executable_sha256 = executable_sha256,
                .source_revision = build_options.benchmark_source_revision,
                .backend = "metal",
                .metal_device = "unit-test-metal-device",
            },
            .bindings = .{
                .oracle_lock_sha256 = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                .model_key = "gemma-4-E2B-it",
                .model_revision = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                .local_artifact_sha256 = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                .base_model_sha256 = provenance.base_model_sha256,
                .initial_adapter_sha256 = adapter_sha256,
                .train_prepared_sha256 = prepared_sha256,
                .source_dataset_sha256 = train_source_digest,
                .example_index = @as(usize, 0),
                .target_preset = "peft-qv",
                .rank = @as(usize, 2),
                .alpha = @as(f64, 2),
                .target_count = @as(usize, 2),
            },
            .training = .{
                .optimizer = "adamw",
                .seed = @as(u64, 42),
                .steps = @as(usize, 1),
                .learning_rate = @as(f64, 0.001),
                .betas = [_]f64{ 0.9, 0.999 },
                .eps = @as(f64, 1e-8),
                .weight_decay = @as(f64, 0.01),
                .max_grad_norm = @as(f64, 1.0),
                .grad_accum_steps = @as(u32, 1),
                .supervised_token_normalization = "mean",
                .dropout = @as(f64, 0),
                .use_cache = false,
            },
        };
        const request_json = try std.json.Stringify.valueAlloc(allocator, request_payload, .{});
        defer allocator.free(request_json);
        try compat.cwd().writeFile(io, .{ .sub_path = oracle_request_path, .data = request_json });
        const oracle_args = [_][]const u8{
            base_dir,
            adapter_dir,
            prepared_path,
            oracle_out_dir,
            "--trainer",
            "autodiff",
            "--backend",
            "metal",
            "--lr",
            "0.001",
            "--max-examples",
            "1",
            "--eval-prepared",
            prepared_path,
            "--eval-max-examples",
            "1",
            "--epochs",
            "1",
            "--grad-accum",
            "1",
            "--seed",
            "42",
            "--oracle-request",
            oracle_request_path,
            "--oracle-capture-out",
            oracle_capture_dir,
        };
        try train_command.runFromArgsWithoutSummary(allocator, io, &oracle_args);

        inline for (.{
            "capture.json",
            "raw_gradients.safetensors",
            "trainer_checkpoint.safetensors",
            "run_manifest.json",
        }) |name| {
            const path = try std.fs.path.join(allocator, &.{ oracle_capture_dir, name });
            defer allocator.free(path);
            try compat.cwd().access(io, path, .{});
        }
        const capture_path = try std.fs.path.join(allocator, &.{ oracle_capture_dir, "capture.json" });
        defer allocator.free(capture_path);
        const capture_bytes = try compat.cwd().readFileAlloc(io, capture_path, allocator, .limited(1024 * 1024));
        defer allocator.free(capture_bytes);
        const Captured = struct {
            schema_version: []const u8,
            result: struct {
                loss_history: []const f64,
                raw_gradient_norm: f64,
                supervised_tokens: usize,
                logit_probes: []const std.json.Value,
                targets: []const std.json.Value,
                execution: struct {
                    optimizer_steps: u64,
                    micro_batch_steps: u64,
                    metal_optimizer_steps: u64,
                    graph_executor_steps: u64,
                },
            },
        };
        const captured = try std.json.parseFromSlice(Captured, allocator, capture_bytes, .{ .ignore_unknown_fields = true });
        defer captured.deinit();
        try std.testing.expectEqualStrings("antfly_gemma4_lora_zig_oracle_capture/v1", captured.value.schema_version);
        try std.testing.expectEqual(@as(usize, 1), captured.value.result.loss_history.len);
        try std.testing.expect(std.math.isFinite(captured.value.result.loss_history[0]));
        try std.testing.expect(captured.value.result.raw_gradient_norm > 0);
        try std.testing.expectEqual(@as(usize, 1), captured.value.result.supervised_tokens);
        try std.testing.expectEqual(@as(usize, 1), captured.value.result.logit_probes.len);
        try std.testing.expectEqual(@as(usize, 4), captured.value.result.targets.len);
        try std.testing.expectEqual(@as(u64, 1), captured.value.result.execution.optimizer_steps);
        try std.testing.expectEqual(@as(u64, 1), captured.value.result.execution.micro_batch_steps);
        try std.testing.expectEqual(@as(u64, 1), captured.value.result.execution.metal_optimizer_steps);
        try std.testing.expectEqual(@as(u64, 1), captured.value.result.execution.graph_executor_steps);
    }
}
