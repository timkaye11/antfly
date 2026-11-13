// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Retained-input adapter for test-only native whole-head quality batches.
//! Preparation artifacts are re-tokenized using the pinned loaded model.
const std = @import("std");
const generation = @import("pipelines/generation.zig");
const quality = @import("runtime/kv/metal_checkpoint_quality.zig");
const ordered = @import("context_serving_messages.zig");
const serving = @import("context_serving_driver.zig");
const server = @import("server/server.zig");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const runtime = @import("runtime/root.zig");
const context_study_templates = @import("backends/context_study_templates.zig");

pub const File = struct { path: []const u8, sha256: []const u8 };
pub const Case = struct {
    id: []const u8,
    bucket: []const u8,
    messages: File,
    rendered_prompt: File,
    prompt_tokens: File,
    prompt_count: usize,
    reference_utf8: File,
    reference_tokens: File,
    reference_count: usize,
    reference_sequence_sha256: []const u8,
    request: quality.Request,
};
pub const Input = struct {
    schema: []const u8,
    model: []const u8,
    main_gguf_sha256: []const u8,
    native_identity: checkpoint.Identity,
    cases: []const Case,
};

pub fn parseDigest(raw: []const u8) ![32]u8 {
    if (raw.len != 64) return error.InvalidNativeQualityDigest;
    var result: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&result, raw) catch return error.InvalidNativeQualityDigest;
    if (std.mem.allEqual(u8, &result, 0)) return error.InvalidNativeQualityDigest;
    return result;
}

fn digest(raw: []const u8) [32]u8 {
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &result, .{});
    return result;
}

pub fn validPath(path: []const u8) bool {
    if (path.len == 0 or std.fs.path.isAbsolute(path)) return false;
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part|
        if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    return true;
}

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    inputs: std.Io.Dir,
    outputs: std.Io.Dir,
    model: *server.model_manager.LoadedModel,
    pipeline: *generation.NativeGenerationPipeline,
    manifest_sha256: [32]u8,
    capture_accepted_snapshot: bool = false,
    staged_private_policy: bool = false,
    staged_future_training_policy: bool = false,
    staged_resident_hybrid_policy: bool = false,
    resident_hybrid_authority: ?generation.ResidentHybridAuthority = null,
    route_parity_diagnostic: bool = false,
    supplied_input: ?@import("context_supplied_checkpoint_quality.zig").Input = null,
    supplied_root: ?std.Io.Dir = null,
    prepared_input: ?@import("context_prepared_serving_quality.zig").Input = null,
    prepared_root: ?std.Io.Dir = null,
    prepared_producer: ?@import("server/native_kv_verified_sidecar.zig").ExpectedProducer = null,

    fn read(self: *Session, file_ref: File) ![]u8 {
        if (!validPath(file_ref.path)) return error.InvalidNativeQualityPath;
        const expected = try parseDigest(file_ref.sha256);
        var dir = self.inputs;
        var owned = false;
        defer if (owned) dir.close(self.io);
        var parts = std.mem.splitScalar(u8, file_ref.path, '/');
        var part = parts.next() orelse return error.InvalidNativeQualityPath;
        while (parts.next()) |next| {
            const child = try dir.openDir(self.io, part, .{ .follow_symlinks = false });
            if (owned) dir.close(self.io);
            dir = child;
            owned = true;
            part = next;
        }
        const file = try dir.openFile(self.io, part, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false, .resolve_beneath = true });
        defer file.close(self.io);
        const stat = try file.stat(self.io);
        if (stat.kind != .file or stat.size == 0 or stat.size > 16 * 1024 * 1024) return error.InvalidNativeQualityPath;
        const result = try self.allocator.alloc(u8, @intCast(stat.size));
        errdefer self.allocator.free(result);
        if (try file.readPositionalAll(self.io, result, 0) != result.len or
            !std.mem.eql(u8, &digest(result), &expected)) return error.NativeQualityInputChanged;
        return result;
    }

    fn readTokens(self: *Session, file_ref: File, count: usize) ![]i64 {
        const bytes = try self.read(file_ref);
        defer self.allocator.free(bytes);
        if (count == 0 or count > checkpoint.max_context_tokens or bytes.len != try std.math.mul(usize, count, 8))
            return error.InvalidNativeQualityTokens;
        const result = try self.allocator.alloc(i64, count);
        errdefer self.allocator.free(result);
        for (result, 0..) |*token, i| {
            token.* = std.mem.readInt(i64, bytes[i * 8 ..][0..8], .little);
            if (token.* < 0 or token.* >= self.pipeline.gpt_config.vocab_size) return error.InvalidNativeQualityTokens;
        }
        return result;
    }

    const Gold = struct {
        session: *Session,
        case: Case,
        prompt: []const u8,
        prompt_ids: []const i64,
        tokens: ?[]i64 = null,
        calls: usize = 0,

        fn reveal(raw: *anyopaque) ![]const i64 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (self.calls != 0 or self.tokens != null) return error.NativeQualityReferenceAlreadyRevealed;
            self.calls += 1;
            const session = self.session;
            const model = session.model;
            const text = try session.read(self.case.reference_utf8);
            defer session.allocator.free(text);
            if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidNativeQualityReference;
            const tokens = try session.readTokens(self.case.reference_tokens, self.case.reference_count);
            errdefer session.allocator.free(tokens);
            const combined = try std.mem.concat(session.allocator, u8, &.{ self.prompt, text });
            defer session.allocator.free(combined);
            var encoded = try generation.encodeNativeGenerationPrompt(model.getTokenizer(), session.allocator, combined, self.prompt_ids.len + self.case.request.output_tokens, model.manifest.add_bos_token, model.manifest.bos_token);
            defer encoded.deinit();
            var count: usize = 0;
            while (count < encoded.attention_mask.len and encoded.attention_mask[count] != 0) : (count += 1) {}
            if (count != self.prompt_ids.len + tokens.len) return error.UnstableQualityReferenceBoundary;
            for (self.prompt_ids, encoded.ids[0..self.prompt_ids.len]) |expected, actual|
                if (expected != actual) return error.UnstableQualityReferenceBoundary;
            for (tokens, encoded.ids[self.prompt_ids.len..count]) |expected, actual|
                if (expected != actual) return error.NativeQualityInputChanged;
            const signed = try session.allocator.alloc(i32, tokens.len);
            defer session.allocator.free(signed);
            for (tokens, signed) |token, *id| id.* = @intCast(token);
            const decoded = try model.getTokenizer().decode(session.allocator, signed);
            defer session.allocator.free(decoded);
            if (!std.mem.eql(u8, decoded, text)) return error.QualityReferenceTokenizerMismatch;
            self.tokens = tokens;
            return tokens;
        }
    };

    pub fn execute(self: *Session, case: Case, index: usize) !void {
        if (!@import("builtin").is_test or case.id.len == 0 or case.id.len > 256 or
            (case.prompt_count != 7936 and case.prompt_count != 32512) or
            case.request.output_tokens != quality.max_output_tokens or case.reference_count < 2 or case.reference_count > quality.max_output_tokens)
            return error.InvalidNativeQualityRequest;
        _ = try case.request.capacity(case.prompt_count, self.pipeline.gpt_config.max_position_embeddings);
        const message_bytes = try self.read(case.messages);
        defer self.allocator.free(message_bytes);
        var messages = try ordered.parseOwned(self.allocator, message_bytes);
        defer messages.deinit();
        const semantic = try ordered.semanticDigest(messages.value);
        var message_storage: [ordered.max_messages]generation.Message = undefined;
        var parts: [ordered.max_messages * ordered.max_parts_per_message]generation.Message.ContentPart = undefined;
        var calls: [ordered.max_total_tool_calls]generation.Message.ToolCall = undefined;
        const serving_request: serving.Request = .{
            .route = .native,
            .variant = .attention_matching,
            .model = "qualification",
            .prompt = "",
            .ordered = .{ .artifact = messages.value, .semantic_sha256 = semantic },
            .max_context_tokens = @intCast(case.prompt_count + quality.max_output_tokens),
            .max_kv_tokens = @intCast((case.prompt_count + quality.max_output_tokens) / 2),
            .keep_prefix_tokens = 512,
            .keep_recent_tokens = 512,
            .reserved_output_tokens = quality.max_output_tokens,
        };
        const native = try serving.Driver.nativeMessages(serving_request, &message_storage, &parts, &calls);
        const prompt = try (self.model.chat_tmpl orelse return error.MissingChatTemplate).applyWithOptions(self.allocator, native, .{ .enable_thinking = false });
        defer self.allocator.free(prompt);
        const expected_prompt = try self.read(case.rendered_prompt);
        defer self.allocator.free(expected_prompt);
        if (!std.mem.eql(u8, prompt, expected_prompt)) return error.NativeQualityInputChanged;
        const tokens = try self.readTokens(case.prompt_tokens, case.prompt_count);
        defer self.allocator.free(tokens);
        var encoded = try generation.encodeNativeGenerationPrompt(self.model.getTokenizer(), self.allocator, prompt, case.prompt_count + quality.max_output_tokens, self.model.manifest.add_bos_token, self.model.manifest.bos_token);
        defer encoded.deinit();
        var count: usize = 0;
        while (count < encoded.attention_mask.len and encoded.attention_mask[count] != 0) : (count += 1) {}
        if (count != case.prompt_count) return error.NativeQualityInputChanged;
        for (tokens, encoded.ids[0..count]) |expected, actual|
            if (expected != actual) return error.NativeQualityInputChanged;
        var supplied_loaded: ?@import("context_supplied_checkpoint_quality.zig").Loaded = null;
        defer if (supplied_loaded) |*loaded| loaded.deinit();
        if (self.supplied_input) |supplied_input| {
            const root = self.supplied_root orelse return error.InvalidSuppliedCheckpointQualityInput;
            supplied_loaded = try @import("context_supplied_checkpoint_quality.zig").load(self.io, self.allocator, root, supplied_input, self.pipeline.attention_match_identity orelse return error.UnsupportedKvIdentity, tokens);
        }
        var prepared_loaded: ?@import("context_prepared_serving_quality.zig").Loaded = null;
        defer if (prepared_loaded) |*loaded| loaded.deinit();
        if (self.prepared_input) |prepared_input| {
            if (supplied_loaded != null) return error.InvalidPreparedServingQualityInput;
            const root = self.prepared_root orelse return error.InvalidPreparedServingQualityInput;
            const producer = self.prepared_producer orelse return error.InvalidPreparedServingQualityInput;
            prepared_loaded = try @import("context_prepared_serving_quality.zig").load(
                self.io,
                self.allocator,
                root,
                prepared_input,
                self.pipeline.attention_match_identity orelse return error.UnsupportedKvIdentity,
                producer,
                tokens,
            );
        }
        var gold: Gold = .{ .session = self, .case = case, .prompt = prompt, .prompt_ids = tokens };
        defer if (gold.tokens) |ids| self.allocator.free(ids);
        const self_study_enabled = @import("builtin").is_test and std.c.getenv("ANTFLY_METAL_QUALITY_SELF_STUDY_TRAINING_DIR") != null;
        var study_payload: ?generation.PreparedCompactionPayload = null;
        defer if (study_payload) |*owned| owned.deinit();
        var fit_study: ?generation.PreparedCompactionStudy = null;
        defer if (fit_study) |*owned| owned.deinit();
        if (self_study_enabled) {
            const policy = try serving.Driver.policy(serving_request);
            study_payload = try generation.prepareCompactionPayload(self.allocator, self.model.chat_tmpl.?, self.model.getTokenizer(), native, .{ .enable_thinking = false }, policy, case.prompt_count + quality.max_output_tokens, self.model.manifest.add_bos_token);
            if (!std.mem.eql(u8, study_payload.?.rendered.text, prompt)) return error.NativeQualityInputChanged;
            fit_study = try generation.prepareCompactionStudy(self.allocator, self.model.chat_tmpl.?, self.model.getTokenizer(), native, .{ .enable_thinking = false }, &study_payload.?, .fit_training_v1, context_study_templates.authority(.fit_training_v1), self.pipeline.gpt_config.max_position_embeddings, self.model.manifest.add_bos_token);
            const suffix_count = fit_study.?.ids.len;
            const repeat_count = try std.math.add(usize, suffix_count, context_study_templates.response_tokens);
            std.debug.print("attention_match_self_study_preflight original_count={d} suffix_count={d} generated_count={d} repeat_count={d} template_variant=fit_training_v1\n", .{ case.prompt_count, suffix_count, context_study_templates.response_tokens, repeat_count });
            if (suffix_count < 64 or repeat_count < 320) return error.SelfStudyRepeatTooShort;
        }
        const reference: quality.Reference = .{
            .count = case.reference_count,
            .sha256 = try parseDigest(case.reference_sequence_sha256),
            .provider = .{ .ctx = &gold, .reveal_fn = Gold.reveal },
        };
        var quality_request = case.request;
        if (self.staged_private_policy) {
            quality_request.fit_strategy = .identity_chunk_coefficients_adaptive_v18;
            quality_request.admission = .cumulative_ladder_4_3_2;
            quality_request.chunk_rows = 256;
            quality_request.reject_exact_active_fixed_point = true;
            quality_request.private_dense512_policy = true;
            quality_request.private_future_decode_training_v1 = self.staged_future_training_policy;
            quality_request.private_resident_hybrid_policy = self.staged_resident_hybrid_policy;
            quality_request.resident_hybrid_authority = if (self.resident_hybrid_authority) |authority| .{
                .configuration_sha256 = authority.configuration_sha256,
                .source_freeze_sha256 = authority.source_freeze_sha256,
                .shape_authority_sha256 = authority.shape_authority_sha256,
                .policy_module_sha256 = authority.policy_module_sha256,
                .e2_guarded_development_trial = authority.e2_guarded_development_trial,
                .e2_fit_source_rows = authority.e2_fit_source_rows,
                .e2_formal_owner_inventory_sha256 = authority.e2_formal_owner_inventory_sha256,
                .e2_formal_owner_inventory = authority.e2_formal_owner_inventory,
                .e4_formal_authority = authority.e4_formal_authority,
                .cohort = switch (authority.cohort) {
                    .e4_8128_v1 => .e4_8128_v1,
                    .e2_8128_v1 => .e2_8128_v1,
                    .e4_formal_7936_256_v1 => .e4_formal_7936_256_v1,
                    .e4_formal_32512_256_v1 => .e4_formal_32512_256_v1,
                },
            } else null;
        }
        quality_request.route_parity_diagnostic = self.route_parity_diagnostic;
        if (comptime @import("builtin").is_test) if (std.c.getenv("ANTFLY_METAL_QUALITY_CUMULATIVE_ADMISSION")) |raw| {
            if (!std.mem.eql(u8, std.mem.span(raw), "1")) return error.InvalidNativeQualityRequest;
            quality_request.admission = .cumulative_ladder_4_3_2;
        };
        if (self.capture_accepted_snapshot and (supplied_loaded != null or prepared_loaded != null or quality_request.admission != .cumulative_ladder_4_3_2 or quality_request.encoding != .f16))
            return error.InvalidNativeQualityRequest;
        var prepared_candidate: ?quality.PreparedServingCandidate = if (prepared_loaded) |*loaded| try loaded.prepared.takeCandidate() else null;
        defer if (prepared_candidate) |*candidate| candidate.deinit();
        var report = if (prepared_candidate) |*candidate|
            try self.pipeline.qualifyPreparedServingMetalCheckpointQuality(tokens, prompt, quality_request, reference, candidate)
        else if (supplied_loaded) |*loaded|
            try self.pipeline.qualifySuppliedMetalCheckpointQuality(tokens, prompt, quality_request, reference, try loaded.candidate(self.pipeline.attention_match_identity orelse return error.UnsupportedKvIdentity, tokens))
        else if (self.capture_accepted_snapshot)
            try self.pipeline.qualifyMetalCheckpointQualityCapturing(tokens, prompt, quality_request, reference, if (fit_study) |*owned| owned else null, .{ .output = self.outputs })
        else
            try self.pipeline.qualifyMetalCheckpointQuality(tokens, prompt, quality_request, reference, if (fit_study) |*owned| owned else null);
        defer report.deinit();
        if (gold.calls != 1 or !report.collection_complete or !report.reference_revealed_after_both_greedy_branches)
            return error.IncompleteNativeQualityCollection;
        try report.validateV18Evidence();
        const body = try std.json.Stringify.valueAlloc(self.allocator, .{
            .schema = "antfly.native-checkpoint-quality.case.v2",
            .allocation_scope = "private_native_kv_owners",
            .full_serving_owner_qualified = false,
            .release_eligible = false,
            .collection_complete = report.collection_complete,
            .case_index = index,
            .case = case,
            .manifest_sha256 = self.manifest_sha256,
            .load_generation = self.model.load_generation,
            .frozen = report.frozen,
            .calibration_consumers = if (report.frozen.calibration) |calibration| report.calibration_consumers[0..calibration.consumer_count] else &.{},
            .phase_durations_ns = report.phase_durations_ns,
            .v18_evidence_profile = @tagName(report.v18_evidence_profile),
            .v18_evidence_profile_sha256 = std.fmt.bytesToHex(report.v18_evidence_profile_sha256, .lower),
            .v18_codec_physical_profile = @tagName(report.v18_codec_physical_profile),
            .v18_codec_physical_profile_sha256 = std.fmt.bytesToHex(report.v18_codec_physical_profile_sha256, .lower),
            .v18_bounded_tsqr_policy_sha256 = std.fmt.bytesToHex(report.v18_bounded_tsqr_policy_sha256, .lower),
            .v18_bounded_tsqr_heads = report.v18_bounded_tsqr_heads,
            .v18_bounded_tsqr_calls = report.v18_bounded_tsqr_calls,
            .v18_bounded_tsqr_workspace_peak_bytes = report.v18_bounded_tsqr_workspace_peak_bytes,
            .v18_bounded_tsqr_cancellation_checks = report.v18_bounded_tsqr_cancellation_checks,
            .primary_query_collection = report.primary_query_collection,
            .dense512_query_collection = report.dense512_query_collection,
            .query_collection_combined_before_fit_ns = report.query_collection_combined_before_fit_ns,
            .future_training_authority = report.future_training_authority,
            .source_serving_prefill = report.source_serving_prefill,
            .source_route = report.source_route,
            .source_prefill_full_chunk_rows = report.source_prefill_full_chunk_rows,
            .source_prefill_full_chunk_count = report.source_prefill_full_chunk_count,
            .source_prefill_tail_rows = report.source_prefill_tail_rows,
            .source_prefill_final_token_rows = report.source_prefill_final_token_rows,
            .source_prefill_e4_fused_half_attempts = report.source_prefill_e4_fused_half_attempts,
            .source_prefill_e4_fused_half_hits = report.source_prefill_e4_fused_half_hits,
            .source_prefill_e4_fused_half_pair_dispatches = report.source_prefill_e4_fused_half_pair_dispatches,
            .source_prefill_e4_fused_half_down_dispatches = report.source_prefill_e4_fused_half_down_dispatches,
            .source_prefill_e4_fused_half_fallbacks = report.source_prefill_e4_fused_half_fallbacks,
            .source_prefill_e4_fused_half_fatal_failures = report.source_prefill_e4_fused_half_fatal_failures,
            .collection_elapsed_ns = report.collection_elapsed_ns,
            .model_slot_runtime_kv_bytes_before = report.model_slot_runtime_kv_bytes_before,
            .model_slot_runtime_kv_bytes_after = report.model_slot_runtime_kv_bytes_after,
            .continuation_runtime_kv_bytes = report.continuation_runtime_kv_bytes,
            .compacted_greedy_runtime_kv_bytes_before = report.compacted_greedy_runtime_kv_bytes_before,
            .compacted_greedy_runtime_kv_bytes_after = report.compacted_greedy_runtime_kv_bytes_after,
            .compacted_teacher_runtime_kv_bytes_before = report.compacted_teacher_runtime_kv_bytes_before,
            .compacted_teacher_runtime_kv_bytes_after = report.compacted_teacher_runtime_kv_bytes_after,
            .model_slots_prepared_without_kv_allocation = report.model_slots_prepared_without_kv_allocation,
            .exact_greedy = report.exact_greedy,
            .compacted_greedy = report.compacted_greedy,
            .exact_teacher = report.exact_teacher,
            .compacted_teacher = report.compacted_teacher,
            .exact_raw_greedy_diagnostic = report.exact_raw_greedy_diagnostic,
            .exact_prepared_teacher_diagnostic = report.exact_prepared_teacher_diagnostic,
            .route_parity_diagnostic_unqualified = report.route_parity_diagnostic_unqualified,
            .reference_sha256 = report.reference_sha256,
            .reference_revealed_after_both_greedy_branches = report.reference_revealed_after_both_greedy_branches,
            .reference_provider_calls = gold.calls,
            .source_unchanged = report.source_unchanged,
            .source_owner_id = report.source_owner_id,
            .compacted_owner_id = report.compacted_owner_id,
            .query_scope_released = report.query_scope_released,
            .query_scope_delta_bytes = report.query_scope_delta_bytes,
            .admitted_peak_bytes = report.admitted_peak_bytes,
            .source_global_bytes = report.source_global_bytes,
            .candidate_global_bytes = report.candidate_global_bytes,
            .max_head_mass_error = report.max_head_mass_error,
            .max_head_output_rmse = report.max_head_output_rmse,
            .fp64_accelerated_value_solves = report.fp64_accelerated_value_solves,
            .candidate_source = report.candidate_source,
            .fit_calls = report.fit_calls,
            .calibration_executed = report.calibration_executed,
            .admission_strategy = @tagName(report.admission),
            .chunk_rows = report.chunk_rows,
            .reject_exact_active_fixed_point = report.reject_exact_active_fixed_point,
            .fit_strategy = @tagName(report.fit_strategy),
            .v18_policy_sha256 = report.v18_policy_sha256,
            .v18_eligible_heads = report.v18_eligible_heads,
            .v18_attempted_heads = report.v18_attempted_heads,
            .v18_accepted_heads = report.v18_accepted_heads,
            .v18_preserved_heads = report.v18_preserved_heads,
            .v18_fallback_heads = report.v18_fallback_heads,
            .v18_cache_ready_heads = report.v18_cache_ready_heads,
            .v18_planned_exact_heads = report.v18_planned_exact_heads,
            .v18_query_batch_rows = report.v18_query_batch_rows,
            .v18_planned_head_workspace_bytes = report.v18_planned_head_workspace_bytes,
            .v18_cache_phase_peak_bound_bytes = report.v18_cache_phase_peak_bound_bytes,
            .supplied_closure_sha256 = report.supplied_closure_sha256,
            .accepted_snapshot = report.accepted_snapshot,
            .source_release = report.source_release,
            .compacted_release = report.compacted_release,
        }, .{ .whitespace = .indent_2 });
        defer self.allocator.free(body);
        var filename: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&filename, "case-{d:0>4}.json", .{index});
        var output = try self.outputs.createFile(self.io, name, .{ .exclusive = true, .truncate = false });
        defer output.close(self.io);
        try output.writeStreamingAll(self.io, body);
        try output.writeStreamingAll(self.io, "\n");
        try output.sync(self.io);
        std.debug.print("native_checkpoint_quality_completed index={d} case={s} encoding={s} exact_tokens={d} compacted_tokens={d} reference_tokens={d}\n", .{ index, case.id, @tagName(case.request.encoding), report.exact_greedy.count, report.compacted_greedy.count, report.exact_teacher.count });
    }
};

test "attention_match Metal native quality input paths and hashes fail closed" {
    for ([_][]const u8{ "", "/absolute", "a/../b", "./x", "a//b", "a/" }) |path|
        try std.testing.expect(!validPath(path));
    try std.testing.expect(validPath("prepared/case/messages.json"));
    try std.testing.expectError(error.InvalidNativeQualityDigest, parseDigest("abc"));
    try std.testing.expectError(error.InvalidNativeQualityDigest, parseDigest(&@as([64]u8, @splat('0'))));
}
