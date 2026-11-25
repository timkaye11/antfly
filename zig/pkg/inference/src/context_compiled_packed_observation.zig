const std = @import("std");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");

pub const schema_name = "antfly.compiled-afkv-exact-kv-equivalence.v3";
pub const export_abi_name = "ExactLayerRangeExporter.v1";

pub const SourceAuthority = struct {
    model_weights_sha256: [32]u8,
    model_identity_layout_sha256: [32]u8,
    context_layout_sha256: [32]u8,
    token_stream_sha256: [32]u8,
    source_revision: u64,
    source_sequence: u32,
    logical_tokens: usize,
};

pub const Document = struct {
    schema: []const u8,
    diagnostic_only: bool,
    qualification_credit: bool,
    serving_activation: bool,
    compiled_prefill_completed: bool,
    exact_export_completed: bool,
    exact_export_abi: []const u8,
    exact_export_peak_host_staging_bytes: usize,
    source_checkpoint_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    test_binary_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    model_identity: checkpoint.Identity,
    process_rss_limit_bytes: usize,
    layout_census: ?struct {
        attention_layers: u32,
        owner_count: u32,
        context_layout_sha256: [32]u8,
        owner_projection_sha256: [32]u8,
    } = null,
    result: struct {
        source_authority: SourceAuthority,
        copied_bytes: usize,
        imported_owners: usize,
        source_unchanged: bool,
        release_complete: bool,
        compared_owners: u32,
        total_key_mismatches: u64,
        total_value_mismatches: u64,
        all_f16_identical: bool,
    },
};

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(Document) {
    return std.json.parseFromSlice(Document, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }) catch error.InvalidCompiledPackedObservation;
}

test "compiled packed observation producer and loader share exact JSON wire" {
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const document: Document = .{
        .schema = schema_name,
        .diagnostic_only = true,
        .qualification_credit = false,
        .serving_activation = false,
        .compiled_prefill_completed = true,
        .exact_export_completed = true,
        .exact_export_abi = export_abi_name,
        .exact_export_peak_host_staging_bytes = 0,
        .source_checkpoint_sha256 = @splat(5),
        .source_freeze_sha256 = @splat(6),
        .test_binary_sha256 = @splat(7),
        .implementation_semantic_sha256 = @splat(8),
        .model_identity = identity,
        .process_rss_limit_bytes = 8 * 1024 * 1024 * 1024,
        .layout_census = .{ .attention_layers = 42, .owner_count = 24, .context_layout_sha256 = @splat(9), .owner_projection_sha256 = @splat(11) },
        .result = .{
            .source_authority = .{ .model_weights_sha256 = identity.weights, .model_identity_layout_sha256 = identity.layout, .context_layout_sha256 = @splat(9), .token_stream_sha256 = @splat(10), .source_revision = 11, .source_sequence = 12, .logical_tokens = 8128 },
            .copied_bytes = 13,
            .imported_owners = 24,
            .source_unchanged = true,
            .release_complete = true,
            .compared_owners = 24,
            .total_key_mismatches = 0,
            .total_value_mismatches = 0,
            .all_f16_identical = true,
        },
    };
    const bytes = try std.json.Stringify.valueAlloc(std.testing.allocator, document, .{});
    defer std.testing.allocator.free(bytes);
    var parsed = try parse(std.testing.allocator, bytes);
    defer parsed.deinit();
    const roundtrip = try std.json.Stringify.valueAlloc(std.testing.allocator, parsed.value, .{});
    defer std.testing.allocator.free(roundtrip);
    try std.testing.expectEqualStrings(bytes, roundtrip);
}
