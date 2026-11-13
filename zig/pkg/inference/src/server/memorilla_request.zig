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

const std = @import("std");
const platform = @import("antfly_platform");
const grounding = @import("../pipelines/memorilla_grounding.zig");

pub const ContextMode = enum { latent, source_verified };
pub const ReaderKind = platform.inference_settings.MemorillaProfile.ReaderKind;

// Ordinary Zig numeric parsing permits quoted numbers. The API wire contract
// requires actual numeric tokens, including inside each feature vector.
fn JsonNumber(comptime T: type) type {
    return struct {
        value: T,
        pub fn jsonParse(a: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
            if (try source.peekNextTokenType() != .number) return error.UnexpectedToken;
            return .{ .value = try std.json.innerParse(T, a, source, options) };
        }
    };
}

pub const SourceTexts = struct {
    value: ?[]const []const u8 = null,

    pub fn jsonParse(a: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
        if (try source.peekNextTokenType() != .array_begin) return error.UnexpectedToken;
        return .{ .value = try std.json.innerParse([]const []const u8, a, source, options) };
    }
};

pub const Request = struct {
    model: []const u8,
    question: []const u8,
    /// Required for released profiles, rejected for research profiles. The
    /// wrapper distinguishes an absent field from an explicit empty array.
    document_embeddings: Embeddings = .{},
    encoder_execution_identity: []const u8,
    max_new_tokens: JsonNumber(i64) = .{ .value = 64 },
    context_mode: ContextMode = .latent,
    document_texts: SourceTexts = .{},

    pub const Embeddings = struct {
        value: ?[]const []const JsonNumber(f32) = null,

        pub fn jsonParse(a: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
            if (try source.peekNextTokenType() != .array_begin) return error.UnexpectedToken;
            return .{ .value = try std.json.innerParse([]const []const JsonNumber(f32), a, source, options) };
        }
    };

    /// Supplied feature rows, empty when the field was absent.
    pub fn embeddingRows(self: @This()) []const []const JsonNumber(f32) {
        return self.document_embeddings.value orelse &.{};
    }

    /// Released profiles only: texts accompany embeddings and must be aligned.
    pub fn validateSourceContext(self: @This()) !void {
        switch (self.context_mode) {
            .latent => if (self.document_texts.value != null) return error.UnexpectedMemorillaSourceContext,
            .source_verified => try grounding.validateSources(self.document_texts.value orelse return error.MemorillaSourceContextRequired, self.embeddingRows().len),
        }
    }

    /// Profile-kind shape rules, checked before any lane, slot or model work.
    /// Released readers consume caller-supplied embeddings; the research
    /// encoder-LoRA candidate encodes caller-supplied texts itself and never
    /// accepts vectors, and its source-verified mode is not implemented. The
    /// hybrid profile takes the same texts (the server also pools them).
    pub fn validateShape(self: @This(), kind: ReaderKind) !void {
        switch (kind) {
            .released => {
                if (self.embeddingRows().len == 0) return error.EmptyMemorillaCollection;
                try self.validateSourceContext();
            },
            .encoder_lora_research, .hybrid_research => {
                if (self.document_embeddings.value != null) return error.UnexpectedMemorillaEmbeddings;
                if (self.context_mode == .source_verified) return error.UnsupportedMemorillaProfile;
                const texts = self.document_texts.value orelse return error.MemorillaSourceContextRequired;
                if (texts.len == 0) return error.EmptyMemorillaCollection;
                if (texts.len > grounding.max_documents) return error.MemorillaCollectionBudgetExceeded;
                try grounding.validateSources(texts, texts.len);
            },
        }
    }
};

test "Memorilla request legacy defaults preserve latent-only source semantics" {
    const a = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(Request, a,
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id"}
    , .{});
    defer parsed.deinit();
    try parsed.value.validateSourceContext();
    try parsed.value.validateShape(.released);
    try std.testing.expect(parsed.value.context_mode == .latent);
    try std.testing.expect(parsed.value.document_texts.value == null);
    try std.testing.expectEqual(@as(usize, 1), parsed.value.embeddingRows().len);
    try std.testing.expectEqual(@as(i64, 64), parsed.value.max_new_tokens.value);
}

test "Memorilla request requires explicit mode and aligned nonblank sources" {
    const a = std.testing.allocator;
    const cases = [_]struct { body: []const u8, failure: anyerror }{
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","document_texts":["fact"]}
        , .failure = error.UnexpectedMemorillaSourceContext },
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","context_mode":"source_verified"}
        , .failure = error.MemorillaSourceContextRequired },
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","context_mode":"source_verified","document_texts":[]}
        , .failure = error.InvalidMemorillaSourceCount },
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","context_mode":"source_verified","document_texts":["a","b"]}
        , .failure = error.InvalidMemorillaSourceCount },
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","context_mode":"source_verified","document_texts":[" \n\t"]}
        , .failure = error.InvalidMemorillaSourceText },
    };
    for (cases) |case| {
        var parsed = try std.json.parseFromSlice(Request, a, case.body, .{});
        defer parsed.deinit();
        try std.testing.expectError(case.failure, parsed.value.validateSourceContext());
        try std.testing.expectError(case.failure, parsed.value.validateShape(.released));
    }
    var valid = try std.json.parseFromSlice(Request, a,
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","context_mode":"source_verified","document_texts":["A fact."]}
    , .{});
    defer valid.deinit();
    try valid.value.validateSourceContext();
    try valid.value.validateShape(.released);
}

test "Memorilla request shape separates released embeddings from research texts" {
    const a = std.testing.allocator;
    const long: [8193]u8 = @splat('x');
    const cases = [_]struct { body: []const u8, kind: ReaderKind, failure: ?anyerror }{
        // Released: embeddings are mandatory even though the wire field is optional.
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id"}
        , .kind = .released, .failure = error.EmptyMemorillaCollection },
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[],"encoder_execution_identity":"id"}
        , .kind = .released, .failure = error.EmptyMemorillaCollection },
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id","document_texts":["A fact."]}
        , .kind = .released, .failure = error.EmptyMemorillaCollection },
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id"}
        , .kind = .released, .failure = null },
        // Research: texts replace embeddings in latent mode.
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id","document_texts":["A fact."]}
        , .kind = .encoder_lora_research, .failure = null },
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id","context_mode":"latent","document_texts":["A fact.","Another fact."]}
        , .kind = .encoder_lora_research, .failure = null },
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id"}
        , .kind = .encoder_lora_research, .failure = error.MemorillaSourceContextRequired },
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id","document_texts":[]}
        , .kind = .encoder_lora_research, .failure = error.EmptyMemorillaCollection },
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","document_texts":["A fact."]}
        , .kind = .encoder_lora_research, .failure = error.UnexpectedMemorillaEmbeddings },
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[],"encoder_execution_identity":"id","document_texts":["A fact."]}
        , .kind = .encoder_lora_research, .failure = error.UnexpectedMemorillaEmbeddings },
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id","context_mode":"source_verified","document_texts":["A fact."]}
        , .kind = .encoder_lora_research, .failure = error.UnsupportedMemorillaProfile },
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","context_mode":"source_verified","document_texts":["A fact."]}
        , .kind = .encoder_lora_research, .failure = error.UnexpectedMemorillaEmbeddings },
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id","document_texts":[" \n\t"]}
        , .kind = .encoder_lora_research, .failure = error.InvalidMemorillaSourceText },
        .{ .body = "{\"model\":\"m\",\"question\":\"q\",\"encoder_execution_identity\":\"id\",\"document_texts\":[\"" ++ long ++ "\"]}", .kind = .encoder_lora_research, .failure = error.MemorillaSourceTextLimitExceeded },
        // Hybrid: the research text rules, with the server pooling the texts.
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id","document_texts":["A fact.","Another fact."]}
        , .kind = .hybrid_research, .failure = null },
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id"}
        , .kind = .hybrid_research, .failure = error.MemorillaSourceContextRequired },
        .{ .body =
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","document_texts":["A fact."]}
        , .kind = .hybrid_research, .failure = error.UnexpectedMemorillaEmbeddings },
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id","context_mode":"source_verified","document_texts":["A fact."]}
        , .kind = .hybrid_research, .failure = error.UnsupportedMemorillaProfile },
        .{ .body =
        \\{"model":"m","question":"q","encoder_execution_identity":"id","document_texts":[" \n\t"]}
        , .kind = .hybrid_research, .failure = error.InvalidMemorillaSourceText },
    };
    for (cases) |case| {
        var parsed = try std.json.parseFromSlice(Request, a, case.body, .{});
        defer parsed.deinit();
        if (case.failure) |failure| {
            try std.testing.expectError(failure, parsed.value.validateShape(case.kind));
        } else {
            try parsed.value.validateShape(case.kind);
        }
    }
    // Sixty-five texts exceed the research document bound before text checks.
    var too_many: std.ArrayList(u8) = .empty;
    defer too_many.deinit(a);
    try too_many.appendSlice(a, "{\"model\":\"m\",\"question\":\"q\",\"encoder_execution_identity\":\"id\",\"document_texts\":[");
    for (0..grounding.max_documents + 1) |i| {
        if (i > 0) try too_many.append(a, ',');
        try too_many.appendSlice(a, "\"t\"");
    }
    try too_many.appendSlice(a, "]}");
    var parsed = try std.json.parseFromSlice(Request, a, too_many.items, .{});
    defer parsed.deinit();
    try std.testing.expectError(error.MemorillaCollectionBudgetExceeded, parsed.value.validateShape(.encoder_lora_research));
}

test "Memorilla request wire rejects ambiguous modes null sources and number coercion" {
    const a = std.testing.allocator;
    const cases = [_][]const u8{
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","context_mode":null}
        ,
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","context_mode":"automatic"}
        ,
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","context_mode":"latent","context_mode":"source_verified"}
        ,
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","document_texts":null}
        ,
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","document_texts":[1]}
        ,
        \\{"model":"m","question":"q","document_embeddings":[["1"]],"encoder_execution_identity":"id"}
        ,
        \\{"model":"m","question":"q","document_embeddings":null,"encoder_execution_identity":"id"}
        ,
        \\{"model":"m","question":"q","document_embeddings":[1],"encoder_execution_identity":"id"}
        ,
        \\{"model":"m","question":"q","document_embeddings":[[1]],"document_embeddings":[[1]],"encoder_execution_identity":"id"}
        ,
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","max_new_tokens":"3"}
        ,
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","max_new_tokens":null}
        ,
        \\{"model":"m","question":"q","document_embeddings":[[1]],"encoder_execution_identity":"id","stream":false}
        ,
    };
    for (cases) |body| {
        if (std.json.parseFromSlice(Request, a, body, .{})) |parsed_value| {
            var parsed = parsed_value;
            parsed.deinit();
            return error.TestUnexpectedResult;
        } else |_| {}
    }
}
