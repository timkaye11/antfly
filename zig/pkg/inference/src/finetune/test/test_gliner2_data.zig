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
const gliner2_data = @import("inference_finetune_data").gliner2_data;

test {
    std.testing.refAllDecls(gliner2_data);
}

test "GLiNER2 smoke NER fixture has stable stats and span shape" {
    const allocator = std.testing.allocator;
    const entity_types = [_][]const u8{ "person", "organization", "location" };

    var loaded = try gliner2_data.loadExamples(allocator, "testdata/gliner2_ner_smoke.jsonl", null);
    defer loaded.deinit();

    const stats = try gliner2_data.computeStats(allocator, loaded.examples);
    try std.testing.expectEqual(@as(usize, 3), stats.num_examples);
    try std.testing.expectApproxEqAbs(@as(f64, 27.0), stats.avg_text_chars, 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), stats.avg_entities, 0.01);
    try std.testing.expectEqual(@as(usize, 3), stats.unique_labels);

    const coverage = gliner2_data.computeTargetCoverageStats(loaded.examples, &entity_types);
    try std.testing.expectEqual(@as(usize, 3), coverage.num_samples);
    try std.testing.expectEqual(@as(usize, 9), coverage.total_entities);
    try std.testing.expectEqual(@as(usize, 9), coverage.target_entities);
    try std.testing.expectEqual(@as(usize, 3), coverage.samples_with_target);
    try std.testing.expectEqual(@as(usize, 0), coverage.samples_without_target);

    const label_vocab = try gliner2_data.buildLabelVocab(allocator, loaded.examples, &entity_types);
    defer {
        for (label_vocab) |label| allocator.free(label);
        allocator.free(label_vocab);
    }
    try std.testing.expectEqual(@as(usize, 3), label_vocab.len);
    try std.testing.expectEqualStrings("location", label_vocab[0]);
    try std.testing.expectEqualStrings("organization", label_vocab[1]);
    try std.testing.expectEqualStrings("person", label_vocab[2]);

    const batch_shape = try gliner2_data.buildSimpleBatchShapeSummary(
        allocator,
        loaded.examples,
        &entity_types,
        256,
        8,
        4,
    );
    try std.testing.expectEqual(@as(usize, 3), batch_shape.batch_size);
    try std.testing.expectEqual(@as(usize, 256), batch_shape.max_length);
    try std.testing.expectEqual(@as(usize, 3), batch_shape.num_entity_types);
    try std.testing.expectEqual(@as(usize, 244), batch_shape.max_words_per_sample);
    try std.testing.expectEqual(@as(usize, 1952), batch_shape.max_spans);
    try std.testing.expectEqual(@as(usize, 51), batch_shape.valid_spans);
    try std.testing.expectEqual(@as(usize, 9), batch_shape.positive_labels);

    const span_targets = try gliner2_data.summarizeSpanTargetsForExamples(
        allocator,
        loaded.examples,
        &entity_types,
        256,
        8,
    );
    try std.testing.expectEqual(@as(usize, 3), span_targets.num_examples);
    try std.testing.expectEqual(@as(usize, 51), span_targets.valid_spans);
    try std.testing.expectEqual(@as(usize, 9), span_targets.positive_labels);
}

test "GLiNER2 dataset readiness exposes non-toy gate failures" {
    const allocator = std.testing.allocator;
    const entity_types = [_][]const u8{ "person", "organization", "location" };

    var loaded = try gliner2_data.loadExamples(allocator, "testdata/gliner2_ner_smoke.jsonl", null);
    defer loaded.deinit();

    var smoke = try gliner2_data.evaluateDatasetReadiness(
        allocator,
        loaded.examples,
        &entity_types,
        256,
        8,
        4,
        .{
            .min_examples = 3,
            .min_total_entities = 9,
            .min_unique_labels = 3,
            .min_target_entities = 9,
            .min_target_coverage_ratio = 1.0,
            .require_all_examples_with_target = true,
            .min_positive_span_labels = 9,
        },
    );
    defer gliner2_data.freeDatasetReadinessSummary(allocator, &smoke);
    try std.testing.expect(smoke.passed);
    try std.testing.expectEqual(@as(usize, 0), smoke.failed_reasons.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), smoke.target_coverage_ratio, 0.000001);
    try std.testing.expectEqual(@as(usize, 9), smoke.span_targets.positive_labels);

    var small_preview = try gliner2_data.evaluateDatasetReadiness(
        allocator,
        loaded.examples,
        &entity_types,
        256,
        8,
        1,
        .{
            .min_examples = 3,
            .min_total_entities = 9,
            .min_unique_labels = 3,
            .min_target_entities = 9,
            .min_target_coverage_ratio = 1.0,
            .require_all_examples_with_target = true,
            .min_positive_span_labels = 9,
        },
    );
    defer gliner2_data.freeDatasetReadinessSummary(allocator, &small_preview);
    try std.testing.expect(small_preview.passed);
    try std.testing.expectEqual(@as(usize, 1), small_preview.batch_shape.batch_size);
    try std.testing.expectEqual(@as(usize, 9), small_preview.span_targets.positive_labels);

    var non_toy = try gliner2_data.evaluateDatasetReadiness(
        allocator,
        loaded.examples,
        &entity_types,
        256,
        8,
        4,
        .{
            .min_examples = 100,
            .min_total_entities = 100,
            .min_unique_labels = 3,
            .min_target_entities = 100,
            .min_target_coverage_ratio = 1.0,
            .require_all_examples_with_target = true,
            .min_positive_span_labels = 100,
        },
    );
    defer gliner2_data.freeDatasetReadinessSummary(allocator, &non_toy);
    try std.testing.expect(!non_toy.passed);
    try std.testing.expect(containsReason(non_toy.failed_reasons, "min_examples"));
    try std.testing.expect(containsReason(non_toy.failed_reasons, "min_total_entities"));
    try std.testing.expect(containsReason(non_toy.failed_reasons, "min_target_entities"));
    try std.testing.expect(containsReason(non_toy.failed_reasons, "min_positive_span_labels"));
}

test "GLiNER2 upstream all-task fixture exposes extractive span labels" {
    const allocator = std.testing.allocator;

    var loaded = try gliner2_data.loadExamples(allocator, "testdata/gliner2_all_task_smoke.jsonl", null);
    defer loaded.deinit();

    const stats = try gliner2_data.computeStats(allocator, loaded.examples);
    try std.testing.expectEqual(@as(usize, 4), stats.num_examples);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), stats.avg_entities, 0.01);
    try std.testing.expectEqual(@as(usize, 10), stats.unique_labels);

    try std.testing.expectEqual(@as(usize, 3), loaded.examples[0].entities.len);
    try std.testing.expectEqual(@as(usize, 0), loaded.examples[1].entities.len);
    try std.testing.expectEqual(@as(usize, 4), loaded.examples[2].entities.len);
    try std.testing.expectEqual(@as(usize, 3), loaded.examples[3].entities.len);

    const label_vocab = try gliner2_data.buildLabelVocab(allocator, loaded.examples, null);
    defer {
        for (label_vocab) |label| allocator.free(label);
        allocator.free(label_vocab);
    }

    try std.testing.expect(containsLabel(label_vocab, "person"));
    try std.testing.expect(containsLabel(label_vocab, "product.name"));
    try std.testing.expect(containsLabel(label_vocab, "product.status"));
    try std.testing.expect(containsLabel(label_vocab, "founded.head"));
    try std.testing.expect(containsLabel(label_vocab, "founded.tail"));
}

test "GLiNER2 upstream all-task fixture preserves task-level labels" {
    const allocator = std.testing.allocator;

    var loaded = try gliner2_data.loadTrainingRecords(allocator, "testdata/gliner2_all_task_smoke.jsonl", null);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 4), loaded.records.len);
    const stats = gliner2_data.computeUpstreamTaskStats(loaded.records);
    try std.testing.expectEqual(@as(usize, 4), stats.num_records);
    try std.testing.expectEqual(@as(usize, 1), stats.entity_tasks);
    try std.testing.expectEqual(@as(usize, 2), stats.classification_tasks);
    try std.testing.expectEqual(@as(usize, 1), stats.json_structure_tasks);
    try std.testing.expectEqual(@as(usize, 1), stats.relation_tasks);
    try std.testing.expectEqual(@as(usize, 4), stats.non_entity_task_annotations);
    try std.testing.expectEqual(@as(usize, 5), stats.classification_label_count);
    try std.testing.expectEqual(@as(usize, 2), stats.classification_true_label_count);
    try std.testing.expectEqual(@as(usize, 10), stats.span_field_annotations);

    const entities = loaded.records[0].tasks[0];
    try std.testing.expectEqual(gliner2_data.UpstreamTaskKind.entities, entities.kind);
    try std.testing.expectEqualStrings("entities", entities.name);
    try std.testing.expectEqual(@as(usize, 3), entities.fields.len);
    try std.testing.expectEqualStrings("person", entities.fields[0].name);

    const cls = loaded.records[1].tasks[0];
    try std.testing.expectEqual(gliner2_data.UpstreamTaskKind.classifications, cls.kind);
    try std.testing.expectEqualStrings("sentiment", cls.name);
    try std.testing.expectEqual(@as(usize, 2), cls.labels.len);
    try std.testing.expectEqualStrings("negative", cls.true_labels[0]);

    const structure = loaded.records[2].tasks[0];
    try std.testing.expectEqual(gliner2_data.UpstreamTaskKind.json_structures, structure.kind);
    try std.testing.expectEqualStrings("product", structure.name);
    try std.testing.expectEqual(@as(usize, 4), structure.fields.len);
    try std.testing.expectEqualStrings("status", structure.fields[3].name);
    try std.testing.expectEqualStrings("available", structure.fields[3].value);
    try std.testing.expectEqual(@as(usize, 4), structure.fields[3].target_word_start.?);
    try std.testing.expectEqual(@as(usize, 4), structure.fields[3].target_word_end.?);
    try std.testing.expectEqual(@as(usize, 9), loaded.records[2].prefix_tokens.len);

    const relation = loaded.records[3].tasks[0];
    try std.testing.expectEqual(gliner2_data.UpstreamTaskKind.relations, relation.kind);
    try std.testing.expectEqualStrings("founded", relation.name);
    try std.testing.expectEqual(@as(usize, 3), relation.fields.len);
    try std.testing.expectEqualStrings("head", relation.fields[0].name);
    try std.testing.expect(relation.fields[0].start != null);
}

test "GLiNER2 upstream all-task label vocab includes classification and structured labels" {
    const allocator = std.testing.allocator;

    var loaded = try gliner2_data.loadTrainingRecords(allocator, "testdata/gliner2_all_task_smoke.jsonl", null);
    defer loaded.deinit();

    const labels = try gliner2_data.buildUpstreamTaskLabelVocab(allocator, loaded.records, null);
    defer {
        for (labels) |label| allocator.free(label);
        allocator.free(labels);
    }

    try std.testing.expect(containsLabel(labels, "person"));
    try std.testing.expect(containsLabel(labels, "negative"));
    try std.testing.expect(containsLabel(labels, "urgent"));
    try std.testing.expect(containsLabel(labels, "product.name"));
    try std.testing.expect(containsLabel(labels, "product.status"));
    try std.testing.expect(containsLabel(labels, "founded.head"));
    try std.testing.expect(containsLabel(labels, "founded.tail"));
    try std.testing.expectEqualStrings("person", labels[0]);
    try std.testing.expectEqualStrings("organization", labels[1]);
    try std.testing.expectEqualStrings("location", labels[2]);
    try std.testing.expectEqualStrings("positive", labels[3]);
    try std.testing.expectEqualStrings("negative", labels[4]);
    try std.testing.expectEqualStrings("product.name", labels[8]);
    try std.testing.expectEqualStrings("product.price", labels[9]);
    try std.testing.expectEqualStrings("product.color", labels[10]);
    try std.testing.expectEqualStrings("product.status", labels[11]);
}

test "GLiNER2 upstream task batch carries schema markers and structured span labels" {
    const allocator = std.testing.allocator;

    var loaded = try gliner2_data.loadTrainingRecords(allocator, "testdata/gliner2_all_task_smoke.jsonl", null);
    defer loaded.deinit();

    const labels = try gliner2_data.buildUpstreamTaskLabelVocab(allocator, loaded.records, null);
    defer {
        for (labels) |label| allocator.free(label);
        allocator.free(labels);
    }

    var tokenizer = try gliner2_data.Tokenizer.initDefault(allocator);
    defer tokenizer.deinit(allocator);

    var batch = try gliner2_data.buildUpstreamTaskBatch(allocator, &tokenizer, loaded.records, labels, 96, 4, loaded.records.len);
    defer batch.deinit();

    try std.testing.expectEqual(@as(usize, 4), batch.batch_size);
    try std.testing.expectEqual(@as(usize, labels.len), batch.num_entity_types);
    try std.testing.expectEqual(@as(i32, 1), batch.schema_counts[0]);
    try std.testing.expectEqual(@as(i32, 2), batch.schema_counts[1]);
    try std.testing.expectEqual(@as(i32, 1), batch.schema_counts[2]);
    try std.testing.expectEqual(@as(i32, 1), batch.schema_counts[3]);
    try std.testing.expectEqual(gliner2_data.upstreamTaskTypeId(.entities), batch.task_type_ids[0]);
    try std.testing.expectEqual(gliner2_data.upstreamTaskTypeId(.classifications), batch.task_type_ids[batch.max_schemas]);
    try std.testing.expect(batch.schema_special_counts[0] > 0);
    try std.testing.expect(batch.schema_special_positions[0] >= 0);

    const negative_idx = labelIndex(labels, "negative").?;
    const urgent_idx = labelIndex(labels, "urgent").?;
    const product_name_idx = labelIndex(labels, "product.name").?;
    try std.testing.expect(batch.e_token_positions[1 * labels.len + negative_idx] >= 0);
    try std.testing.expect(batch.e_token_positions[1 * labels.len + urgent_idx] >= 0);
    try std.testing.expect(batch.e_token_positions[2 * labels.len + product_name_idx] >= 0);
    try std.testing.expectEqual(@as(i32, 2), batch.entity_type_kind[2 * labels.len + product_name_idx]);
}

test "GLiNER2 upstream negative entity task keeps schema active" {
    const allocator = std.testing.allocator;

    var loaded = try gliner2_data.loadTrainingRecords(allocator, "testdata/gliner2_negative_smoke.jsonl", null);
    defer loaded.deinit();

    const labels = try gliner2_data.buildUpstreamTaskLabelVocab(allocator, loaded.records, null);
    defer {
        for (labels) |label| allocator.free(label);
        allocator.free(labels);
    }

    var tokenizer = try gliner2_data.Tokenizer.initDefault(allocator);
    defer tokenizer.deinit(allocator);

    var batch = try gliner2_data.buildUpstreamTaskBatch(allocator, &tokenizer, loaded.records[0..2], labels, 64, 4, 2);
    defer batch.deinit();

    const person_idx = labelIndex(labels, "person").?;
    try std.testing.expect(batch.e_token_positions[1 * labels.len + person_idx] >= 0);
    try std.testing.expectEqual(@as(i32, 2), batch.entity_type_kind[1 * labels.len + person_idx]);

    var positives: usize = 0;
    for (0..batch.max_spans) |span_idx| {
        if (batch.span_labels[(1 * batch.max_spans + span_idx) * labels.len + person_idx] > 0.0) {
            positives += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), positives);
}

test "GLiNER2 label class capacity rejects collapsed label mappings" {
    const allocator = std.testing.allocator;

    var loaded = try gliner2_data.loadExamples(allocator, "testdata/gliner2_ner_smoke.jsonl", null);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 3), try gliner2_data.validateLabelClassCapacity(allocator, loaded.examples, 4));
    try std.testing.expectError(error.TooManyEntityTypes, gliner2_data.validateLabelClassCapacity(allocator, loaded.examples, 3));
    try std.testing.expectError(error.InvalidNumClasses, gliner2_data.validateLabelClassCapacity(allocator, loaded.examples, 1));
}

fn containsReason(reasons: []const []const u8, needle: []const u8) bool {
    for (reasons) |reason| {
        if (std.mem.eql(u8, reason, needle)) return true;
    }
    return false;
}

fn containsLabel(labels: []const []const u8, needle: []const u8) bool {
    for (labels) |label| {
        if (std.mem.eql(u8, label, needle)) return true;
    }
    return false;
}

fn labelIndex(labels: []const []const u8, needle: []const u8) ?usize {
    for (labels, 0..) |label, idx| {
        if (std.mem.eql(u8, label, needle)) return idx;
    }
    return null;
}
