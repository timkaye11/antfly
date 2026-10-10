// Copyright 2026 Antfly, Inc. SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { test } from "node:test";
import { resolveAdapter, inspectModel } from "../dist/runtime-assets/runtime/model-adapters.js";
const marker = {
  model_type: "extractor",
  architecture: "span",
  config_version: 3,
  architecture_version: 1,
  span_head: { span_mode: "markerV0" },
};
test("configuration adapters preserve metadata and restrict executable tasks", () => {
  const model = resolveAdapter(
    marker,
    { model_type: "deberta-v2" },
    { tasks: ["extract", "decide", "future"], capabilities: ["classification", "future"] }
  );
  assert.deepEqual(model.tasks, ["extract", "decide", "future"]);
  assert.deepEqual(model.execution.tasks, ["extract"]);
  assert.deepEqual(model.execution.capabilities, ["classification"]);
  assert.deepEqual(model.execution.decisionKinds, []);
  assert.throws(() => resolveAdapter(marker, {}, { tasks: "extract" }), /metadata/);
});
test("unsupported native decider routes report reasons rather than trained substitutes", () => {
  for (const [config, encoder, manifest, reason] of [
    [marker, { model_type: "modernbert" }, undefined, /Decide-1B/],
    [{ model_type: "embedding_gemma2_text" }, undefined, undefined, /calibration/],
    [{ architecture: "boundary" }, { model_type: "modernbert" }, undefined, /boundary/],
    [
      { model_type: "modernbert", vocab_size: 262144, laya: {} },
      undefined,
      undefined,
      /vocabulary/,
    ],
    [
      { model_type: "embedding_gemma2" },
      undefined,
      { tasks: ["embed", "decide"], capabilities: ["embedding_similarity"] },
      /calibration/,
    ],
  ]) {
    const result = resolveAdapter(config, encoder, manifest);
    assert.equal(result.availability.available, false);
    assert.match(result.availability.reason, reason);
    assert.deepEqual(result.execution.tasks, []);
  }
});
test("metadata inspection needs no weights and supports upstream Laya folders", async () => {
  const files = new Map([
    [
      "encoder/config.json",
      new Blob([JSON.stringify({ model_type: "modernbert", vocab_size: 100 })]),
    ],
    ["rl_agent_config.json", new Blob(["{}"])],
    [
      "model_manifest.json",
      new Blob([JSON.stringify({ tasks: ["decide"], capabilities: ["typed_decisions"] })]),
    ],
  ]);
  const result = await inspectModel(files);
  assert.equal(result.family, "laya");
  assert.deepEqual(result.execution.tasks, ["decide"]);
  assert.deepEqual(result.execution.capabilities, ["typed_decisions"]);
  assert(!files.has("config.json"));
});

test("native role corrections keep older Laya metadata usable as a decider", () => {
  const result = resolveAdapter({ model_type: "modernbert", laya: {} }, undefined, {
    tasks: ["extract", "classify"],
    capabilities: ["classification", "extraction", "typed_decisions"],
  });
  assert.deepEqual(result.tasks, ["decide"]);
  assert.deepEqual(result.capabilities, ["typed_decisions"]);
  assert.deepEqual(result.execution.tasks, ["decide"]);
});
