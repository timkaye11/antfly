"use client";

import type { ComponentType } from "react";
import type { ModelFrames } from "@/app/models/[slug]/model-page-client";
import type { KernelRoute, ModelSpec } from "@/lib/schema";
import { Gemma4Chapters } from "./gemma4/chapters";
import { Gliner2Chapters } from "./gliner2/chapters";
import { Qwen3EmbeddingChapters } from "./qwen3-embedding/chapters";
import { Qwen3VlChapters } from "./qwen3-vl/chapters";

export interface ChaptersProps {
  spec: ModelSpec;
  routes: KernelRoute[];
  frames: ModelFrames;
}

const registry: Record<string, ComponentType<ChaptersProps>> = {
  "gemma4-e2b": Gemma4Chapters,
  "gemma4-e4b": Gemma4Chapters,
  gliner2: Gliner2Chapters,
  "qwen3-embedding": Qwen3EmbeddingChapters,
  "qwen3-vl": Qwen3VlChapters,
};

function MissingChapters({ spec }: ChaptersProps) {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16 text-muted-foreground">
      Chapters for {spec.displayName} are not written yet — explore the graph via the DAG explorer above.
    </div>
  );
}

export function getChapters(slug: string): ComponentType<ChaptersProps> {
  return registry[slug] ?? MissingChapters;
}
