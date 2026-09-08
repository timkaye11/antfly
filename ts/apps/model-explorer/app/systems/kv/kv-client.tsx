"use client";

import { Tabs, TabsList, TabsTrigger } from "@antfly/design-system";
import { parseAsString, useQueryState } from "nuqs";
import { Suspense } from "react";
import { CodeLink } from "@/components/code/code-link";
import { type ClientSnippet, SnippetProvider } from "@/components/code/snippet-context";
import { KvCacheBlocks } from "@/components/viz/kv-cache-blocks";
import { L } from "@/lib/links";
import type { KvTrace } from "@/lib/schema";

/* ------------------------------------------------------------------ */
/* Deterministic trace synthesis (page size 16 tokens throughout).     */
/* These replay the paged-KV allocation rules; they are not captures.  */
/* ------------------------------------------------------------------ */

const PAGE = 16;

type Steps = KvTrace["steps"];
type Events = Steps[number]["events"];

/**
 * Gemma4 E4B: of 42 layers only 24 own KV — 4 global (keep everything),
 * 20 sliding-window (evict whole pages behind a 512-token window) — and the
 * 18 tail layers share a donor's blocks, so their lane stays empty.
 */
function gemma4Trace(): KvTrace {
  const WINDOW = 512;
  const TOTAL = 768;
  const steps: Steps = [];
  for (let t = 0; t <= TOTAL; t++) {
    const events: Events = [];
    if (t > 0 && t % PAGE === 0) {
      const block = t / PAGE - 1;
      events.push({ kind: "alloc", lane: "global", blockId: block });
      events.push({ kind: "alloc", lane: "swa", blockId: block });
      const evictBefore = Math.floor((t - WINDOW) / PAGE) - 1;
      if (evictBefore >= 0) events.push({ kind: "evict", lane: "swa", blockId: evictBefore });
    }
    steps.push({ t, events });
  }
  return {
    schemaVersion: 1,
    modelId: "gemma4-e4b",
    synthesized: true,
    config: {
      blockTokens: PAGE,
      lanes: [
        { id: "global", label: "Global layers (own KV)", layers: 4 },
        { id: "swa", label: "Sliding-window layers (own KV)", layers: 20, windowTokens: WINDOW },
        { id: "shared", label: "Shared-KV tail — zero pages", layers: 18, sharedWith: "donor layers above" },
      ],
      dtypes: [{ id: "f16", label: "f16 KV (2 KV heads × 256)", bytesPerTokenLayer: 2048 }],
    },
    steps,
  };
}

/**
 * Qwen3 Embedding: KV grows through one forward pass over the input, the
 * pooled hidden state is read out, and the whole allocation is released.
 */
function qwen3EmbeddingTrace(): KvTrace {
  const INPUT = 512;
  const steps: Steps = [];
  for (let t = 0; t <= INPUT; t++) {
    const events: Events = [];
    if (t > 0 && t % PAGE === 0) {
      events.push({ kind: "alloc", lane: "layers", blockId: t / PAGE - 1 });
    }
    steps.push({ t, events });
  }
  // The pass is over: every block goes back to the pool at once.
  const releaseAll: Events = Array.from({ length: INPUT / PAGE }, (_, b) => ({
    kind: "compact" as const,
    lane: "layers",
    blockId: b,
    note: "released",
  }));
  steps.push({ t: INPUT + 1, events: releaseAll });
  return {
    schemaVersion: 1,
    modelId: "qwen3-embedding",
    synthesized: true,
    config: {
      blockTokens: PAGE,
      lanes: [{ id: "layers", label: "Decoder layers (one forward pass)", layers: 28 }],
      dtypes: [{ id: "f16", label: "f16 KV (8 KV heads × 128)", bytesPerTokenLayer: 4096 }],
    },
    steps,
  };
}

/**
 * Qwen3-VL: a large prefill burst of image tokens (allocated as fast as the
 * vision tower emits them), then steady one-token-at-a-time decode.
 */
function qwen3VlTrace(): KvTrace {
  const IMAGE_BLOCKS = 64; // ~1024 image tokens
  const DECODE_TOKENS = 256;
  const steps: Steps = [];
  // t 0..IMAGE_BLOCKS: prefill burst, one page per step (plays back fast).
  for (let t = 0; t <= IMAGE_BLOCKS; t++) {
    const events: Events = [];
    if (t > 0) events.push({ kind: "alloc", lane: "layers", blockId: t - 1, note: "image-token prefill" });
    steps.push({ t, events });
  }
  // then decode: one new page every 16 steps.
  for (let d = 1; d <= DECODE_TOKENS; d++) {
    const t = IMAGE_BLOCKS + d;
    const events: Events = [];
    if (d % PAGE === 0) events.push({ kind: "alloc", lane: "layers", blockId: IMAGE_BLOCKS + d / PAGE - 1 });
    steps.push({ t, events });
  }
  return {
    schemaVersion: 1,
    modelId: "qwen3-vl",
    synthesized: true,
    config: {
      blockTokens: PAGE,
      lanes: [{ id: "layers", label: "Decoder layers", layers: 36 }],
      dtypes: [{ id: "f16", label: "f16 KV (8 KV heads × 128)", bytesPerTokenLayer: 4096 }],
    },
    steps,
  };
}

const TRACES: Record<string, KvTrace> = {
  gemma4: gemma4Trace(),
  "qwen3-embedding": qwen3EmbeddingTrace(),
  "qwen3-vl": qwen3VlTrace(),
};

const MODEL_NOTES: Record<string, { title: string; body: string }> = {
  gemma4: {
    title: "Three kinds of layer, three lanes",
    body:
      "Global layers keep every page. Sliding-window layers evict whole pages as they fall 512 tokens behind " +
      "(hatched blocks) — the pool trims on page granularity, not per token. The shared-KV tail lane stays " +
      "empty on purpose: those 18 layers never project K/V and read a donor layer's blocks instead.",
  },
  "qwen3-embedding": {
    title: "Alloc, pool, release",
    body:
      "An embedding request still builds KV — causal attention over the input needs it — but only for the " +
      "duration of one forward pass. Drag the slider to the end: after last-token pooling reads out the hidden " +
      "state, every page returns to the pool at once. Nothing persists between requests.",
  },
  "qwen3-vl": {
    title: "The image burst",
    body:
      "Press play: the vision tower's output lands as ~1,024 image tokens of prefill, allocating 64 pages in a " +
      "burst before the first generated token. Decode then resumes the familiar rhythm — one new page per 16 " +
      "tokens. Image tokens are ordinary KV once written; only their RoPE positions (m-RoPE) know they were pixels.",
  },
};

const TABS = [
  { id: "gemma4", label: "gemma4" },
  { id: "qwen3-embedding", label: "qwen3-embedding" },
  { id: "qwen3-vl", label: "qwen3-vl" },
  { id: "gliner2", label: "gliner2" },
];

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

interface KvClientProps {
  snippets: Record<string, ClientSnippet>;
  gitCommit: string;
  permalinkBase?: string;
}

export function KvClient(props: KvClientProps) {
  return (
    <Suspense fallback={<div className="min-h-screen" />}>
      <KvInner {...props} />
    </Suspense>
  );
}

function KvInner({ snippets, gitCommit, permalinkBase }: KvClientProps) {
  const [model, setModel] = useQueryState("model", parseAsString.withDefault("gemma4"));
  const trace = TRACES[model];
  const note = MODEL_NOTES[model];

  return (
    <SnippetProvider snippets={snippets} gitCommit={gitCommit} permalinkBase={permalinkBase}>
      <div className="mx-auto max-w-6xl space-y-8 px-4 py-8">
        <header className="max-w-3xl">
          <h1 className="text-3xl font-bold tracking-tight">KV cache</h1>
          <p className="mt-2 text-muted-foreground">
            KV memory is paged: 16-token blocks in a shared pool, a block table per sequence. The same machinery
            produces very different pictures per model — scrub the timeline to watch pages allocate, evict
            behind sliding windows, and return to the pool.
          </p>
        </header>

        <Tabs value={model} onValueChange={setModel}>
          <TabsList className="h-8">
            {TABS.map((t) => (
              <TabsTrigger key={t.id} value={t.id} className="h-7 px-3 font-mono text-xs">
                {t.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>

        {model === "gliner2" ? (
          <div className="max-w-2xl rounded-lg border bg-muted/20 p-6">
            <h2 className="text-lg font-semibold">GLiNER2 is an encoder — there is no KV cache</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              KV caching exists to avoid recomputing keys and values for past tokens during autoregressive
              decode. GLiNER2 runs its DeBERTa encoder over the whole input in one bidirectional pass and emits
              spans — there is no token-by-token loop, so there is nothing to cache and nothing to page. Every
              request is a fresh forward pass.
            </p>
          </div>
        ) : (
          trace &&
          note && (
            <section className="space-y-4">
              <div className="rounded-lg border bg-card p-4">
                <KvCacheBlocks trace={trace} />
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                <div className="rounded-lg border p-4">
                  <h3 className="mb-2 text-sm font-semibold">{note.title}</h3>
                  <p className="text-xs text-muted-foreground">{note.body}</p>
                </div>
                <div className="rounded-lg border p-4">
                  <h3 className="mb-2 text-sm font-semibold">Bytes per block</h3>
                  <p className="text-xs text-muted-foreground">
                    Sizes above assume f16 K and V. With TurboQuant, keys can be stored polar4 (4-bit polar
                    encoding, two elements per byte) — a 16-token key page shrinks 4× while values stay f16, and
                    attention scores keys directly in the packed form. For a 2-KV-head, head-dim-256 layer
                    that's 16 KB → 4 KB of keys per page per layer.
                  </p>
                </div>
              </div>
            </section>
          )
        )}

        <p className="text-xs text-muted-foreground">
          <CodeLink link={L("kv-manager")} /> · <CodeLink link={L("kv-sliding-window")} /> ·{" "}
          <CodeLink link={L("kv-turboquant-polar4")} /> · <CodeLink link={L("config-shared-kv")} />
        </p>
      </div>
    </SnippetProvider>
  );
}
