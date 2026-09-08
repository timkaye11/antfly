"use client";

/**
 * Bespoke figures for Gemma4 chapters 2–6 (tokens, PLE, attention, KV, A4B).
 * Everything draws with the shared glyph vocabulary from components/viz/glyphs.
 */
import {
  ActivationGlyph,
  AttentionGlyph,
  EmbeddingGlyph,
  Figure,
  FlowArrow,
  ForkGlyph,
  KvBlockGlyph,
  MatmulGlyph,
  NormGlyph,
  WeightGlyph,
} from "@/components/viz/glyphs";
import type { KvTrace } from "@/lib/schema";

/* ------------------------------------------------------------------ */
/* Ch 2 — tokens in                                                    */
/* ------------------------------------------------------------------ */

/** Illustrative SentencePiece split — plausible pieces, invented ids. */
const TOKEN_PIECES = [
  { piece: "▁Why", id: 10906 },
  { piece: "▁do", id: 776 },
  { piece: "▁ant", id: 4501 },
  { piece: "s", id: 236751 },
  { piece: "▁never", id: 6231 },
  { piece: "▁sleep", id: 12873 },
  { piece: "?", id: 236881 },
];

export function TokenPiecesFigure() {
  return (
    <Figure
      viewBox="0 0 480 200"
      title="prompt → SentencePiece pieces"
      caption="Illustrative split and ids — the real vocabulary has 262,144 pieces. ▁ marks a leading space."
    >
      {/* the raw prompt */}
      <rect
        x={40}
        y={20}
        width={400}
        height={34}
        rx={6}
        fill="color-mix(in oklch, var(--dtype-f32) 10%, transparent)"
        stroke="var(--dtype-f32)"
        strokeWidth={1.25}
      />
      <text
        x={240}
        y={37}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={13}
        className="fill-foreground font-mono"
      >
        &quot;Why do ants never sleep?&quot;
      </text>
      <FlowArrow x1={240} y1={58} x2={240} y2={100} label="tokenize" />
      {/* the pieces */}
      {TOKEN_PIECES.map((t, i) => {
        const w = 60;
        const x = 14 + i * (w + 6);
        return (
          <g key={t.id}>
            <rect
              x={x}
              y={108}
              width={w}
              height={34}
              rx={4}
              fill="color-mix(in oklch, var(--dtype-f16) 14%, transparent)"
              stroke="var(--dtype-f16)"
              strokeWidth={1.25}
            />
            <text
              x={x + w / 2}
              y={125}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={11}
              className="fill-foreground font-mono"
            >
              {t.piece}
            </text>
            <text
              x={x + w / 2}
              y={158}
              textAnchor="middle"
              fontSize={9}
              className="fill-muted-foreground font-mono"
            >
              {t.id}
            </text>
          </g>
        );
      })}
      <text
        x={240}
        y={186}
        textAnchor="middle"
        fontSize={9}
        className="fill-muted-foreground font-mono"
      >
        7 token ids, i32 — the only thing the model ever sees
      </text>
    </Figure>
  );
}

export function SessionRoutingFigure() {
  const rows = [
    {
      label: "HTTP",
      detail: "POST /v1/chat/completions → chatCompletions()",
      color: "var(--dtype-f32)",
    },
    {
      label: "session_factory",
      detail: "reads the manifest, detects the architecture",
      color: "var(--kfam-fusion)",
    },
    {
      label: "ModelFamily.gemma",
      detail: "one enum tag in the unified GPT config",
      color: "var(--kfam-attention)",
    },
    {
      label: "GPT runtime",
      detail: "the same decode loop llama/qwen/phi use — Gemma is a config, not a codepath",
      color: "var(--kfam-matvec)",
    },
  ];
  return (
    <div className="flex h-full flex-col justify-center gap-2">
      <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
        request → session routing
      </div>
      {rows.map((r, i) => (
        <div key={r.label} className="flex items-center gap-3">
          <div className="w-4 text-center font-mono text-[10px] text-muted-foreground">
            {i < rows.length - 1 ? "↓" : ""}
          </div>
          <div className="flex-1 rounded-md border px-3 py-2" style={{ borderColor: r.color }}>
            <span className="font-mono text-xs font-semibold" style={{ color: r.color }}>
              {r.label}
            </span>
            <span className="ml-2 text-xs text-muted-foreground">{r.detail}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Ch 3 — embeddings, then embeddings again (PLE)                      */
/* ------------------------------------------------------------------ */

export function EmbedLookupFigure({ hidden }: { hidden: number }) {
  return (
    <Figure
      viewBox="0 0 480 220"
      title="the ordinary embedding lookup"
      caption={`One row of a [262144 × ${hidden}] table, scaled — the token id becomes a hidden-state vector.`}
    >
      <ActivationGlyph x={30} y={90} w={70} h={30} label="token id" sublabel="i32" dtype="f32" />
      <FlowArrow x1={104} y1={105} x2={170} y2={105} />
      <EmbeddingGlyph
        x={176}
        y={78}
        w={130}
        h={54}
        label="embed_tokens"
        sublabel={`[262144 × ${hidden}]`}
        dtype="q4_0"
      />
      <FlowArrow x1={310} y1={105} x2={376} y2={105} />
      <ActivationGlyph
        x={380}
        y={90}
        w={80}
        h={30}
        label="hidden"
        sublabel={`[1 × ${hidden}]`}
        dtype="f16"
      />
    </Figure>
  );
}

export function PleRibbonFigure({ layers }: { layers: number }) {
  const stackTop = 60;
  const rowH = 10;
  const shown = Math.min(layers, 14); // draw a compressed stack; label the true count
  const stackH = shown * rowH;
  const H = stackTop + stackH + 40;
  return (
    <Figure
      viewBox={`0 0 480 ${H}`}
      title="the second embedding: a per-layer lane"
      caption={`A second lookup feeds a thin per-layer ribbon that runs beside all ${layers} layers and taps each one — no other Metal runtime executes this full path.`}
    >
      {/* main embedding into the stack */}
      <EmbeddingGlyph x={30} y={stackTop - 44} w={100} h={36} label="embed_tokens" dtype="q4_0" />
      <FlowArrow x1={80} y1={stackTop - 6} x2={80} y2={stackTop + 8} />
      {/* PLE embedding into the ribbon */}
      <EmbeddingGlyph
        x={330}
        y={stackTop - 44}
        w={100}
        h={36}
        label="per-layer embed"
        sublabel="PLE"
        dtype="f32"
      />
      <FlowArrow x1={380} y1={stackTop - 6} x2={380} y2={stackTop + 8} />
      {/* layer stack */}
      {Array.from({ length: shown }, (_, i) => {
        const y = stackTop + 12 + i * rowH;
        const gap = i === Math.floor(shown / 2); // ellipsis row
        if (gap)
          return (
            <text
              key="ellipsis"
              x={170}
              y={y + rowH / 2}
              textAnchor="middle"
              fontSize={9}
              className="fill-muted-foreground font-mono"
            >
              ⋮ {layers} layers ⋮
            </text>
          );
        return (
          <g key={y}>
            <rect
              x={40}
              y={y}
              width={260}
              height={rowH - 3}
              rx={2}
              fill="var(--kfam-attention)"
              opacity={0.3}
            />
            {/* the tap from the ribbon into this layer */}
            <line
              x1={376}
              y1={y + (rowH - 3) / 2}
              x2={302}
              y2={y + (rowH - 3) / 2}
              stroke="var(--kfam-fusion)"
              strokeWidth={1.25}
              markerEnd="url(#arrowhead)"
            />
          </g>
        );
      })}
      {/* the ribbon itself */}
      <rect
        x={376}
        y={stackTop + 10}
        width={8}
        height={stackH + 2}
        rx={4}
        fill="color-mix(in oklch, var(--kfam-fusion) 30%, transparent)"
        stroke="var(--kfam-fusion)"
        strokeWidth={1.25}
      />
      <text
        x={392}
        y={stackTop + stackH / 2}
        fontSize={9}
        className="fill-muted-foreground font-mono"
        writingMode="vertical-rl"
      >
        PLE lane
      </text>
      <text
        x={170}
        y={H - 12}
        textAnchor="middle"
        fontSize={9}
        className="fill-muted-foreground font-mono"
      >
        every layer mixes in its own slice of the PLE vector
      </text>
    </Figure>
  );
}

export function PleCostFigure({ isE4b }: { isE4b: boolean }) {
  return (
    <Figure
      viewBox="0 0 480 250"
      title="what the PLE lane costs per token"
      caption={
        isE4b
          ? "E4B: the per_layer_model_proj matvec alone streams 55 MB per token in F16 — Q8_0 staging cuts the slot to roughly half."
          : "E2B: slot 351 (8960×1536) shipped as dense F32 — Q8_0 staging saves ~41 MB per token."
      }
    >
      <ActivationGlyph x={30} y={60} w={70} h={28} label="hidden" dtype="f16" />
      <FlowArrow x1={104} y1={74} x2={160} y2={74} />
      <WeightGlyph
        x={166}
        y={52}
        w={150}
        h={44}
        label="per_layer_model_proj"
        sublabel={isE4b ? "10752 × 2560" : "8960 × 1536"}
        quant={isE4b ? "bf16" : "f32"}
        dim
      />
      <FlowArrow x1={320} y1={74} x2={376} y2={74} />
      <ActivationGlyph x={380} y={60} w={76} h={28} label="PLE vecs" dtype="f16" />
      {/* staged replacement */}
      <FlowArrow x1={241} y1={100} x2={241} y2={140} label="staged at load" />
      <WeightGlyph
        x={166}
        y={146}
        w={150}
        h={44}
        label="same slot, staged"
        sublabel="default-on"
        quant="q8_0"
        highlight
      />
      <text x={241} y={216} textAnchor="middle" fontSize={10} className="fill-foreground font-mono">
        {isE4b ? "bf16 → Q8_0 · ~13 MB/token saved" : "F32 → Q8_0 · ~41 MB/token saved"}
      </text>
      <text
        x={241}
        y={234}
        textAnchor="middle"
        fontSize={9}
        className="fill-muted-foreground font-mono"
      >
        tokens identical · E2B +1.10% on the M4 Pro testbed
      </text>
    </Figure>
  );
}

/* ------------------------------------------------------------------ */
/* Ch 4 — attention block, GQA grouping, range masks                   */
/* ------------------------------------------------------------------ */

export function AttentionBlockFigure({ isE4b }: { isE4b: boolean }) {
  const kvHeads = isE4b ? 2 : 1;
  return (
    <Figure
      viewBox="0 0 480 290"
      title="one attention block, exploded"
      caption={`QKV → per-head Q/K RMS norms (Gemma's unusual placement) → the fused ⟨head_rms ⋄ rope⟩ node → grouped-query attention (8 query heads, ${kvHeads} KV head${kvHeads > 1 ? "s" : ""}).`}
    >
      <ActivationGlyph x={20} y={120} w={64} h={28} label="hidden" dtype="f16" />
      <FlowArrow x1={88} y1={134} x2={124} y2={134} />
      <MatmulGlyph x={128} y={118} w={90} h={32} label="QKV proj" sublabel="q4_0" dtype="q4_0" />
      {/* Q and K branches into thin norm bars */}
      <FlowArrow x1={218} y1={126} x2={262} y2={72} label="Q" />
      <FlowArrow x1={218} y1={142} x2={262} y2={196} label="K" />
      <NormGlyph x={266} y={64} w={56} label="q_norm" dtype="f32" />
      <NormGlyph x={266} y={192} w={56} label="k_norm" dtype="f32" />
      {/* fused zipper node */}
      <FlowArrow x1={324} y1={70} x2={352} y2={118} />
      <FlowArrow x1={324} y1={196} x2={352} y2={150} />
      <MatmulGlyph x={340} y={118} w={110} h={32} fused={["head_rms", "rope"]} dtype="f16" />
      <text
        x={395}
        y={106}
        textAnchor="middle"
        fontSize={8}
        className="fill-muted-foreground font-mono"
      >
        one kernel, per head
      </text>
      {/* GQA hexagon */}
      <FlowArrow x1={395} y1={152} x2={395} y2={196} />
      <AttentionGlyph x={335} y={200} w={120} h={44} label="" dtype="f16" />
      <text
        x={395}
        y={222}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={9}
        className="fill-foreground font-mono"
      >
        GQA 8q/{kvHeads}kv
      </text>
      <text
        x={240}
        y={278}
        textAnchor="middle"
        fontSize={9}
        className="fill-muted-foreground font-mono"
      >
        norms are thin bars on purpose — tiny work, load-bearing placement
      </text>
    </Figure>
  );
}

export function GqaGroupingFigure({ isE4b }: { isE4b: boolean }) {
  const kvHeads = isE4b ? 2 : 1;
  const qPerKv = 8 / kvHeads;
  return (
    <Figure
      viewBox="0 0 480 240"
      title="grouped-query attention"
      caption={`8 query heads share ${kvHeads} KV head${kvHeads > 1 ? "s" : ""} — ${qPerKv} queries read each K/V bank, dividing KV memory and KV bandwidth by ${qPerKv}.`}
    >
      {[0, 1, 2, 3, 4, 5, 6, 7].map((q) => {
        const x = 30 + q * 55;
        const kv = Math.floor(q / qPerKv);
        const kvX = kvHeads === 1 ? 215 : 130 + kv * 170;
        return (
          <g key={`q${q}`}>
            <rect
              x={x}
              y={30}
              width={40}
              height={26}
              rx={6}
              fill="color-mix(in oklch, var(--kfam-attention) 14%, transparent)"
              stroke="var(--kfam-attention)"
              strokeWidth={1.25}
            />
            <text
              x={x + 20}
              y={43}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={9}
              className="fill-foreground font-mono"
            >
              q{q}
            </text>
            <line
              x1={x + 20}
              y1={58}
              x2={kvX + 25}
              y2={140}
              stroke="var(--muted-foreground)"
              strokeWidth={1}
              opacity={0.6}
            />
          </g>
        );
      })}
      {(kvHeads === 1 ? [0] : [0, 1]).map((k) => {
        const kvX = kvHeads === 1 ? 215 : 130 + k * 170;
        return (
          <g key={`kv${k}`}>
            <rect
              x={kvX}
              y={144}
              width={50}
              height={30}
              rx={3}
              fill="color-mix(in oklch, var(--kfam-kv) 20%, transparent)"
              stroke="var(--kfam-kv)"
              strokeWidth={1.5}
            />
            <text
              x={kvX + 25}
              y={159}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={9}
              className="fill-foreground font-mono"
            >
              kv{k}
            </text>
          </g>
        );
      })}
      <text
        x={240}
        y={210}
        textAnchor="middle"
        fontSize={9}
        className="fill-muted-foreground font-mono"
      >
        {isE4b ? "E4B: 4 query heads per KV head" : "E2B: all 8 query heads share a single KV head"}
      </text>
    </Figure>
  );
}

/** Honest range-mask view: which past tokens each attention kind may read. */
export function RangeMaskFigure() {
  const N = 20;
  const cur = N - 1;
  const windowTokens = 6; // illustrative window, in cells
  const cell = 21;
  const row = (y: number, label: string, canSee: (i: number) => boolean, color: string) => (
    <g>
      <text x={4} y={y - 8} fontSize={9} className="fill-foreground font-mono">
        {label}
      </text>
      {Array.from({ length: N }, (_, i) => {
        const x = 30 + i * cell;
        const visible = canSee(i);
        const isCur = i === cur;
        return (
          <rect
            key={x}
            x={x}
            y={y}
            width={cell - 3}
            height={cell - 3}
            rx={2}
            fill={isCur ? "var(--primary)" : visible ? color : "none"}
            opacity={isCur ? 0.9 : visible ? 0.55 : 1}
            stroke={visible || isCur ? "none" : "var(--border)"}
            strokeWidth={1}
          />
        );
      })}
    </g>
  );
  return (
    <Figure
      viewBox="0 0 480 200"
      title="what each layer kind may read"
      caption="A range mask, not attention scores: sliding layers read a fixed trailing window; every 6th layer reads the whole sequence. The bright cell is the current token."
    >
      {row(50, "sliding", (i) => i >= cur - windowTokens && i < cur, "var(--dtype-f16)")}
      {row(120, "global", (i) => i < cur, "var(--kfam-attention)")}
      <text
        x={240}
        y={175}
        textAnchor="middle"
        fontSize={9}
        className="fill-muted-foreground font-mono"
      >
        window drawn at 6 cells for legibility — the real window is measured in hundreds of tokens
      </text>
    </Figure>
  );
}

/* ------------------------------------------------------------------ */
/* Ch 5 — KV cache trace (deterministically synthesized)               */
/* ------------------------------------------------------------------ */

/**
 * Synthesized replay of the paged-KV allocation rules for the Gemma4 layer
 * plan: page_size_tokens=16, SWA lanes evict behind the window, shared-KV
 * tail layers never allocate. Window shortened to 128 tokens so eviction is
 * visible inside a 320-token replay.
 */
export function makeGemmaKvTrace(isE4b: boolean): KvTrace {
  const globalLayers = isE4b ? 4 : 2;
  const swaLayers = isE4b ? 20 : 13;
  const sharedLayers = isE4b ? 18 : 20;
  const windowTokens = 128;
  const steps: KvTrace["steps"] = [];
  for (let t = 0; t <= 320; t++) {
    const events: KvTrace["steps"][number]["events"] = [];
    if (t > 0 && t % 16 === 0) {
      const block = t / 16 - 1;
      events.push({ kind: "alloc", lane: "global", blockId: block });
      events.push({ kind: "alloc", lane: "swa", blockId: block });
      const evictBefore = Math.floor((t - windowTokens) / 16) - 1;
      if (evictBefore >= 0) events.push({ kind: "evict", lane: "swa", blockId: evictBefore });
    }
    steps.push({ t, events });
  }
  return {
    schemaVersion: 1,
    modelId: isE4b ? "gemma4-e4b" : "gemma4-e2b",
    synthesized: true,
    config: {
      blockTokens: 16,
      lanes: [
        { id: "global", label: "Global layers (own KV)", layers: globalLayers },
        { id: "swa", label: "SWA layers (own KV, window evicts)", layers: swaLayers, windowTokens },
        {
          id: "shared",
          label: "Shared-KV tail (no pages)",
          layers: sharedLayers,
          sharedWith: "donor layers",
        },
      ],
      dtypes: [{ id: "f16", label: "f16 KV (approx.)", bytesPerTokenLayer: 2048 }],
    },
    steps,
  };
}

export function KvExtrasFigure() {
  return (
    <Figure
      viewBox="0 0 480 240"
      title="two more tricks in the same manager"
      caption="Prefix caching re-attaches pages of an earlier prompt instead of re-prefilling them; TurboQuant Polar4 compresses cached keys to ~4 bits."
    >
      {/* prefix cache row */}
      <text x={14} y={40} fontSize={10} className="fill-foreground font-mono">
        prompt-prefix cache
      </text>
      {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
        <KvBlockGlyph
          key={i}
          x={160 + i * 20}
          y={26}
          size={16}
          state={i < 5 ? "shared" : "filled"}
        />
      ))}
      <text x={330} y={40} fontSize={9} className="fill-muted-foreground font-mono">
        ← 5 pages reused, 3 new
      </text>
      {/* turboquant row */}
      <text x={14} y={120} fontSize={10} className="fill-foreground font-mono">
        TurboQuant Polar4
      </text>
      <rect
        x={160}
        y={102}
        width={120}
        height={26}
        rx={3}
        fill="color-mix(in oklch, var(--dtype-f16) 14%, transparent)"
        stroke="var(--dtype-f16)"
        strokeWidth={1.25}
      />
      <text
        x={220}
        y={115}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={9}
        className="fill-foreground font-mono"
      >
        f16 keys
      </text>
      <FlowArrow x1={284} y1={115} x2={330} y2={115} label="encode" />
      <rect
        x={334}
        y={106}
        width={40}
        height={18}
        rx={3}
        fill="color-mix(in oklch, var(--dtype-sub4) 20%, transparent)"
        stroke="var(--dtype-sub4)"
        strokeWidth={1.25}
      />
      <text
        x={354}
        y={115}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={8}
        className="fill-foreground font-mono"
      >
        4-bit
      </text>
      <text
        x={240}
        y={190}
        textAnchor="middle"
        fontSize={9}
        className="fill-muted-foreground font-mono"
      >
        paged + sliding-window + shared-KV + compressed KV — one KvManager
      </text>
    </Figure>
  );
}

/* ------------------------------------------------------------------ */
/* Ch 6 — the MoE cousin (Gemma4 26B-A4B)                              */
/* ------------------------------------------------------------------ */

export function MoeRoutingFigure() {
  return (
    <Figure
      viewBox="0 0 480 260"
      title="A4B: routed experts (this page's model is dense)"
      caption="26B-A4B: 30 MoE layers, 128 experts at hidden 2816. The router picks a few experts per token; the rest of the layer never runs."
    >
      <ActivationGlyph x={20} y={110} w={70} h={28} label="hidden" dtype="f16" />
      <FlowArrow x1={94} y1={124} x2={140} y2={124} />
      <ForkGlyph
        x={144}
        y={94}
        w={80}
        h={60}
        label="router"
        sublabel="top-k"
        branches={5}
        dtype="f32"
      />
      {Array.from({ length: 5 }, (_, i) => {
        const y = 40 + i * 42;
        const active = i === 1 || i === 3;
        return (
          <g key={y} opacity={active ? 1 : 0.3}>
            <MatmulGlyph
              x={260}
              y={y}
              w={110}
              h={30}
              label={i === 0 ? "experts" : undefined}
              sublabel={i === 0 ? "128 total" : undefined}
              dtype="q4_0"
            />
            <text
              x={315}
              y={y + 15}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={8.5}
              className="fill-foreground font-mono"
            >
              expert {[7, 23, 64, 101, 119][i]}
            </text>
          </g>
        );
      })}
      <FlowArrow x1={374} y1={97} x2={420} y2={122} />
      <FlowArrow x1={374} y1={181} x2={420} y2={136} />
      <ActivationGlyph x={410} y={116} w={60} h={26} label="mix" dtype="f16" />
    </Figure>
  );
}

export function MoeResidencyFigure() {
  const perRow = 16;
  const rows = 8;
  const residentCount = 16;
  return (
    <Figure
      viewBox="0 0 480 240"
      title="expert residency: resident vs streamed"
      caption="A4bMappedMoeRoute: the resident route keeps all mapped expert weights on the GPU; otherwise experts stream from host memory on demand — a shelf per expert."
    >
      {Array.from({ length: rows * perRow }, (_, i) => {
        const x = 60 + (i % perRow) * 23;
        const y = 30 + Math.floor(i / perRow) * 23;
        const resident = i < residentCount;
        return (
          <rect
            key={`${x}-${y}`}
            x={x}
            y={y}
            width={19}
            height={19}
            rx={2}
            fill={resident ? "color-mix(in oklch, var(--kfam-moe) 55%, transparent)" : "none"}
            stroke="var(--kfam-moe)"
            strokeWidth={1}
            strokeDasharray={resident ? undefined : "3 2"}
            opacity={resident ? 1 : 0.5}
          />
        );
      })}
      <text x={60} y={222} fontSize={9} className="fill-foreground font-mono">
        ■ resident on GPU
      </text>
      <text x={200} y={222} fontSize={9} className="fill-muted-foreground font-mono">
        ▢ streamed from host when routed to
      </text>
    </Figure>
  );
}

export function MoeForkJoinFigure() {
  const resources = ["input", "route_plan", "shared_gated", "routed_gated", "…projected", "output"];
  return (
    <Figure
      viewBox="0 0 480 230"
      title="the A4B fork/join lowering"
      caption="A4bForkJoinCommandLowerer describes the decode-FFN fork/join as planner resources and scopes — it deliberately stops at planning; no encoder consumes it yet."
    >
      {/* fork/join spine */}
      <ForkGlyph x={20} y={70} w={60} h={50} label="fork" branches={2} dtype="f16" />
      {/* shared branch */}
      <rect
        x={110}
        y={44}
        width={150}
        height={30}
        rx={4}
        fill="color-mix(in oklch, var(--kfam-matvec) 14%, transparent)"
        stroke="var(--kfam-matvec)"
        strokeWidth={1.25}
      />
      <text
        x={185}
        y={59}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={9}
        className="fill-foreground font-mono"
      >
        shared expert path
      </text>
      {/* routed branch */}
      <rect
        x={110}
        y={116}
        width={150}
        height={30}
        rx={4}
        fill="color-mix(in oklch, var(--kfam-moe) 14%, transparent)"
        stroke="var(--kfam-moe)"
        strokeWidth={1.25}
      />
      <text
        x={185}
        y={131}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={9}
        className="fill-foreground font-mono"
      >
        routed experts path
      </text>
      <FlowArrow x1={264} y1={59} x2={320} y2={90} />
      <FlowArrow x1={264} y1={131} x2={320} y2={100} />
      <rect
        x={324}
        y={80}
        width={80}
        height={30}
        rx={4}
        fill="color-mix(in oklch, var(--kfam-fusion) 14%, transparent)"
        stroke="var(--kfam-fusion)"
        strokeWidth={1.25}
      />
      <text
        x={364}
        y={95}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={9}
        className="fill-foreground font-mono"
      >
        join → output
      </text>
      {/* resource strip */}
      {resources.map((r, i) => (
        <text
          key={r}
          x={30 + i * 75}
          y={200}
          fontSize={8}
          className="fill-muted-foreground font-mono"
        >
          {r}
        </text>
      ))}
      <line x1={24} y1={186} x2={456} y2={186} stroke="var(--border)" strokeWidth={1} />
      <text x={24} y={176} fontSize={8} className="fill-muted-foreground font-mono">
        planner resources (enum, in order):
      </text>
    </Figure>
  );
}
