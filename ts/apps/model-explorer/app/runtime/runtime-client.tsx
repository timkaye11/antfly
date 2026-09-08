"use client";

import Link from "next/link";
import { Fragment } from "react";
import { CodeLink } from "@/components/code/code-link";
import { type ClientSnippet, SnippetProvider } from "@/components/code/snippet-context";
import { Divergence, Scene, ScrollyChapter } from "@/components/scrollytelling/scrolly";
import { SpineStrip } from "@/components/spine-strip";
import {
  AttentionGlyph,
  ElementwiseGlyph,
  EmbeddingGlyph,
  Figure,
  FlowArrow,
  ForkGlyph,
  KvBlockGlyph,
  MatmulGlyph,
  NormGlyph,
  SamplerGlyph,
  WeightGlyph,
} from "@/components/viz/glyphs";
import { L } from "@/lib/links";

/* ------------------------------------------------------------------ */
/* Small shared bits                                                   */
/* ------------------------------------------------------------------ */

function ModelUses({ items }: { items: Array<{ slug: string; label: string; note?: string }> }) {
  return (
    <p className="mt-4 border-t pt-2 text-xs text-muted-foreground">
      <span className="font-mono text-[10px] uppercase tracking-wider">How each model uses this:</span>{" "}
      {items.map((m, i) => (
        <Fragment key={m.slug}>
          {i > 0 && <span> · </span>}
          <Link href={`/models/${m.slug}`} className="text-primary underline">
            {m.label}
          </Link>
          {m.note && <span className="text-muted-foreground"> ({m.note})</span>}
        </Fragment>
      ))}
    </p>
  );
}

function Box({
  x,
  y,
  w,
  h,
  label,
  sublabel,
  dashed,
  accent,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  sublabel?: string;
  dashed?: boolean;
  accent?: boolean;
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={6}
        fill={accent ? "color-mix(in oklch, var(--primary) 10%, transparent)" : "var(--muted)"}
        fillOpacity={accent ? 1 : 0.4}
        stroke={accent ? "var(--primary)" : "var(--border)"}
        strokeWidth={accent ? 1.5 : 1}
        strokeDasharray={dashed ? "5 4" : undefined}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 - (sublabel ? 6 : 0)}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={10}
        className="fill-foreground font-mono"
      >
        {label}
      </text>
      {sublabel && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 8}
          textAnchor="middle"
          dominantBaseline="central"
          fontSize={8}
          className="fill-muted-foreground font-mono"
        >
          {sublabel}
        </text>
      )}
    </g>
  );
}

/* ------------------------------------------------------------------ */
/* Figures                                                             */
/* ------------------------------------------------------------------ */

function HttpDoorsFigure() {
  const doors = [
    { label: "generateEmbeddings", sub: "/v1/embeddings" },
    { label: "rerankPrompts", sub: "/v1/rerank" },
    { label: "generateContent", sub: "generateContent" },
    { label: "chatCompletions", sub: "/v1/chat/completions" },
  ];
  return (
    <Figure
      viewBox="0 0 460 250"
      title="four doors, one hall"
      caption="Every public endpoint is a handler on the same server; past the door, the pipeline below is shared."
    >
      {doors.map((d, i) => {
        const y = 15 + i * 58;
        return (
          <g key={d.label}>
            <Box x={10} y={y} w={175} h={44} label={d.label} sublabel={d.sub} />
            {/* door hinge stripe */}
            <line x1={14} y1={y + 4} x2={14} y2={y + 40} stroke="var(--primary)" strokeWidth={2} opacity={0.6} />
            <FlowArrow x1={185} y1={y + 22} x2={300} y2={125} />
          </g>
        );
      })}
      <Box x={302} y={95} w={148} h={60} label="server.zig" sublabel="one inference server" accent />
    </Figure>
  );
}

function SessionFactoryFigure() {
  const families = ["gemma", "qwen3", "qwen3_vl", "gliner2 / deberta", "bert …"];
  return (
    <Figure
      viewBox="0 0 460 250"
      title="route by ModelFamily"
      caption="One switch on the detected family picks the architecture session; the model manager keeps loaded sessions warm."
    >
      <Box x={10} y={95} w={150} h={56} label="session_factory" sublabel="detect family → session" accent />
      {families.map((f, i) => {
        const y = 12 + i * 46;
        return (
          <g key={f}>
            <FlowArrow x1={160} y1={123} x2={280} y2={y + 16} />
            <Box x={282} y={y} w={168} h={32} label={f} />
          </g>
        );
      })}
      <rect x={10} y={190} width={150} height={40} rx={6} fill="none" stroke="var(--border)" strokeDasharray="4 3" />
      <text x={85} y={205} textAnchor="middle" fontSize={9} className="fill-muted-foreground font-mono">
        model manager cache
      </text>
      <text x={85} y={218} textAnchor="middle" fontSize={8} className="fill-muted-foreground font-mono">
        second request skips the load
      </text>
    </Figure>
  );
}

function TokenizerFigure() {
  // Illustrative pieces — not a captured tokenization.
  const rows: Array<{ title: string; pieces: string[]; color: string }> = [
    {
      title: "SentencePiece · Gemma, DeBERTa-v3",
      pieces: ["▁Ant", "fly", "▁plans", "▁frames", "▁once", "."],
      color: "var(--dtype-f16)",
    },
    {
      title: "HF BPE tokenizer.json · Qwen3 family",
      pieces: ["Ant", "fly", "Ġplans", "Ġframes", "Ġonce", "."],
      color: "var(--dtype-q8)",
    },
  ];
  return (
    <Figure
      viewBox="0 0 460 220"
      title="same string, different pieces"
      caption="Illustrative split of “Antfly plans frames once.” — the point is the two piece vocabularies, not these exact ids."
    >
      <text x={230} y={24} textAnchor="middle" fontSize={12} className="fill-foreground font-mono">
        “Antfly plans frames once.”
      </text>
      {rows.map((row, r) => {
        const y = 60 + r * 75;
        let x = 12;
        return (
          <g key={row.title}>
            <text x={12} y={y - 8} fontSize={9} className="fill-muted-foreground font-mono">
              {row.title}
            </text>
            {row.pieces.map((p) => {
              const w = 14 + p.length * 9;
              const pill = (
                <g key={`${row.title}-${p}-${x}`}>
                  <rect
                    x={x}
                    y={y}
                    width={w}
                    height={26}
                    rx={6}
                    fill={`color-mix(in oklch, ${row.color} 14%, transparent)`}
                    stroke={row.color}
                    strokeWidth={1.25}
                  />
                  <text
                    x={x + w / 2}
                    y={y + 13}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={10}
                    className="fill-foreground font-mono"
                  >
                    {p}
                  </text>
                </g>
              );
              x += w + 6;
              return pill;
            })}
          </g>
        );
      })}
    </Figure>
  );
}

function OpGalleryFigure() {
  return (
    <Figure
      viewBox="0 0 470 320"
      title="the op vocabulary (12 of 72)"
      caption="34 primitive + 38 fused op kinds. Shapes follow the legend: hexagon = attention (shutter = sliding window), wide rect = matmul, thick-bottom rect = weight, die = sampling."
    >
      {/* row 1 */}
      <EmbeddingGlyph x={15} y={40} label="embedding_lookup" />
      <NormGlyph x={140} y={55} label="rms_norm" />
      <MatmulGlyph x={235} y={42} label="linear_no_bias" />
      <MatmulGlyph x={358} y={42} w={100} label="…_pair" fused={["gate", "up", "silu"]} />
      {/* row 2 */}
      <AttentionGlyph x={15} y={125} label="gqa_paged" sublabel="_attention" />
      <AttentionGlyph x={130} y={125} shutter label="windowed" sublabel="_self_attention" />
      <AttentionGlyph x={245} y={125} label="disentangled" sublabel="_relative_attention" />
      <ElementwiseGlyph x={375} y={135} label="silu" />
      <ElementwiseGlyph x={425} y={135} label="rope" />
      {/* row 3 */}
      <ForkGlyph x={15} y={230} label="moe_select_routes" branches={4} />
      <WeightGlyph x={140} y={230} label="parameter" quant="q4_0" />
      <MatmulGlyph x={255} y={228} w={110} label="moe_linear_no_bias" />
      <SamplerGlyph x={405} y={226} label="sample" />
    </Figure>
  );
}

function GraphToFrameFigure() {
  return (
    <Figure
      viewBox="0 0 460 240"
      title="trace once → plan once"
      caption="The graph is traced and cached per shape class; the planner lowers it to encoder scopes of planned ops."
    >
      {/* mini DAG */}
      <Box x={15} y={20} w={90} h={28} label="rms_norm" />
      <Box x={15} y={70} w={90} h={28} label="qkv" />
      <Box x={15} y={120} w={90} h={28} label="attention" />
      <Box x={15} y={170} w={90} h={28} label="ffn" />
      <FlowArrow x1={60} y1={48} x2={60} y2={68} />
      <FlowArrow x1={60} y1={98} x2={60} y2={118} />
      <FlowArrow x1={60} y1={148} x2={60} y2={168} />
      <text x={60} y={225} textAnchor="middle" fontSize={9} className="fill-muted-foreground font-mono">
        op graph (IR)
      </text>

      <FlowArrow x1={120} y1={110} x2={195} y2={110} label="plan" />

      {/* frame descriptor: scope brackets over op rects */}
      <text x={330} y={30} textAnchor="middle" fontSize={9} className="fill-muted-foreground font-mono">
        FrameDescriptor
      </text>
      {[0, 1, 2].map((s) => {
        const y = 45 + s * 55;
        return (
          <g key={s}>
            <path
              d={`M 205 ${y} v -5 h 240 v 5`}
              fill="none"
              stroke="var(--muted-foreground)"
              strokeWidth={1}
              opacity={0.7}
            />
            {[0, 1, 2, 3].map((o) => (
              <rect
                key={o}
                x={207 + o * 60}
                y={y + 4}
                width={55}
                height={30}
                rx={3}
                fill={
                  s === 1 && o === 1
                    ? "var(--kfam-attention)"
                    : o % 2 === 0
                      ? "var(--kfam-matvec)"
                      : "var(--kfam-fusion)"
                }
                opacity={0.75}
              />
            ))}
            <text x={455} y={y + 20} fontSize={8} className="fill-muted-foreground font-mono" textAnchor="end">
              {["scope: attn setup", "scope: attention", "scope: ffn"][s]}
            </text>
          </g>
        );
      })}
      <text x={330} y={225} textAnchor="middle" fontSize={9} className="fill-muted-foreground font-mono">
        PlannedOps grouped into EncoderScopes
      </text>
    </Figure>
  );
}

function RebindFigure() {
  return (
    <Figure
      viewBox="0 0 460 200"
      title="re-bind, don't re-encode"
      caption="The same planned frame is submitted every step; only buffer bindings (token id, KV page tables) change."
    >
      {[0, 1, 2].map((i) => {
        const y = 30 + i * 50;
        return (
          <g key={i} opacity={1 - i * 0.25}>
            <text x={12} y={y + 18} fontSize={9} className="fill-muted-foreground font-mono">
              step {i + 1}
            </text>
            {[0, 1, 2, 3, 4, 5].map((o) => (
              <rect
                key={o}
                x={70 + o * 55}
                y={y}
                width={50}
                height={28}
                rx={3}
                fill={["var(--kfam-fusion)", "var(--kfam-matvec)", "var(--kfam-attention)", "var(--kfam-fusion)", "var(--kfam-matvec)", "var(--kfam-sampling)"][o]}
                opacity={0.75}
              />
            ))}
            <FlowArrow x1={405} y1={y + 14} x2={440} y2={y + 14} label={i === 0 ? "token" : undefined} />
          </g>
        );
      })}
      <text x={230} y={185} textAnchor="middle" fontSize={9} className="fill-emerald-500 font-mono">
        1 encoder · 143 scopes · planned_barriers = 0 (live Q4_0 frame)
      </text>
    </Figure>
  );
}

function BackendVtableFigure() {
  return (
    <Figure
      viewBox="0 0 460 240"
      title="one vtable, many backends"
      caption="Solid = Metal, dashed = native CPU — the same convention every figure on this site uses."
    >
      <Box x={140} y={15} w={180} h={48} label="ComputeBackend" sublabel="vtable: one fn per op" accent />
      <FlowArrow x1={200} y1={63} x2={110} y2={105} />
      <FlowArrow x1={260} y1={63} x2={350} y2={105} />
      <Box x={20} y={108} w={185} h={48} label="Metal executor" sublabel="DeviceMesh priority 10" />
      <Box x={255} y={108} w={185} h={48} label="native CPU (SIMD Zig)" sublabel="priority 0 · supports all" dashed />
      <FlowArrow x1={112} y1={156} x2={112} y2={185} />
      <Box x={20} y={188} w={185} h={40} label="generated quant routes" sublabel="schedule table → MSL" />
      <text x={347} y={205} textAnchor="middle" fontSize={8} className="fill-muted-foreground font-mono">
        DeviceMesh partitions each graph by
      </text>
      <text x={347} y={216} textAnchor="middle" fontSize={8} className="fill-muted-foreground font-mono">
        priority; native is the total fallback
      </text>
    </Figure>
  );
}

function KvPagesFigure() {
  const size = 20;
  const gap = 5;
  const row = (y: number, states: Array<"empty" | "filled" | "evicted" | "shared">) =>
    states.map((s, i) => <KvBlockGlyph key={`${y}-${i}`} x={15 + i * (size + gap)} y={y} size={size} state={s} />);
  return (
    <Figure
      viewBox="0 0 460 230"
      title="paged KV: 16-token blocks"
      caption="A sequence is a block table over a shared pool — full pages behind a sliding window are evicted; shared prefixes point at the same physical blocks."
    >
      <text x={15} y={30} fontSize={9} className="fill-muted-foreground font-mono">
        global layer · keeps everything
      </text>
      {row(38, ["filled", "filled", "filled", "filled", "filled", "filled", "filled", "filled", "filled", "filled", "empty", "empty"])}

      <text x={15} y={95} fontSize={9} className="fill-muted-foreground font-mono">
        sliding-window layer · evicts behind the window
      </text>
      {row(103, ["evicted", "evicted", "evicted", "evicted", "filled", "filled", "filled", "filled", "filled", "filled", "empty", "empty"])}
      <path d="M 115 130 v 5 h 149 v -5" fill="none" stroke="var(--primary)" strokeWidth={1} />
      <text x={190} y={148} textAnchor="middle" fontSize={8} className="fill-primary font-mono">
        window
      </text>

      <text x={15} y={175} fontSize={9} className="fill-muted-foreground font-mono">
        second request, same prompt prefix · shares blocks
      </text>
      {row(183, ["shared", "shared", "shared", "shared", "filled", "filled", "empty", "empty", "empty", "empty", "empty", "empty"])}
    </Figure>
  );
}

function GenerationLoopFigure() {
  return (
    <Figure
      viewBox="0 0 460 230"
      title="one scheduler step"
      caption="Each step packs prefill chunks and decode tokens into one batch; the sampled token never has to visit the host."
    >
      <text x={12} y={22} fontSize={9} className="fill-muted-foreground font-mono">
        batched step
      </text>
      {/* prefill chunk */}
      <rect x={12} y={32} width={190} height={30} rx={3} fill="var(--kfam-mmsg)" opacity={0.7} />
      <text x={107} y={47} textAnchor="middle" dominantBaseline="central" fontSize={9} className="fill-foreground font-mono">
        prefill chunk · seq A
      </text>
      {/* decode tokens */}
      {[0, 1, 2, 3].map((i) => (
        <g key={i}>
          <rect x={212 + i * 36} y={32} width={30} height={30} rx={3} fill="var(--kfam-matvec)" opacity={0.7} />
          <text
            x={227 + i * 36}
            y={47}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={8}
            className="fill-foreground font-mono"
          >
            t{i}
          </text>
        </g>
      ))}
      <text x={280} y={78} textAnchor="middle" fontSize={8} className="fill-muted-foreground font-mono">
        decode tokens · seqs B–E
      </text>

      <FlowArrow x1={180} y1={90} x2={180} y2={120} />
      <Box x={100} y={124} w={160} h={36} label="forward → logits" />
      <FlowArrow x1={260} y1={142} x2={310} y2={142} />
      <SamplerGlyph x={318} y={125} label="device-resident sample" />
      <ForkGlyph x={318} y={185} w={60} h={30} branches={2} label="speculative / MTP hooks" />
      <FlowArrow x1={335} y1={162} x2={335} y2={182} dashed />
    </Figure>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export function RuntimeClient({
  snippets,
  gitCommit,
  permalinkBase,
}: {
  snippets: Record<string, ClientSnippet>;
  gitCommit: string;
  permalinkBase?: string;
}) {
  return (
    <SnippetProvider snippets={snippets} gitCommit={gitCommit} permalinkBase={permalinkBase}>
      <div className="py-8">
        <header className="mx-auto max-w-7xl px-4">
          <h1 className="text-3xl font-bold tracking-tight">The spine</h1>
          <p className="mt-2 max-w-3xl text-muted-foreground">
            Every model on this site — generative, embedder, or extractor — rides the same eight-stage pipeline
            from HTTP request to output. This page walks the shared runtime once; the model pages then only have
            to explain where they diverge from it.
          </p>
          <SpineStrip className="mt-4" />
        </header>

        {/* ── 1 · HTTP ─────────────────────────────────────────────── */}
        <ScrollyChapter id="http" number={1} title="A request arrives">
          <Scene id="doors" graphic={<HttpDoorsFigure />}>
            <p>
              Four doors into one hall. <code>generateEmbeddings</code>, <code>rerankPrompts</code>,{" "}
              <code>generateContent</code>, and <code>chatCompletions</code> are four handlers on the same
              server binary — there is no separate embedding service or chat service. Past request parsing and
              admission, all four converge on the same session, graph, and kernel machinery below.
            </p>
            <p>
              <CodeLink link={L("server-embeddings")} /> · <CodeLink link={L("server-rerank")} /> ·{" "}
              <CodeLink link={L("server-generate")} /> · <CodeLink link={L("server-chat")} />
            </p>
            <ModelUses
              items={[
                { slug: "gemma4-e4b", label: "gemma4", note: "chat + generate" },
                { slug: "gliner2", label: "gliner2", note: "extraction" },
                { slug: "qwen3-embedding", label: "qwen3-embedding", note: "embeddings" },
                { slug: "qwen3-vl", label: "qwen3-vl", note: "multimodal chat" },
              ]}
            />
          </Scene>
        </ScrollyChapter>

        {/* ── 2 · Session ──────────────────────────────────────────── */}
        <ScrollyChapter id="session" number={2} title="The session factory">
          <Scene id="factory" graphic={<SessionFactoryFigure />}>
            <p>
              The factory detects a <code>ModelFamily</code> from the model's config — gemma, qwen3, qwen3_vl,
              gliner2/deberta, and the rest — and builds the matching architecture session around the loaded
              weights. The switch is boring on purpose: one detection, one construction, and everything
              downstream is family code.
            </p>
            <p>
              The model manager caches loaded sessions, so the first request pays the weight load and every
              later request finds the session warm.
            </p>
            <p>
              <CodeLink link={L("session-factory")} />
            </p>
            <ModelUses
              items={[
                { slug: "gemma4-e4b", label: "gemma4", note: "gpt/gemma family" },
                { slug: "gliner2", label: "gliner2", note: "deberta encoder" },
                { slug: "qwen3-embedding", label: "qwen3-embedding" },
                { slug: "qwen3-vl", label: "qwen3-vl", note: "vision tower + decoder" },
              ]}
            />
          </Scene>
        </ScrollyChapter>

        {/* ── 3 · Tokenizer ────────────────────────────────────────── */}
        <ScrollyChapter id="tokenizer" number={3} title="Two tokenizers">
          <Scene id="pieces" graphic={<TokenizerFigure />}>
            <p>
              The runtime carries two tokenizer implementations behind one interface. Gemma and DeBERTa-v3
              (GLiNER2) ship SentencePiece models; the Qwen3 family ships a HuggingFace-style BPE{" "}
              <code>tokenizer.json</code>. Same input string, different piece inventories, different ids — which
              is why the tokenizer is a per-model artifact, not a runtime constant.
            </p>
            <p>
              <CodeLink link={L("tokenizer-main")} /> · <CodeLink link={L("tokenizer-sentencepiece")} /> ·{" "}
              <CodeLink link={L("tokenizer-hf")} />
            </p>
            <ModelUses
              items={[
                { slug: "gemma4-e4b", label: "gemma4", note: "SentencePiece, 262k vocab" },
                { slug: "gliner2", label: "gliner2", note: "SentencePiece via DeBERTa-v3" },
                { slug: "qwen3-embedding", label: "qwen3-embedding", note: "BPE" },
                { slug: "qwen3-vl", label: "qwen3-vl", note: "BPE + image placeholder tokens" },
              ]}
            />
          </Scene>
        </ScrollyChapter>

        {/* ── 4 · Graph ────────────────────────────────────────────── */}
        <ScrollyChapter id="graph" number={4} title="The op graph">
          <Scene id="gallery" graphic={<OpGalleryFigure />}>
            <p>
              Forward passes are described in an XLA-inspired IR: trace the computation once, cache the graph
              per shape class, replay it thereafter. The vocabulary is deliberately small — 34 primitive ops
              plus 38 fused ops, 72 total — and the fused tier is where architectures live:{" "}
              <code>gqa_paged_attention</code>, <code>windowed_self_attention</code>,{" "}
              <code>disentangled_relative_attention</code>, the <code>moe_*</code> quartet.
            </p>
            <p>
              A new model that fits this vocabulary needs no new backend code; a model that doesn't gets one new
              fused op, implemented once per backend.
            </p>
            <p>
              <CodeLink link={L("node-primitive-op")} /> · <CodeLink link={L("node-fused-op")} />
            </p>
            <ModelUses
              items={[
                { slug: "gemma4-e4b", label: "gemma4", note: "windowed + moe_*" },
                { slug: "gliner2", label: "gliner2", note: "disentangled_relative_attention" },
                { slug: "qwen3-embedding", label: "qwen3-embedding", note: "gqa attention" },
                { slug: "qwen3-vl", label: "qwen3-vl", note: "conv2d patches + gqa" },
              ]}
            />
          </Scene>
        </ScrollyChapter>

        {/* ── 5 · Frames ───────────────────────────────────────────── */}
        <ScrollyChapter id="frames" number={5} title="Planned frames, not walked graphs">
          <Scene id="plan" graphic={<GraphToFrameFigure />}>
            <p>
              On Metal, the graph is not walked at decode time. The planner compiles it — once — into a{" "}
              <code>FrameDescriptor</code>: an ordered list of <code>PlannedOp</code>s grouped into{" "}
              <code>EncoderScope</code>s, with barriers derived from actual read/write byte ranges rather than
              sprinkled defensively.
            </p>
            <p>
              <CodeLink link={L("planner-frame-descriptor")} />
            </p>
          </Scene>
          <Scene id="rebind" graphic={<RebindFigure />}>
            <p>
              Every subsequent step re-binds and re-submits the same frame. The live Gemma4 Q4_0 decode frame is
              one compute encoder, 143 planned scopes, zero planned barriers — and with device-resident token
              handoff, frame N+1 can be encoded before frame N finishes.
            </p>
            <Divergence
              others={<p>llama.cpp re-encodes its graph op-by-op, every token.</p>}
              antfly={<p>plan once, re-bind per step; barriers only where byte ranges actually collide.</p>}
              link={<CodeLink link={L("runtime-submit-frame")} />}
            />
            <p className="text-xs">
              The full story, with the barrier census:{" "}
              <Link className="text-primary underline" href="/models/gemma4-e4b#ch-7">
                Gemma4 chapter 7 →
              </Link>{" "}
              ·{" "}
              <Link className="text-primary underline" href="/systems/timeline">
                frame timeline →
              </Link>
            </p>
            <ModelUses
              items={[
                { slug: "gemma4-e4b", label: "gemma4", note: "pipelined decode frames" },
                { slug: "qwen3-embedding", label: "qwen3-embedding", note: "batched prefill frames" },
                { slug: "qwen3-vl", label: "qwen3-vl" },
              ]}
            />
          </Scene>
        </ScrollyChapter>

        {/* ── 6 · Kernels ──────────────────────────────────────────── */}
        <ScrollyChapter id="kernels" number={6} title="One vtable, many backends">
          <Scene id="vtable" graphic={<BackendVtableFigure />}>
            <p>
              Every op dispatches through the <code>ComputeBackend</code> vtable to a native CPU implementation
              (SIMD Zig) or a Metal one. The <code>DeviceMesh</code> partitions each graph by backend priority —
              Metal at 10, native at 0 — so anything Metal can't run falls back to the CPU implementation of the
              same op, not to an error.
            </p>
            <p>
              The small-batch quantized matvec kernels aren't hand-written at all: the quant-kernel compiler
              renders them from one schedule table into MSL at build time.
            </p>
            <p>
              <CodeLink link={L("ops-compute-backend")} /> · <CodeLink link={L("multi-executor")} /> ·{" "}
              <CodeLink link={L("compiler-schedules")} />
            </p>
            <p className="text-xs">
              Full route table:{" "}
              <Link className="text-primary underline" href="/systems/kernels">
                kernel routing →
              </Link>
            </p>
            <ModelUses
              items={[
                { slug: "gemma4-e4b", label: "gemma4" },
                { slug: "gliner2", label: "gliner2", note: "fastest backend is batch-dependent" },
                { slug: "qwen3-embedding", label: "qwen3-embedding" },
                { slug: "qwen3-vl", label: "qwen3-vl" },
              ]}
            />
          </Scene>
        </ScrollyChapter>

        {/* ── 7 · KV ───────────────────────────────────────────────── */}
        <ScrollyChapter id="kv" number={7} title="Paged KV">
          <Scene id="pages" graphic={<KvPagesFigure />}>
            <p>
              KV cache is paged: 16-token blocks in a shared pool, with a block table per sequence. Sliding-window
              layers evict whole pages behind the window; identical prompt prefixes attach to the same physical
              blocks instead of recomputing them; and TurboQuant can store keys at 4 bits (polar4) instead of
              f16.
            </p>
            <p>
              <CodeLink link={L("kv-manager")} /> · <CodeLink link={L("kv-pool-config")} /> ·{" "}
              <CodeLink link={L("kv-prompt-cache")} />
            </p>
            <p className="text-xs">
              Watch the blocks move:{" "}
              <Link className="text-primary underline" href="/systems/kv">
                KV cache tool →
              </Link>
            </p>
            <ModelUses
              items={[
                { slug: "gemma4-e4b", label: "gemma4", note: "window eviction + shared-KV tail" },
                { slug: "qwen3-embedding", label: "qwen3-embedding", note: "one pass, then released" },
                { slug: "qwen3-vl", label: "qwen3-vl", note: "image-token prefill burst" },
                { slug: "gliner2", label: "gliner2", note: "encoder — no KV at all" },
              ]}
            />
          </Scene>
        </ScrollyChapter>

        {/* ── 8 · Sample ───────────────────────────────────────────── */}
        <ScrollyChapter id="sample" number={8} title="The generation loop">
          <Scene id="loop" graphic={<GenerationLoopFigure />}>
            <p>
              The batched step scheduler packs work, not requests: each step mixes prefill chunks from new
              sequences with single decode tokens from running ones. Sampling happens on-device — logits never
              round-trip to the host for a Gumbel-max draw — and the loop carries hooks for speculative decoding
              and MTP drafts where a model provides them.
            </p>
            <p>
              <CodeLink link={L("generation-config")} /> · <CodeLink link={L("native-generate-scheduler")} /> ·{" "}
              <CodeLink link={L("kernel-gumbel")} />
            </p>
            <ModelUses
              items={[
                { slug: "gemma4-e4b", label: "gemma4", note: "device-resident Gumbel-max + MTP" },
                { slug: "qwen3-vl", label: "qwen3-vl", note: "decode after vision prefill" },
                { slug: "qwen3-embedding", label: "qwen3-embedding", note: "no sampling — pooled hidden state" },
              ]}
            />
          </Scene>
        </ScrollyChapter>
      </div>
    </SnippetProvider>
  );
}
