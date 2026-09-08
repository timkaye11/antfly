"use client";

import { EnvFlagChip, FusionChip, OpKindBadge, QuantChip, TensorShapeBadge } from "@/components/primitives/chips";
import { Divergence } from "@/components/scrollytelling/scrolly";
import {
  ActivationGlyph,
  AttentionGlyph,
  ElementwiseGlyph,
  EmbeddingGlyph,
  Figure,
  ForkGlyph,
  KvBlockGlyph,
  MatmulGlyph,
  NormGlyph,
  SamplerGlyph,
  WeightGlyph,
} from "@/components/viz/glyphs";

const DTYPE_RAMP = [
  ["f32", "var(--dtype-f32)", "full precision"],
  ["f16 / bf16", "var(--dtype-f16)", "half precision"],
  ["q8", "var(--dtype-q8)", "8-bit blocks"],
  ["q4–q6 / iq4 / mxfp4", "var(--dtype-q4)", "4–6-bit blocks"],
  ["sub-4-bit (tl1/tl2, polar4)", "var(--dtype-sub4)", "ternary / compressed KV"],
] as const;

const KERNEL_FAMILIES = [
  ["matvec", "var(--kfam-matvec)", "decode-time quantized matrix-vector"],
  ["mm_sg", "var(--kfam-mmsg)", "simdgroup tensor-core matmul (batch ≥ 8)"],
  ["attention", "var(--kfam-attention)", "flash / paged / disentangled attention"],
  ["fusion", "var(--kfam-fusion)", "pair-fusion, norm⋄rope, fused epilogues"],
  ["moe", "var(--kfam-moe)", "expert routing, scatter, slot arena"],
  ["sampling", "var(--kfam-sampling)", "device-resident Gumbel-max / argmax / top-8"],
  ["kv", "var(--kfam-kv)", "KV seed / compress (polar4, turbo3)"],
] as const;

function Swatch({ color, label, note }: { color: string; label: string; note: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="size-5 shrink-0 rounded" style={{ background: color }} />
      <span className="w-56 font-mono text-sm">{label}</span>
      <span className="text-sm text-muted-foreground">{note}</span>
    </div>
  );
}

export function LegendClient() {
  return (
    <div className="mx-auto max-w-4xl space-y-12 px-4 py-10">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">Visual vocabulary</h1>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          One legend for the whole explorer. Once a shape or color means something here, it never means
          anything else — every figure states which color axis it uses (<code>colorBy</code>), and axes are
          never mixed in one figure.
        </p>
      </header>

      <section id="shapes">
        <h2 className="mb-4 text-xl font-semibold">Shapes</h2>
        <div className="rounded-lg border bg-card p-4">
          <Figure viewBox="0 0 760 260">
            <ActivationGlyph x={20} y={30} label="activation tensor" sublabel="width ∝ log(dim)" dtype="f16" />
            <WeightGlyph x={170} y={30} label="weight (on disk)" quant="q4_0" />
            <MatmulGlyph x={320} y={28} label="matmul / linear" dtype="f16" />
            <AttentionGlyph x={490} y={24} label="attention" dtype="f16" />
            <AttentionGlyph x={620} y={24} label="+ sliding window" dtype="f16" shutter />

            <NormGlyph x={20} y={120} label="norm (small on purpose)" dtype="f32" />
            <ElementwiseGlyph x={170} y={108} label="elementwise" dtype="f16" />
            <EmbeddingGlyph x={260} y={100} label="embedding lookup" dtype="f16" />
            <ForkGlyph x={420} y={100} label="routing (MoE / MTP)" dtype="q4_0" />
            <SamplerGlyph x={560} y={100} label="sampling" dtype="f32" />
            <MatmulGlyph x={620} y={96} w={120} label="fused (zipper)" fused={["gate", "up", "silu"]} dtype="q4_0" />

            <g transform="translate(20, 190)">
              <KvBlockGlyph x={0} y={0} state="filled" />
              <KvBlockGlyph x={20} y={0} state="evicted" />
              <KvBlockGlyph x={40} y={0} state="shared" />
              <KvBlockGlyph x={60} y={0} state="empty" />
              <text x={85} y={11} fontSize={10} className="fill-muted-foreground font-mono">
                KV blocks: filled · evicted (hatched) · shared (split) · free
              </text>
            </g>
          </Figure>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Solid outline = runs on Metal; dashed outline = native CPU path. A thick bottom border marks weights
          that stream from disk. Norm bars are deliberately tiny — they set up the systems story about hundreds
          of small dispatches.
        </p>
      </section>

      <section id="colors">
        <h2 className="mb-4 text-xl font-semibold">Color axes</h2>
        <div className="grid gap-8 md:grid-cols-2">
          <div>
            <h3 className="mb-3 font-mono text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              colorBy: dtype (default)
            </h3>
            <div className="space-y-2">
              {DTYPE_RAMP.map(([label, color, note]) => (
                <Swatch key={label} color={color} label={label} note={note} />
              ))}
            </div>
          </div>
          <div>
            <h3 className="mb-3 font-mono text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              colorBy: kernel family
            </h3>
            <div className="space-y-2">
              {KERNEL_FAMILIES.map(([label, color, note]) => (
                <Swatch key={label} color={color} label={label} note={note} />
              ))}
            </div>
          </div>
        </div>
      </section>

      <section id="chips">
        <h2 className="mb-4 text-xl font-semibold">Chips</h2>
        <div className="flex flex-wrap items-center gap-3 rounded-lg border bg-card p-4">
          <QuantChip format="q4_k" />
          <QuantChip format="q8_0" />
          <QuantChip format="iq4_xs" />
          <TensorShapeBadge shape={{ dims: ["B", "T", 2048], dtype: "f16" }} />
          <OpKindBadge opKind="dot_general" group="primitive" />
          <OpKindBadge opKind="gqa_paged_attention" group="fused" />
          <FusionChip ops={["head_rms", "rope"]} />
          <EnvFlagChip name="TERMITE_METAL_ENABLE_PIPELINED_DECODE_FRAME" defaultOn />
          <EnvFlagChip name="TERMITE_METAL_DISABLE_A4B_ZERO_BIAS_ELISION" defaultOn={false} />
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Quant chips use the precision ramp. Fused op kinds get the accent tint. Env-flag chips show the{" "}
          <em>current default</em> in the runtime (green dot = the behavior is on by default). File chips like{" "}
          <code className="font-mono text-xs">gpt.zig:1039</code> open a commit-pinned GitHub permalink and show
          a code peek on hover.
        </p>
      </section>

      <section id="divergences">
        <h2 className="mb-4 text-xl font-semibold">"Where Antfly diverges" callouts</h2>
        <p className="mb-3 max-w-2xl text-sm text-muted-foreground">
          The explorer's comparisons with llama.cpp / vLLM / PyTorch always appear in this exact form — muted
          column for them, full-color for Antfly, never more than three rows, always ending in a measurable or
          linkable claim:
        </p>
        <Divergence
          others={<p>reads logits back to the host and samples on the CPU.</p>}
          antfly={<p>keeps logits device-resident: top-8 rescore + Gumbel-max on the GPU, token id handed to the next frame without a round-trip.</p>}
        />
      </section>
    </div>
  );
}
