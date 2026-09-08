"use client";

import { Figure } from "@/components/viz/glyphs";

/* ------------------------------------------------------------------ */
/* Ch 1 — a decoder whose output bends away from the LM head           */
/* ------------------------------------------------------------------ */

export function VectorNotTokenFigure() {
  return (
    <Figure
      viewBox="0 0 420 230"
      title="same stack, different exit"
      caption="The 28 decoder layers are stock Qwen3. What changes is the exit ramp: instead of the vocabulary matvec and a sampler, the last hidden state leaves sideways as a 1,024-dim vector."
    >
      {/* compressed layer stack */}
      {Array.from({ length: 9 }, (_, i) => (
        <rect
          key={i}
          x={60}
          y={30 + i * 14}
          width={180}
          height={10}
          rx={2}
          fill="var(--kfam-attention)"
          opacity={0.28 + (i % 3) * 0.08}
        />
      ))}
      <text x={150} y={22} textAnchor="middle" fontSize={9} className="fill-muted-foreground font-mono">
        28 × Qwen3 decoder layer
      </text>
      {/* ghosted LM head path */}
      <line x1={150} y1={158} x2={150} y2={186} stroke="var(--muted-foreground)" strokeWidth={1.25} strokeDasharray="5 4" opacity={0.4} />
      <g opacity={0.35}>
        <rect x={95} y={190} width={110} height={24} rx={3} fill="none" stroke="var(--kfam-sampling)" strokeWidth={1} strokeDasharray="4 3" />
        <text x={150} y={206} textAnchor="middle" fontSize={8.5} className="fill-muted-foreground font-mono">
          LM head · sampler
        </text>
        <line x1={120} y1={186} x2={180} y2={218} stroke="var(--destructive)" strokeWidth={1.25} />
        <line x1={120} y1={218} x2={180} y2={186} stroke="var(--destructive)" strokeWidth={1.25} />
      </g>
      {/* the bend */}
      <path d="M 240 152 C 290 152, 300 120, 330 110" fill="none" stroke="var(--primary)" strokeWidth={2} />
      <rect x={318} y={78} width={84} height={56} rx={6} fill="color-mix(in oklch, var(--dtype-f32) 14%, transparent)" stroke="var(--dtype-f32)" strokeWidth={1.5} />
      <text x={360} y={100} textAnchor="middle" fontSize={9} className="fill-foreground font-mono">
        [1024]
      </text>
      <text x={360} y={116} textAnchor="middle" fontSize={8} className="fill-muted-foreground font-mono">
        ‖x‖ = 1
      </text>
      <text x={360} y={150} textAnchor="middle" fontSize={8.5} className="fill-primary font-mono">
        the embedding
      </text>
    </Figure>
  );
}

/* ------------------------------------------------------------------ */
/* Ch 2 — BPE vs SentencePiece (illustrative)                          */
/* ------------------------------------------------------------------ */

const BPE_PIECES = ["ant", "fly", " runs", " on", " Metal"];
const SP_PIECES = ["▁ant", "fly", "▁runs", "▁on", "▁Metal"];

export function TokenizerContrastFigure() {
  const row = (label: string, pieces: string[], color: string, note: string) => (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">{label}</span>
        <span className="font-mono text-[10px] text-muted-foreground/70">{note}</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {pieces.map((p, i) => (
          <span
            key={i}
            className="rounded border px-1.5 py-0.5 font-mono text-xs"
            style={{ borderColor: color, color }}
          >
            {p.replace(/ /g, "␣")}
          </span>
        ))}
      </div>
    </div>
  );
  return (
    <div className="flex h-full flex-col justify-center gap-6">
      <div className="rounded-lg border bg-muted/30 p-3 text-center font-mono text-sm">"antfly runs on Metal"</div>
      {row("Qwen3 · byte-level BPE", BPE_PIECES, "var(--kfam-attention)", "byte fallback — no unknown tokens, ever")}
      {row("Gemma · SentencePiece", SP_PIECES, "var(--kfam-fusion)", "▁ marks word starts; unigram LM pieces")}
      <p className="text-center font-mono text-[10px] text-muted-foreground">
        illustrative split — the point is the two families, not these exact pieces
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Ch 3 — last-token pooling + unit sphere                             */
/* ------------------------------------------------------------------ */

export function LastTokenPoolFigure({ step }: { step: 0 | 1 }) {
  const rows = 9;
  return (
    <Figure
      viewBox="0 0 420 220"
      title={step === 0 ? "T hidden states in, one comes out" : "…then scaled onto the unit sphere"}
      caption={
        step === 0
          ? "qwen3.pooling_type = 3 in the GGUF metadata: keep the last token's hidden state, discard the rest. The trailing EOS token (add_eos = 1) is the one that pooled — it has attended to the entire input."
          : "L2 normalization puts every embedding on the unit sphere, so cosine similarity downstream is a plain dot product."
      }
    >
      {Array.from({ length: rows }, (_, i) => {
        const last = i === rows - 1;
        return (
          <g key={i}>
            <rect
              x={40}
              y={24 + i * 19}
              width={170}
              height={14}
              rx={3}
              fill="var(--dtype-f32)"
              opacity={last ? 0.85 : Math.max(0.06, 0.35 - i * 0.03)}
              stroke={last ? "var(--primary)" : "none"}
              strokeWidth={1.5}
            />
            {last && (
              <text x={218} y={35 + i * 19} fontSize={8.5} className="fill-primary font-mono">
                ← ⟨EOS⟩: saw everything
              </text>
            )}
          </g>
        );
      })}
      {step === 1 && (
        <>
          <circle cx={330} cy={110} r={52} fill="none" stroke="var(--muted-foreground)" strokeWidth={1} strokeDasharray="3 3" />
          <line x1={330} y1={110} x2={366} y2={73} stroke="var(--primary)" strokeWidth={2} />
          <circle cx={366} cy={73} r={3.5} fill="var(--primary)" />
          <line x1={330} y1={110} x2={352} y2={88} stroke="var(--muted-foreground)" strokeWidth={1.25} strokeDasharray="4 3" />
          <text x={330} y={185} textAnchor="middle" fontSize={8.5} className="fill-muted-foreground font-mono">
            x / ‖x‖ → cos(a,b) = a·b
          </text>
        </>
      )}
    </Figure>
  );
}

/* ------------------------------------------------------------------ */
/* Ch 4 — cosine vs reference across context length                    */
/* ------------------------------------------------------------------ */

export function CosineFidelityFigure() {
  const W = 400;
  const H = 170;
  const xOf = (t: number) => 50 + (Math.log2(t / 128) / Math.log2(8192 / 128)) * (W - 80);
  const yOf = (c: number) => 30 + (1 - (c - 0.999) / 0.001) * (H - 60);
  return (
    <Figure
      viewBox={`0 0 ${W} ${H + 30}`}
      title="Q8_0 vs reference, cosine of embeddings"
      caption="The qualification gate: cosine similarity between Antfly's Q8_0 Metal embedding and the reference implementation at the full 8,192-token qualification input — 0.99976. The line is a guide; the circled point is the measured gate. Axis starts at 0.999: the interesting failure modes live in the third decimal."
    >
      <line x1={50} y1={H - 30} x2={W - 25} y2={H - 30} stroke="var(--border)" strokeWidth={1} />
      <line x1={50} y1={H - 30} x2={50} y2={24} stroke="var(--border)" strokeWidth={1} />
      <text x={30} y={34} fontSize={8} className="fill-muted-foreground font-mono">1.0</text>
      <text x={22} y={H - 26} fontSize={8} className="fill-muted-foreground font-mono">0.999</text>
      <line
        x1={xOf(128)}
        y1={yOf(0.99976) - 4}
        x2={xOf(8192)}
        y2={yOf(0.99976)}
        stroke="var(--kfam-attention)"
        strokeWidth={1.5}
        strokeDasharray="5 4"
        opacity={0.6}
      />
      <circle cx={xOf(8192)} cy={yOf(0.99976)} r={3} fill="var(--kfam-attention)" />
      <circle cx={xOf(8192)} cy={yOf(0.99976)} r={6} fill="none" stroke="var(--primary)" strokeWidth={1.5} />
      <text x={xOf(8192) - 8} y={yOf(0.99976) - 12} textAnchor="end" fontSize={8.5} className="fill-primary font-mono">
        8,192 tok · 0.99976
      </text>
      {[128, 512, 2048, 8192].map((t) => (
        <text key={t} x={xOf(t)} y={H - 16} textAnchor="middle" fontSize={8} className="fill-muted-foreground font-mono">
          {t >= 1024 ? `${t / 1024}k` : t}
        </text>
      ))}
      <text x={(W + 25) / 2} y={H - 2} textAnchor="middle" fontSize={8} className="fill-muted-foreground font-mono">
        input length (log) →
      </text>
    </Figure>
  );
}

/* ------------------------------------------------------------------ */
/* Ch 5 — three stacked perf wins                                      */
/* ------------------------------------------------------------------ */

const WINS = [
  {
    label: "batched FFN",
    detail: "run [T×F] matmuls, not T matvecs — the single largest lever",
    color: "var(--kfam-matvec)",
    w: 0.95,
  },
  {
    label: "simdgroup flash attention",
    detail: "sg_q16 tiles, online softmax, no [T,T] score matrix",
    color: "var(--kfam-attention)",
    w: 0.7,
  },
  {
    label: "f16-KV direct load",
    detail: "attention reads f16 K/V straight — half the KV bytes",
    color: "var(--kfam-kv)",
    w: 0.45,
  },
];

export function BatchingWinsFigure() {
  return (
    <div className="flex h-full flex-col justify-center gap-5">
      <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
        the three levers behind 1,217 tok/s
      </div>
      <div className="grid gap-4">
        {WINS.map((win) => (
          <div key={win.label}>
            <div className="mb-1 flex items-baseline justify-between">
              <span className="font-mono text-xs text-foreground">{win.label}</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-muted">
              <div className="h-full rounded-full" style={{ width: `${win.w * 100}%`, background: win.color }} />
            </div>
            <div className="mt-1 text-[11px] text-muted-foreground">{win.detail}</div>
          </div>
        ))}
      </div>
      <div className="rounded-lg border bg-muted/30 p-3">
        <span className="text-2xl font-bold tabular-nums">1,217</span>
        <span className="ml-2 text-sm text-muted-foreground">embed tok/s · 511-token input · M4 Pro · Q8_0</span>
        <div className="mt-1 font-mono text-[10px] text-muted-foreground">
          bar lengths are ordinal (relative impact), not measured ratios
        </div>
      </div>
    </div>
  );
}
