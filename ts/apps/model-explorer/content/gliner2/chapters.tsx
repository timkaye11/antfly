"use client";

import Link from "next/link";
import { CodeLink } from "@/components/code/code-link";
import { EnvFlagChip } from "@/components/primitives/chips";
import { Divergence, Scene, ScrollyChapter } from "@/components/scrollytelling/scrolly";
import { L } from "@/lib/links";
import type { ChaptersProps } from "../registry";
import {
  DisentangledScoresFigure,
  EncoderVsDecoderFigure,
  FusedKernelPairFigure,
  GlinerSpineFigure,
  LogBucketFigure,
  SchemaExtractionFigure,
  SpanHeadFigure,
} from "./figures";

export function Gliner2Chapters({ spec }: ChaptersProps) {
  return (
    <div>
      {/* ── Ch 1 · Extraction by schema, not prompt ─────────────────── */}
      <ScrollyChapter
        id="ch-1"
        number={1}
        title="Extraction by schema, not prompt"
        intro="You don't ask GLiNER2 a question. You hand it a schema."
      >
        <Scene id="schema" graphic={<SchemaExtractionFigure />}>
          <p>
            The input is a list of label names — <code>company</code>, <code>person</code>,{" "}
            <code>amount</code>, anything — encoded into the <em>same</em> token sequence as the text,
            separated by entity-marker tokens. One encoder pass later, every candidate span in the text has a
            score against every label. The spans above light up when their sigmoid score crosses the
            threshold; the example is illustrative, the mechanism is exactly what chapters 3–5 unpack.
          </p>
          <p>
            No generation, no prompt engineering, no decoding loop. Zero-shot means the labels are just
            embeddings — swap the schema and the same weights extract different things.
          </p>
          <p>
            <CodeLink link={L("gliner-pipeline")} />
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 2 · An encoder in a decoder world ────────────────────── */}
      <ScrollyChapter
        id="ch-2"
        number={2}
        title="An encoder in a decoder world"
        intro={
          <p>
            Everything else on this site decodes: token by token, causally masked, KV-cached. GLiNER2 is a{" "}
            {spec.stats.layers}-layer, hidden-{spec.stats.hidden} DeBERTa-v3 <em>encoder</em> — and that
            changes what the runtime even needs.
          </p>
        }
      >
        <Scene id="mask" graphic={<EncoderVsDecoderFigure emphasis="mask" />}>
          <p>
            <strong>No causal mask.</strong> Every token attends to every other token in both directions. For
            extraction that's not a nicety, it's the point: whether <em>"Ruiz"</em> is a person may hinge on
            the <em>"Dr."</em> before it and the <em>"hired"</em> before that — and on words that come after.
            The only mask GLiNER2 ever applies is padding.
          </p>
        </Scene>
        <Scene id="nokv" graphic={<EncoderVsDecoderFigure emphasis="kv" />}>
          <p>
            <strong>No decode loop, so no KV cache.</strong> The model runs one batched forward pass over{" "}
            <code>[B·T]</code> tokens (context {spec.stats.context}) and is done. There are no paged pools, no
            eviction, no sampler — the KV and sampling stages of the shared spine simply don't exist for this
            model.
          </p>
          <Divergence
            others={<p>serving stacks built around autoregression treat encoders as an afterthought bolted onto a decode engine.</p>}
            antfly={<p>the graph runtime doesn't care: an encoder is just a DAG with no KV edges — same planner, same frames, same kernels.</p>}
            link={<CodeLink link={L("gliner-deberta-graph")} />}
          />
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 3 · Disentangled attention ───────────────────────────── */}
      <ScrollyChapter
        id="ch-3"
        number={3}
        title="Disentangled attention: content and position, separately"
        intro={
          <p>
            DeBERTa's core idea: don't fold position into the token embedding. Keep <em>what a token is</em>{" "}
            and <em>where it is</em> as separate vectors, and let attention combine them three ways.
          </p>
        }
      >
        <Scene id="c2c" graphic={<DisentangledScoresFigure highlight="c2c" />}>
          <p>
            <strong>C2C</strong> is ordinary attention: content query dot content key, <code>Q_c · K_cᵀ</code>.
            If this were all, the model would be position-blind — "Acme hired Elena" and "Elena hired Acme"
            would score identically.
          </p>
        </Scene>
        <Scene id="c2p" graphic={<DisentangledScoresFigure highlight="all" />}>
          <p>
            <strong>C2P and P2C</strong> add position back — from both sides. Content-to-position asks "how
            much does <em>this word</em> care about something <em>k slots away</em>?"; position-to-content
            asks the reverse. The three score matrices sum and scale by <code>1/√(3·d)</code> before softmax.
          </p>
          <p>
            The position terms are Toeplitz — every diagonal shares one relative-position bucket — so Antfly
            computes each as a single <code>[T × num_rel]</code> GEMM and gathers scores along diagonals
            (<code>qi−ki+S−1</code>), never materializing the <code>[S·S, H]</code> expansion.
          </p>
          <p>
            <CodeLink link={L("gliner-fused-attn-gate")} />
          </p>
        </Scene>
        <Scene id="buckets" graphic={<LogBucketFigure />}>
          <p>
            <strong>Distance is bucketed, not raw.</strong> {String(spec.stats.relBuckets)} buckets cover
            ±511 relative positions: exact out to ±128, then logarithmically wider. Word order nearby is
            preserved precisely; far context blurs gracefully. And because position lives entirely inside
            attention, there are <em>no absolute position embeddings at all</em> — the embedding stage is
            lookup, LayerNorm, mask, nothing else.
          </p>
          <p>
            <CodeLink link={L("gliner-rel-bucket")} />
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 4 · One kernel, forward and backward ─────────────────── */}
      <ScrollyChapter
        id="ch-4"
        number={4}
        title="One kernel, forward and backward"
        intro={
          <p>
            The whole decomposition — C2C, C2P, P2C, softmax, context matmul — is one fused Metal kernel.
            Including the backward pass.
          </p>
        }
      >
        <Scene id="fused" graphic={<FusedKernelPairFigure />}>
          <p>
            <code>termite_disentangled_relative_attention_f32</code> (with a <code>_flash4</code> tiled
            variant) folds the five-op attention core into a single dispatch. The rarer half: matching{" "}
            <code>_bwd_scores</code>, <code>_bwd_dv</code>, <code>_bwd_dq_dk</code> and{" "}
            <code>_bwd_dqr_dkr</code> kernels make the fusion hold during <em>training</em> — LoRA fine-tuning
            of GLiNER2 runs through the same autodiff graph with hand-written gradients for the fused node.
          </p>
          <Divergence
            others={
              <p>
                PyTorch on MPS runs DeBERTa's disentangled attention as eager op soup — a chain of matmuls,
                gathers and adds per layer, and autograd replays a second chain backward.
              </p>
            }
            antfly={
              <p>
                one fused kernel pair, both directions, dispatched from the same planned-frame machinery as
                the decoders on this site.
              </p>
            }
            link={<CodeLink link={L("gliner-fused-attn-kernel")} />}
          />
          <p>
            <CodeLink link={L("gliner-fused-attn-bwd-kernel")} />{" "}
            <EnvFlagChip name="TERMITE_METAL_DISABLE_DEBERTA_FLASH_ATTENTION" defaultOn={false} />
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 5 · The span head ────────────────────────────────────── */}
      <ScrollyChapter
        id="ch-5"
        number={5}
        title="The span head: score everything at once"
        intro="After the encoder, extraction is three gathers and a matmul."
      >
        <Scene id="gather" graphic={<SpanHeadFigure step={0} />}>
          <p>
            The schema's label names went through the encoder alongside the text, so their representations are{" "}
            <em>contextualized against this document</em>. The head gathers them back out — hidden states at
            the positions where <code>input_ids == entity_token_id</code> — while token states pool into word
            representations (first sub-token per word, a dedicated Metal kernel).
          </p>
          <p>
            <CodeLink link={L("gliner-label-gather")} /> · <CodeLink link={L("gliner-word-embeddings-kernel")} />
          </p>
        </Scene>
        <Scene id="spans" graphic={<SpanHeadFigure step={1} />}>
          <p>
            <strong>Every candidate span, materialized cheaply.</strong> Word reps run through{" "}
            <code>project_start</code> and <code>project_end</code> MLPs; for each span up to{" "}
            {String(spec.stats.maxSpanWidth)} words wide, its start and end vectors concatenate to{" "}
            <code>[total_spans, 2H]</code>, ReLU, and project back down to a span representation. Labels get a
            parallel treatment through a downscaled mini-transformer with a GRU combine step — also a bespoke
            kernel.
          </p>
          <p>
            <CodeLink link={L("gliner-span-marker")} /> · <CodeLink link={L("gliner-gru-combine-kernel")} />
          </p>
        </Scene>
        <Scene id="score" graphic={<SpanHeadFigure step={2} />}>
          <p>
            Then the whole extraction problem collapses into <code>span_rep @ label_projᵀ</code> — one{" "}
            <code>[S × L]</code> matmul, a sigmoid, a threshold. Those are the highlights from chapter 1:
            every span-label pair scored simultaneously, independent of how many labels your schema declares.
          </p>
          <p>
            <CodeLink link={L("gliner-score-spans")} />
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 6 · The shared spine ─────────────────────────────────── */}
      <ScrollyChapter
        id="ch-6"
        number={6}
        title="Where GLiNER2 rides the shared spine"
        intro="Different model class, same machinery."
      >
        <Scene id="spine" graphic={<GlinerSpineFigure />}>
          <p>
            GLiNER2 enters through the same server, session manager and tokenizer as every decoder here, and
            its encoder layers are planned into Metal frames by the same command planner — its Q4_K linears
            dispatch through the same generated quant-kernel routes. What it skips, it skips structurally: no
            KV stage, no sampler.
          </p>
          <p>
            It also stresses the spine differently: at batch ≥ 8 the Metal path overtakes the native CPU
            backend for GLiNER2 — encoder workloads batch wide instead of decoding deep.
          </p>
          <p className="text-xs">
            The spine in full: <Link className="text-primary underline" href="/runtime">runtime walkthrough →</Link>
          </p>
        </Scene>
      </ScrollyChapter>
    </div>
  );
}
