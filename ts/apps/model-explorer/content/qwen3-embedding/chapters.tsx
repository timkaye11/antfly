"use client";

import Link from "next/link";
import { CodeLink } from "@/components/code/code-link";
import { EnvFlagChip, QuantChip } from "@/components/primitives/chips";
import { Divergence, Scene, ScrollyChapter } from "@/components/scrollytelling/scrolly";
import { L } from "@/lib/links";
import type { ChaptersProps } from "../registry";
import {
  BatchingWinsFigure,
  CosineFidelityFigure,
  LastTokenPoolFigure,
  TokenizerContrastFigure,
  VectorNotTokenFigure,
} from "./figures";

export function Qwen3EmbeddingChapters({ spec }: ChaptersProps) {
  return (
    <div>
      {/* ── Ch 1 · A decoder that outputs one vector ────────────────── */}
      <ScrollyChapter
        id="ch-1"
        number={1}
        title="A decoder that outputs one vector"
        intro={
          <p>
            {spec.stats.layers} layers, hidden {spec.stats.hidden}, GQA {spec.stats.queryHeads}/
            {spec.stats.kvHeads} — a perfectly ordinary Qwen3 decoder. Then it refuses to decode.
          </p>
        }
      >
        <Scene id="bend" graphic={<VectorNotTokenFigure />}>
          <p>
            An embedding request runs the whole prompt through the stack as one batched prefill and stops.
            There is no LM head matvec, no sampler, no second forward pass — the final hidden state exits
            sideways as a {spec.stats.hidden}-dimensional vector. The decode loop that dominates every other
            decoder page on this site simply never starts.
          </p>
          <p>
            The same <code>gpt.zig</code> runtime serves it; the embedding pipeline just wires{" "}
            <code>tokenize → encode → pool → normalize</code> around a forward pass.
          </p>
          <p>
            <CodeLink link={L("embed-pipeline")} /> <QuantChip format="q8_0" />
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 2 · BPE, not SentencePiece ───────────────────────────── */}
      <ScrollyChapter
        id="ch-2"
        number={2}
        title="BPE, not SentencePiece"
        intro="Two tokenizer families live in this runtime; embeddings ride the byte-level one."
      >
        <Scene id="tok" graphic={<TokenizerContrastFigure />}>
          <p>
            Qwen3 uses byte-level BPE (the <code>gpt2</code>-style vocab with the <code>qwen2</code>{" "}
            pre-tokenizer): any byte sequence tokenizes, so there is no unknown-token path at all. Gemma, one
            page over, uses SentencePiece with its ▁-prefixed unigram pieces. Same tokenizer stage in the
            spine, different machinery behind it.
          </p>
          <p>
            One detail that matters here: the tokenizer appends EOS (<code>add_eos_token = 1</code>, no BOS).
            Chapter 3 explains why that trailing token is the entire point.
          </p>
          <p>
            <CodeLink link={L("tokenizer-hf")} /> · <CodeLink link={L("qwen3-embed-catalog")} />
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 3 · Last-token pooling ───────────────────────────────── */}
      <ScrollyChapter
        id="ch-3"
        number={3}
        title="Last-token pooling"
        intro={
          <p>
            The GGUF metadata says it in one key: <code>qwen3.pooling_type = 3</code>.
          </p>
        }
      >
        <Scene id="pool" graphic={<LastTokenPoolFigure step={0} />}>
          <p>
            <strong>Why the last token?</strong> In a causal model, only the final position has attended to
            everything before it — every earlier hidden state is blind to what follows. Mean pooling (the BERT
            default) would average in those half-informed states; last-token pooling takes the one vector that
            saw the whole input. That's the appended EOS token doing its job.
          </p>
          <Divergence
            others={
              <p>
                encoder embedders mean-pool or CLS-pool, because every bidirectional position sees everything
                anyway.
              </p>
            }
            antfly={
              <p>
                the pooling strategy is data, not code: read from model metadata, dispatched in the embedding
                pipeline — the same pipeline serves mean-, CLS- and last-token models.
              </p>
            }
            link={<CodeLink link={L("embed-pooling-strategy")} />}
          />
        </Scene>
        <Scene id="norm" graphic={<LastTokenPoolFigure step={1} />}>
          <p>
            The pooled vector is L2-normalized onto the unit sphere. Downstream, cosine similarity degrades
            into a dot product — which is exactly what a vector index wants to compute millions of times.
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 4 · Faithful at 8,192 tokens ─────────────────────────── */}
      <ScrollyChapter
        id="ch-4"
        number={4}
        title="Faithful at 8,192 tokens"
        intro="Quantized inference is only useful if the vectors still point the same way."
      >
        <Scene id="cosine" graphic={<CosineFidelityFigure />}>
          <p>
            The Q8_0 Metal path was qualified against the reference implementation end-to-end: at the full
            8,192-token qualification input the embedding cosine is <strong>0.99976</strong>. Getting there
            surfaced real bugs — the most instructive: the GGUF bundle omits{" "}
            <code>qwen3.context_length</code>, and a permissive default once <em>silently truncated</em>{" "}
            every input to 512 tokens. The embedding you got was a valid vector of the wrong text. That
            failure mode is now fail-loud.
          </p>
          <p>
            Context window: {Number(spec.stats.context).toLocaleString()} tokens; the qualification gate runs
            at 8,192.
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 5 · The batching path ────────────────────────────────── */}
      <ScrollyChapter
        id="ch-5"
        number={5}
        title="1,217 tok/s: the batching path"
        intro={
          <p>
            Embedding is prefill-shaped: all T tokens arrive at once. The perf story is the opposite of
            decode — compute-dense matmuls, not memory-bound matvecs.
          </p>
        }
      >
        <Scene id="wins" graphic={<BatchingWinsFigure />}>
          <p>
            Three levers took the 511-token embed path to 1,217 tok/s on an M4 Pro. First,{" "}
            <strong>batched FFN</strong>: run the feed-forward as <code>[T × F]</code> simdgroup matmuls (
            <code>termite_q8_0_linear_mm_sg</code>) instead of T separate matvecs — the largest single win.
            Second, <strong>simdgroup flash attention</strong>: <code>sg_q16</code> tiles with online softmax,
            never materializing the <code>[T, T]</code> score matrix. Third, <strong>f16-KV direct load</strong>:
            the attention kernel reads K/V in f16 natively, halving KV traffic.
          </p>
          <p>
            <CodeLink link={L("kernel-q8-mm-sg")} /> · <CodeLink link={L("kernel-sg-q16-f16kv-gqa2")} />
          </p>
          <p>
            <EnvFlagChip name="TERMITE_METAL_DISABLE_DENSE_CAUSAL_SG_ATTENTION_Q16" defaultOn={false} />{" "}
            <EnvFlagChip name="TERMITE_METAL_DISABLE_DENSE_CAUSAL_SG_ATTENTION_F16KV" defaultOn={false} />
          </p>
          <p className="text-xs">
            This is the same batch-size fork every model here crosses at batch ≥ 8:{" "}
            <Link className="text-primary underline" href="/systems/kernels?batch=32">
              the mm_sg kernel family →
            </Link>
          </p>
        </Scene>
      </ScrollyChapter>
    </div>
  );
}
