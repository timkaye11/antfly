"use client";

/**
 * Gemma4 chapters 2–6: tokens in, PLE, attention, KV, and the MoE cousin.
 */
import { useMemo } from "react";
import { CodeLink } from "@/components/code/code-link";
import { EnvFlagChip, QuantChip } from "@/components/primitives/chips";
import { Divergence, Scene, ScrollyChapter } from "@/components/scrollytelling/scrolly";
import { KvCacheBlocks } from "@/components/viz/kv-cache-blocks";
import { L } from "@/lib/links";
import type { ChaptersProps } from "../registry";
import {
  AttentionBlockFigure,
  EmbedLookupFigure,
  GqaGroupingFigure,
  KvExtrasFigure,
  MoeForkJoinFigure,
  MoeResidencyFigure,
  MoeRoutingFigure,
  makeGemmaKvTrace,
  PleCostFigure,
  PleRibbonFigure,
  RangeMaskFigure,
  SessionRoutingFigure,
  TokenPiecesFigure,
} from "./figures-early";

export function Gemma4EarlyChapters({ spec }: ChaptersProps) {
  const isE4b = spec.id === "gemma4-e4b";
  const layers = Number(spec.stats.layers);
  const hidden = Number(spec.stats.hidden);
  const kvTrace = useMemo(() => makeGemmaKvTrace(isE4b), [isE4b]);

  return (
    <div>
      {/* ── Ch 2 · Tokens in ─────────────────────────────────────── */}
      <ScrollyChapter
        id="ch-2"
        number={2}
        title="Tokens in"
        intro="Before any tensor exists, a string becomes ids and a request becomes a session."
      >
        <Scene id="pieces" graphic={<TokenPiecesFigure />}>
          <p>
            The prompt is split by a SentencePiece tokenizer into pieces from a 262,144-entry
            vocabulary. Common words are single pieces; rarer ones shatter (<code>ants</code> →{" "}
            <code>▁ant</code> + <code>s</code>). From here on the model only ever sees the ids.
          </p>
          <p>
            <CodeLink link={L("tokenizer-sentencepiece")} />
          </p>
        </Scene>
        <Scene id="routing" graphic={<SessionRoutingFigure />}>
          <p>
            The HTTP handler hands the request to the session factory, which reads the model
            manifest and resolves it to <code>ModelFamily.gemma</code> — one tag in the unified GPT
            config. There is no Gemma-specific runtime: iSWA, PLE, and shared KV are all config
            fields on the same decode loop that serves llama, qwen, and phi.
          </p>
          <p>
            <CodeLink link={L("server-chat")} /> · <CodeLink link={L("session-factory")} />
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 3 · Embeddings, and then embeddings again ─────────── */}
      <ScrollyChapter
        id="ch-3"
        number={3}
        title="Embeddings, and then embeddings again (PLE)"
        intro={
          <p>
            Gemma-4 looks its tokens up twice: once for the hidden state, and once for a per-layer
            embedding lane that most runtimes don't bother to run.
          </p>
        }
      >
        <Scene id="lookup" graphic={<EmbedLookupFigure hidden={hidden} />}>
          <p>
            The ordinary part first: the token id selects one row of the embedding table and becomes
            a {hidden}-wide hidden state, scaled by <code>√{hidden}</code>-style normalization.
            Every transformer since 2017 starts this way.
          </p>
        </Scene>
        <Scene id="ple" graphic={<PleRibbonFigure layers={layers} />}>
          <p>
            <strong>Then the second lookup.</strong> Gemma-4's per-layer embeddings (PLE) run a
            separate projection whose output is sliced per layer: a thin lane that rides alongside
            all {layers} layers and feeds each one its own gated slice. It's part of the model —
            skip it and you are running a different network.
          </p>
          <p>
            <CodeLink link={L("gpt-compute-ple")} /> · <CodeLink link={L("config-ple-hidden")} />
          </p>
        </Scene>
        <Scene id="cost" graphic={<PleCostFigure isE4b={isE4b} />}>
          <p>
            <strong>PLE is not free.</strong> On E4B the <code>per_layer_model_proj</code> matvec
            streams 55 MB per token in F16 — 86 MB/token for the PLE lane in total. The runtime
            stages the slot to <QuantChip format="q8_0" /> at load time, default-on:{" "}
            {isE4b
              ? "E4B's 10752×2560 bf16 slot drops ~13 MB/token"
              : "E2B's slot 351 (8960×1536) was shipping as dense F32 — staging saves ~41 MB/token"}
            , token-identical, worth +1.10% on the M4 Pro testbed.
          </p>
          <p>
            <EnvFlagChip name="TERMITE_METAL_DISABLE_PLE_MODEL_PROJ_Q8" defaultOn={false} />
          </p>
          <Divergence
            others={
              <p>
                no other Metal runtime executes Gemma-4's full PLE path — some llama.cpp builds skip
                it (open issue #22243), and a build doing less work per token flatters its tok/s.
              </p>
            }
            antfly={
              <p>
                the full PLE lane runs every token, with the heavy slot quantized instead of
                dropped.
              </p>
            }
            link={<CodeLink link={L("gpt-compute-ple")} />}
          />
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 4 · Attention ─────────────────────────────────────── */}
      <ScrollyChapter
        id="ch-4"
        number={4}
        title="Attention: windows, groups, and norms in the right places"
        intro={
          <p>
            One attention block, taken apart: Gemma-4 puts RMS norms on the Q and K heads
            themselves, and the runtime fuses that oddity with rope into a single kernel.
          </p>
        }
      >
        <Scene id="block" graphic={<AttentionBlockFigure isE4b={isE4b} />}>
          <p>
            After the QKV projection, each query and key head is RMS-normalized <em>per head</em> —
            an unusual placement that stabilizes Gemma's attention logits. Antfly lowers the pair as
            one fused <code>⟨head_rms ⋄ rope⟩</code> dispatch instead of four small ones.
          </p>
          <p>
            <CodeLink link={L("gpt-qk-head-norm")} /> ·{" "}
            <CodeLink link={L("kernel-head-rms-rope")} />{" "}
            <EnvFlagChip name="TERMITE_METAL_DISABLE_A4B_HEAD_NORM_ROPE_FUSION" defaultOn={false} />
          </p>
        </Scene>
        <Scene id="gqa" graphic={<GqaGroupingFigure isE4b={isE4b} />}>
          <p>
            <strong>Grouped-query attention.</strong> {spec.displayName} runs 8 query heads against{" "}
            {isE4b ? "2 KV heads" : "a single KV head"} — {isE4b ? "4 queries" : "all 8 queries"}{" "}
            share each K/V bank. Since decode is memory-bound and the KV read is the hot loop,
            shrinking the KV side by {isE4b ? "4×" : "8×"} matters far more than the query-side
            arithmetic.
          </p>
        </Scene>
        <Scene id="mask" graphic={<RangeMaskFigure />}>
          <p>
            <strong>The two masks.</strong> A sliding layer may only read a fixed trailing window of
            the sequence; the global layer every sixth slot reads everything. That's the whole
            difference — the same kernels run both, with different valid ranges and head dims (512
            global, 256 sliding, from chapter 1).
          </p>
          <p>
            <CodeLink link={L("config-layer-uses-sliding")} />
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 5 · The KV cache is a filing system ───────────────── */}
      <ScrollyChapter
        id="ch-5"
        number={5}
        title="The KV cache is a filing system"
        intro={
          <p>
            K/V tensors live in 16-token pages owned by a pool. Drag the slider: this is a
            deterministic replay of the allocation rules, not a screenshot.
          </p>
        }
      >
        <Scene
          id="alloc"
          graphic={
            <div className="flex h-full flex-col">
              <KvCacheBlocks trace={kvTrace} initialStep={96} />
            </div>
          }
        >
          <p>
            As the sequence grows past each 16-token boundary, every KV-owning layer files a new
            page (<code>page_size_tokens = 16</code>). Pages, not one monolithic buffer — so memory
            grows in steps and pages can be reclaimed, shared, or compressed individually.
          </p>
          <p>
            <CodeLink link={L("kv-pool-config")} />
          </p>
        </Scene>
        <Scene
          id="evict"
          graphic={
            <div className="flex h-full flex-col">
              <KvCacheBlocks trace={kvTrace} initialStep={288} />
            </div>
          }
        >
          <p>
            <strong>The SWA lane evicts behind the window.</strong> Sliding layers only ever need
            the trailing window, so their old pages are marked reusable (hatched) while the global
            lane keeps growing. Long contexts cost global-layer memory, not whole-model memory.
          </p>
          <p className="text-xs text-muted-foreground">
            The replay draws a 128-token window so eviction is visible; the pool uses the model's
            real per-layer window.
          </p>
        </Scene>
        <Scene
          id="shared"
          graphic={
            <div className="flex h-full flex-col">
              <KvCacheBlocks trace={kvTrace} initialStep={288} />
            </div>
          }
        >
          <p>
            <strong>The third lane never fills.</strong> The {spec.stats.sharedKv} shared-KV tail
            layers hold zero pages at every step — they read a donor layer's pages instead of
            writing their own. An empty lane is the visualization of a predicate that is one
            comparison in the config.
          </p>
          <p>
            <CodeLink link={L("config-shared-kv")} /> · <CodeLink link={L("kv-manager")} />
          </p>
        </Scene>
        <Scene id="extras" graphic={<KvExtrasFigure />}>
          <p>
            Two more behaviors share this machinery: a prompt-prefix cache re-attaches pages from a
            previous request instead of re-prefilling them, and TurboQuant's Polar4 codec stores
            cached keys at ~4 bits.
          </p>
          <p>
            <CodeLink link={L("kv-prompt-cache")} /> · <CodeLink link={L("kv-turboquant-polar4")} />
          </p>
          <Divergence
            others={
              <p>
                vLLM invented paged KV to pack thousands of concurrent sequences onto CUDA fleets.
              </p>
            }
            antfly={
              <p>
                the same idea runs paged + sliding-window + shared-KV + compressed KV in one manager
                — on a laptop GPU.
              </p>
            }
            link={<CodeLink link={L("kv-manager")} />}
          />
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 6 · The MoE cousin ────────────────────────────────── */}
      <ScrollyChapter
        id="ch-6"
        number={6}
        title="The MoE cousin: Gemma4 26B-A4B"
        intro={
          <p>
            To be clear: {spec.displayName} is dense — every FFN runs for every token. But the same
            runtime serves Gemma-4's 26B-A4B mixture-of-experts variant, and its machinery leaks
            useful ideas back into the dense path.
          </p>
        }
      >
        <Scene id="route" graphic={<MoeRoutingFigure />}>
          <p>
            A4B replaces the FFN in 30 of its layers with 128 experts at hidden size 2816. A router
            scores the token and activates only the top-k experts (the repo's own docs state both
            top-8 and top-2 in different sections — the honest summary is &quot;a few of 128&quot;).
            Most of the 26B parameters sleep through any given token.
          </p>
        </Scene>
        <Scene id="residency" graphic={<MoeResidencyFigure />}>
          <p>
            <strong>Experts don't all fit on the GPU.</strong> The Metal path picks an{" "}
            <code>A4bMappedMoeRoute</code>: the explicit high-memory route keeps every mapped expert
            resident; otherwise expert weights stream from host memory when the router lands on
            them. A shelf of warm experts, a warehouse behind it.
          </p>
          <p>
            <CodeLink link={L("moe-mapped-route")} />{" "}
            <EnvFlagChip name="TERMITE_METAL_DISABLE_A4B_FUSED_GATE_UP" defaultOn={false} />
          </p>
        </Scene>
        <Scene id="forkjoin" graphic={<MoeForkJoinFigure />}>
          <p>
            The planner describes the A4B decode-FFN as an explicit fork/join: shared-expert and
            routed-expert paths as named resources that join back into the layer output. Today it
            deliberately stops at planning — a typed description no encoder consumes yet — which is
            how this codebase grows: the shape first, the lowering when it's earned.
          </p>
          <p>
            <CodeLink link={L("planner-a4b-forkjoin")} />
          </p>
        </Scene>
      </ScrollyChapter>
    </div>
  );
}
