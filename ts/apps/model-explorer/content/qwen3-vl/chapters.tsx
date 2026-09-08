"use client";

import Link from "next/link";
import { CodeLink } from "@/components/code/code-link";
import { EnvFlagChip, QuantChip } from "@/components/primitives/chips";
import { Divergence, Scene, ScrollyChapter } from "@/components/scrollytelling/scrolly";
import { L } from "@/lib/links";
import type { ChaptersProps } from "../registry";
import {
  MRopeClocksFigure,
  PixelsToTokensFigure,
  RerankerFigure,
  TwoVsThreeAxisFigure,
  VisionTowerFigure,
  VlSpineNotesFigure,
} from "./figures";

export function Qwen3VlChapters({ spec }: ChaptersProps) {
  return (
    <div>
      {/* ── Ch 1 · Pixels are tokens too ────────────────────────────── */}
      <ScrollyChapter
        id="ch-1"
        number={1}
        title="Pixels are tokens too"
        intro="The spine strip above has a chip no other decoder page has: Vision. This page is about what flows through it."
      >
        <Scene id="pixels" graphic={<PixelsToTokensFigure />}>
          <p>
            An image is chopped into 16×16-pixel patches (doubled across a 2-frame temporal window), pushed
            through a 24-block vision tower, and merged 4-into-1 — a 768×768 image becomes 576 visual tokens
            of width {spec.stats.hidden}. From the decoder's perspective they are just embeddings in the
            sequence: attended to causally, cached in the same paged KV pools, costing exactly what text
            costs.
          </p>
          <p>
            <CodeLink link={L("qwen3vl-projector")} /> <QuantChip format="q8_0" />
            <span className="ml-1 text-xs text-muted-foreground">(mmproj) +</span> <QuantChip format="q4_k" />
            <span className="ml-1 text-xs text-muted-foreground">(decoder)</span>
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 2 · The vision tower ─────────────────────────────────── */}
      <ScrollyChapter
        id="ch-2"
        number={2}
        title="The vision tower"
        intro={
          <p>
            Not a CLIP pooling head: a temporal-split Conv3D patch embed, learned-position interpolation,
            two-axis RoPE, {spec.stats.visionBlocks} attention blocks, and feature taps partway up.
          </p>
        }
      >
        <Scene id="patch" graphic={<VisionTowerFigure step={0} />}>
          <p>
            <strong>Patches first.</strong> The Conv3D patch embed spans 16×16 pixels × 2 frames — video is
            native, and still images simply duplicate their frame. Positions come from a learned 48×48 table
            (2,304 entries) that is <em>bilinearly interpolated</em> to whatever patch grid the actual image
            produced, so arbitrary resolutions never leave the training distribution's position manifold.
          </p>
          <p>
            <CodeLink link={L("qwen3vl-patchify")} /> · <CodeLink link={L("qwen3vl-pos-interp")} />
          </p>
        </Scene>
        <Scene id="merge" graphic={<VisionTowerFigure step={1} />}>
          <p>
            <strong>The 2×2 merger</strong> zips each block of four neighboring patch vectors into one:
            concat to 4,096 dims, MLP down to the decoder's {spec.stats.hidden}. Four times fewer tokens hit
            the expensive causal stack, and the spatial relationships survive inside the concatenation rather
            than being averaged away.
          </p>
          <p>
            <CodeLink link={L("qwen3vl-merger")} />
          </p>
        </Scene>
        <Scene id="deepstack" graphic={<VisionTowerFigure step={2} />}>
          <p>
            <strong>DeepStack.</strong> After vision blocks 5, 11 and 17, features exit sideways through
            dedicated merger heads and are <em>added into the hidden states</em> of the first decoder layers
            at visual-token positions — mid-level texture and layout information that the top of the tower
            would have abstracted away. If Gemma4's PLE ribbon looked familiar, it should: this is the same
            visual grammar — a side lane feeding the main stack — carrying different physics.
          </p>
          <Divergence
            others={<p>most VLM runtimes flatten the projector to "encode image, get tokens" and would silently drop non-token outputs.</p>}
            antfly={<p>the projector keeps DeepStack outputs structurally separate from tokens, so the decoder cannot mistake the concatenated payload for sequence content.</p>}
            link={<CodeLink link={L("qwen3vl-deepstack-tap")} />}
          />
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 3 · m-RoPE: three clocks ─────────────────────────────── */}
      <ScrollyChapter
        id="ch-3"
        number={3}
        title="m-RoPE: three clocks"
        intro="One rotary embedding, three position streams. Drag the slider."
      >
        <Scene id="clocks" graphic={<MRopeClocksFigure />}>
          <p>
            Every token carries <em>three</em> positions — text-time, image-height, image-width — and the
            head dimension is partitioned into interleaved sections rotating on each stream. In plain text the
            three clocks tick in lockstep, which collapses m-RoPE back to ordinary RoPE. Inside an image, the
            temporal hand freezes while h and w advance with the patch grid: two patches in the same row share
            an h phase, two in the same column share w.
          </p>
          <p>
            The position plan is built once per request; decode continues at{" "}
            <code>tokenCount + mrope_position_delta</code>, because a 2D grid consumes fewer sequential
            positions than it has tokens.
          </p>
          <p>
            <CodeLink link={L("qwen3vl-plan")} /> · <CodeLink link={L("qwen3vl-mrope-kernel")} />
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 4 · Two-axis vs three-axis ───────────────────────────── */}
      <ScrollyChapter
        id="ch-4"
        number={4}
        title="Two-axis vs three-axis"
        intro="The tower and the decoder both rotate positions — but they're solving different problems."
      >
        <Scene id="axes" graphic={<TwoVsThreeAxisFigure />}>
          <p>
            The vision tower's RoPE is <strong>two-axis</strong>: patch row and patch column, bidirectional,
            scoped to one image — it only ever needs to know where a patch sits in <em>its</em> grid. The
            decoder's m-RoPE is <strong>three-axis</strong> because its problem is disambiguation across an
            interleaved conversation: the word after an image, the second of two images, a video frame versus
            the frame before it. Without the text-time axis, "patch (0,0) of image one" and "patch (0,0) of
            image two" would be positionally identical.
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 5 · Same decoder, new inputs ─────────────────────────── */}
      <ScrollyChapter
        id="ch-5"
        number={5}
        title="Same decoder, new inputs"
        intro={
          <p>
            The text stack is a stock {spec.stats.layers}-layer Qwen3: GQA {spec.stats.queryHeads}/
            {spec.stats.kvHeads}, Q/K head norm, SwiGLU. It runs in the same <code>gpt.zig</code> that serves
            Gemma4 and Qwen3 Embedding.
          </p>
        }
      >
        <Scene id="interleave" graphic={<VlSpineNotesFigure />}>
          <p>
            Visual tokens are spliced into the embedding stream at the <code>&lt;|image_pad|&gt;</code>{" "}
            markers before layer 0; from there the runtime is family-gated, not forked —{" "}
            <code>family == .qwen3_vl</code> switches on m-RoPE positions and DeepStack injection, and
            everything else is the shared machinery: planned Metal frames, paged KV, the generated quant
            routes.
          </p>
          <Divergence
            others={<p>llama.cpp runs multimodal through a separate clip.cpp path with its own execution model bolted on.</p>}
            antfly={<p>one decoder runtime; vision is an input transformation plus two family-gated features, behind the same fail-closed qualification gates as every other model.</p>}
            link={<CodeLink link={L("qwen3vl-family-gate")} />}
          />
          <p>
            <CodeLink link={L("qwen3vl-splice")} /> ·{" "}
            <Link className="text-primary underline" href="/explore/qwen3-vl">
              walk the full DAG →
            </Link>
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 6 · The reranker variant ─────────────────────────────── */}
      <ScrollyChapter
        id="ch-6"
        number={6}
        title="The reranker variant"
        intro="Take the same model, ask it one question, never let it answer in words."
      >
        <Scene id="rerank" graphic={<RerankerFigure />}>
          <p>
            Qwen3-VL-Reranker wraps query, document and image into a prompt whose system message pins the
            answer space to "yes" or "no". The relevance score is{" "}
            <code>sigmoid(logit_yes − logit_no)</code> — computed directly from the final hidden state against
            the two corresponding LM-head rows. One prefill per candidate, no generation, no sampling; a
            generative architecture used as a calibrated binary classifier for multimodal search ranking.
          </p>
          <p>
            <CodeLink link={L("qwen3vl-reranker-score")} />
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 7 · Spine notes ──────────────────────────────────────── */}
      <ScrollyChapter
        id="ch-7"
        number={7}
        title="Spine notes"
        intro="What this model added to the kernel inventory, and what it borrowed."
      >
        <Scene id="notes" graphic={<VlSpineNotesFigure />}>
          <p>
            VL-only surface: <code>termite_apply_mrope</code>, the Conv3D/merger vision path, and DeepStack
            injection. Everything else — attention, the Q4_K matvec routes, frame planning — is shared, and
            the VL-specific fast paths ship behind their own flags:
          </p>
          <p>
            <EnvFlagChip name="TERMITE_METAL_DISABLE_QWEN3VL_DECODE_FRAME" defaultOn={false} />{" "}
            <EnvFlagChip name="TERMITE_METAL_DISABLE_QWEN3VL_PREPARED_FFN" defaultOn={false} />{" "}
            <EnvFlagChip name="TERMITE_QWEN3VL_PROFILE" defaultOn={false} />
          </p>
          <p>
            The promotion is deliberately narrow — exact managed artifacts, Metal only, fail-closed for
            everything else. The operational contract lives in the repo:
          </p>
          <p>
            <CodeLink link={L("qwen3vl-support-doc")} /> ·{" "}
            <Link className="text-primary underline" href="/systems/kernels">
              kernel inventory →
            </Link>
          </p>
        </Scene>
      </ScrollyChapter>
    </div>
  );
}
