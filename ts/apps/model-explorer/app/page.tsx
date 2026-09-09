import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle } from "@antfly/design-system";
import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { SpineStrip } from "@/components/spine-strip";
import { manifest } from "@/lib/data";

const MODEL_CARDS = [
  {
    slug: "gemma4-e4b",
    name: "Gemma4 E2B / E4B",
    kind: "Generative LLM · deep dive",
    hook: "Per-layer embeddings, variant-specific sliding-window attention, shared KV, and Metal decode frames. Related family chapters explain MoE and optional MTP.",
    badges: ["PLE", "iSWA", "shared KV", "Q4_0"],
  },
  {
    slug: "gliner2",
    name: "GLiNER2",
    kind: "Schema-driven extraction",
    hook: "A DeBERTa-v3 encoder with disentangled relative attention and a schema-conditioned head that scores candidate entity spans.",
    badges: ["DeBERTa", "C2P/P2C", "span head"],
  },
  {
    slug: "qwen3-embedding",
    name: "Qwen3 Embedding",
    kind: "Text embeddings",
    hook: "A causal transformer used as an encoder: query instructions, last-token pooling, and normalized vectors. Follow the 0.6B graph and its configured context limit.",
    badges: ["last-token pool", "8k ctx", "Q8_0"],
  },
  {
    slug: "qwen3-vl",
    name: "Qwen3-VL",
    kind: "Vision-language",
    hook: "Pixels become tokens: Conv3D patch embedding, a 2×2 merger, DeepStack feature taps, and three-axis m-RoPE feeding the same decoder runtime.",
    badges: ["vision tower", "m-RoPE", "DeepStack"],
  },
];

export default function HomePage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-12">
      <section className="mx-auto max-w-3xl text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          How Antfly runs models<span className="text-primary">.</span>
        </h1>
        <p className="mt-4 text-lg text-muted-foreground">
          An interactive tour of the Zig inference runtime — from an HTTP request to a sampled token,
          with model-specific graph execution and Metal command frames. The source index contains {manifest.counts.kernels} Metal
          kernel entries. Source links are pinned to commit{" "}
          <code className="font-mono text-sm">{manifest.gitCommit.slice(0, 10)}</code>.
        </p>
        <div className="mt-6 flex justify-center">
          <SpineStrip />
        </div>
      </section>

      <section className="mt-12 grid gap-4 sm:grid-cols-2">
        {MODEL_CARDS.map((m) => (
          <Link key={m.slug} href={`/models/${m.slug}`} className="group">
            <Card className="h-full transition-colors group-hover:border-primary/50">
              <CardHeader>
                <CardDescription>{m.kind}</CardDescription>
                <CardTitle className="flex items-center gap-2">
                  {m.name}
                  <ArrowRight className="size-4 opacity-0 transition-opacity group-hover:opacity-100" />
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">{m.hook}</p>
                <div className="flex flex-wrap gap-1.5">
                  {m.badges.map((b) => (
                    <Badge key={b} className="font-mono text-xs">
                      {b}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </section>

      <section className="mt-12 grid gap-4 text-center sm:grid-cols-4">
        {[
          { label: "graph op kinds", value: manifest.counts.opKinds, href: "/runtime#graph" },
          { label: "Metal kernels", value: manifest.counts.kernels, href: "/systems/kernels" },
          { label: "compiled kernel routes", value: manifest.counts.routes, href: "/systems/kernels" },
          { label: "environment-name references", value: manifest.counts.envFlags, href: "/systems/flags" },
        ].map((s) => (
          <Link key={s.label} href={s.href} className="rounded-lg border p-6 transition-colors hover:border-primary/50">
            <div className="text-3xl font-bold tabular-nums">{s.value}</div>
            <div className="mt-1 text-sm text-muted-foreground">{s.label}</div>
          </Link>
        ))}
      </section>
    </div>
  );
}
