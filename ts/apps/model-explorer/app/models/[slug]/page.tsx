import { notFound } from "next/navigation";
import { collectSourceLinks, kernels, manifest, namedLinks, snippetsFor } from "@/lib/data";
import { frameQ40, frameQ80Anchor } from "@/lib/frames";
import { getModelSpec, modelSlugs } from "@/lib/models";
import { ModelPageClient } from "./model-page-client";

export function generateStaticParams() {
  return modelSlugs().map((slug) => ({ slug }));
}

export default async function ModelPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const spec = getModelSpec(slug);
  if (!spec) notFound();
  const snippets = snippetsFor([...collectSourceLinks(spec), ...Object.values(namedLinks)]);
  return (
    <ModelPageClient
      spec={spec}
      routes={kernels.routes}
      frames={{ q40: frameQ40, q80Anchor: frameQ80Anchor }}
      snippets={snippets}
      gitCommit={manifest.gitCommit}
      permalinkBase={manifest.permalinkBase}
    />
  );
}
