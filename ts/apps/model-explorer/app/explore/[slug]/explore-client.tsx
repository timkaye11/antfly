"use client";

import { useRouter } from "next/navigation";
import { type ClientSnippet, SnippetProvider } from "@/components/code/snippet-context";
import { ChoiceGroup } from "@/components/primitives/choice-group";
import { OpDagExplorer } from "@/components/viz/op-dag-explorer";
import type { KernelRoute, ModelSpec } from "@/lib/schema";

export function ExploreClient({
  spec,
  routes,
  snippets,
  gitCommit,
  permalinkBase,
  allSlugs,
}: {
  spec: ModelSpec;
  routes: KernelRoute[];
  snippets: Record<string, ClientSnippet>;
  gitCommit: string;
  permalinkBase?: string;
  allSlugs: string[];
}) {
  const router = useRouter();
  return (
    <SnippetProvider snippets={snippets} gitCommit={gitCommit} permalinkBase={permalinkBase}>
      <div className="relative">
        <header className="border-b px-4 py-3">
          <h1 className="text-lg font-semibold">{spec.displayName} · operation explorer</h1>
          <p className="mb-3 text-xs text-muted-foreground">
            Curated architecture schematic. Shapes and backend labels illustrate the documented
            path; repeated layers and operations are collapsed. Select a node for tensors and source
            links.
          </p>
          <ChoiceGroup
            label="Model operation explorer"
            value={spec.id}
            onValueChange={(value) => router.push(`/explore/${value}`)}
            options={allSlugs.map((slug) => ({ value: slug, label: slug }))}
            buttonClassName="h-7 px-3"
          />
        </header>
        <OpDagExplorer spec={spec} routes={routes} />
      </div>
    </SnippetProvider>
  );
}
