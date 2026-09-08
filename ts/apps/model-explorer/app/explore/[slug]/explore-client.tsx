"use client";

import { Tabs, TabsList, TabsTrigger } from "@antfly/design-system";
import { useRouter } from "next/navigation";
import { type ClientSnippet, SnippetProvider } from "@/components/code/snippet-context";
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
        <div className="absolute left-1/2 top-3 z-20 -translate-x-1/2">
          <Tabs value={spec.id} onValueChange={(v) => router.push(`/explore/${v}`)}>
            <TabsList className="h-8">
              {allSlugs.map((s) => (
                <TabsTrigger key={s} value={s} className="h-7 px-3 text-xs">
                  {s}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </div>
        <OpDagExplorer spec={spec} routes={routes} />
      </div>
    </SnippetProvider>
  );
}
