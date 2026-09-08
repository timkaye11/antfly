import { kernels, manifest, snippetsFor } from "@/lib/data";
import { KitchenSinkClient } from "./sink-client";

export default function KitchenSinkPage() {
  const routes = kernels.routes.slice(0, 3);
  const snippets = snippetsFor(routes.map((r) => r.source));
  return (
    <KitchenSinkClient
      routes={routes}
      snippets={snippets}
      gitCommit={manifest.gitCommit}
      permalinkBase={manifest.permalinkBase}
    />
  );
}
