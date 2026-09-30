import { Button, Input } from "@antfly/design-system";
import { ArrowUpRight, Box, ChevronDown, Laptop, Network, Plus, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useId, useState } from "react";
import { TrainingSectionHeading, TrainingStatus } from "@/components/training-ui";
import { useApiConfig } from "@/hooks/use-api-config";
import { useTrainingResource } from "@/hooks/use-training";

export function TrainingPeers({
  inventoryOnly = false,
  embedded = false,
}: {
  inventoryOnly?: boolean;
  embedded?: boolean;
}) {
  const destinationId = useId();
  const { client, apiUrl } = useApiConfig();
  const load = useCallback((signal: AbortSignal) => client.training.peers(signal), [client]);
  const { data, error, refresh } = useTrainingResource(load, 10000);
  const [destination, setDestination] = useState("");
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState("");
  const [controller, setController] = useState(() => new AbortController());
  // biome-ignore lint/correctness/useExhaustiveDependencies: Mutation ownership follows endpoint and credential changes.
  useEffect(() => {
    const next = new AbortController();
    setController(next);
    setDestination("");
    setActionError("");
    setBusy(false);
    return () => next.abort();
  }, [apiUrl, client]);
  const act = async (operation: () => Promise<unknown>) => {
    setBusy(true);
    setActionError("");
    try {
      await operation();
      if (!controller.signal.aborted) refresh();
    } catch (failure) {
      if (!controller.signal.aborted) setActionError(String(failure));
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  };
  if (!data && error?.includes("404")) return null;
  return (
    <section
      id="training-machines"
      className={`scroll-mt-24 space-y-5 ${embedded ? "" : "border border-border bg-card p-5 sm:p-6"}`}
    >
      <TrainingSectionHeading
        eyebrow={inventoryOnly ? "Network inventory" : "Network"}
        title={inventoryOnly ? "Models on training peers" : "Nearby machines"}
        description={
          inventoryOnly
            ? "Peer inventory is available for training. Inference uses your configured connections."
            : "Bring a second Mac into your training workspace. Connect using a trusted SSH alias."
        }
        action={
          <Button type="button" variant="ghost" size="sm" onClick={refresh}>
            <RefreshCw aria-hidden="true" className="size-3.5" />
            Refresh
          </Button>
        }
      />
      {(error || actionError) && (
        <p
          role="alert"
          className="border-l-2 border-destructive bg-destructive/5 px-3 py-2 text-sm text-destructive"
        >
          {actionError || error}
        </p>
      )}
      {!inventoryOnly && (
        <form
          className="flex flex-col gap-2 sm:max-w-2xl sm:flex-row sm:items-end"
          onSubmit={(event) => {
            event.preventDefault();
            void act(() =>
              client.training.registerPeer({ ssh_destination: destination }, controller.signal)
            );
          }}
        >
          <label
            htmlFor={destinationId}
            className="flex min-w-0 flex-1 flex-col gap-2 text-xs font-medium"
          >
            SSH destination
            <Input
              id={destinationId}
              aria-label="SSH destination"
              className="h-10 font-mono text-xs"
              placeholder="tim@mac-mini.local"
              value={destination}
              onChange={(event) => setDestination(event.target.value)}
            />
          </label>
          <Button type="submit" size="lg" disabled={busy || !destination.trim()}>
            <Plus aria-hidden="true" className="size-4" />
            Connect Mac
          </Button>
        </form>
      )}
      {data?.discovery_error && (
        <p className="text-sm text-muted-foreground">{data.discovery_error}</p>
      )}
      {!inventoryOnly &&
        data?.nearby.map((peer) => (
          <div
            key={peer.id}
            className="flex flex-wrap items-center justify-between gap-3 border-l-2 border-primary bg-muted/30 px-4 py-3 text-sm"
          >
            <span>
              {peer.name}{" "}
              <span className="text-muted-foreground">
                {peer.hostname} · SSH {peer.ssh_port}
              </span>
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={peer.ssh_port !== 22}
              title={
                peer.ssh_port === 22
                  ? undefined
                  : `Configure an SSH alias with Port ${peer.ssh_port}`
              }
              onClick={() => setDestination(peer.hostname)}
            >
              <ArrowUpRight aria-hidden="true" className="size-3.5" />
              Use hostname
            </Button>
          </div>
        ))}
      <div className={`grid gap-3 ${(data?.peers.length ?? 0) > 1 ? "lg:grid-cols-2" : ""}`}>
        {data?.peers.map((peer) => {
          const inventory = peer.inventory as
            | {
                chip?: string;
                memory_bytes?: number;
                metal?: boolean;
                tcp?: boolean;
                jaccl?: boolean;
                models?: { name: string; path: string; kind: string }[];
              }
            | undefined;
          return (
            <div
              key={peer.id}
              className="min-w-0 space-y-4 border border-border bg-background/50 p-4 text-sm"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="flex min-w-0 items-center gap-3">
                  <span className="grid size-10 shrink-0 place-items-center border border-border bg-card">
                    <Laptop aria-hidden="true" className="size-5 text-muted-foreground" />
                  </span>
                  <div className="min-w-0">
                    <h3 className="break-words font-medium">{peer.name}</h3>
                    <p className="mt-1 break-all font-mono text-[11px] text-muted-foreground">
                      {peer.ssh_destination}
                    </p>
                  </div>
                </div>
                <TrainingStatus status={peer.status} />
              </div>
              {inventory && (
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
                  <span>
                    {inventory.chip || "Hardware details unavailable"}
                    {inventory.memory_bytes
                      ? ` · ${(inventory.memory_bytes / 1024 ** 3).toFixed(0)} GiB`
                      : ""}
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <Network aria-hidden="true" className="size-3" />
                    TCP {inventory.tcp ? "available" : "missing"}
                  </span>
                  <span>{inventory.metal ? "Metal + CPU" : "CPU"}</span>
                </div>
              )}
              {peer.error && <p className="break-words text-xs text-destructive">{peer.error}</p>}
              {inventory && (
                <details className="group border-t border-border pt-3" open={inventoryOnly}>
                  <summary className="flex cursor-pointer list-none items-center gap-2 text-xs text-muted-foreground [&::-webkit-details-marker]:hidden">
                    <Box aria-hidden="true" className="size-3.5" />
                    {inventory.models?.length ?? 0} models on this Mac
                    <ChevronDown
                      aria-hidden="true"
                      className="ml-auto size-3.5 transition-transform group-open:rotate-180"
                    />
                  </summary>
                  <div className="mt-3 space-y-3">
                    <p className="text-xs text-muted-foreground">
                      GLiNER2.5 {inventory.metal ? "CPU / Metal" : "CPU"} · Gemma4 CPU · JACCL{" "}
                      {inventory.jaccl ? "available" : "not installed"}
                    </p>
                    {inventory.models?.map((model) => (
                      <div key={model.path} className="border-l-2 border-border pl-3">
                        <span className="text-xs font-medium">{model.name}</span>
                        <p className="mt-1 break-all font-mono text-[11px] text-muted-foreground">
                          {model.path}
                        </p>
                      </div>
                    ))}
                  </div>
                </details>
              )}
              {!inventoryOnly && (
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={busy}
                    onClick={() =>
                      void act(() => client.training.refreshPeer(peer.id, controller.signal))
                    }
                  >
                    Check connection
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={busy}
                    onClick={() =>
                      void act(() => client.training.removePeer(peer.id, controller.signal))
                    }
                  >
                    Remove
                  </Button>
                </div>
              )}
            </div>
          );
        })}
      </div>
      {data && data.peers.length === 0 && (
        <div className="flex items-center gap-3 border border-dashed border-border px-4 py-5 text-sm text-muted-foreground">
          <Laptop aria-hidden="true" className="size-5 shrink-0" />
          <div>
            <p className="font-medium text-foreground">No training peers connected</p>
            <p className="mt-1 text-xs">
              Connect a Mac above to see its hardware and available models.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
