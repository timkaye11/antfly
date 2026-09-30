import { MonoLabel } from "@antfly/design-system";
import { Check, Circle, Loader2, Pause, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export const trainingField = "flex min-w-0 flex-col gap-2 text-sm font-medium";
export const trainingSelect =
  "h-10 w-full min-w-0 border border-input bg-background px-3 text-sm font-normal transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50";
export const trainingAction = "h-auto min-h-9 max-w-full whitespace-normal py-2";

export function TrainingSectionHeading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="min-w-0 space-y-1.5">
        <MonoLabel className="block text-[10px]">{eyebrow}</MonoLabel>
        <h2 className="font-display text-xl tracking-tight">{title}</h2>
        {description && (
          <p className="max-w-2xl text-sm font-normal leading-relaxed text-muted-foreground">
            {description}
          </p>
        )}
      </div>
      {action}
    </div>
  );
}

export function TrainingStatus({ status }: { status: string }) {
  const success = ["ready", "complete", "connected"].includes(status);
  const failed = ["failed", "offline", "cleanup_unconfirmed"].includes(status);
  const working = [
    "queued",
    "checking",
    "importing",
    "preparing",
    "uploading",
    "running",
    "preflight_running",
    "pausing",
    "cancelling",
  ].includes(status);
  const Icon = success
    ? Check
    : failed
      ? TriangleAlert
      : working
        ? Loader2
        : status === "paused"
          ? Pause
          : Circle;
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 border px-2 py-1 text-[11px] font-medium leading-none",
        success
          ? "border-success/25 bg-success/5 text-success"
          : failed
            ? "border-destructive/25 bg-destructive/5 text-destructive"
            : working
              ? "border-info/25 bg-info/5 text-info"
              : "border-border bg-muted/40 text-muted-foreground"
      )}
    >
      <Icon aria-hidden="true" className={cn("size-3", working && "motion-safe:animate-spin")} />
      <span className="capitalize">{status.replaceAll("_", " ")}</span>
    </span>
  );
}
