import {
  Button,
  DashboardPage,
  DashboardPageDescription,
  DashboardPageHeader,
  DashboardPageTitle,
  Input,
} from "@antfly/design-system";
import type { TrainingJob, TrainingJobSpec } from "@antfly/sdk";
import {
  Activity,
  ArrowDown,
  Check,
  ChevronDown,
  ChevronRight,
  FileCode2,
  History,
  Laptop,
  Network,
  Play,
  ShieldCheck,
  Terminal,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  defaultGlinerOptions,
  GlinerTrainingConfig,
  type GlinerTrainingOptions,
} from "@/components/gliner-training-config";
import { TrainingDatasets } from "@/components/training-datasets";
import { TrainingPeers } from "@/components/training-peers";
import {
  TrainingSectionHeading,
  TrainingStatus,
  trainingAction,
  trainingField,
  trainingSelect,
} from "@/components/training-ui";
import { useApiConfig } from "@/hooks/use-api-config";
import { useTrainingResource } from "@/hooks/use-training";

const active = new Set(["queued", "preflight_running", "running", "pausing", "cancelling"]);
const fieldClass = trainingField;
const selectClass = trainingSelect;

function isTwoMac(spec: TrainingJobSpec) {
  return spec.execution_mode === "two_mac" || (!spec.execution_mode && !!spec.peer_id);
}

function JobLogs({ job }: { job: TrainingJob }) {
  const { client, apiUrl } = useApiConfig();
  const [rank, setRank] = useState<"launcher" | "0" | "1">("launcher");
  // biome-ignore lint/correctness/useExhaustiveDependencies: Reset the byte cursor when the API endpoint changes, even if a caller reuses its client.
  const load = useMemo(() => {
    let cursor = 0;
    let text = "";
    return async (signal: AbortSignal) => {
      const chunk = await client.training.logs(job.id, rank, cursor, signal);
      if (!signal.aborted) {
        cursor = chunk.cursor;
        text = (text + chunk.text).slice(-128 * 1024);
      }
      return { ...chunk, text };
    };
  }, [client, apiUrl, job.id, rank]);
  const { data, error } = useTrainingResource(load, 3000);
  let progress = "";
  for (const line of (data?.text || "").split("\n").reverse()) {
    try {
      const event = JSON.parse(line);
      if (event.event === "progress" && event.family === "gemma4") {
        progress = `Epoch ${event.epoch}/${event.epochs} · loss ${Number(event.metrics.average_loss).toFixed(4)} · ${event.metrics.optimizer_steps} updates`;
        break;
      }
      if (event.event === "step" && event.report) {
        progress = `Epoch ${event.report.epoch} · batch ${event.report.batch} · ${event.report.optimizer.identity.optimizer_step} updates`;
        break;
      }
    } catch {
      /* Logs also contain plain diagnostic lines. */
    }
  }
  return (
    <div className="space-y-3">
      {progress && (
        <p aria-live="polite" className="text-sm font-medium">
          {progress}
        </p>
      )}
      <label className={`${fieldClass} max-w-xs`}>
        Log source
        <select
          className={selectClass}
          value={rank}
          onChange={(event) => setRank(event.target.value as typeof rank)}
        >
          <option value="launcher">Coordinator</option>
          <option value="0">{isTwoMac(job.spec) ? "This Mac · rank 0" : "This Mac"}</option>
          {isTwoMac(job.spec) && <option value="1">Remote Mac · rank 1</option>}
        </select>
      </label>
      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}
      <pre
        role="region"
        aria-label="Training logs"
        className="min-h-44 max-h-96 overflow-auto whitespace-pre-wrap break-all border border-border bg-muted/50 p-4 font-mono text-xs leading-relaxed"
      >
        {data?.text || "Waiting for output…"}
      </pre>
      {data && data.cursor >= 128 * 1024 && (
        <p className="text-sm text-muted-foreground">
          Showing the latest 128 KiB received. Full logs are retained on the coordinator.
        </p>
      )}
    </div>
  );
}

export default function TrainingPage() {
  const { client, apiUrl } = useApiConfig();
  const loadPeers = useCallback((signal: AbortSignal) => client.training.peers(signal), [client]);
  const loadJobs = useCallback((signal: AbortSignal) => client.training.jobs(signal), [client]);
  const peers = useTrainingResource(loadPeers, 10000);
  const jobs = useTrainingResource(loadJobs, 3000);
  const [distributed, setDistributed] = useState(false);
  const [peerId, setPeerId] = useState("");
  const [family, setFamily] = useState<"gliner25" | "gemma4">("gliner25");
  const [datasetId, setDatasetId] = useState("");
  const [calibrationId, setCalibrationId] = useState("template");
  const [testId, setTestId] = useState("template");
  const [coordinator, setCoordinator] = useState("");
  const [config, setConfig] = useState("/Users/Shared/antfly-training/data/gliner25.json");
  const [useTemplate, setUseTemplate] = useState(false);
  const [glinerModel, setGlinerModel] = useState("/Users/Shared/antfly-training/models/gliner25");
  const [trainFile, setTrainFile] = useState("");
  const [glinerOptions, setGlinerOptions] = useState<GlinerTrainingOptions>(defaultGlinerOptions);
  const [base, setBase] = useState("/Users/Shared/antfly-training/models/gemma4");
  const [adapter, setAdapter] = useState("/Users/Shared/antfly-training/models/gemma4-adapter");
  const [prepared, setPrepared] = useState(
    "/Users/Shared/antfly-training/data/gemma4-prepared.json"
  );
  const [overrideGliner, setOverrideGliner] = useState(false);
  const [examples, setExamples] = useState(2);
  const [epochs, setEpochs] = useState(1);
  const [rate, setRate] = useState(0.0001);
  const [timeout, setTimeout] = useState(7200);
  const [selectedId, setSelectedId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [ready, setReady] = useState("");
  const [checking, setChecking] = useState<{ id: string; key: string }>();
  const pending = useRef<{ key: string; requestId: string } | undefined>(undefined);
  const controller = useRef(new AbortController());
  // biome-ignore lint/correctness/useExhaustiveDependencies: Abort mutations and clear run selections on every endpoint or credential change.
  useEffect(() => {
    const next = new AbortController();
    controller.current = next;
    setSelectedId("");
    setPeerId("");
    setDistributed(false);
    setDatasetId("");
    setCalibrationId("template");
    setTestId("template");
    setReady("");
    setChecking(undefined);
    setError("");
    setBusy(false);
    pending.current = undefined;
    return () => next.abort();
  }, [client, apiUrl]);
  const spec: Omit<TrainingJobSpec, "request_id"> = {
    execution_mode: distributed ? "two_mac" : "local",
    ...(distributed ? { peer_id: peerId, coordinator } : {}),
    family,
    timeout_seconds: timeout,
    ...(datasetId ? { dataset_id: datasetId } : {}),
    ...(family === "gliner25"
      ? {
          ...(useTemplate
            ? { gliner25_config: config }
            : { base_model: glinerModel, ...(datasetId ? {} : { train_file: trainFile.trim() }) }),
          ...(calibrationId !== "template" || !useTemplate
            ? {
                calibration_dataset_id: ["none", "template"].includes(calibrationId)
                  ? null
                  : calibrationId,
              }
            : {}),
          ...(testId !== "template" || !useTemplate
            ? { test_dataset_id: ["none", "template"].includes(testId) ? null : testId }
            : {}),
          ...(!useTemplate || overrideGliner ? { gliner25_options: glinerOptions } : {}),
        }
      : {
          base_model: base,
          adapter,
          ...(datasetId ? {} : { prepared_inputs: prepared }),
          max_examples: examples,
          epochs,
          learning_rate: rate,
        }),
  };
  const specKey = JSON.stringify(spec);
  const loadSelected = useCallback(
    (signal: AbortSignal) =>
      selectedId ? client.training.job(selectedId, signal) : Promise.resolve(undefined),
    [client, selectedId]
  );
  const selectedResource = useTrainingResource(loadSelected, 3000);
  const selected = selectedResource.data;
  useEffect(() => {
    if (
      checking &&
      jobs.data?.jobs.some((job) => job.id === checking.id && job.status === "complete")
    )
      setReady(checking.key);
  }, [checking, jobs.data]);
  const act = async (operation: (signal: AbortSignal) => Promise<TrainingJob>) => {
    const signal = controller.current.signal;
    setBusy(true);
    setError("");
    try {
      const job = await operation(signal);
      if (!signal.aborted) {
        setSelectedId(job.id);
        jobs.refresh();
      }
    } catch (failure) {
      if (!signal.aborted) setError(String(failure));
    } finally {
      if (!signal.aborted) setBusy(false);
    }
  };
  const submit = async (kind: "transport" | "preflight" | "train") => {
    const key = `${kind}:${specKey}`;
    if (pending.current?.key !== key) pending.current = { key, requestId: crypto.randomUUID() };
    const requestId = pending.current.requestId;
    await act(async (signal) => {
      const request = {
        ...spec,
        request_id: requestId,
        ...(kind === "transport" ? { kind: "transport" as const } : {}),
      };
      const job =
        kind === "train"
          ? await client.training.start(request, signal)
          : await client.training.preflight(request, signal);
      if (!signal.aborted) {
        pending.current = undefined;
        if (kind === "preflight") setChecking({ id: job.id, key: specKey });
      }
      return job;
    });
  };
  const hasActive = jobs.data?.jobs.some((job) => active.has(job.status)) ?? false;
  const missingGlinerInput =
    family === "gliner25" &&
    (useTemplate ? !config.trim() : !glinerModel.trim() || (!datasetId && !trainFile.trim()));
  const unavailable = peers.error?.includes("404") || jobs.error?.includes("404");
  return (
    <DashboardPage className="mx-auto w-full max-w-[1400px] gap-6 pb-8 [&_input:not([type=checkbox]):not([type=file])]:h-10">
      <DashboardPageHeader>
        <div className="space-y-2">
          <DashboardPageTitle>Training</DashboardPageTitle>
          <DashboardPageDescription>
            Fine-tune GLiNER2.5 and Gemma4 on this Mac, with an optional second Mac.
          </DashboardPageDescription>
        </div>
        <span className="inline-flex w-fit items-center gap-2 border border-border bg-card px-3 py-2 font-mono text-[11px] text-muted-foreground">
          {distributed ? (
            <Network aria-hidden="true" className="size-3.5" />
          ) : (
            <Laptop aria-hidden="true" className="size-3.5" />
          )}
          {distributed ? "Two Macs · TCP" : "This Mac"}
        </span>
      </DashboardPageHeader>
      {unavailable ? (
        <div className="border border-border p-6 space-y-2">
          <h2 className="font-medium">Enable training on this server</h2>
          <p className="text-sm text-muted-foreground">
            Configure the standalone server’s training toolchain, input roots, output directory, and
            private state directory. Python 3 is required. Add SSH access when using a second Mac.
          </p>
        </div>
      ) : (
        <>
          <nav
            aria-label="Training sections"
            className="grid grid-cols-2 border-y border-border sm:grid-cols-4"
          >
            {[
              ["01", "Model & run", "training-configuration"],
              ["02", "Dataset", "training-datasets"],
              ["03", "Execution", "training-execution"],
              ["04", "Run history", "training-runs"],
            ].map(([number, label, anchor]) => (
              <a
                key={anchor}
                href={`#${anchor}`}
                className="group flex items-center gap-3 px-3 py-3.5 text-sm transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 sm:px-4"
              >
                <span className="font-mono text-[10px] text-muted-foreground">{number}</span>
                <span>{label}</span>
                <ArrowDown
                  aria-hidden="true"
                  className="ml-auto size-3 text-muted-foreground opacity-50 group-hover:opacity-100"
                />
              </a>
            ))}
          </nav>
          {(error || peers.error || jobs.error) && (
            <p
              role="alert"
              className="border-l-2 border-destructive bg-destructive/5 px-4 py-3 text-sm text-destructive"
            >
              {error || peers.error || jobs.error}
            </p>
          )}
          <section
            id="training-configuration"
            className="scroll-mt-24 space-y-6 border border-border bg-card p-5 sm:p-6"
          >
            <TrainingSectionHeading
              eyebrow="01 / Configure"
              title="New training run"
              description="Choose your model and training settings."
            />
            <div className="grid gap-x-6 gap-y-5 md:grid-cols-2">
              <label className={fieldClass}>
                Model family
                <select
                  className={selectClass}
                  value={family}
                  onChange={(event) => {
                    setFamily(event.target.value as typeof family);
                    setDatasetId("");
                    setCalibrationId("template");
                    setTestId("template");
                  }}
                >
                  <option value="gliner25">GLiNER2.5 · Metal or CPU</option>
                  <option value="gemma4">Gemma4 · text-only CPU</option>
                </select>
              </label>
              <label className={fieldClass} htmlFor="training-field-2">
                Run timeout (seconds)
                <Input
                  id="training-field-2"
                  type="number"
                  min={1}
                  max={604800}
                  value={timeout}
                  onChange={(event) => setTimeout(Number(event.target.value))}
                />
              </label>
              {family === "gemma4" && (
                <>
                  <label className={fieldClass} htmlFor="training-field-4">
                    Base model directory
                    <Input
                      id="training-field-4"
                      value={base}
                      onChange={(event) => setBase(event.target.value)}
                    />
                  </label>
                  <label className={fieldClass} htmlFor="training-field-5">
                    Initial adapter directory
                    <Input
                      id="training-field-5"
                      value={adapter}
                      onChange={(event) => setAdapter(event.target.value)}
                    />
                  </label>
                  {!datasetId && (
                    <label className={`${fieldClass} md:col-span-2`} htmlFor="training-field-6">
                      Prepared text inputs
                      <Input
                        id="training-field-6"
                        value={prepared}
                        onChange={(event) => setPrepared(event.target.value)}
                      />
                    </label>
                  )}
                  <label className={fieldClass} htmlFor="training-field-7">
                    {distributed ? "Selected examples (even)" : "Selected examples"}
                    <Input
                      id="training-field-7"
                      type="number"
                      min={distributed ? 2 : 1}
                      step={distributed ? 2 : 1}
                      value={examples}
                      onChange={(event) => setExamples(Number(event.target.value))}
                    />
                  </label>
                  <label className={fieldClass} htmlFor="training-field-8">
                    Epochs
                    <Input
                      id="training-field-8"
                      type="number"
                      min={1}
                      value={epochs}
                      onChange={(event) => setEpochs(Number(event.target.value))}
                    />
                  </label>
                  <label className={fieldClass} htmlFor="training-field-9">
                    Learning rate
                    <Input
                      id="training-field-9"
                      type="number"
                      min={0.000001}
                      step={0.0001}
                      value={rate}
                      onChange={(event) => setRate(Number(event.target.value))}
                    />
                  </label>
                  <p className="text-sm text-muted-foreground">
                    Autodiff on CPU; gradient accumulation is fixed at 1. Gemma4 pause/resume is
                    unavailable.
                  </p>
                </>
              )}
            </div>
            {family === "gliner25" && (
              <GlinerTrainingConfig
                useTemplate={useTemplate}
                onUseTemplate={setUseTemplate}
                model={glinerModel}
                onModel={setGlinerModel}
                template={config}
                onTemplate={setConfig}
                overrideTemplate={overrideGliner}
                onOverrideTemplate={setOverrideGliner}
                value={glinerOptions}
                onChange={setGlinerOptions}
              />
            )}
            <TrainingDatasets
              key={family}
              family={family}
              distributed={distributed}
              useJobTemplate={useTemplate}
              trainFile={trainFile}
              onTrainFile={setTrainFile}
              modelDir={base}
              selectedId={datasetId}
              onSelect={setDatasetId}
              calibrationId={calibrationId}
              onCalibration={setCalibrationId}
              testId={testId}
              onTest={setTestId}
            />
            <section
              id="training-execution"
              aria-label="Execution"
              className="scroll-mt-24 space-y-5 border-t border-border pt-6"
            >
              <TrainingSectionHeading
                eyebrow="03 / Execution"
                title="Where to run"
                description="Training runs on this Mac by default. Add a second Mac when you need it."
              />
              <label className="flex cursor-pointer items-start gap-3 border border-border bg-muted/20 p-4 transition-colors hover:bg-muted/40">
                <input
                  type="checkbox"
                  className="mt-1 size-4 shrink-0 accent-primary"
                  checked={distributed}
                  onChange={(event) => setDistributed(event.target.checked)}
                  aria-controls="training-remote-configuration"
                  aria-expanded={distributed}
                />
                <span className="min-w-0 space-y-1">
                  <span className="flex flex-wrap items-center gap-2 text-sm font-medium">
                    <Network aria-hidden="true" className="size-4 text-muted-foreground" />
                    Use another Mac
                    <span className="border border-border px-1.5 py-0.5 font-mono text-[10px] font-normal text-muted-foreground">
                      Optional
                    </span>
                  </span>
                  <span className="block text-xs leading-relaxed text-muted-foreground">
                    Split training examples across two Macs over TCP. Each Mac holds the full model.
                  </span>
                </span>
              </label>
              {distributed && (
                <div id="training-remote-configuration" className="space-y-5">
                  <TrainingPeers embedded />
                  <div className="grid gap-x-6 gap-y-5 md:grid-cols-2">
                    <label className={fieldClass}>
                      Remote Mac
                      <select
                        className={selectClass}
                        value={peerId}
                        onChange={(event) => setPeerId(event.target.value)}
                      >
                        <option value="">Choose a connected Mac</option>
                        {peers.data?.peers.map((peer) => (
                          <option key={peer.id} value={peer.id}>
                            {peer.name} · {peer.status}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className={fieldClass} htmlFor="training-field-1">
                      This Mac’s reachable address
                      <Input
                        id="training-field-1"
                        placeholder="192.168.1.20:32132"
                        value={coordinator}
                        onChange={(event) => setCoordinator(event.target.value)}
                      />
                    </label>
                  </div>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    Use a trusted network and stage matching models on both Macs. Imported datasets
                    are copied automatically during readiness checks.
                  </p>
                </div>
              )}
            </section>
            <details className="group border-t border-border pt-4">
              <summary className="flex cursor-pointer list-none items-center gap-2 text-xs text-muted-foreground [&::-webkit-details-marker]:hidden">
                <FileCode2 aria-hidden="true" className="size-4" />
                Review configuration
                <ChevronDown
                  aria-hidden="true"
                  className="ml-auto size-4 transition-transform group-open:rotate-180"
                />
              </summary>
              <pre className="mt-3 max-h-80 overflow-auto border border-border bg-muted/40 p-4 text-xs leading-relaxed">
                {JSON.stringify(spec, null, 2)}
              </pre>
            </details>
            <div className="-mx-5 -mb-5 space-y-4 border-t border-border bg-muted/25 p-5 sm:-mx-6 sm:-mb-6 sm:p-6">
              <div className="flex items-start gap-3">
                <ShieldCheck
                  aria-hidden="true"
                  className="mt-0.5 size-5 shrink-0 text-muted-foreground"
                />
                <div className="space-y-1">
                  <h3 className="text-sm font-medium">Verify and launch</h3>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {distributed
                      ? "Check the connection and dataset before training. Both Macs need room for the full model and training state."
                      : "Check the model, dataset, and available resources on this Mac before training."}
                  </p>
                </div>
              </div>
              {missingGlinerInput && (
                <p className="text-xs text-muted-foreground">
                  {useTemplate
                    ? "Enter your job JSON path to check readiness."
                    : "Choose a base model and a training dataset to check readiness."}
                </p>
              )}
              <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
                {distributed && (
                  <Button
                    variant="outline"
                    className={trainingAction}
                    disabled={busy || hasActive || !peerId || !coordinator}
                    onClick={() => void submit("transport")}
                  >
                    <Network aria-hidden="true" className="size-4" />
                    Test TCP connection
                  </Button>
                )}
                <Button
                  variant="outline"
                  className={trainingAction}
                  disabled={
                    busy ||
                    hasActive ||
                    missingGlinerInput ||
                    (distributed && (!peerId || !coordinator))
                  }
                  onClick={() => void submit("preflight")}
                >
                  <ShieldCheck aria-hidden="true" className="size-4" />
                  Check training readiness
                </Button>
                <Button
                  variant="brand"
                  className="sm:ml-auto"
                  disabled={busy || hasActive || ready !== specKey}
                  onClick={() => void submit("train")}
                >
                  <Play aria-hidden="true" className="size-4" />
                  Start training
                </Button>
                {ready === specKey && (
                  <span className="inline-flex items-center gap-1.5 self-center text-xs text-success">
                    <Check aria-hidden="true" className="size-3.5" />
                    Readiness checks passed
                  </span>
                )}
              </div>
            </div>
          </section>
          <section id="training-runs" className="scroll-mt-24 border border-border bg-card">
            <div className="border-b border-border p-5 sm:p-6">
              <TrainingSectionHeading
                eyebrow="04 / Monitor"
                title="Run history"
                description="Follow progress, inspect logs, and pick up from a saved checkpoint."
                action={
                  <span className="inline-flex items-center gap-2 text-xs text-muted-foreground">
                    <History aria-hidden="true" className="size-4" />
                    {jobs.data?.jobs.length ?? 0} {jobs.data?.jobs.length === 1 ? "run" : "runs"}
                  </span>
                }
              />
            </div>
            <div className="grid min-w-0 lg:grid-cols-[minmax(0,280px)_minmax(0,1fr)]">
              <div className="min-w-0 space-y-2 border-b border-border bg-muted/20 p-4 lg:border-r lg:border-b-0">
                {jobs.data?.jobs.map((job) => (
                  <button
                    type="button"
                    key={job.id}
                    onClick={() => setSelectedId(job.id)}
                    aria-pressed={job.id === selectedId}
                    className={`group block w-full space-y-3 border border-l-2 p-3 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 ${job.id === selectedId ? "border-border border-l-primary bg-card" : "border-transparent hover:border-border hover:bg-card"}`}
                  >
                    <div className="flex items-center gap-2 font-medium">
                      <Activity aria-hidden="true" className="size-3.5 text-muted-foreground" />
                      {job.spec.family === "gliner25"
                        ? "GLiNER2.5"
                        : job.spec.family === "gemma4"
                          ? "Gemma4"
                          : "TCP"}
                      <ChevronRight
                        aria-hidden="true"
                        className="ml-auto size-3.5 text-muted-foreground"
                      />
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <TrainingStatus status={job.status} />
                      <span className="text-xs capitalize text-muted-foreground">{job.kind}</span>
                    </div>
                    <div className="truncate font-mono text-[10px] text-muted-foreground">
                      {job.id.slice(0, 12)}
                    </div>
                  </button>
                ))}
                {!jobs.loading && jobs.data?.jobs.length === 0 && (
                  <p className="px-2 py-4 text-xs leading-relaxed text-muted-foreground">
                    Your readiness checks and training runs will appear here.
                  </p>
                )}
              </div>
              <div className="min-w-0 space-y-4 p-5 sm:p-6">
                {selected ? (
                  <>
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="mr-auto">
                        <TrainingStatus status={selected.status} />
                      </div>
                      <Button
                        variant="outline"
                        disabled={
                          busy ||
                          selected.kind !== "training" ||
                          selected.spec.family !== "gliner25" ||
                          selected.status !== "running"
                        }
                        onClick={() =>
                          void act((signal) => client.training.pause(selected.id, signal))
                        }
                      >
                        Pause
                      </Button>
                      <Button
                        variant="outline"
                        disabled={busy || !active.has(selected.status)}
                        onClick={() =>
                          void act((signal) => client.training.cancel(selected.id, signal))
                        }
                      >
                        Cancel
                      </Button>
                      <Button
                        variant="outline"
                        className={trainingAction}
                        disabled={
                          busy ||
                          hasActive ||
                          selected.kind !== "training" ||
                          selected.spec.family !== "gliner25" ||
                          active.has(selected.status)
                        }
                        onClick={() => {
                          const key = `resume:${selected.id}`;
                          if (pending.current?.key !== key)
                            pending.current = { key, requestId: crypto.randomUUID() };
                          const requestId = pending.current.requestId;
                          void act((signal) =>
                            client.training.resume(selected.id, requestId, signal)
                          );
                        }}
                      >
                        {isTwoMac(selected.spec)
                          ? "Resume from common checkpoint"
                          : "Resume from checkpoint"}
                      </Button>
                    </div>
                    {selected.error && (
                      <p role="alert" className="text-destructive">
                        {selected.error}
                      </p>
                    )}
                    {selected.output_dir && (
                      <p className="break-all text-xs font-mono">{selected.output_dir}</p>
                    )}
                    <JobLogs key={selected.id} job={selected} />
                    <details>
                      <summary className="cursor-pointer text-sm">
                        Verification and checkpoint report
                      </summary>
                      <pre className="mt-2 max-h-96 overflow-auto bg-muted p-3 text-xs">
                        {JSON.stringify(
                          {
                            configuration: selected.configuration,
                            transport: selected.transport_report,
                            report: selected.report,
                            checkpoint: selected.checkpoint,
                          },
                          null,
                          2
                        )}
                      </pre>
                    </details>
                  </>
                ) : (
                  <div className="flex min-h-48 flex-col items-center justify-center gap-3 px-4 text-center">
                    <span className="grid size-11 place-items-center border border-border bg-muted/40">
                      <Terminal aria-hidden="true" className="size-5 text-muted-foreground" />
                    </span>
                    <div>
                      <h3 className="text-sm font-medium">Run details</h3>
                      <p className="mt-1 max-w-xs text-xs leading-relaxed text-muted-foreground">
                        Select a run to inspect progress, logs, and saved artifacts.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        </>
      )}
    </DashboardPage>
  );
}
