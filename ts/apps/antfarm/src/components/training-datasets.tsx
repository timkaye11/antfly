import { Button, Input } from "@antfly/design-system";
import type {
  TrainingDataset,
  TrainingDatasetSpec,
  TrainingHuggingFaceResponse,
} from "@antfly/sdk";
import {
  ArrowRight,
  Check,
  ChevronDown,
  Database,
  FileJson2,
  FileUp,
  Globe2,
  Loader2,
  Plus,
  Rows3,
  Settings2,
  Upload,
} from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import {
  TrainingSectionHeading,
  TrainingStatus,
  trainingAction,
  trainingField,
  trainingSelect,
} from "@/components/training-ui";
import { useApiConfig } from "@/hooks/use-api-config";
import { useTrainingResource } from "@/hooks/use-training";

const field = trainingField;
const select = trainingSelect;
const active = new Set(["importing", "preparing"]);
const formats = {
  gliner25: [
    ["gliner25", "Native GLiNER2.5 JSONL"],
    ["gliner_bio", "NER tokens + BIO tags"],
  ],
  gemma4: [
    ["gemma_instruction", "Instruction + response"],
    ["gemma_chat", "Chat messages"],
    ["gemma_completion", "Completion text"],
  ],
} as const;
const mappings = {
  gliner25: [],
  gliner_bio: [
    ["tokens", "Tokens column"],
    ["tags", "BIO tags column"],
  ],
  gemma_instruction: [
    ["prompt", "Prompt column"],
    ["response", "Response column"],
    ["input", "Extra input column (optional)"],
  ],
  gemma_chat: [["messages", "Messages column"]],
  gemma_completion: [["text", "Text column"]],
} as const;

type Props = {
  family: "gliner25" | "gemma4";
  distributed?: boolean;
  useJobTemplate?: boolean;
  trainFile?: string;
  onTrainFile?: (value: string) => void;
  modelDir: string;
  selectedId: string;
  onSelect: (id: string) => void;
  calibrationId: string;
  onCalibration: (id: string) => void;
  testId: string;
  onTest: (id: string) => void;
};

export function TrainingDatasets(props: Props) {
  const { family, modelDir, selectedId, onSelect } = props;
  const useJobTemplate = props.useJobTemplate ?? true;
  const { client, apiUrl } = useApiConfig();
  const load = useCallback((signal: AbortSignal) => client.training.datasets(signal), [client]);
  const catalog = useTrainingResource(load, 3000);
  const id = useId();
  const [source, setSource] = useState<"upload" | "huggingface">("upload");
  const [name, setName] = useState("");
  const [format, setFormat] = useState<TrainingDatasetSpec["format"]>(formats[family][0][0]);
  const [file, setFile] = useState<File>();
  const [filePreview, setFilePreview] = useState("");
  const [repository, setRepository] = useState("");
  const [subset, setSubset] = useState("");
  const [split, setSplit] = useState("");
  const [splits, setSplits] = useState<{ dataset: string; value: TrainingHuggingFaceResponse }>();
  const [hfPreview, setHfPreview] = useState<{ key: string; value: TrainingHuggingFaceResponse }>();
  const [maxRows, setMaxRows] = useState(1000);
  const [offset, setOffset] = useState(0);
  const [sequence, setSequence] = useState(512);
  const [columns, setColumns] = useState({
    tokens: "tokens",
    tags: "ner_tags",
    prompt: "instruction",
    response: "output",
    input: "",
    messages: "messages",
    text: "text",
  });
  const [labels, setLabels] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState("");
  const [importId, setImportId] = useState("");
  const controller = useRef(new AbortController());
  const pending = useRef<{ key: string; requestId: string } | undefined>(undefined);
  const currentFile = useRef<File | undefined>(undefined);
  // biome-ignore lint/correctness/useExhaustiveDependencies: An endpoint change must cancel uploads and discard credentials-bound requests.
  useEffect(() => {
    const next = new AbortController();
    controller.current = next;
    pending.current = undefined;
    setBusy(false);
    setError("");
    setProgress("");
    setImportId("");
    setSplits(undefined);
    setHfPreview(undefined);
    // Stop request replaces the controller so a retry gets a fresh signal.
    return () => controller.current.abort();
  }, [apiUrl, client]);
  const datasets = catalog.data?.datasets.filter((row) => row.family === family) ?? [];
  const ready = datasets.filter((row) => row.status === "ready");
  const selected = ready.find((row) => row.id === selectedId);
  const imported = datasets.find((row) => row.id === importId);
  const hfKey = JSON.stringify([repository, subset, split]);
  const preview = hfPreview?.key === hfKey ? hfPreview.value : undefined;
  const availableSplits = splits?.dataset === repository ? splits.value.splits : [];
  const availableColumns =
    preview?.features.map((feature) => String(feature.name ?? "")).filter(Boolean) ?? [];
  const compatible = (row: TrainingDataset) =>
    family !== "gemma4" || row.spec.model_dir === modelDir;
  const canTrain = (row: TrainingDataset) =>
    compatible(row) &&
    (row.row_count ?? 0) >= (props.distributed ? 2 : 1) &&
    (!props.distributed || family === "gemma4" || (row.row_count ?? 0) % 2 === 0);

  const action = async (operation: (signal: AbortSignal) => Promise<void>) => {
    const signal = controller.current.signal;
    setBusy(true);
    setError("");
    try {
      await operation(signal);
      if (!signal.aborted) catalog.refresh();
    } catch (failure) {
      if (!signal.aborted) setError(String(failure));
    } finally {
      if (!signal.aborted) setBusy(false);
    }
  };
  const importDataset = () =>
    action(async (signal) => {
      const spec: TrainingDatasetSpec = {
        request_id: "pending-request",
        name: name.trim() || (source === "upload" ? file?.name : repository) || "Dataset",
        family,
        source,
        format,
        max_rows: maxRows,
        columns: Object.fromEntries(
          mappings[format].map(([key]) => [key, columns[key]]).filter(([, value]) => value.trim())
        ),
        ...(format === "gliner_bio" && labels.trim()
          ? { label_names: labels.split(",").map((label) => label.trim()) }
          : {}),
        ...(source === "upload"
          ? { filename: file?.name, size_bytes: file?.size }
          : { hf_dataset: repository, hf_config: subset, hf_split: split, hf_offset: offset }),
        ...(family === "gemma4" ? { model_dir: modelDir, max_seq_len: sequence } : {}),
      };
      const key = JSON.stringify(spec);
      if (pending.current?.key !== key) pending.current = { key, requestId: crypto.randomUUID() };
      spec.request_id = pending.current.requestId;
      let record = await client.training.createDataset(spec, signal);
      if (signal.aborted) return;
      setImportId(record.id);
      if (source === "upload" && file) {
        for (let position = record.uploaded_bytes ?? 0; position < file.size; position += 32768) {
          signal.throwIfAborted();
          const bytes = new Uint8Array(await file.slice(position, position + 32768).arrayBuffer());
          signal.throwIfAborted();
          record = await client.training.uploadDatasetChunk(
            record.id,
            position,
            btoa(String.fromCharCode(...bytes)),
            signal
          );
          if (signal.aborted) return;
          setProgress(`Uploading ${Math.round(((record.uploaded_bytes ?? 0) / file.size) * 100)}%`);
        }
      }
      await client.training.prepareDataset(record.id, signal);
      if (!signal.aborted) {
        pending.current = undefined;
        setProgress("Import submitted. Preparation continues on this Mac.");
      }
    });
  const cancelUpload = () => {
    controller.current.abort();
    controller.current = new AbortController();
    setBusy(false);
    setProgress("Upload stopped. Import again with the same file and settings to resume.");
    catalog.refresh();
  };
  const labelFor = (row: TrainingDataset) =>
    `${row.name} · ${row.row_count ?? "?"} rows${compatible(row) ? "" : " · different tokenizer"}`;

  return (
    <section
      id="training-datasets"
      aria-label="Dataset configuration"
      className="scroll-mt-24 space-y-5 border-t border-border pt-6"
    >
      <TrainingSectionHeading
        eyebrow="02 / Data"
        title="Dataset configuration"
        description="Upload a file or import from Hugging Face, then select a prepared dataset."
        action={
          <span className="inline-flex items-center gap-1.5 border border-border bg-muted/30 px-2 py-1 text-[11px] text-muted-foreground">
            <Database aria-hidden="true" className="size-3" />
            {ready.length} ready
          </span>
        }
      />
      <label className={field}>
        Training dataset
        <select
          className={select}
          value={selectedId}
          onChange={(event) => onSelect(event.target.value)}
        >
          <option value="">
            {family === "gliner25"
              ? useJobTemplate
                ? "Use dataset from job template"
                : "Select a prepared dataset"
              : "Use staged prepared-input path above"}
          </option>
          {ready.map((row) => (
            <option key={row.id} value={row.id} disabled={!canTrain(row)}>
              {labelFor(row)}
            </option>
          ))}
        </select>
        <span className="text-xs font-normal leading-relaxed text-muted-foreground">
          {props.distributed
            ? family === "gliner25"
              ? "Two-Mac training requires at least two examples and an even row count."
              : "Two-Mac training requires an even selected example count, with at least two examples."
            : "Local training requires at least one example."}
        </span>
      </label>
      {family === "gliner25" && !useJobTemplate && !selectedId && props.onTrainFile && (
        <details className="group border-b border-border pb-4">
          <summary className="flex cursor-pointer list-none items-center gap-2 text-xs text-muted-foreground [&::-webkit-details-marker]:hidden">
            <FileJson2 aria-hidden="true" className="size-3.5" />
            Use a training file already on this Mac
            <ChevronDown
              aria-hidden="true"
              className="ml-auto size-3.5 transition-transform group-open:rotate-180"
            />
          </summary>
          <label className={`${field} mt-4`} htmlFor={`${id}-train-file`}>
            Training JSONL path
            <Input
              id={`${id}-train-file`}
              placeholder="/Users/Shared/antfly-training/data/train.jsonl"
              value={props.trainFile ?? ""}
              onChange={(event) => props.onTrainFile?.(event.target.value)}
            />
            <span className="text-xs font-normal leading-relaxed text-muted-foreground">
              Use an existing native GLiNER2.5 dataset, or import a file below. For two Macs,
              manually supplied files must exist at the same path on both machines.
            </span>
          </label>
        </details>
      )}
      {family === "gliner25" && (
        <details className="group border-b border-border pb-4">
          <summary className="flex cursor-pointer list-none items-center gap-2 text-xs text-muted-foreground [&::-webkit-details-marker]:hidden">
            <Settings2 aria-hidden="true" className="size-3.5" />
            Calibration &amp; evaluation datasets
            <ChevronDown
              aria-hidden="true"
              className="ml-auto size-3.5 transition-transform group-open:rotate-180"
            />
          </summary>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            {(
              [
                ["Calibration dataset", props.calibrationId, props.onCalibration],
                ["Held-out test dataset", props.testId, props.onTest],
              ] as const
            ).map(([label, value, change]) => (
              <label key={label} className={field}>
                {label}
                <select
                  className={select}
                  value={!useJobTemplate && value === "template" ? "none" : value}
                  onChange={(event) => change(event.target.value)}
                >
                  {useJobTemplate && <option value="template">Use job template</option>}
                  <option value="none">None</option>
                  {ready.map((row) => (
                    <option key={row.id} value={row.id}>
                      {labelFor(row)}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
        </details>
      )}
      {selected && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer">Prepared file location</summary>
          <p className="mt-2 break-all font-mono text-[11px]">{selected.path}</p>
        </details>
      )}
      <div className="space-y-5 border border-border bg-background/40 p-4 sm:p-5">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Plus aria-hidden="true" className="size-4 text-muted-foreground" />
          Import a dataset
        </div>
        <div className="grid grid-cols-2 gap-3" role="group" aria-label="Dataset source">
          <button
            type="button"
            aria-label="Upload file"
            aria-pressed={source === "upload"}
            className={`flex min-w-0 flex-col items-start gap-2 border p-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50 sm:flex-row sm:gap-3 sm:p-4 ${source === "upload" ? "border-foreground bg-card" : "border-border hover:border-input hover:bg-muted/40"}`}
            disabled={busy}
            onClick={() => setSource("upload")}
          >
            <FileUp aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
            <span className="min-w-0">
              <span className="block text-sm font-medium">Upload file</span>
              <span className="mt-1 hidden text-xs leading-relaxed text-muted-foreground sm:block">
                A CSV or JSONL from your computer
              </span>
            </span>
            {source === "upload" && (
              <Check aria-hidden="true" className="ml-auto hidden size-3.5 shrink-0 sm:block" />
            )}
          </button>
          <button
            type="button"
            aria-label="Hugging Face"
            aria-pressed={source === "huggingface"}
            className={`flex min-w-0 flex-col items-start gap-2 border p-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50 sm:flex-row sm:gap-3 sm:p-4 ${source === "huggingface" ? "border-foreground bg-card" : "border-border hover:border-input hover:bg-muted/40"}`}
            disabled={busy}
            onClick={() => setSource("huggingface")}
          >
            <Globe2 aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
            <span className="min-w-0">
              <span className="block text-sm font-medium">Hugging Face</span>
              <span className="mt-1 hidden text-xs leading-relaxed text-muted-foreground sm:block">
                Choose a public dataset and split
              </span>
            </span>
            {source === "huggingface" && (
              <Check aria-hidden="true" className="ml-auto hidden size-3.5 shrink-0 sm:block" />
            )}
          </button>
        </div>
        {source === "upload" ? (
          <label
            className="group relative flex cursor-pointer flex-col items-center gap-3 border border-dashed border-input bg-card px-4 py-6 text-center transition-colors hover:bg-muted/30 focus-within:ring-2 focus-within:ring-ring/50"
            htmlFor={`${id}-file`}
          >
            <span className="grid size-10 place-items-center bg-muted/60">
              <FileJson2 aria-hidden="true" className="size-5 text-muted-foreground" />
            </span>
            <span className="min-w-0 max-w-full">
              <span className="block break-all text-sm font-medium">
                {file?.name || "Choose a CSV or JSONL file"}
              </span>
              <span className="mt-1 block text-xs text-muted-foreground">
                {file
                  ? `${(file.size / 1024).toFixed(1)} KiB · Click to choose another file`
                  : "Browse your computer · up to 64 MiB"}
              </span>
            </span>
            <Input
              id={`${id}-file`}
              aria-label="CSV or JSONL file"
              type="file"
              className="sr-only"
              accept=".csv,.jsonl,.ndjson"
              disabled={busy}
              onChange={async (event) => {
                const next = event.target.files?.[0];
                currentFile.current = next;
                pending.current = undefined;
                setFile(next);
                setFilePreview("");
                setError("");
                if (next) {
                  if (next.size > 64 * 1024 ** 2) {
                    setError("Choose a file of 64 MiB or less.");
                    return;
                  }
                  const text = await next.slice(0, 4000).text();
                  if (currentFile.current === next) setFilePreview(text);
                }
              }}
            />
            <span className="max-w-md text-[11px] leading-relaxed text-muted-foreground">
              UTF-8 · CSV list columns use JSON arrays · Native GLiNER2.5 uses JSONL
            </span>
          </label>
        ) : (
          <div className="space-y-4">
            <label className={field} htmlFor={`${id}-repository`}>
              Hugging Face dataset
              <Input
                id={`${id}-repository`}
                placeholder="organization/dataset"
                value={repository}
                disabled={busy}
                onChange={(event) => {
                  setRepository(event.target.value.trim());
                  setSubset("");
                  setSplit("");
                }}
              />
            </label>
            <Button
              variant="outline"
              className={trainingAction}
              disabled={busy || !repository}
              onClick={() =>
                void action(async (signal) => {
                  const value = await client.training.huggingFace({ dataset: repository }, signal);
                  if (!signal.aborted) {
                    setSplits({ dataset: repository, value });
                    setSubset(value.splits[0]?.config ?? "");
                    setSplit(value.splits[0]?.split ?? "");
                  }
                })
              }
            >
              <Globe2 aria-hidden="true" className="size-4" />
              Load subsets and splits
            </Button>
            {availableSplits.length > 0 && (
              <>
                <label className={field}>
                  Subset / split
                  <select
                    className={select}
                    value={JSON.stringify([subset, split])}
                    disabled={busy}
                    onChange={(event) => {
                      const [config, value] = JSON.parse(event.target.value);
                      setSubset(config);
                      setSplit(value);
                    }}
                  >
                    {availableSplits.map((row) => (
                      <option
                        key={JSON.stringify([row.config, row.split])}
                        value={JSON.stringify([row.config, row.split])}
                      >
                        {row.config} / {row.split}
                      </option>
                    ))}
                  </select>
                </label>
                <Button
                  variant="outline"
                  disabled={busy || !subset || !split}
                  onClick={() =>
                    void action(async (signal) => {
                      const value = await client.training.huggingFace(
                        { dataset: repository, config: subset, split },
                        signal
                      );
                      if (!signal.aborted) setHfPreview({ key: hfKey, value });
                    })
                  }
                >
                  <Rows3 aria-hidden="true" className="size-4" />
                  Preview rows
                </Button>
              </>
            )}
            <p className="text-xs leading-relaxed text-muted-foreground">
              Public datasets supported by Hugging Face Dataset Viewer. No dataset scripts or media
              are downloaded.
            </p>
            {preview?.total_rows != null && (
              <p className="inline-flex items-center gap-2 text-xs text-muted-foreground">
                <Rows3 aria-hidden="true" className="size-3.5" />
                {preview.total_rows.toLocaleString()} rows in this split
              </p>
            )}
          </div>
        )}
        {((filePreview && source === "upload") || (preview && source === "huggingface")) && (
          <details className="group border border-border bg-card">
            <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2.5 text-xs [&::-webkit-details-marker]:hidden">
              <FileJson2 aria-hidden="true" className="size-3.5 text-muted-foreground" />
              Source preview (truncated)
              <ChevronDown
                aria-hidden="true"
                className="ml-auto size-3.5 text-muted-foreground transition-transform group-open:rotate-180"
              />
            </summary>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all border-t border-border bg-muted/30 p-3 text-[11px] leading-relaxed">
              {source === "upload" ? filePreview : preview?.rows.join("\n\n")}
            </pre>
          </details>
        )}
        <div className="grid gap-x-6 gap-y-5 border-t border-border pt-5 md:grid-cols-2">
          <label className={field} htmlFor={`${id}-name`}>
            Dataset name
            <Input
              id={`${id}-name`}
              value={name}
              placeholder={file?.name || "My training data"}
              disabled={busy}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <label className={field}>
            Input format
            <select
              className={select}
              value={format}
              disabled={busy}
              onChange={(event) => setFormat(event.target.value as TrainingDatasetSpec["format"])}
            >
              {formats[family].map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className={field} htmlFor={`${id}-rows`}>
            Maximum rows to import
            <Input
              id={`${id}-rows`}
              type="number"
              min={1}
              max={10000}
              value={maxRows}
              disabled={busy}
              onChange={(event) => setMaxRows(Number(event.target.value))}
            />
          </label>
          {source === "huggingface" && (
            <label className={field} htmlFor={`${id}-offset`}>
              Starting row (zero-based)
              <Input
                id={`${id}-offset`}
                type="number"
                min={0}
                value={offset}
                disabled={busy}
                onChange={(event) => setOffset(Number(event.target.value))}
              />
            </label>
          )}
          {family === "gemma4" && (
            <label className={field} htmlFor={`${id}-sequence`}>
              Maximum sequence tokens
              <Input
                id={`${id}-sequence`}
                type="number"
                min={8}
                max={8192}
                value={sequence}
                disabled={busy}
                onChange={(event) => setSequence(Number(event.target.value))}
              />
            </label>
          )}
          {mappings[format].map(([key, label]) => (
            <label key={key} className={field} htmlFor={`${id}-${key}`}>
              {label}
              <Input
                id={`${id}-${key}`}
                list={`${id}-columns`}
                value={columns[key]}
                disabled={busy}
                onChange={(event) =>
                  setColumns((values) => ({ ...values, [key]: event.target.value }))
                }
              />
            </label>
          ))}
          <datalist id={`${id}-columns`}>
            {availableColumns.map((column) => (
              <option key={column} value={column} />
            ))}
          </datalist>
          {format === "gliner_bio" && (
            <label className={`${field} md:col-span-2`} htmlFor={`${id}-labels`}>
              Ordered BIO labels
              <Input
                id={`${id}-labels`}
                value={labels}
                placeholder="O, B-PER, I-PER, B-ORG, I-ORG"
                disabled={busy}
                onChange={(event) => setLabels(event.target.value)}
              />
              <span className="text-xs font-normal leading-relaxed text-muted-foreground">
                For integer tags, order must match their IDs. Leave blank to use Hugging Face
                ClassLabel metadata. Tokens are joined with spaces and entity offsets are rebuilt.
              </span>
            </label>
          )}
        </div>
        {format === "gliner25" && (
          <p className="text-xs leading-relaxed text-muted-foreground">
            Each row needs version 1, a unique id, text, an explicit schema, and annotations. Native
            validation checks labels and offsets before the dataset becomes ready.
          </p>
        )}
        {family === "gemma4" && (
          <p className="break-all text-xs leading-relaxed text-muted-foreground">
            Prepared with the tokenizer in {modelDir}. Text chat supports system, user, and
            assistant messages.
          </p>
        )}
        {(error || catalog.error) && (
          <p
            role="alert"
            className="border-l-2 border-destructive bg-destructive/5 px-3 py-2 text-sm text-destructive"
          >
            {error || catalog.error}
          </p>
        )}
        {progress && (
          <p role="status" className="flex items-center gap-2 text-xs text-muted-foreground">
            {busy && <Loader2 aria-hidden="true" className="size-3.5 motion-safe:animate-spin" />}
            {progress}
          </p>
        )}
        {imported && (
          <p role="status" className="flex flex-wrap items-center gap-2 text-xs">
            {imported.name}
            <TrainingStatus status={imported.status} />
            {imported.error ? ` · ${imported.error}` : ""}
          </p>
        )}
        <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
          <Button
            className={trainingAction}
            disabled={
              busy ||
              maxRows < 1 ||
              maxRows > 10000 ||
              (source === "upload" ? !file || file.size > 64 * 1024 ** 2 : !preview) ||
              (family === "gemma4" && !modelDir)
            }
            onClick={() => void importDataset()}
          >
            <Upload aria-hidden="true" className="size-4" />
            Import and prepare dataset
          </Button>
          {busy && (
            <Button variant="outline" onClick={cancelUpload}>
              Stop request
            </Button>
          )}
        </div>
      </div>
      {datasets.length > 0 && (
        <details className="group/library" open>
          <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-medium [&::-webkit-details-marker]:hidden">
            <Database aria-hidden="true" className="size-4 text-muted-foreground" />
            Dataset library{" "}
            <span className="font-mono text-xs font-normal text-muted-foreground">
              {datasets.length}
            </span>
            <ChevronDown
              aria-hidden="true"
              className="ml-auto size-4 text-muted-foreground transition-transform group-open/library:rotate-180"
            />
          </summary>
          <div className="mt-3 space-y-3">
            {datasets.map((row) => (
              <div
                key={row.id}
                className={`space-y-3 border border-l-2 p-4 text-sm ${row.id === selectedId ? "border-border border-l-primary bg-muted/20" : "border-border bg-background/40"}`}
              >
                <div className="flex flex-wrap justify-between gap-2">
                  <div className="min-w-0">
                    <h3 className="break-all font-medium">{row.name}</h3>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {row.spec.source === "huggingface" ? "Hugging Face" : "File upload"}
                      {row.row_count != null ? ` · ${row.row_count.toLocaleString()} rows` : ""}
                    </p>
                  </div>
                  <TrainingStatus status={row.status} />
                </div>
                {row.error && <p className="text-destructive">{row.error}</p>}
                <div className="flex flex-wrap gap-2">
                  {row.status === "ready" && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={!canTrain(row)}
                      onClick={() => onSelect(row.id)}
                    >
                      {row.id === selectedId ? (
                        <Check aria-hidden="true" className="size-3.5" />
                      ) : (
                        <ArrowRight aria-hidden="true" className="size-3.5" />
                      )}
                      Use for training
                    </Button>
                  )}
                  {["failed", "cancelled", "pending"].includes(row.status) && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={busy}
                      onClick={() =>
                        void action(async (signal) => {
                          await client.training.prepareDataset(row.id, signal);
                        })
                      }
                    >
                      Retry preparation
                    </Button>
                  )}
                  {active.has(row.status) && (
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={busy}
                      onClick={() =>
                        void action(async (signal) => {
                          await client.training.cancelDataset(row.id, signal);
                        })
                      }
                    >
                      Cancel preparation
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={
                      busy ||
                      active.has(row.status) ||
                      row.id === selectedId ||
                      row.id === props.calibrationId ||
                      row.id === props.testId
                    }
                    onClick={() =>
                      void action(async (signal) => {
                        await client.training.removeDataset(row.id, signal);
                      })
                    }
                  >
                    Remove
                  </Button>
                </div>
                {row.preview && (
                  <details>
                    <summary className="cursor-pointer text-xs text-muted-foreground">
                      Prepared rows and fingerprint
                    </summary>
                    <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-all bg-muted p-3 text-xs">
                      {row.preview.join("\n\n")}
                    </pre>
                    <p className="mt-2 break-all font-mono text-xs">SHA-256 {row.sha256}</p>
                  </details>
                )}
              </div>
            ))}
          </div>
        </details>
      )}
      {!catalog.loading && datasets.length === 0 && !catalog.error && (
        <div className="flex items-start gap-3 px-1 py-2 text-muted-foreground">
          <Database aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
          <p className="text-xs leading-relaxed">
            Your dataset library is empty. Imported datasets will appear here with their validation
            status and row count.
          </p>
        </div>
      )}
    </section>
  );
}
