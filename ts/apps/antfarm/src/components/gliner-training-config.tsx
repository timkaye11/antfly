import { Input } from "@antfly/design-system";
import type { TrainingJobSpec } from "@antfly/sdk";
import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useId } from "react";
import { trainingField, trainingSelect } from "@/components/training-ui";

export type GlinerTrainingOptions = NonNullable<TrainingJobSpec["gliner25_options"]>;

export const defaultGlinerOptions: GlinerTrainingOptions = {
  execution: "resident_metal",
  mode: "lora",
  rank: 8,
  alpha: 16,
  epochs: 1,
  batch_size: 2,
  accumulation: 1,
  encoder_lr: 0.00001,
  task_lr: 0.0005,
  scheduler: "linear",
  warmup_ratio: 0.1,
  weight_decay: 0.01,
  max_grad_norm: 1,
  seed: 42,
  shuffle: true,
  dropout: 0,
  targets: ["encoder"],
  max_text_words: 128,
  max_sequence_tokens: 512,
  max_queries: 64,
  checkpoint_every_microbatches: 100,
  memory_total_gib: 12,
  memory_host_gib: 5.875,
  memory_backend_gib: 4,
  dataset_memory_mib: 512,
  source_auxiliary_mib: 384,
};

type NumericKey = {
  [K in keyof GlinerTrainingOptions]-?: NonNullable<GlinerTrainingOptions[K]> extends number
    ? K
    : never;
}[keyof GlinerTrainingOptions];

type NumberField = {
  key: NumericKey;
  label: string;
  min: number;
  max: number;
  step?: number | "any";
};

const basicFields: NumberField[] = [
  { key: "epochs", label: "Epochs", min: 1, max: 10000 },
  { key: "batch_size", label: "Batch size per Mac", min: 1, max: 8 },
  { key: "rank", label: "Adapter rank", min: 1, max: 1024 },
  { key: "task_lr", label: "Learning rate", min: 0.00000001, max: 1, step: "any" },
];

const advancedFields: NumberField[] = [
  { key: "alpha", label: "Adapter alpha", min: 0.00000001, max: 65536, step: "any" },
  { key: "dropout", label: "Adapter dropout", min: 0, max: 0.99, step: 0.01 },
  { key: "accumulation", label: "Gradient accumulation", min: 1, max: 65536 },
  { key: "warmup_ratio", label: "Warmup fraction", min: 0, max: 1, step: 0.01 },
  { key: "weight_decay", label: "Weight decay", min: 0, max: 1, step: 0.01 },
  { key: "max_grad_norm", label: "Gradient clipping", min: 0, max: 1000, step: "any" },
  { key: "seed", label: "Random seed", min: 0, max: 4294967295 },
  { key: "max_text_words", label: "Maximum text words", min: 1, max: 8192 },
  { key: "max_sequence_tokens", label: "Maximum sequence tokens", min: 1, max: 16384 },
  { key: "max_queries", label: "Maximum queries", min: 1, max: 256 },
  {
    key: "checkpoint_every_microbatches",
    label: "Save checkpoint every (microbatches)",
    min: 1,
    max: 1000000,
  },
];

const memoryFields: NumberField[] = [
  {
    key: "memory_total_gib",
    label: "Total memory budget (GiB)",
    min: 0.25,
    max: 1024,
    step: 0.125,
  },
  { key: "memory_host_gib", label: "Host memory budget (GiB)", min: 0.25, max: 1024, step: 0.125 },
  {
    key: "memory_backend_gib",
    label: "Compute memory budget (GiB)",
    min: 0.25,
    max: 1024,
    step: 0.125,
  },
  { key: "dataset_memory_mib", label: "Dataset memory budget (MiB)", min: 4, max: 4096 },
  { key: "source_auxiliary_mib", label: "Model loading overhead (MiB)", min: 4, max: 1024 },
];

type Props = {
  useTemplate: boolean;
  onUseTemplate: (value: boolean) => void;
  model: string;
  onModel: (value: string) => void;
  template: string;
  onTemplate: (value: string) => void;
  overrideTemplate: boolean;
  onOverrideTemplate: (value: boolean) => void;
  value: GlinerTrainingOptions;
  onChange: (value: GlinerTrainingOptions) => void;
};

export function GlinerTrainingConfig(props: Props) {
  const id = useId();
  const { value, onChange } = props;
  const numberField = ({ key, label, min, max, step }: NumberField) => (
    <label key={key} className={trainingField} htmlFor={`${id}-${key}`}>
      {label}
      <Input
        id={`${id}-${key}`}
        type="number"
        min={min}
        max={max}
        step={step ?? 1}
        value={value[key] ?? ""}
        onChange={(event) => onChange({ ...value, [key]: Number(event.target.value) })}
      />
    </label>
  );

  return (
    <div className="space-y-5">
      <label className={`${trainingField} max-w-sm`}>
        Configuration method
        <select
          className={trainingSelect}
          value={props.useTemplate ? "template" : "form"}
          onChange={(event) => props.onUseTemplate(event.target.value === "template")}
        >
          <option value="form">Configure in Antfarm</option>
          <option value="template">Existing job JSON (advanced)</option>
        </select>
      </label>
      {props.useTemplate ? (
        <>
          <label className={trainingField} htmlFor={`${id}-template`}>
            Job JSON path
            <Input
              id={`${id}-template`}
              value={props.template}
              onChange={(event) => props.onTemplate(event.target.value)}
            />
            <span className="text-xs font-normal leading-relaxed text-muted-foreground">
              Use an existing version 1 LoRA or DoRA job. Each run uses a copy and a fresh output
              directory.
            </span>
          </label>
          <label className="flex items-center gap-3 text-sm font-medium">
            <input
              type="checkbox"
              className="size-4 accent-primary"
              checked={props.overrideTemplate}
              onChange={(event) => props.onOverrideTemplate(event.target.checked)}
            />
            Override the JSON training settings for this run
          </label>
        </>
      ) : (
        <label className={trainingField} htmlFor={`${id}-model`}>
          Base model directory
          <Input
            id={`${id}-model`}
            value={props.model}
            onChange={(event) => props.onModel(event.target.value)}
          />
          <span className="text-xs font-normal leading-relaxed text-muted-foreground">
            Choose a local GLiNER2.5 FP32 model package. Antfarm creates the job configuration and
            output directory for you.
          </span>
        </label>
      )}
      {(!props.useTemplate || props.overrideTemplate) && (
        <>
          <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
            <label className={trainingField}>
              Compute
              <select
                className={trainingSelect}
                value={value.execution}
                onChange={(event) =>
                  onChange({
                    ...value,
                    execution: event.target.value as GlinerTrainingOptions["execution"],
                  })
                }
              >
                <option value="resident_metal">Metal</option>
                <option value="native">CPU</option>
              </select>
            </label>
            <label className={trainingField}>
              Adapter
              <select
                className={trainingSelect}
                value={value.mode}
                onChange={(event) =>
                  onChange({ ...value, mode: event.target.value as GlinerTrainingOptions["mode"] })
                }
              >
                <option value="lora">LoRA</option>
                <option value="dora">DoRA</option>
              </select>
            </label>
            {basicFields.map(numberField)}
          </div>
          <details className="group border-t border-border pt-4">
            <summary className="flex cursor-pointer list-none items-center gap-2 text-xs text-muted-foreground [&::-webkit-details-marker]:hidden">
              <SlidersHorizontal aria-hidden="true" className="size-4" />
              Advanced training settings
              <ChevronDown
                aria-hidden="true"
                className="ml-auto size-4 transition-transform group-open:rotate-180"
              />
            </summary>
            <div className="mt-5 space-y-5">
              <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
                <label className={trainingField}>
                  Learning rate schedule
                  <select
                    className={trainingSelect}
                    value={value.scheduler}
                    onChange={(event) =>
                      onChange({
                        ...value,
                        scheduler: event.target.value as GlinerTrainingOptions["scheduler"],
                      })
                    }
                  >
                    <option value="linear">Linear decay</option>
                    <option value="cosine">Cosine decay</option>
                    <option value="constant">Constant</option>
                  </select>
                </label>
                {advancedFields.map(numberField)}
              </div>
              <label className={trainingField} htmlFor={`${id}-targets`}>
                Adapter target modules
                <Input
                  id={`${id}-targets`}
                  value={(value.targets ?? []).join(", ")}
                  onChange={(event) =>
                    onChange({
                      ...value,
                      targets: event.target.value.split(",").map((target) => target.trim()),
                    })
                  }
                />
                <span className="text-xs font-normal leading-relaxed text-muted-foreground">
                  Comma-separated aliases or module names. The default encoder target trains
                  attention and feed-forward adapters.
                </span>
              </label>
              <label className="flex items-center gap-3 text-sm font-medium">
                <input
                  type="checkbox"
                  className="size-4 accent-primary"
                  checked={value.shuffle ?? true}
                  onChange={(event) => onChange({ ...value, shuffle: event.target.checked })}
                />
                Shuffle training examples
              </label>
              <div className="space-y-4 border-t border-border pt-4">
                <p className="text-xs leading-relaxed text-muted-foreground">
                  Memory budgets apply to each Mac. They cap resource use; readiness and the trainer
                  still check whether the model fits.
                </p>
                <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
                  {memoryFields.map(numberField)}
                </div>
              </div>
            </div>
          </details>
        </>
      )}
    </div>
  );
}
