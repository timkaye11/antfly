# Training in Antfarm

Antfarm trains locally by default. Optionally, it can discover nearby Antfly
installations, inspect a trusted SSH peer, check TCP collectives and staged
inputs, and manage a two-rank training job.
The standalone Antfly backend owns the job; closing Antfarm leaves it running.

This first version supports **GLiNER2.5 FP32 LoRA/DoRA on CPU or Metal** and
**text-only Gemma4 CPU LoRA with the autodiff trainer and accumulation 1**.
Local training requires at least one example; odd row counts are supported.
With a second Mac, each host must fit the full model and its training state.
Data parallelism splits examples, not model weights. Two-Mac GLiNER2.5 requires
an even number of JSONL training examples. Two-Mac Gemma4 requires an even
selected example count. Every selected Gemma example needs supervised tokens.
Gemma4 has cancel, but no pause/resume in either mode.
JACCL availability appears in inventory; launching JACCL remains a
[CLI workflow](DISTRIBUTED_JACCL.md).

## Install the local toolchain

Use macOS with Python **3.9 or newer**, Zig 0.16, and Xcode Command Line Tools.
Choose an absolute directory; use the same path on both Macs if adding a peer.
These examples use `/Users/Shared/antfly-training`, owned by the account that
runs Antfly locally and by the SSH account on the mini.

From the repository root on the build Mac:

```sh
mkdir -p /Users/Shared/antfly-training/{toolchain,models,data,runs,state}
chmod 700 /Users/Shared/antfly-training/state
ANTFLY_TCP_OUTPUT_ROOT=/Users/Shared/antfly-training/toolchain/lib \
  zig/pkg/inference/scripts/build_tcp_bridge.sh
cd zig/pkg/inference
zig build -Doptimize=ReleaseFast -Dmetal=true -j1 \
  --prefix /Users/Shared/antfly-training/toolchain
```

The installation contains `bin/antfly-inference`, `lib/libantfly_tcp.dylib`, and
the Python helpers in `share/antfly/training`. The TCP bridge is optional for
local training; skip `build_tcp_bridge.sh` if only running locally.

### Optional second Mac

For two-Mac training, copy that exact toolchain to the
same path on the mini. Copy `bin`, `lib`, and `share`; leave `var` local to each
Mac because it contains that machine's discovery identity. Independently built
binaries can differ; readiness requires matching binary and bridge hashes.
Stage models and adapters at identical absolute paths with identical contents
on both hosts. An existing job JSON, if used, must also be staged on both.
Antfarm generates form-configured job snapshots on each host automatically.
Datasets imported through Antfarm are staged
automatically. For manually configured dataset paths, copy identical files to
both hosts yourself. Models and adapters are not uploaded by the service.

Enable **Remote Login** on the mini. Set up SSH keys and verify its host key
interactively before using Antfarm:

```sh
ssh tim@mac-mini.local true
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes tim@mac-mini.local true
```

Use your actual account or an alias in `~/.ssh/config`. Configure nonstandard
SSH ports and identity files in that alias. The backend uses the credentials
and SSH configuration of its own OS account; SSH options cannot be supplied
through the API.

## Enable the local coordinator

Build the Antfly server from this branch, including its regenerated Antfarm
assets, on a machine with enough memory for the repository's build reservations:

```sh
cd zig
zig build antfly -Doptimize=ReleaseFast -Dmetal=true -j1
```

Start the updated server with the supplied configuration from the repository
root:

```sh
zig/zig-out/bin/antfly standalone --config configs/config-training-local.json
```

If transferring this build to another Mac, copy both `bin/antfly` and
`share/antfly/antfarm` from `zig/zig-out`, preserving that layout, and use the
transferred binary's path. The example uses a separate database under the training directory.
To use an existing standalone server, merge only its `training` section into
that server's configuration and restart it. Training is disabled when this
section is absent. Distributed and serverless coordinators reject enabled
training configuration.

`state_dir` is private coordinator state and must have a short absolute path
(at most 85 bytes, because it contains a Unix socket). Keep it on local disk.
`input_roots` restricts every model, template, and dataset path, including
symlink targets. `output_root` must already exist and be writable on every participating Mac.
The same configured paths are used when inspecting the remote peer. Managed
imports under `output_root/datasets` are also allowed as training inputs.

Without authentication, open the bundled Antfarm UI through `localhost` or
`127.0.0.1`. Training requests must come from a loopback socket and a browser
origin matching that host. To manage it remotely, enable Antfly authentication
and use an administrator account. The underlying TCP training traffic has no
encryption or authentication; use a trusted LAN, Thunderbolt IP link, or private
VPN, and permit the chosen rendezvous port on the coordinator.

## Optional: discover and connect the mini

The coordinator's `training.discovery: true` advertises and browses Bonjour
`_antfly._tcp`. The mini can advertise while serving inference:

```sh
/Users/Shared/antfly-training/toolchain/bin/antfly-inference run \
  --models-dir /Users/Shared/antfly-training/models \
  --advertise-training /Users/Shared/antfly-training/toolchain
```

`antfly inference run` accepts the same advertisement flag when its training
helpers are installed. An enabled standalone training coordinator also advertises.
Only protocol version, a persistent node ID, and SSH service information are
broadcast. Model paths, inventory, and capabilities are fetched after you connect
through trusted SSH. Discovery never enrolls a machine or launches training.

In **Connections → Nearby machines**, choose a discovered hostname or enter an
SSH alias and select **Connect Mac**. Inventory includes models, chip, RAM,
Metal availability, and installed TCP/JACCL bridges. **Models** also shows the
peer inventory. Inference routing continues to use configured inference
connections. Bonjour is scoped to the local network; manual aliases work across
routed networks and VPNs. Advertising SSH does not enable macOS Remote Login.

## Import and configure datasets

The Training page's **Dataset configuration** section supports two import sources:

- **Upload file** accepts UTF-8 CSV or JSONL up to 64 MiB. Uploads use 32 KiB,
  offset-checked chunks; retrying the same file and settings in the open page
  resumes an interrupted upload. Choose the input format, column names, and
  maximum rows to import.
- **Hugging Face** accepts a public dataset ID. Load its subsets/splits, preview
  rows, choose column mappings, and set a starting row and maximum row count.
  Imports use the [Dataset Viewer API](https://huggingface.co/docs/dataset-viewer/rows)
  without executing dataset scripts or fetching media. Private/gated datasets and
  datasets without viewer support require exporting a file and uploading it.
  Truncated source cells are rejected rather than used as training examples.

The formats are **native GLiNER2.5 JSONL**, **NER tokens and BIO tags**, and Gemma
**instruction/response**, **text chat**, or **completion text**. For BIO data,
map token and tag columns and supply the ordered label names for integer IDs;
Hugging Face `ClassLabel` metadata can supply those names. Conversion joins tokens
with spaces and rebuilds UTF-8 entity offsets. Native GLiNER validation checks
schemas, annotation offsets, labels, and duplicate row IDs before publication.

Gemma preparation uses the selected local base model's tokenizer files and
masks prompt tokens through the native preparation command. It supports plain
system/user/assistant text messages. Set a maximum sequence length; the
configured maximum rows times sequence length is limited to 2,097,152 tokens.
The prepared artifact is bound to the tokenizer/configuration fingerprint, so
changing tokenizer files requires preparing it again. Tokenizer files must be
present beside the model; a bare GGUF without tokenizer files is not supported
by this import workflow.

Imports are limited to 10,000 rows, 1 MiB per normalized row, and 64 MiB per
artifact. Preparation is a backend-owned, cancellable operation with a
15-minute limit. A coordinator runs one preparation or training operation at a
time. Interrupted preparation is marked failed after restart and can be retried.
The library stores up to 50 datasets with row previews, counts, source settings,
and SHA-256 fingerprints. Remove unused local imports through the library;
datasets referenced by retained training jobs cannot be removed.
Readiness and launch verify each selected training, calibration, and test
artifact against its saved SHA-256 in both local and two-Mac modes. Changed
bytes require importing the dataset again.

Choose **Use for training** after preparation. GLiNER also supports separate
calibration and held-out datasets. Both default to **None** for a form-configured
run; existing JSON jobs can retain or disable the template's files.
Local training accepts any nonzero row count. Two-Mac GLiNER training requires
an even row count; Gemma requires an even selected example count. Calibration
and test files can have odd counts. Selecting an imported dataset changes the per-run snapshot
without editing any original template. For a form-configured GLiNER run, the
optional **Use a training file already on this Mac** control accepts a native
training JSONL path. Existing job JSON and Gemma prepared-input paths remain
available as alternatives to imported datasets.

When using a second Mac, readiness and launch stream the selected artifacts over trusted SSH to the
same managed path on the peer. Publication is atomic and SHA-256 checked, and an
existing matching file is reused. Models and tokenizers stay pre-staged. Local
removal does not delete remote copies. Uploads are authenticated like every
training operation; no filesystem path or executable can be chosen as an upload
destination.

## Run a job

Open **Inference → Training**. Configure the model, dataset, and run settings
first. **Where to run** defaults to this Mac. To distribute a run, enable
**Use another Mac** near the end of the form, connect/select a peer, and enter
this Mac's address (for example `192.168.1.20:32132`). That address must be
reachable **from the mini**; a loopback address is not suitable. Local runs do
not require a peer, SSH, coordinator address, or TCP bridge.

The API accepts `execution_mode: "local"` or `"two_mac"`. Omit `peer_id` and
`coordinator` for local runs. For compatibility, requests without an explicit
mode use two-Mac execution when a peer is supplied, and local execution otherwise.

For GLiNER2.5, **Configure in Antfarm** is the default. Choose a local compatible
FP32 base-model directory, select or import a dataset, and set compute, LoRA or
DoRA, epochs, batch size, adapter rank, and learning rate. **Advanced training
settings** covers accumulation, scheduling, warmup, clipping, dropout, target
modules, seed/shuffling, tokenization, checkpoint cadence, and per-Mac memory
budgets, including dataset and model-loading overhead. Batch size is per Mac;
accumulation combines consecutive microbatches.
The learning rate applies to the trainable adapters. Antfarm writes the native
version-1 configuration into each run directory and validates it before launch.
Creating a job JSON manually is unnecessary.

**Existing job JSON (advanced)** keeps the previous template workflow available,
including native options not exposed in the form. Its settings remain intact
unless **Override the JSON training settings for this run** is enabled; that
option applies the displayed form values. The template stays unchanged and each
run receives a fresh output directory. See the [finetuning guide](FINETUNING.md)
and [distributed GLiNER2.5 contract](DISTRIBUTED_JACCL.md#launch-gliner25).

API clients configure a form-based GLiNER run with `base_model`, either
`dataset_id` or `train_file`, and optional `gliner25_options`. Alternatively,
supply `gliner25_config`. Mixing that template path with `base_model` or
`train_file` is rejected. Output and recovery paths stay coordinator-owned.
Memory budgets are ceilings, not estimates of actual usage; live native
admission can reject a job even after structural readiness succeeds. Reduce
the workload and its budgets or free memory before retrying an admission denial.

For Gemma4, supply the native base-model directory, initial adapter directory,
and a prepared-input JSON or imported dataset, then choose example count, epochs,
and learning rate.
Start with a small example count and one epoch. For two-Mac training, both
ranks must use the same selected inputs, and the full model must fit on each Mac.

1. **Check training readiness** checks the local inventory, path containment,
   fresh output, machine lock, disk headroom, and native job/input contract.
   It hashes inputs and compares a configured GLiNER memory envelope with RAM.
   Two-Mac runs also check the peer, TCP collectives, and matching hashes.
   **Test TCP connection** is available separately when a second Mac is enabled.
   Readiness does not load the full training graph or prove that a real model
   fits; native admission still applies.
2. Review the effective configuration and reports, then **Start training**.
   The backend repeats readiness checks, so old browser state cannot bypass them.
3. Follow progress and bounded live logs for this Mac, or both ranks when
   distributed. Success requires clean process exit and a final adapter hash;
   two-Mac runs also require matching adapter hashes across both ranks.

Each job writes under `output_root/<job-id>/output` separately on each host.
The coordinator keeps job records, reports, and participating rank logs under `state_dir`.
The UI lists the latest 200 records; records and logs remain on disk.

## Stop, recover, and troubleshoot

**Pause** on a local GLiNER2.5 run saves `latest.safetensors` at a cooperative
boundary and confirms process cleanup. **Resume from checkpoint** starts a fresh
output directory, restores the local checkpoint, and pins the saved state hash
when a completed result is available. Native restore fingerprint checks reject
changed model, data, or semantic settings.

For two-Mac GLiNER2.5 runs, **Pause** asks both running parents to stop at a cooperative boundary.
A paused job is reported only after both ranks acknowledge cleanup and a common
checkpoint receipt is found. **Resume from common checkpoint** scans both hosts,
selects the newest matching generation and shared state hash, and starts a fresh
job/output directory. Each rank restores its own checkpoint; no checkpoint is
copied between machines. Changed model/data/semantic settings remain subject to
the native restore fingerprint checks. Gemma4 exposes **Cancel** only.

**Cancel**, backend shutdown, or loss of the backend lifetime pipe stops the
launcher. Independent rank leases stop remote work after launcher/SSH loss.
The default lease and shutdown grace are 30 seconds each. A pause has a
five-minute limit. The job timeout also covers preparation and verification.
Missing cleanup receipts after a rank launch, or an exception that prevents
reading its report, produce `cleanup_unconfirmed`. A successful TCP check's
cleanup receipts cannot confirm cleanup of a later training launch. After an
unexpected manager restart, previously active jobs keep that status. Inspect
the participating machines and rank logs before retrying. Jobs are never
silently restarted.

An SSH or fingerprint failure usually means a missing trusted host key, a
different toolchain/data copy, or paths that differ between machines. A failed
TCP check usually means an unreachable coordinator address, firewall, or occupied
port. A busy machine is protected by an exclusive lock in its output root.
Use one shared output root for all training coordinators targeting that machine.
CLI users should also pass `--lock-file <output_root>/.training.lock` to cooperate.

For scripted use, the generated API and TypeScript SDK expose peers, preflights,
jobs, bounded log cursors, cancel, pause, and resume at `/db/v1/training/*` and
`client.training`. Creation and resume require a stable `request_id`; retrying
the same request returns the original operation. Reusing it for another
specification returns a conflict.

## Reproduce the local checks

From the repository root, after installing the inference toolchain (no model
downloads are needed):

```sh
export ANTFLY_TRAINING_TEST_BINARY=/Users/Shared/antfly-training/toolchain/bin/antfly-inference
cd zig/pkg/inference
python3 -m unittest scripts.test_training_service scripts.test_training_datasets scripts.test_distributed_training -v
zig build -Doptimize=ReleaseFast -Dmetal=true test-finetune-unit -j1 -- \
  --test-filter 'distributed GLiNER2.5'
```

`ANTFLY_TRAINING_TEST_BINARY` enables native GLiNER2.5 validation and Gemma4
tokenization/readiness checks. Without it, those tests are skipped. These checks
use small fixtures; they do not qualify full-model training.

From the repository root, check the native API boundary:

```sh
cd zig
zig build public-api-parity-test -Doptimize=ReleaseFast -j1 -- \
  --test-filter 'httpx training' \
  --test-filter 'httpx antfly routes require auth and enforce admin middleware'
```

Use the repository's pinned Node 24.16.0 and pnpm 11.10.0 for the frontend and
SDK. From the repository root, run the build before the tests:

```sh
cd ts
node scripts/run-pinned-toolchain.mjs pnpm --filter antfarm... build
node scripts/run-pinned-toolchain.mjs pnpm --filter antfarm test \
  src/pages/TrainingPage.test.tsx src/components/training-datasets.test.tsx \
  src/hooks/use-training.test.ts
node scripts/run-pinned-toolchain.mjs pnpm --filter @antfly/sdk test
diff -qr apps/antfarm/dist ../zig/pkg/antfly/antfarm
```

## Handoff status — 2026-09-30

The reviewed snapshot is `feat/jaccl-gliner` at
`8b1c8020c42d4bcd24e9782e184b3777b354f3b7` plus the accompanying uncommitted
changes. The complete branch patch is based on
`3886b3fb00f625c0a83103043728e6dffa2d46f2`. No changes were committed or pushed
as part of the handoff review.

The final review fixed two lifecycle/data issues: local runs now verify managed
dataset hashes, and launcher failures/timeouts cannot imply confirmed cleanup
without receipts from that launch. Both have regression coverage.
The subsequent form-configuration change removes the mandatory job JSON and
adds native readiness coverage for generated local and two-Mac snapshots.

| Check | Result | Scope |
| --- | --- | --- |
| Python service, datasets, and supervision | 55 passed; no skips | Native input checks enabled; form configuration, loopback TCP, cancellation, cleanup, recovery, and dataset contracts |
| Antfarm focused tests | 12 passed | Form configuration and optional JSON, local defaults, optional second Mac, dataset configuration, and endpoint changes |
| TypeScript SDK suite | 367 passed; 1 skipped | The skipped case is the fallback placeholder for an absent build; both actual CJS bundle tests passed |
| Native HTTP/API tests | 3 passed; no leaks | Administrator authorization, loopback/origin restrictions, and unavailable-manager behavior |
| Native distributed GLiNER2.5 tests | 6 passed | Rank partitioning, gradient union, coordinated pause, and common checkpoint receipts |
| Inference ReleaseFast/Metal build | Passed | Installed CLI and helper scripts |
| Real-model local Metal smoke | Passed | Direct form request, native readiness, two optimizer steps, adapter export/hash, and confirmed rank cleanup |
| Antfarm build and typecheck | Passed | Bundled assets match the local pinned-toolchain build exactly; canonical Linux CI remains pending |
| Browser inspection | Passed | Local/optional-peer forms, responsive layout, light/dark styling, and upload focus |

The local inference toolchain and TCP bridge are installed under
`/Users/Shared/antfly-training/toolchain`. The repository-pinned
`fastino/gliner2.5-small-v1` FP32 model (revision
`cab1bddfd30fda7b803a4691c41f90378a2d517a`) and a two-example synthetic entity
dataset are staged locally for TCP qualification. The development preview at
`http://localhost:4178/inference/training` uses read-only example responses;
it is not a running training coordinator.

The local form-request smoke completed as job
`bc69864b25f4485190b61903f3e26d54`, using the installed manager and native trainer
directly. Its two rows produced two microbatches and two optimizer steps, exit
code 0, and confirmed process cleanup. The 684,016-byte exported adapter has
SHA-256 `4b253c33a88c890bd4a2cb772aacbcb6b875839440ed8f26abbc91a6d55519c7`,
identical to the equivalent template-based run. This proves the direct form
configuration path can train the pinned model; it does not prove convergence
or physical two-Mac training.

To reproduce this small smoke in the form, use
`/Users/Shared/antfly-training/models/gliner25` and the native training file
`/Users/Shared/antfly-training/data/gliner25-tcp-smoke.jsonl`, with these settings:

| Setting | Value |
| --- | --- |
| Compute / adapter | Metal / LoRA |
| Epochs / batch size / accumulation | 1 / 1 / 1 |
| Adapter rank / alpha | 2 / 4 |
| Learning rate / schedule / warmup | 0.0001 / Constant / 0 |
| Shuffle | Off |
| Maximum text words / sequence tokens / queries | 32 / 64 / 16 |
| Checkpoint interval | 1 microbatch |
| Total / host / compute memory budgets | 3 / 0.5 / 0.75 GiB |
| Dataset / model-loading overhead budgets | 32 / 128 MiB |

Leave other fields at their defaults. These bounds fit this small fixture;
larger models or datasets need their own budgets. Earlier attempts were refused
by live memory admission or training limits; the memory guard was not bypassed.
The qualification records are under
`/Users/Shared/antfly-training/qualification-state`, separate from the server's
normal state directory, and the adapter is under
`runs/bc69864b25f4485190b61903f3e26d54/output/model/adapter_model.safetensors`.

The full standalone server build remains blocked on this 16 GiB Mac by the
existing 20 GiB storage-kernel build reservation. That guard was left intact.
The focused API tests do not establish a complete server-to-UI training run.
Physical two-Mac TCP/JACCL, real-model convergence, and performance are also
unqualified. The mini is reachable through Tailscale, but noninteractive SSH
currently rejects this Mac's key; no model or toolchain has been copied there.

The canonical Linux frontend/generated-artifact CI checks remain to be run;
the review did not dispatch CI. The remaining runtime acceptance sequence is:

1. Build the full server on a sufficiently provisioned Mac and start it with
   the supplied training configuration. Confirm Training, Connections, and
   Models against the live backend.
2. Repeat the qualified local GLiNER settings through the live UI. Exercise
   cancel and GLiNER2.5 pause/resume. Qualify Gemma4 separately with a compatible
   model, initial adapter, and prepared inputs.
3. Once noninteractive SSH works, copy the exact toolchain and stage identical
   model and manual dataset paths (and template paths when using existing JSON).
   This CLI/manager test can proceed independently of the full server build.
   Pass TCP collectives and readiness,
   then a tiny synchronized optimizer run with matching final adapter hashes.
4. Exercise GLiNER2.5 common-checkpoint pause/resume and connection-loss cleanup
   on the two physical hosts. Measure convergence and performance only after
   those lifecycle checks pass. Qualify JACCL separately through its CLI workflow.
