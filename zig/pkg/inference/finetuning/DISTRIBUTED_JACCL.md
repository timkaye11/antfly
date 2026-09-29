# Two-Mac LoRA training over Thunderbolt RDMA or TCP/IP

This is an opt-in, synchronous, two-rank data-parallel path for the GLiNER2
real-autodiff trainer and the text-only Gemma4 real-autodiff trainer. Both ranks
load the same base model and adapter, train on disjoint examples, average LoRA
gradients through either standalone JACCL or TCP/IP before clipping and AdamW, and check that
their trainable weights match before and after the run. Antfly loads the C++
bridge at runtime; ordinary builds do not link MLX or JACCL. Select the
transport with `--transport jaccl` (the default) or `--transport tcp`.

GLiNER2.5 has a separate seeded optimizer and replay/checkpoint contract. Its
distributed training path is **not implemented** by this bridge. Do not launch
`train-gliner25` with distributed environment variables expecting data parallelism.
To add it, shard `gliner/boundary_run.zig`'s replay order by rank, reduce the
staged native and resident gradients inside `seeded_gradient_trainer.zig`
before clipping, and bind rank/world size to its durable job fingerprint and
resume validation. Its atomic optimizer transaction must stay intact.

## Prepare both Macs for RDMA

Use macOS 26.2 or newer, Thunderbolt 5 connectivity, and enable RDMA in
Recovery (`rdma_ctl enable`). After reboot, `ibv_devices` must list an RDMA
device on each Mac. The build host needs the macOS 26.2 SDK, CMake, and a C++20
compiler. See the [JACCL standalone documentation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/distributed/jaccl/lib/README.md)
for RDMA setup and topology semantics.

On **both** Macs, from the Antfly repository:

```sh
zig/pkg/inference/scripts/build_jaccl_bridge.sh
cd zig/pkg/inference
zig build -Dmetal=true install
```

The bridge script pins MLX to commit
`1f8e74e3f12f31365464a6867c6579f0e9b29d85` (v0.32.2) and writes
`zig/pkg/inference/zig-out/lib/libantfly_jaccl.dylib`. The installed
`zig/pkg/inference/zig-out/bin/antfly-inference` executable contains both
training commands; use its actual absolute path on both hosts. Stage the same
binary and dylib bytes on both Macs, since the launcher verifies their hashes.
The launcher also needs Python 3 on both Macs and noninteractive SSH access
from rank 0 to rank 1.

For **TCP/IP**, the Macs only need a routable IP connection. Thunderbolt 5,
RDMA setup, `ibv_devices`, a JACCL topology file, and macOS 26.2 are not
required by the transport. On both Macs, build the TCP bridge instead:

```sh
zig/pkg/inference/scripts/build_tcp_bridge.sh
cd zig/pkg/inference
zig build -Dmetal=true install
```

The TCP bridge is `zig/pkg/inference/zig-out/lib/libantfly_tcp.dylib` and has
no MLX dependency. Use an address of rank 0 reachable by rank 1 for
`--coordinator`; allow inbound TCP on that port through the host firewall.
Use it on a trusted network: this first TCP transport does not encrypt or
authenticate the training traffic. The trainer and launcher requirements for
matching staged files, Python 3, SSH, and separate host-local output paths
apply to both transports.

For RDMA, create the same JSON topology path on both hosts. For two ranks with one
Thunderbolt RDMA interface each:

```json
[[null, "rdma_en5"], ["rdma_en5", null]]
```

Use each host's actual `ibv_devices` name in its row; names may differ. The
same JSON content must be available on both hosts. Stage the same model,
adapter, input data, bridge, and executable at the same absolute paths. The
launcher hashes checked paths on both machines before starting either rank.
Use a fresh report path and fresh host-local output directories for each run;
the launcher refuses to overwrite an existing report.

## Preflight and transport smoke

Run these from rank 0 before loading a model. Substitute the actual repository
path, SSH alias, Thunderbolt IP, and RDMA device names. The preflight checks
macOS version, local device names against each rank's topology row, and exact
topology/bridge bytes. It writes a JSON report even when it fails.

```sh
python3 zig/pkg/inference/scripts/launch_jaccl_finetune.py \
  --remote mac2 --coordinator 192.168.1.1:32132 \
  --devices-file /models/jaccl-devices.json \
  --library /repo/antfly/zig/pkg/inference/zig-out/lib/libantfly_jaccl.dylib \
  --report /runs/jaccl-preflight.json --preflight-only
```

Then prove that both ranks can initialize JACCL and complete small and 4 MiB
`all_sum` transfers, `all_gather`, and a barrier over the configured RDMA link:

```sh
python3 zig/pkg/inference/scripts/launch_jaccl_finetune.py \
  --remote mac2 --coordinator 192.168.1.1:32132 \
  --devices-file /models/jaccl-devices.json \
  --library /repo/antfly/zig/pkg/inference/zig-out/lib/libantfly_jaccl.dylib \
  --report /runs/jaccl-smoke.json --timeout-seconds 60 -- \
  /repo/antfly/zig/pkg/inference/scripts/jaccl_smoke.py
```

Both rank logs (`/runs/jaccl-smoke-logs/rank0.log` and `rank1.log`) must contain
a `collective_smoke` JSON line with `status: pass`, and the launcher report must
show exit codes `[0, 0]`. The smoke script is hashed as
the executable, so it must have the same bytes and executable bit on both
hosts. A passing preflight alone does not establish RDMA connectivity; the
smoke does.

For a TCP/IP preflight and smoke, use the same launcher with `--transport tcp`,
omit `--devices-file`, and select `libantfly_tcp.dylib`:

```sh
python3 zig/pkg/inference/scripts/launch_jaccl_finetune.py \
  --transport tcp --remote mac2 --coordinator 192.168.1.1:32132 \
  --library /repo/antfly/zig/pkg/inference/zig-out/lib/libantfly_tcp.dylib \
  --report /runs/tcp-preflight.json --preflight-only

python3 zig/pkg/inference/scripts/launch_jaccl_finetune.py \
  --transport tcp --remote mac2 --coordinator 192.168.1.1:32132 \
  --library /repo/antfly/zig/pkg/inference/zig-out/lib/libantfly_tcp.dylib \
  --report /runs/tcp-smoke.json --timeout-seconds 60 -- \
  /repo/antfly/zig/pkg/inference/scripts/jaccl_smoke.py
```

For a TCP/IP training run, take either GLiNER2 or Gemma4 command below and
replace its bridge options with `--transport tcp` and the TCP library path.
Remove `--devices-file`. Keep the same `--check` and `--compare-adapter` paths.

## Launch GLiNER2

The example assumes 64 examples and `--batch-size 4`: each rank gets 32
examples, eight microbatches per epoch. `--max-steps 1` stops after the first
optimizer update. Set the coordinator to rank 0's
Thunderbolt address. The launcher uses noninteractive SSH to rank 1 and writes
a local run report.

```sh
python3 zig/pkg/inference/scripts/launch_jaccl_finetune.py \
  --remote mac2 --coordinator 192.168.1.1:32132 \
  --devices-file /models/jaccl-devices.json \
  --library /repo/antfly/zig/pkg/inference/zig-out/lib/libantfly_jaccl.dylib \
  --check /models/gliner2 --check /data/train.jsonl \
  --report /runs/gliner2-jaccl-launch.json --timeout-seconds 1800 \
  --compare-adapter /runs/gliner2-lora/adapter_model.safetensors -- \
  /repo/antfly/zig/pkg/inference/zig-out/bin/antfly-inference \
  finetune train run gliner2-autodiff \
  --model-dir /models/gliner2 --train-data /data/train.jsonl \
  --out-dir /runs/gliner2-lora --batch-size 4 --epochs 1 --max-steps 1 \
  --deterministic \
  --lora-only-trainables --backend metal
```

The launcher compares the two `adapter_model.safetensors` hashes after both
ranks finish. The trainer also checks cross-rank weights before training and
again before export. Review the rank logs and adapter hash in the report before
removing `--max-steps 1` for longer runs.

The loaded training dataset must contain an even number of examples and be
divisible by twice the per-rank batch size. This path rejects resume,
periodic checkpointing, evaluation data, early stopping, and non-LoRA
trainables. Output paths are host-local; both ranks write their own artifacts.
The launcher report records host preflight, exact command, and exit codes.
It also points to separate rank logs beside the report file.
The optimizer averages the two rank-local batch gradients. For GLiNER2's
multi-task loss, that objective can differ from evaluating one combined batch
when ranks have different numbers of supervised task rows.

## Launch text Gemma4

Use the same launcher with the Gemma4 CLI command and its four positional
paths, followed by `--trainer autodiff`:

```sh
python3 zig/pkg/inference/scripts/launch_jaccl_finetune.py \
  --remote mac2 --coordinator 192.168.1.1:32132 \
  --devices-file /models/jaccl-devices.json \
  --library /repo/antfly/zig/pkg/inference/zig-out/lib/libantfly_jaccl.dylib \
  --check /models/gemma4 --check /models/gemma4-adapter \
  --check /data/gemma4-prepared.json \
  --report /runs/gemma4-jaccl-launch.json --timeout-seconds 1800 \
  --compare-adapter /runs/gemma4-lora/adapter_model.safetensors -- \
  /repo/antfly/zig/pkg/inference/zig-out/bin/antfly-inference \
  finetune train run gemma4-lora \
  /models/gemma4 /models/gemma4-adapter /data/gemma4-prepared.json \
  /runs/gemma4-lora --trainer autodiff --max-examples 2 --epochs 1
```

Gemma4 uses the native CPU compute backend in this path. The selected
`--max-examples` count must be even, every selected example must have supervised tokens, and
`--grad-accum 1` is currently required. Multimodal prepared inputs and
surrogate training are rejected. The example above trains one example per
rank. Increase to `--max-examples 32` for 16 examples per rank per epoch.
`--eval-max-examples` still evaluates the same
held-out prefix on each rank; it does not affect training sharding.

## Validation and comparison

First run one optimizer step on each Mac and compare the resulting rank-local
adapter files. Then run the same effective global batch on one Mac and two Macs
with fixed seed, learning rate, clipping, precision, and examples. Compare
parameter deltas and loss trajectories before measuring steady-state
examples/s, step time, synchronization time, and peak memory. The launch report
contains setup and exit evidence; trainer metrics contain step timing, and
`distributed_sync` log lines report the synchronization hook time per optimizer step. There
is no performance threshold in this first milestone.

For the later MLX/JACCL comparison, use the same two hosts, topology, LoRA
rank, model, token lengths, global batch, and number of optimizer steps. Record
MLX's own implementation and JACCL version with the measurements. For a
transport comparison, use the same two Macs and optimizer settings over TCP/IP
and RDMA, and record synchronization time and examples/s. exo is a
reference for inference routing and topology discovery only; this training
path does not depend on exo.
