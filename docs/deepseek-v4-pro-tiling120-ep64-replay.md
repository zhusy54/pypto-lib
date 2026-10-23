# PRO CSA tiling experiment and EP64 compute replay

This branch carries the measured tiling changes for
`models/deepseek_v4_flash_dspark`, based on commit
`c0438d4da04de477ad688b61ec3c338ce45e83a0`. The intended scheduling model has
120 AIC and 120 AIV cores. Measurements were taken on A3 with 24 AIC and 48 AIV
cores; they are not measurements on the target device.

## Model changes

| Area | Change |
| --- | --- |
| HcPre, QKV, Indexer, compression | Increase selected persistent worker caps to 120. Actual block counts are still bounded by available independent work. |
| Indexer QR projection | Flatten independent row/column output tiles into one work-unit index. |
| CSA metadata and cache writeback | Cap workers at the number of independent existing token tiles. |
| HcPost, FFN normalization, shared/routed combine | Distribute complete token rows across up to 120 workers, preserving each row's arithmetic and accumulation order. |
| Shared experts | `SH_MM_BLOCKS`: 6 to 24; `W2_INNER`: 2 to 1; `W2_ACT_INNER`: 8 to 2. |

The shared gate/up matmuls now use 24 blocks per call; shared W2 matmul and W2
activation use 28 each. This does not force every task to have 120 blocks.
Independent calls can execute concurrently. Output projection tiling, routed
expert kernel tiling, mixed AIC/AIV protocols, compiler/runtime sources, and model
defaults are unchanged. In particular, this branch does not change the default
model EP setting to 64.

## What the EP64 replay measures

The [replay](../examples/advanced/deepseek_v4_pro_ep64_replay/sample.py) runs one
logical EP64 shard on one physical device. It uses 384 global experts, top-6
routing, and 6 local experts. For batch 180 and MTP 3, the model processes
`180 * (1 + 3) = 720` active tokens per source rank. Synthetic balanced routing
gives each expert 720 rows. With a 64-row expert tile this produces
`6 * ceil(720 / 64) = 72` calls per routed task family. Shared experts use
`ceil(720 / 64) = 12` calls.

The replay prepares post-dispatch data on the CPU and uses a compact receive
capacity of 768 rows per local expert. It does not allocate the full 46,080-row
EP64 receive capacity, run 64 ranks, or measure inter-rank dispatch/combine.
Routing is synthetic, not the observed output of model gating. The test-only
`sample_routed_store` task materializes results and must be excluded from model
operator budgets. Communication estimates still require a separate model.

## Reproduce the local compute captures

Use an environment with the repository's supported PyPTO, CANN, PTOAS, PyTorch,
and NumPy installed. Activate that environment and source CANN before running.
Run all commands below from the repository root.

The input preparer reuses the NZ-packed `shared_w*` and `routed_w*` tensors from
a saved distributed PRO model `data/in` directory. It selects rank 0 and the
first six routed experts, without unpacking or repacking the weights. Supply
your own source path; tensor files are not included in the repository.

```bash
export SOURCE_INPUT_DIR=/path/to/saved/model/data/in
export REPLAY_DIR="$PWD/build_output/ep64_compute_replay"

python examples/advanced/deepseek_v4_pro_ep64_replay/prepare_inputs.py \
  --source-input-dir "$SOURCE_INPUT_DIR" \
  --out-dir "$REPLAY_DIR/inputs"

for case in shared routed; do
  python examples/advanced/deepseek_v4_pro_ep64_replay/sample.py \
    --case "$case" --compile-only \
    --inputs-dir "$REPLAY_DIR/inputs" --out-dir "$REPLAY_DIR"
done
```

Submit the following runtime commands through your machine's device scheduler
(for example, `task-submit`) and use its allocated device ID. On managed
machines, execute these commands inside the submitted job rather than directly
on an unallocated device. Activate the same environment in that job.

```bash
: "${TASK_DEVICE:?Use the device ID allocated to this job}"
for case in shared routed; do
  python examples/advanced/deepseek_v4_pro_ep64_replay/sample.py \
    --case "$case" --device "$TASK_DEVICE" \
    --inputs-dir "$REPLAY_DIR/inputs" --out-dir "$REPLAY_DIR"
done
```

Each case compiles into `builds/<case>` and records execution status in
`results/<case>`. Chip swimlane level 4 is enabled. Compilation and runtime
statuses are separate; use a new output directory for each new capture.
Source hashes are recorded in each status file. Input preparation also writes
a manifest with routing counts and file hashes.

## Validation scope

The original experiment ran the phase-1 full CSA layer once on A3 with TP1,
EP2, batch 180 per rank, MTP 3, layer 10, and history position 524,288. The shared
and routed compute replays then each completed once on a single A3 device.
Model source hashes were checked against those measured versions before
packaging this branch. The replay kernel body is unchanged; the published
entry points replace machine-specific paths with command-line arguments.

These runs skipped golden comparison. Runtime completion and static partition
checks do not establish numerical accuracy or an end-to-end EP64 performance
result. The earlier phase-1 full-layer wrapper reported a post-run bookkeeping
error after the model completed; its source checksum check was rerun separately
and passed. No baseline speedup is claimed.
