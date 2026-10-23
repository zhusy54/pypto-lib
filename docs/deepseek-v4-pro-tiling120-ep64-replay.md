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
| Shared experts | `SH_MM_BLOCKS`: 6 to 24; `W2_INNER`: 2 to 1; `W2_ACT_INNER`: 8 to 2. The later shared-expert commit sets `SH_M_TILE=96` and `SH_ROWS_PER_BLOCK=8`. |
| TP1 output projection B | For `T_PAD=720` and `D=7168`, use 48-row by 896-channel output regions: 15 by 8 gives 120 blocks per call. The inner N tile is 128; the complete K reduction stays within each block. |
| Projection B epilogue | Redistribute complete 8-row by 512-channel units across at most 120 workers, preserving the 16-group accumulation order. |
| Attention-to-MoE packing | Use at most 120 workers to copy active rows and zero padded rows. The SWA, HCA, and CSA bridges use the same partition; CSA was measured. |
| Gate | Assign the existing 32-by-32 compute tiles to at most 240 mixed workers. Keep the cache policy at SPMD scope and preserve the AIC/AIV transfer protocol. |

The shared gate/up matmuls now use 24 blocks per call; shared W2 matmul and W2
activation use 28 each. This does not force every task to have 120 blocks.
Independent calls can execute concurrently. Projection A keeps its existing
partition. Routed expert kernel tiling, mixed AIC/AIV protocols, compiler/runtime
sources, and model defaults are unchanged. In particular, this branch does not
change the default model EP setting to 64.

For the full PRO TP1 shape, projection B changes from 126 to 120 blocks per
call (16 calls), its epilogue from 322 to 120, and attention packing from 768
to 120. Packing still covers 720 active rows plus 48 zero rows. The gate covers
all 276 existing work units with 240 workers: 36 workers execute two units and
204 execute one. The epilogue covers 1,260 work units: 60 workers execute 11
units and 60 execute 10. These changes do not pad the work to an artificial
multiple or discard tail units.

The target estimation convention uses 120 AIC and 120 AIV, with **every MIX
block modeled as one AIC plus one AIV**. This is an analysis assumption only;
the executable A3 kernels retain their actual one-AIC/two-AIV protocol. The
measured block service is not multiplied or divided by two to implement that
assumption. This repository change does not alter the compiler's mixed-kernel
lowering or claim that a one-to-one A3 kernel was measured.

## What the EP64 replay measures

The [replay](../examples/advanced/deepseek_v4_pro_ep64_replay/sample.py) runs one
logical EP64 shard on one physical device. It uses 384 global experts, top-6
routing, and 6 local experts. For batch 180 and MTP 3, the model processes
`180 * (1 + 3) = 720` active tokens per source rank. Synthetic balanced routing
gives each expert 720 rows. With a 64-row expert tile this produces
`6 * ceil(720 / 64) = 72` calls per routed task family. Shared experts now use
`ceil(720 / 96) = 8` calls; the earlier 64-row capture used 12 calls.

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

## Phase-2 full-layer capture

The projection/packing/gate changes were executed once on two A3 devices with
24 AIC and 48 AIV per device: TP1, EP2, batch 180 per rank, MTP 3, layer 10,
and a 180-entry history-position list containing 524,288. The code processes
720 active tokens and uses 768 rows for MoE capacity. Level 4 collected separate
graph and timing passes for one logical invocation; golden comparison was
intentionally skipped.

Both ranks completed with 1,821 kernel calls, 83 task families, 21,732 logical
blocks, and 22,356 physical core-block records. The four changed block counts
matched the intended partitions. Wait-dependency edges normalized by task
family and invocation ordinal matched the preceding capture. This is a
structural and execution check, not a steady-state speedup or accuracy result.

That capture used commit `88155c52f4e66ff52807e7ff0a799de559007548` plus the
three-file phase-2 change. It therefore used shared-expert row settings
`SH_M_TILE=64` and `SH_ROWS_PER_BLOCK=2`. The published branch also retains the
subsequent commit `5fdbdea66a35f30d8a44682da29700a84da0afdf`, which changes those
settings to 96 and 8. The combined published tree has not had another complete
layer capture; do not apply the preceding whole-layer call/block totals or
estimate data to that combined tree without recapturing it.

To capture the current branch, use the model's normal entry point inside a
job allocated two devices. Activate the supported environment and source CANN
first. `TASK_DEVICE` below must contain both allocated IDs, separated by a comma.

```bash
: "${TASK_DEVICE:?Use two device IDs allocated to this job}"
POS_LIST=$(python -c 'print(",".join(["524288"] * 180))')
set -o pipefail
PYPTO_BENCH=0 python -u models/deepseek_v4_flash_dspark/decode_layer.py \
  -p a2a3 -d "$TASK_DEVICE" --tp 1 --ep 2 --layer-id 10 \
  --start-pos "$POS_LIST" --enable-chip-swimlane 4 --skip-golden
```

To reuse frozen inputs, additionally pass `--golden-data /path/to/inputs-only`;
that directory must contain `in/` and no `out/` when using `--skip-golden`.
Use a fresh build after changing tiling. Each rank's `merged_swimlane*.json`,
`deps.json`, and `name_map.json` are under the printed build directory's
`dfx_outputs/rank*/` dispatch folders. Open the merged trace in Perfetto.
