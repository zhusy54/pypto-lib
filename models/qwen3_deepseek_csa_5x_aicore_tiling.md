# Qwen3 and DeepSeek CSA Tiling for 5x AICores

## 1. Goal and scope

This plan targets a machine whose AICore count grows from 24 to 120. It keeps
the repository-default model shapes and does not change the hidden size, head
count, Batch limit, or sequence length.

- Qwen3: `models/qwen3_14b/decode_fwd.py::_decode_layer`
- DeepSeek CSA: `models/deepseek_v4_flash_mtp/decode_csa.py`
- Baseline: `main` at `c3f0dea274f55d9648920f17c968e8564ae9fcdc`

BlockDim means the number of parallel blocks launched by one task invocation.
A regular `pl.at` task has BlockDim 1. Logical block count describes all work
blocks for an operator, but those blocks do not necessarily belong to one task
invocation or execute concurrently.

Batch split markers:

- **Yes**: the block index explicitly includes a Batch or Token dimension.
- **No**: every block processes the complete Batch tile; parallelism comes from
  N, K, Head, or another non-Batch dimension.
- **Indirect**: the expression includes a row group, but this change does not
  split that dimension. The default shape normally has one physical row group.

## 2. Qwen3 plan

### 2.1 Selected tiling

| Task | Baseline blocks | Target blocks | Split dimensions | Batch split | Decision |
|---|---:|---:|---|:---:|---|
| `q_proj` | 50 | 50 | N x Split-K: `10 x 5` | No | Keep |
| `k_proj` | 10 | 10 | N x Split-K: `2 x 5` | No | Keep |
| `v_proj` | 10 | 10 | N x Split-K: `2 x 5` | No | Keep |
| `attn_swpipe_spmd` | BlockDim 24 | BlockDim 120 | Batch x KV Head work items | **Yes** | Change |
| `out_proj` | 50 | 100 | N x Split-K: `20 x 5` | No | Change |
| `gate_proj` | 85 | 85 | N x Split-K: `17 x 5` | No | Keep |
| `up_proj` | 85 | 85 | N x Split-K: `17 x 5` | No | Keep |
| `down_proj` | 85 | 85 | N x K tiles | No | Keep |
| `x_gamma0` | BlockDim 5 | BlockDim 5 | Hidden | No | Keep |
| `silu` | 17 regular tasks | 17 regular tasks | Intermediate N | No | Keep |
| `dcr_xgamma` | BlockDim 5 | BlockDim 5 | Hidden N | No | Keep |

### 2.2 Keep Q/K/V at 50/10/10

The Q, K, and V output workloads are approximately 5:1:1. Their existing
50:10:10 block counts therefore give similar work per block.

Changing only Q to 100 would have these effects:

- Q work per block would halve while K and V retain their longer blocks.
- A total of `100 + 10 + 10 = 120` fills the core count numerically but does
  not balance the work.
- The QKV join could continue to wait for K or V tail work.
- Extra Q blocks could consume dispatch opportunities needed by K and V.

The first target therefore keeps Q/K/V at 50/10/10. Test 100/10/10 only if a
target-machine swimlane proves that Q is independently critical and K/V are
fully hidden. Scaling all three proportionally to 100/20/20 creates 140 blocks,
which necessarily produces a second wave on 120 cores, so it is not the default.

### 2.3 Attention from 24 to 120

Change in `models/qwen3_14b/paged_attention_pypto.py`:

```text
ATTN_SPMD_BLOCKS: 24 -> 120
```

One default Batch window has at most:

```text
active_batch x NUM_KV_HEADS = 16 x 8 = 128 logical work items
```

With BlockDim 120, the first eight lanes execute two items and the other lanes
execute one. This explicitly splits Batch x KV Head. When the active Batch is
below 15, fewer than 120 logical items exist and some AICores remain idle. A
larger Batch should be selected from the serving workload, not solely to fill
the machine.

`ATTN_SPMD_BLOCKS` also sizes transfer and workspace storage, including
`TRANSFER_ROWS`; changing only the launch dimension is insufficient.

The current `a2a3` backend declares 24 Core Groups, while this Attention
implementation contains a hard `syncall`. The 120-block form can compile only
against a target backend that declares 120 Core Groups. A soft barrier is not a
safe emulation on 24 cores because the first 24 resident blocks could wait for
the other 96 blocks before those blocks can be scheduled.

### 2.4 Out Projection from 50 to 100

Increase the N split and keep Split-K unchanged:

```text
N_SPLITS_OUT: 10 -> 20
K_SPLITS_OUT: 5  -> 5
OUT_TN:       512 -> 256
OUT_TK:       1024 (unchanged)
Total:        10x5=50 -> 20x5=100
```

This selection:

- Keeps the atomic-add fan-in at 5 instead of increasing it to 10.
- Limits atomic contention and accumulation-order precision risk.
- Reduces the `[BATCH_PAD, OUT_TN]` accumulator from 512 to 256 columns.
- Does not split Batch; every block still processes the full `BATCH_PAD`.

The baseline organizes 50 logical blocks as 26 regular tasks plus one direct
BlockDim-24 SPMD task. That split is a dispatch-priority design rather than a
matrix-shape requirement. The target uses one direct `20 x 5`, 100-block SPMD
invocation to avoid a 76-regular-task tail followed by BlockDim 24. A
target-machine dependency graph and level-4 swimlane must confirm that this
larger dispatch does not suppress more critical early work.

### 2.5 Keep Gate/Up/Down at 85

Gate and Up each have `17 x 5 = 85` logical blocks, and Down also has 85.
They already expose substantial parallelism for 120 cores. Increasing their
block counts would add dispatch, weight-read, Split-K, and atomic overhead.
Retile them only if a target-machine swimlane shows long blocks alongside
sustained idle cores.

## 3. DeepSeek CSA plan

The DeepSeek changes follow the requested targets. The duplicate
`score 16 -> 80` request is applied once.

The default shape is `B=4`, `S=2`, and `T=B*S=8`.

| Task | Invocations | Baseline BlockDim | Target BlockDim | Parameter change | Batch split |
|---|---:|---:|---:|---|:---:|
| `qproj_matmul` | 1 | 64 | 128 | `QPROJ_MM_N_TILE: 512 -> 256` | No |
| `qr_proj_matmul` | 1 | 16 | 64 | `QR_OK: 2 -> 8` | No |
| `kv_proj_matmul` | 1 | 16 | 32 | `KV_OK: 4 -> 8` | No |
| `idx_qr_proj_matmul` | 1 | 8 | 32 | `Q_OUT_TILE: 1024 -> 256`, `MM_N_TILE: 512 -> 256` | No |
| `idx_qr_proj_dequant` | 1 | 8 | 32 | Follows `Q_OUT_TILE` | No |
| `qr_hadamard_matmul` | 1 | 8 | 32 | `QH_MM_TILE: 64 -> 16` | **Yes** |
| `score` | 1 | 16 | 80 | `REDUCE_NSPLIT: 2 -> 10` | **Yes** |
| `qk_pv` | 1 | 24 | 40 | `NUM_QK_CORES: 24 -> 40` | **Yes** |
| `proj_a_mm` | 8 | 8 | 16 | `PROJ_A_MM_N_TILE: 128 -> 64` | No |
| `proj_b_mm` | 8 | 8 | 16 | `PROJ_B_D_TILE: 512 -> 256` | No |
| `merge_norm` | 1 | 32 | 64 | Merge Head tile: `16 -> 8` | **Yes** |
| Main compressor `kv_score_proj` | 1 | 16 | 64 | `OUT_TILE: 64 -> 16` | Indirect |

The `8 x 16` notation for `proj_a_mm` and `proj_b_mm` means eight task
invocations with BlockDim 16 each. It is not equivalent to one BlockDim-128
invocation.

### 3.1 Batch dimension details

- `qr_hadamard_matmul`: block indexing is Token x HeadTile.
- `score`: block indexing is Token x cache reduction split.
- `qk_pv`: its 40 items are `T x SPARSE_BLOCKS = 8 x 5`.
- `merge_norm`: block indexing is `T x (H / MERGE_H_TILE)`.
- `kv_score_proj`: BlockDim includes a `BS_PAD / MM_B_TILE` row-group
  expression, but the default eight tokens occupy one 16-row physical tile.
  This change obtains 16 to 64 only by reducing the output N tile.
- The remaining matmuls process the complete default Token/Batch rows in each
  block. Their new parallelism comes from N or Split-K.

### 3.2 Merge Norm from 32 to 64

The shared Head tile cannot be changed directly from 16 to 8. Such a change
generates an invalid `[8, 512]` FP32 cube accumulator because physical cube
rows must be a multiple of 16. Eight FP32 scalars also occupy only 32 bytes, so
QK blocks writing adjacent regions could share a 64-byte cache line.

Producer and consumer tiles are therefore decoupled:

```text
QK_STORE_H_TILE = 16  # QK/PV physical storage and cache-line granularity
MERGE_H_TILE    = 8   # Heads consumed by each merge_norm block
```

QK/PV continues writing 16 Heads and fills 64-byte `mi/li` groups. Two
`merge_norm` blocks consume the lower and upper halves of each 16-Head group.
This produces BlockDim 64 without introducing an eight-row cube tile or
cross-block cache-line writes.

### 3.3 Main risks

- `qproj_matmul=128`: a 120-core machine has a `120 + 8` tail wave.
- `qr_proj_matmul=64`: Split-K grows from 2 to 8, increasing atomic fan-in.
- `kv_proj_matmul=32`: Split-K grows from 4 to 8 with the same contention risk.
- `idx_qr_proj_*=32`: `MM_N_TILE` and `Q_OUT_TILE` must both be 256 to
  avoid overlapping neighboring output regions.
- `qr_hadamard_matmul=32`: reducing M from 64 to 16 increases repeated
  Hadamard-weight reads.
- `score=80`: short sequences can create many small reduction tasks.
- `qk_pv=40`: the default shape has exactly 40 useful items, so a larger
  BlockDim cannot expose more work.
- `kv_score_proj=64`: the 16-column output tile can have weak transfer
  efficiency and higher launch overhead.

## 4. Validation requirements

Validate each model with the repository-default shapes:

1. Pass the PyTorch golden comparison.
2. Emit no tile-shape, L0C/L1/UB overflow, or cache-line-sharing diagnostic.
3. Confirm that `deps.json` introduces no unnecessary serial dependency.
4. Use a level-4 target-machine swimlane to prove that tasks run concurrently
   rather than merely creating more short tasks or a new tail wave.
5. Measure end-to-end critical-path latency on the 120-core machine.
6. Test the real serving Batch separately for every Batch-split task before
   deciding whether the Batch configuration should change.

## 5. Current validation status

- Qwen Q/K/V and MLP tiling remain unchanged.
- Qwen Out Projection at 100 blocks passes the default single-layer golden on
  `a2a3` device 1 when Attention is temporarily kept at BlockDim 24.
- Qwen Attention is set to BlockDim 120 in source, but the current 24-Core
  `a2a3` hard-`syncall` backend cannot compile the combined target.
- DeepSeek implements every target BlockDim, including decoupled QK storage and
  Merge Norm consumption, and passes the default golden on `a2a3` device 1.
- No 120-core target-machine performance result is available yet.

Local validation artifacts:

- Qwen 24-core-compatible swimlane:
  `build_output/_jit_decode_fwd_layers_jz59pt7_/dfx_outputs/merged_swimlane_20260914_054016.json`
- Qwen 24-core-compatible dependencies:
  `build_output/_jit_decode_fwd_layers_jz59pt7_/dfx_outputs/deps.json`
- DeepSeek final-tiling swimlane:
  `build_output/_jit_attention_csa_test_v1h8bwzo/dfx_outputs/merged_swimlane_20260914_052747.json`
- DeepSeek final-tiling dependencies:
  `build_output/_jit_attention_csa_test_v1h8bwzo/dfx_outputs/deps.json`

These files were captured on 24-Core `a2a3` device 1. They validate task
structure and numerics but do not represent 120-core scheduling or performance.
