# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import pypto.language as pl

BATCH = 16
HIDDEN = 7168
INTERMEDIATE = 3072  # example/test shape; wse-ffn production shape may vary

# Tiling constants.
K_CHUNK = 128
FFN_OUT_CHUNK = 128
BATCH_TILE = 16


def build_wse_ffn_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    BATCH_SIZE = batch
    HIDDEN_SIZE = hidden_size
    INTER_SIZE = intermediate_size

    HIDDEN_BLOCKS = (HIDDEN_SIZE + K_CHUNK - 1) // K_CHUNK
    FFN_OUT_BLOCKS = (INTER_SIZE + FFN_OUT_CHUNK - 1) // FFN_OUT_CHUNK

    @pl.program
    class WseFFN:
        # ------------------------------------------------------------------
        # Single merged InCore: gate + up + SiLU + mul + down projection.
        # Each matmul chain ends with pl.store(acc, GM_buf) to trigger
        # tpop+tfree, releasing L0C for the next chain.
        # ------------------------------------------------------------------
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_ffn(
            self,
            x: pl.Tensor[[BATCH_TILE, HIDDEN_SIZE], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_down: pl.Tensor[[INTER_SIZE, HIDDEN_SIZE], pl.BF16],
            b0: pl.Scalar[pl.INDEX],
            gate_buf: pl.Out[pl.Tensor[[BATCH_TILE, FFN_OUT_CHUNK], pl.FP32]],
            up_buf: pl.Out[pl.Tensor[[BATCH_TILE, FFN_OUT_CHUNK], pl.FP32]],
            down_buf: pl.Out[pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32]],
            ffn_batch: pl.InOut[pl.Tensor[[BATCH_TILE, INTER_SIZE], pl.BF16]],
            out: pl.Out[pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.FP32]],
        ) -> pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.FP32]:
            # ── Phase 1: gate + up + SiLU + mul → BF16 ffn_batch ──────────
            x_full = pl.load(x, [0, 0], [BATCH_TILE, HIDDEN_SIZE], target_memory=pl.MemorySpace.Mat)
            for ob in pl.range(FFN_OUT_BLOCKS):
                o0 = ob * FFN_OUT_CHUNK

                # Chain: gate matmul → L0C → store GM → load Vec
                x0 = pl.slice(x_full, [BATCH_TILE, K_CHUNK], [0, 0])
                wg0 = pl.load(w_gate, [0, o0], [K_CHUNK, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
                gate_acc = pl.matmul(x0, wg0)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    xi = pl.slice(x_full, [BATCH_TILE, K_CHUNK], [0, k0])
                    wg = pl.load(w_gate, [k0, o0], [K_CHUNK, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
                    gate_acc = pl.matmul_acc(gate_acc, xi, wg)
                gate_vec = pl.move(gate_acc, target_memory=pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)

                # Chain: up matmul → L0C → store GM → load Vec
                x1 = pl.slice(x_full, [BATCH_TILE, K_CHUNK], [0, 0])
                wu0 = pl.load(w_up, [0, o0], [K_CHUNK, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
                up_acc = pl.matmul(x1, wu0)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    xj = pl.slice(x_full, [BATCH_TILE, K_CHUNK], [0, k0])
                    wu = pl.load(w_up, [k0, o0], [K_CHUNK, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
                    up_acc = pl.matmul_acc(up_acc, xj, wu)

                # Vec compute: SiLU(gate) * up → BF16 → ffn_batch[0, o0]
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_vec)), 1.0))
                gate_silu = pl.mul(gate_vec, sigmoid)
                up_vec = pl.move(up_acc, target_memory=pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                ffn_chunk = pl.mul(gate_silu, up_vec)
                ffn_bf16 = pl.cast(ffn_chunk, target_type=pl.BF16)
                ffn_batch = pl.store(ffn_bf16, [0, o0], ffn_batch)

            # ── Phase 2: down projection → FP32 output ────────────────────
            # Load full ffn_batch once (16×3072 BF16 = 96KB < L1 512KB limit); ob loops only slice.
            ffn_full = pl.load(ffn_batch, [0, 0], [BATCH_TILE, INTER_SIZE], target_memory=pl.MemorySpace.Mat)
            for dob in pl.range(HIDDEN_BLOCKS):
                d0 = dob * K_CHUNK

                # Chain: down matmul → L0C → store GM → load Vec → store FP32
                ffn0 = pl.slice(ffn_full, [BATCH_TILE, FFN_OUT_CHUNK], [0, 0])
                wd0 = pl.load(w_down, [0, d0], [FFN_OUT_CHUNK, K_CHUNK], target_memory=pl.MemorySpace.Mat)
                down_acc = pl.matmul(ffn0, wd0)
                for ob in pl.range(1, FFN_OUT_BLOCKS):
                    o0 = ob * FFN_OUT_CHUNK
                    ffn_i = pl.slice(ffn_full, [BATCH_TILE, FFN_OUT_CHUNK], [0, o0])
                    wd_i = pl.load(w_down, [o0, d0], [FFN_OUT_CHUNK, K_CHUNK], target_memory=pl.MemorySpace.Mat)
                    down_acc = pl.matmul_acc(down_acc, ffn_i, wd_i)
                out = pl.store(down_acc, [b0, d0], out)

            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def wse_ffn(
            self,
            post_norm: pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_down: pl.Tensor[[INTER_SIZE, HIDDEN_SIZE], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.FP32]],
        ) -> pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.FP32]:
            for b0 in pl.range(0, BATCH_SIZE, BATCH_TILE):
                x_batch = pl.slice(post_norm, [BATCH_TILE, HIDDEN_SIZE], [b0, 0])
                ffn_batch = pl.create_tensor([BATCH_TILE, INTER_SIZE], dtype=pl.BF16)
                gate_buf = pl.create_tensor([BATCH_TILE, FFN_OUT_CHUNK], dtype=pl.FP32)
                up_buf = pl.create_tensor([BATCH_TILE, FFN_OUT_CHUNK], dtype=pl.FP32)
                down_buf = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.FP32)
                out = self.kernel_ffn(
                    x_batch, w_gate, w_up, w_down, b0,
                    gate_buf, up_buf, down_buf, ffn_batch, out,
                )

            return out

    return WseFFN


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    return [
        TensorSpec(
            "post_norm", [batch, hidden_size], torch.bfloat16,
            init_value=lambda: torch.rand(batch, hidden_size) - 0.5,
        ),
        TensorSpec(
            "w_gate", [hidden_size, intermediate_size], torch.bfloat16,
            init_value=lambda: (torch.rand(hidden_size, intermediate_size) - 0.5) / (hidden_size ** 0.5),
        ),
        TensorSpec(
            "w_up", [hidden_size, intermediate_size], torch.bfloat16,
            init_value=lambda: (torch.rand(hidden_size, intermediate_size) - 0.5) / (hidden_size ** 0.5),
        ),
        TensorSpec(
            "w_down", [intermediate_size, hidden_size], torch.bfloat16,
            init_value=lambda: (torch.rand(intermediate_size, hidden_size) - 0.5) / (intermediate_size ** 0.5),
        ),
        TensorSpec("out", [batch, hidden_size], torch.float32, is_output=True),
    ]


def golden_wse_ffn(tensors):
    """PyTorch reference for wse-ffn SwiGLU feed-forward network.

    Implements SwiGLU gate + up projections followed by down projection.
    Chunked accumulation order matches the hardware kernel to minimize BF16/FP32 drift.
    """
    import torch

    post_norm = tensors["post_norm"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    batch = post_norm.shape[0]
    hidden_size = post_norm.shape[1]
    inter_size = w_gate.shape[1]

    k_chunk = K_CHUNK
    ffn_out_chunk = FFN_OUT_CHUNK

    # SwiGLU: gate + up projections (chunked BF16 matmul, FP32 accumulation).
    ffn_bf16 = torch.zeros(batch, inter_size, dtype=torch.bfloat16)
    for o0 in range(0, inter_size, ffn_out_chunk):
        gate_acc = torch.zeros(batch, ffn_out_chunk, dtype=torch.float32)
        up_acc = torch.zeros(batch, ffn_out_chunk, dtype=torch.float32)
        for k0 in range(0, hidden_size, k_chunk):
            post_chunk = post_norm[:, k0:k0 + k_chunk].float()
            gate_acc += post_chunk @ w_gate[k0:k0 + k_chunk, o0:o0 + ffn_out_chunk].float()
            up_acc += post_chunk @ w_up[k0:k0 + k_chunk, o0:o0 + ffn_out_chunk].float()
        sigmoid = torch.reciprocal(torch.exp(-gate_acc) + 1.0)
        ffn_bf16[:, o0:o0 + ffn_out_chunk] = (gate_acc * sigmoid * up_acc).bfloat16()

    # Down projection (chunked BF16 matmul, FP32 accumulation). Output stays FP32.
    out = torch.zeros(batch, hidden_size, dtype=torch.float32)
    for d0 in range(0, hidden_size, k_chunk):
        down_acc = torch.zeros(batch, k_chunk, dtype=torch.float32)
        for o0 in range(0, inter_size, ffn_out_chunk):
            down_acc += ffn_bf16[:, o0:o0 + ffn_out_chunk].float() @ w_down[o0:o0 + ffn_out_chunk, d0:d0 + k_chunk].float()
        out[:, d0:d0 + k_chunk] = down_acc

    tensors["out"][:] = out


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_wse_ffn_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_wse_ffn,
        config=RunConfig(
            rtol=3e-3,
            atol=5e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
