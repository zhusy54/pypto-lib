# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""WSE-FFN SwiGLU feed-forward network — explicit InCore + Orchestration.

Implements the SwiGLU FFN used in wse-ffn:

    ffn = SiLU(x @ W_gate) * (x @ W_up)
    out = ffn @ W_down

Input x is the post-attention RMSNorm output ([BATCH, HIDDEN] BF16).
All matmuls use BF16 inputs with FP32 accumulation.
Output is FP32 (no final BF16 cast).

Tiling: K_CHUNK=128 over hidden dim, FFN_OUT_CHUNK=256 over intermediate dim.

Three InCore kernels:
  kernel_gate_silu — gate projection + SiLU(gate) → FP32 gate_silu
  kernel_up_mul    — up projection + gate_silu * up → BF16 written
                     directly into ffn_batch at [0, o0]
  kernel_down      — down projection → FP32 store to output at [b0, d0]

Hardware notes:
  - The AICore enforces 'tpop -> use -> tfree -> next tpop'.  A single InCore
    function cannot hold results from two independent matmul tpop chains at the
    same time, which is why Step-1 must remain two separate InCore functions.
  - pl.move(L0C → Vec) generates incorrect ptoas code on this platform; the
    correct pattern is pl.store(acc, GM_buf) then pl.load(GM_buf, ..., Vec).
  - kernel_up_mul writes BF16 directly into ffn_batch at [0, o0] (InOut)
    rather than using a separate ffn_buf + pl.assemble, to avoid a FP32/BF16
    type-aliasing issue where the compiler incorrectly maps the FP32 up_buf
    and BF16 ffn_buf onto the same ffn_batch memory region.
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
HIDDEN = 7168
INTERMEDIATE = 3072  # example/test shape; wse-ffn production shape may vary

# Tiling constants.
K_CHUNK = 128
FFN_OUT_CHUNK = 256
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
        # Step 1a: gate projection + SiLU(gate)  →  FP32 gate_silu (in output)
        # ------------------------------------------------------------------
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_gate_silu(
            self,
            x: pl.Tensor[[BATCH_TILE, HIDDEN_SIZE], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            o0: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[BATCH_TILE, FFN_OUT_CHUNK], pl.FP32]],
        ) -> pl.Tensor[[BATCH_TILE, FFN_OUT_CHUNK], pl.FP32]:
            x0 = pl.load(x, [0, 0], [BATCH_TILE, K_CHUNK], target_memory=pl.MemorySpace.Mat)
            wg0 = pl.load(w_gate, [0, o0], [K_CHUNK, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
            gate_acc = pl.matmul(x0, wg0)
            for kb in pl.range(1, HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xi = pl.load(x, [0, k0], [BATCH_TILE, K_CHUNK], target_memory=pl.MemorySpace.Mat)
                wg = pl.load(w_gate, [k0, o0], [K_CHUNK, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
                gate_acc = pl.matmul_acc(gate_acc, xi, wg)
            # Store gate_acc to GM (use+free gate tpop chain), reload into Vec for SiLU.
            gate_stored = pl.store(gate_acc, [0, 0], output)
            gate_vec = pl.load(gate_stored, [0, 0], [BATCH_TILE, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Vec)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_vec)), 1.0))
            gate_silu = pl.mul(gate_vec, sigmoid)
            # Overwrite output with gate_silu (same FP32 buffer).
            out = pl.store(gate_silu, [0, 0], gate_stored)
            return out

        # ------------------------------------------------------------------
        # Step 1b: up projection + gate_silu * up  →  BF16 written at [0,o0]
        # gate_silu is a plain input parameter (not tainted by any tpop here).
        # Writes the BF16 result directly into ffn_batch at offset [0, o0] to
        # avoid a FP32/BF16 type conflict: if a separate ffn_buf (BF16) were
        # used and then assembled into ffn_batch, the compiler would attempt to
        # alias the FP32 up_buf and BF16 ffn_buf onto the same ffn_batch memory
        # region, causing incorrect strides and data corruption.
        # ------------------------------------------------------------------
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_up_mul(
            self,
            x: pl.Tensor[[BATCH_TILE, HIDDEN_SIZE], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            gate_silu: pl.Tensor[[BATCH_TILE, FFN_OUT_CHUNK], pl.FP32],
            o0: pl.Scalar[pl.INDEX],
            up_buf: pl.Out[pl.Tensor[[BATCH_TILE, FFN_OUT_CHUNK], pl.FP32]],
            ffn_batch: pl.InOut[pl.Tensor[[BATCH_TILE, INTER_SIZE], pl.BF16]],
        ) -> pl.Tensor[[BATCH_TILE, INTER_SIZE], pl.BF16]:
            x0 = pl.load(x, [0, 0], [BATCH_TILE, K_CHUNK], target_memory=pl.MemorySpace.Mat)
            wu0 = pl.load(w_up, [0, o0], [K_CHUNK, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
            up_acc = pl.matmul(x0, wu0)
            for kb in pl.range(1, HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xi = pl.load(x, [0, k0], [BATCH_TILE, K_CHUNK], target_memory=pl.MemorySpace.Mat)
                wu = pl.load(w_up, [k0, o0], [K_CHUNK, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
                up_acc = pl.matmul_acc(up_acc, xi, wu)
            # Store up_acc to GM (use+free up tpop chain), reload both into Vec for mul.
            up_stored = pl.store(up_acc, [0, 0], up_buf)
            gate_silu_vec = pl.load(gate_silu, [0, 0], [BATCH_TILE, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Vec)
            up_vec = pl.load(up_stored, [0, 0], [BATCH_TILE, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Vec)
            # Multiply, cast to BF16, store directly into ffn_batch at [0, o0].
            ffn_chunk = pl.mul(gate_silu_vec, up_vec)
            ffn_bf16 = pl.cast(ffn_chunk, target_type=pl.BF16)
            out = pl.store(ffn_bf16, [0, o0], ffn_batch)
            return out

        # ------------------------------------------------------------------
        # Step 2: down projection + FP32 store to output at [b0, d0]
        # ------------------------------------------------------------------
        @pl.function(type=pl.FunctionType.InCore)
        def kernel_down(
            self,
            ffn_tile: pl.Tensor[[BATCH_TILE, INTER_SIZE], pl.BF16],
            w_down: pl.Tensor[[INTER_SIZE, HIDDEN_SIZE], pl.BF16],
            b0: pl.Scalar[pl.INDEX],
            d0: pl.Scalar[pl.INDEX],
            down_buf: pl.Out[pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32]],
            out: pl.Out[pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.FP32]],
        ) -> pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.FP32]:
            # Down projection: ffn_tile @ W_down[:,d0:d0+K_CHUNK]
            ffn0 = pl.load(ffn_tile, [0, 0], [BATCH_TILE, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
            wd0 = pl.load(w_down, [0, d0], [FFN_OUT_CHUNK, K_CHUNK], target_memory=pl.MemorySpace.Mat)
            acc = pl.matmul(ffn0, wd0)
            for ob in pl.range(1, FFN_OUT_BLOCKS):
                o0 = ob * FFN_OUT_CHUNK
                ffn_i = pl.load(ffn_tile, [0, o0], [BATCH_TILE, FFN_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
                wd_i = pl.load(w_down, [o0, d0], [FFN_OUT_CHUNK, K_CHUNK], target_memory=pl.MemorySpace.Mat)
                acc = pl.matmul_acc(acc, ffn_i, wd_i)
            # Store acc to GM (use+free down tpop chain), reload into Vec, store FP32.
            down_stored = pl.store(acc, [0, 0], down_buf)
            down_vec = pl.load(down_stored, [0, 0], [BATCH_TILE, K_CHUNK], target_memory=pl.MemorySpace.Vec)
            result = pl.store(down_vec, [b0, d0], out)
            return result

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

                # Step 1: gate_silu then up_mul, chunked over FFN_OUT_CHUNK.
                # kernel_up_mul writes BF16 result directly into ffn_batch at [0, o0],
                # avoiding a separate ffn_buf alloc and pl.assemble.
                for ob in pl.range(FFN_OUT_BLOCKS):
                    o0 = ob * FFN_OUT_CHUNK
                    gate_silu_buf = pl.create_tensor([BATCH_TILE, FFN_OUT_CHUNK], dtype=pl.FP32)
                    gate_silu = self.kernel_gate_silu(x_batch, w_gate, o0, gate_silu_buf)
                    up_buf = pl.create_tensor([BATCH_TILE, FFN_OUT_CHUNK], dtype=pl.FP32)
                    ffn_batch = self.kernel_up_mul(x_batch, w_up, gate_silu, o0, up_buf, ffn_batch)

                # Step 2: down projection + FP32 store, chunked over K_CHUNK / hidden dim.
                for dob in pl.range(HIDDEN_BLOCKS):
                    d0 = dob * K_CHUNK
                    down_buf = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.FP32)
                    out = self.kernel_down(ffn_batch, w_down, b0, d0, down_buf, out)

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
