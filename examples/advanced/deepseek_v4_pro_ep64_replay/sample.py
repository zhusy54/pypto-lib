# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sample one logical EP64 shard on one physical NPU; no communication emulation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
MODEL = REPO / 'models/deepseek_v4_flash_dspark'
sys.path[:0] = [str(MODEL), str(REPO)]
import config

config.configure_pro_decode_layer(1, batch_per_rank=180)
config.EP = 64
# Compact, post-dispatch replay storage. This is not the physical EP64 receive layout.
config.RECV_MAX = 768

import pypto.language as pl
from expert_routed import expert_routed_tile
from expert_shared import expert_shared_test

E = 6
R = 768
D = 7168
I = 3072
TILE = 64


@pl.jit(auto_scope=False)
def ep64_routed_compute(
    recv_x: pl.Tensor[[E, R, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[E, R], pl.FP32],
    recv_weights: pl.Tensor[[E, R], pl.FP32],
    recv_count: pl.Tensor[[E, 1], pl.INT32],
    routed_w1: pl.Tensor[[E, I, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[E, I], pl.FP32],
    routed_w3: pl.Tensor[[E, I, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[E, I], pl.FP32],
    routed_w2: pl.Tensor[[E, D, I], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[E, D], pl.FP32],
    recv_y: pl.InOut[pl.Tensor[[E, R, D], pl.BF16]],
):
    y_flat = pl.reshape(recv_y, [E * R, D])
    inputs_ready = pl.system.task_dummy(deps=[])
    for e in pl.parallel(E):
        rows = pl.cast(pl.read(recv_count, [e, 0]), pl.INDEX)
        tiles = (rows + TILE - 1) // TILE
        for tile in pl.parallel(tiles):
            row = tile * TILE
            valid_rows = pl.min(TILE, rows - row)
            flat_row = e * R + row
            y_tile = pl.create_tensor([TILE, D], dtype=pl.BF16)
            ready = expert_routed_tile(
                recv_x, recv_scale_dq, recv_weights,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale, y_tile,
                e, row, valid_rows, inputs_ready,
            )
            # Test-only local materialization, excluded from operator budgets.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint='sample_routed_store', deps=[ready]):
                for col in pl.range(0, D, 256):
                    y_flat[flat_row:flat_row + TILE, col:col + 256] = y_tile[:, col:col + 256]
    return recv_y


def make_specs(case, inputs_dir):
    import torch
    from golden import TensorSpec
    if case == 'routed':
        shapes = [
            ('recv_x', [E, R, D], torch.int8),
            ('recv_scale_dq', [E, R], torch.float32),
            ('recv_weights', [E, R], torch.float32),
            ('recv_count', [E, 1], torch.int32),
        ]
        for key, shape in [('w1', [E, I, D]), ('w3', [E, I, D]), ('w2', [E, D, I])]:
            shapes += [('routed_' + key, shape, torch.int8), ('routed_' + key + '_scale', shape[:-1], torch.float32)]
        shapes += [('recv_y', [E, R, D], torch.bfloat16)]
    else:
        shapes = [('x_local_i8', [R, D], torch.int8), ('x_local_scale_dq', [R, 1], torch.float32)]
        for key, shape in [('w1', [I, D]), ('w3', [I, D]), ('w2', [D, I])]:
            shapes += [('shared_' + key, shape, torch.int8), ('shared_' + key + '_scale', shape[:-1], torch.float32)]
        shapes += [('sh', [R, D], torch.bfloat16)]
    specs = []
    for name, shape, dtype in shapes:
        path = inputs_dir / case / (name + '.pt')
        init = (lambda p=path: torch.load(p, mmap=True, weights_only=True)) if name not in ('recv_y', 'sh') else None
        specs.append(TensorSpec(name, shape, dtype, init_value=init))
    return specs


def source_hashes():
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(MODEL.glob('*.py'))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['shared', 'routed'], required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--inputs-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.inputs_dir = args.inputs_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    import torch
    from golden import run
    torch.set_num_threads(8)
    os.environ['PYPTO_BENCH'] = '0'
    before_hashes = source_hashes()
    build = args.out_dir / 'builds' / args.case
    out = args.out_dir / 'results' / args.case
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / ('compile.json' if args.compile_only else 'status.json')
    if status_path.exists():
        raise FileExistsError(f'Preserve previous execution evidence: {status_path}')
    state = {'case': args.case, 'logical_ep': 64, 'physical_devices': 1,
             'device': args.device, 'validation': 'NOT_PERFORMED',
             'routing': 'synthetic balanced', 'logical_dispatches': 1,
             'swimlane_level': 4, 'start_time': time.time(), 'source_sha256': before_hashes}
    status_path.write_text(json.dumps(state, indent=2))
    try:
        kwargs = dict(fn=expert_shared_test if args.case == 'shared' else ep64_routed_compute,
                      specs=make_specs(args.case, args.inputs_dir), golden_fn=None,
                      config=dict(platform='a2a3', device_id=args.device, enable_chip_swimlane=4,
                                  save_kernels_dir=str(build)))
        if args.compile_only:
            kwargs['compile_only'] = True
        else:
            kwargs['runtime_dir'] = str(build)
        result = run(**kwargs)
        state.update(passed=result.passed, error=result.error, work_dir=str(result.work_dir),
                     execution_time=result.execution_time)
        if not result.passed:
            raise RuntimeError(result.error)
    except BaseException as exc:
        state['exception'] = repr(exc)
        raise
    finally:
        state.update(end_time=time.time(), source_unchanged=source_hashes() == before_hashes)
        status_path.write_text(json.dumps(state, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
