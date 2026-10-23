# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Construct balanced EP64 routes and reuse original NZ-packed model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parents[2] / 'models/deepseek_v4_flash_dspark')]
import numpy as np
import torch
from utils import int8_quant_per_row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-input-dir', type=Path, required=True,
                        help='Saved distributed model data/in with NZ-packed weights.')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20261022)
    source = args.source_input_dir.resolve()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f'Do not replace a saved fixture: {out}')
    for prefix in ['shared', 'routed']:
        for suffix in ['w1', 'w1_scale', 'w3', 'w3_scale', 'w2', 'w2_scale']:
            path = source / f'{prefix}_{suffix}.pt'
            if not path.is_file():
                raise FileNotFoundError(path)
    for case in ['shared', 'routed']:
        (out / case).mkdir(parents=True)

    # Every token selects six distinct experts. Every global expert receives 720 rows.
    routes = (np.arange(64 * 720 * 6, dtype=np.int32) % 384).reshape(64, 720, 6)
    assert np.all(np.diff(np.sort(routes, axis=2), axis=2) > 0)
    global_counts = np.bincount(routes.ravel(), minlength=384)
    assert np.all(global_counts == 720)
    np.save(out / 'routes.npy', routes)
    source_counts = np.zeros((64, 6), dtype=np.int32)

    shared_float = torch.randn(768, 7168, dtype=torch.bfloat16)
    shared_float[720:] = 0
    shared_i8, shared_scale = int8_quant_per_row(shared_float)
    torch.save(shared_i8, out / 'shared/x_local_i8.pt')
    torch.save(shared_scale.float().reshape(768, 1), out / 'shared/x_local_scale_dq.pt')

    # This balanced pattern sends the same selected token set to experts 0..5.
    pairs = np.argwhere(np.any(routes == 0, axis=2))
    assert pairs.shape == (720, 2)
    payload = torch.randn(720, 7168, dtype=torch.bfloat16)
    for j, (rank, token) in enumerate(pairs):
        if rank == 0:
            payload[j] = shared_float[int(token)]
    q, scale = int8_quant_per_row(payload)
    recv = torch.zeros(6, 768, 7168, dtype=torch.int8)
    scales = torch.zeros(6, 768, dtype=torch.float32)
    weights = torch.zeros(6, 768, dtype=torch.float32)
    for e in range(6):
        epairs = np.argwhere(np.any(routes == e, axis=2))
        assert np.array_equal(epairs, pairs)
        source_counts[:, e] = np.bincount(epairs[:, 0], minlength=64)
        recv[e, :720] = q
        scales[e, :720] = scale.reshape(-1)
        weights[e, :720] = 2.5 / 6
    for name, tensor in [('recv_x', recv), ('recv_scale_dq', scales), ('recv_weights', weights),
                         ('recv_count', torch.full((6, 1), 720, dtype=torch.int32))]:
        torch.save(tensor, out / 'routed' / (name + '.pt'))

    provenance = []
    for prefix, case in [('shared', 'shared'), ('routed', 'routed')]:
        for suffix in ['w1', 'w1_scale', 'w3', 'w3_scale', 'w2', 'w2_scale']:
            name = prefix + '_' + suffix
            path = source / (name + '.pt')
            tensor = torch.load(path, weights_only=True, mmap=True)
            shard = tensor[0] if prefix == 'shared' else tensor[0, :6]
            target = out / case / path.name
            torch.save(shard.clone(), target)
            provenance.append({'name': name, 'source_file': path.name, 'source_shape': list(tensor.shape),
                               'selected': 'rank0' if prefix == 'shared' else 'rank0 experts0:6',
                               'shape': list(shard.shape), 'dtype': str(shard.dtype)})

    manifest = {'logical_ep': 64, 'global_experts': 384, 'local_experts': 6, 'target_rank': 0,
                'batch_per_source_rank': 180, 'mtp': 3, 'tokens_per_request': 4,
                'active_tokens_per_source_rank': 720, 'shared_padded_tokens': 768,
                'topk': 6, 'routing': 'synthetic exactly balanced cyclic; not measured gating',
                'source_expert_counts': source_counts.tolist(), 'local_counts': [720] * 6,
                'tiles_per_expert': [12] * 6, 'total_expert_tiles': 72,
                'compact_receive_capacity': 768, 'full_ep64_receive_capacity_not_allocated': 46080,
                'communication': 'CPU-prepared post-dispatch data; no inter-rank transfer',
                'weights': provenance,
                'sha256': {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in out.rglob('*') if p.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['local_counts', 'tiles_per_expert', 'total_expert_tiles']}, indent=2))


if __name__ == '__main__':
    main()
