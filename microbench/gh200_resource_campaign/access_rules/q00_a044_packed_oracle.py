#!/usr/bin/env python3
"""a044 后六资源家族的 CPU 数值检查；调用原冻结 oracle。"""
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_short_values(directory, device, row, case, profile, seed, validate_metadata):
    """按原 S15 独立 stream_values.py 的逐 CTA 公式检查完整数组。"""
    import numpy as np
    validate_metadata(device, row, case, profile, seed)
    parameters = case['parameters']
    width, height = parameters['box_dim']
    q, pitch = parameters['payload_bytes'], parameters['row_stride_bytes']
    n, blocks = profile['target_iterations'][0], row['blocks']
    g2s, sw = parameters['direction'] == 'gmem_to_smem', parameters['swizzle'] == 'SW128'
    yy = np.arange(height, dtype=np.uint32)[:, None]
    xx = np.arange(width, dtype=np.uint32)[None, :]
    ii = np.arange(n, dtype=np.uint32)[:, None, None]
    base = (17*xx + 31*yy + (seed & 65535) +
            (73*(ii % 32) if g2s else np.zeros((n, 1, 1), dtype=np.uint32))) & 65535
    physical_index = yy*width + (((xx//8) ^ (yy % 8))*8 + xx % 8) if sw else yy*width + xx
    physical = np.empty(base.shape, dtype=np.uint16)
    physical.reshape(n, -1)[:, physical_index.astype(np.intp).ravel()] = base.reshape(n, -1)
    rr = np.arange(32*height, dtype=np.uint32)[:, None]
    cc = np.arange(pitch//2, dtype=np.uint32)[None, :]
    slots, gy = (rr//height) % 32, rr % height
    global_base = (17*cc + 31*gy + (seed & 65535) + (73*slots if g2s else 0)) & 65535
    checked = 0
    for item in row['checks'][0]['output_artifacts']:
        name, path = item['path'], Path(directory) / item['path']
        dtype = {'uint8': '<u1', 'uint16': '<u2', 'uint32': '<u4'}[item['dtype']]
        assert path.stat().st_size == math.prod(item['shape']) * np.dtype(dtype).itemsize
        assert file_sha(path) == item['sha256']
        actual = np.memmap(path, dtype=dtype, mode='r')
        if name in ('tensor_logical.u16le', 'tensor_physical.u16le'):
            assert len(actual) == blocks*n*q//2
            shaped = actual.reshape(blocks, n, height, width)
            expected = base if name == 'tensor_logical.u16le' else physical
            for b in range(blocks):
                assert np.array_equal(shaped[b], ((expected.astype(np.uint32)+151*b) & 65535).astype(np.uint16)), (name, b)
            checked += len(actual)
        elif name == 'tensor_global.u16le':
            assert len(actual) == blocks*32*height*(pitch//2)+128
            assert np.array_equal(actual[:64], np.arange(64)+0xa900)
            assert np.array_equal(actual[-64:], np.arange(64)+0xa940)
            shaped = actual[64:-64].reshape(blocks, 32*height, pitch//2)
            for b in range(blocks):
                expected = (global_base+151*b) & 65535
                if not g2s and n < 32:
                    expected = np.where(slots >= n, expected ^ 65535, expected)
                expected = np.where(cc >= width, (0x7d00+(rr+b*32*height)*11+cc) & 65535, expected).astype(np.uint16)
                assert np.array_equal(shaped[b], expected), (name, b)
            checked += len(actual)
        elif name in ('tensor_completion.u32le', 'tensor_stamps.u32le'):
            values = actual.view('<u8').reshape(blocks, 5)
            if name == 'tensor_completion.u32le':
                assert np.all(values[:, 0] == n) and np.all(values[:, 2:4] == 0)
                assert np.all(values[:, 4] == (0 if g2s else n))
                assert (np.all(values[:, 1] >= n) and np.all(values[:, 1] < 2**64-1)) if g2s else np.all(values[:, 1] == 0)
            else:
                assert np.all(values[:, 0] <= values[:, 1]) and np.all(values[:, 1] < 2**64-1)
                assert np.all(values[:, 2] <= values[:, 3]) and np.all(values[:, 3] < 2**64-1)
                assert np.all(values[:, 4] < 2**32-1)
        elif name == 'tensor_descriptor.u8':
            assert file_sha(path) == row['resource_identity']['extensions']['descriptor_sha256']
        else:
            raise ValueError(name)
    assert checked == row['checks'][0]['checked_elements'] == row['checks'][0]['expected_elements']
    return dict(complete_values_verified=True, checked_uint16_elements=checked)


class ResourceOracle:
    def __init__(self, family, source_root):
        self.family, self.root = family, Path(source_root)
        sys.path.insert(0, str(self.root))
        self.code = self.root / 'microbench/gh200_resource_campaign'
        sys.path.insert(0, str(self.code))
        from common.suite_io import verify_files
        manifest = self.root / ('formal-source-manifest.json' if family == 'tma_tensor_2d' else 'source-manifest.json')
        verify_files(self.root, read(manifest))
        self.source_manifest_sha256 = file_sha(manifest)
        self.protocol = read(self.code / 'contracts/protocol.json')
        if family == 'tma_tensor_2d':
            cases = read(self.code / 'contracts/tma_tensor_2d.json')['cases']
            self.profiles = {p['id']: p for p in read(self.code / 'contracts/tma_tensor_2d_validation_profiles_v1.draft.json')['profiles']}
            self.metadata = importlib.import_module('run_candidate').check_metadata
            self.values = importlib.import_module('reference_rows').verify_values
            self.short_metadata = importlib.import_module('auditors.tma_tensor_2d_validation_v1').validate_validation
            self.observation = importlib.import_module('auditors.observation').validate_observation
        elif family == 'low_precision':
            cases = read(self.root / 'cases.json')['cases']
            self.bind_values = importlib.import_module('launch_binding').bind_values
            self.capture = importlib.import_module('read_capture').check
            self.parent = importlib.import_module('auditors.low_precision').validate_trial
            self.bounded = importlib.import_module('auditors.low_precision_bounded_fp8').validate_trial
        else:
            cases = read(self.root / 'cases.json')['cases']
            self.control = importlib.import_module('control')
        self.cases = {c['id']: c for c in cases}
        self.resources = {}

    def check(self, directory, task, rows):
        device = rows[0]
        kind = task['protocol']['original_kind']
        point = task['origins'][0]['point']
        argv = task['argv']
        family = self.family
        if kind == 'resources':
            if family == 'tma_concurrency':
                row = rows[1]
                legal = self.control.check_resources(row, self.cases[row['case_id']], device)
                self.resources[row['case_id']] = row
                return dict(resource_query=True, legal=legal)
            if family == 'dsm_multicast':
                measured = argv[1] == 'measured-cluster-device'
                self.resources['measured' if measured else 'legacy'] = self.control.capability_matrix(rows[1:], device, measured)
            elif family == 'fp32_combination':
                self.resources['measured'] = self.control.check_capability(rows[1], device)
            elif family == 'bf16_wgmma_combination':
                measured = argv[1] == 'measured-capability'
                self.resources['measured' if measured else 'legacy'] = self.control.check_capability(rows[1], device, measured)
            else:
                raise ValueError(family)
            return dict(resource_query=True)
        row = rows[1]
        assert len(rows) == 2
        case = self.cases[point['case_id']]
        seed = int(argv[-1])
        pilot = kind == 'pilot'
        if family == 'tma_tensor_2d':
            if kind == 'short':
                return tensor_short_values(directory, device, row, case, self.profiles[argv[3]], seed, self.short_metadata)
            iterations = int(argv[3])
            self.metadata(directory, device, row, case, dict(mode=argv[1], case_id=case['id'], iterations=iterations, seed=seed))
            self.values(directory, row, case)
            if pilot:
                return dict(pilot=True, complete_values_verified=True, event_ms=row['event_ms'])
            w, h = case['parameters']['box_dim']
            amount = row['blocks'] * iterations * w * h * 2
            g2s = case['parameters']['direction'] == 'gmem_to_smem'
            checked = row['blocks']*32*h*(case['parameters']['row_stride_bytes']//2)+128+(2*row['blocks']*w*h if g2s else 0)
            expected = dict(blocks=row['blocks'], work=amount, read=amount if g2s else 0,
                write=0 if g2s else amount, operations=row['blocks']*iterations,
                correctness=dict(method='S15_final_tile_or_complete32slot_ring_and_all_CTA_completion_v1',
                    checked_elements=checked, input_conditions='nonuniform_uint16_tensor_pattern;32slot_ring;capture_false;all final values exact'))
            return self.observation(row, dict(case, iterations=iterations), device, seed, self.protocol, expected)
        if family == 'low_precision':
            self.bind_values(directory, case, row, seed, kind == 'short', device)
            metadata = [read(path) for path in sorted(Path(directory).glob('*.metadata.json'))]
            assert metadata and all(m['case_id'] == case['id'] and m['seed'] == seed for m in metadata)
            checks = [self.capture(directory, item) for item in metadata]
            if kind == 'short':
                assert row['type'] == 'validation' and row['errors'] == 0 and len(checks) == 4
                return dict(complete_values_verified=True, captures=len(checks))
            return (self.bounded if case['id'].startswith('bounded_v1_') else self.parent)(row, case, device, seed, self.protocol)
        if family == 'tma_concurrency':
            if kind == 'short':
                return self.control.check_short(directory, row, case, device)
            resource = self.resources[case['id']]
        elif family == 'dsm_multicast':
            measured = kind != 'short'
            symbol = self.control.symbol(case['parameters']['form'], case['parameters']['cluster_size'], measured)
            resource = self.resources['measured' if measured else 'legacy'][symbol]
            if kind == 'short':
                return self.control.check_short(directory, row, case, device, resource)
        else:
            resource = self.resources['legacy' if kind == 'short' and family == 'bf16_wgmma_combination' else 'measured']
            if kind == 'short':
                return self.control.check_short(directory, row, case, device, resource, argv[3])
        return self.control.check_measured(directory, row, case, device, resource, seed, int(argv[3]), pilot, self.protocol)
