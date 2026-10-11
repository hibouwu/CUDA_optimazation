#!/usr/bin/env python3
"""Generate the fixed R18 first-fill/input-map pairs; retain JSONs as snapshots."""
import argparse
import json
from pathlib import Path


def cases(family, axis):
    if family not in ('first-fill', 'input-map') or axis not in ('m', 'n'):
        raise ValueError('choose first-fill/input-map and axis m/n')
    if family == 'first-fill':
        config, m, n, storage_m, storage_n = (
            ('cfg_a', 832, 4096, 1024, 4096) if axis == 'm'
            else ('cfg_c', 1536, 2624, 1536, 2816))
        label, case_set, swizzle = 'first_fill', 'calib', 1
    else:
        config = 'cfg_b'
        m, n, storage_m, storage_n = (
            (2304, 4096, 3072, 4096) if axis == 'm'
            else (4096, 2304, 4096, 3072))
        label, case_set, swizzle = 'pad'+axis.upper(), 'ctrl', 8
    rows = []
    for k in (1024, 4096):
        for input_path in ('oob', 'address_zero'):
            row = {
                'id': f'{config}_{label}_{input_path}_k{k}',
                'config': config, 'set': case_set, 'kind': 'ordinary',
                'm': m, 'n': n, 'k': k, 'lda': k,
                'ldb': storage_n, 'ldd': storage_n,
                'storage_m': storage_m, 'storage_n': storage_n,
                'zero_m': m if axis == 'm' else -1,
                'zero_n': n if axis == 'n' else -1,
                'swizzle': swizzle, 'group': f'{label}_k{k}',
                'original_experiment': 'R18', 'input_path': input_path,
                'input_mode': 'dyadic', 'seed': 17, 'sm_count': 0,
                'alloc_lda': k, 'alloc_ldb': storage_n,
                'input_map_m': m if input_path == 'oob' else storage_m,
                'input_map_n': n if input_path == 'oob' else storage_n,
            }
            if family == 'first-fill':
                row['evict'] = 0
            rows.append(row)
    return rows


def check_snapshots():
    configs = Path(__file__).resolve().parent/'configs'
    snapshots = [
        ('first-fill', 'm', 'r18-first-fill-calibration.json'),
        ('first-fill', 'n', 'r18-first-fill-calibration-c.json'),
        ('input-map', 'm', 'r18-input-map-padding.json'),
        ('input-map', 'n', 'r18-input-map-padding-n.json'),
    ]
    counts = {}
    for family, axis, name in snapshots:
        generated = cases(family, axis)
        expected = json.loads((configs/name).read_text())
        if generated != expected:
            raise ValueError('generated conditions differ from snapshot: '+name)
        counts[name] = len(generated)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('first-fill', 'input-map'))
    parser.add_argument('--axis', choices=('m', 'n'))
    parser.add_argument('--check', action='store_true',
                        help='check all four snapshots, including condition order and every field')
    args = parser.parse_args()
    if args.check:
        result = check_snapshots()
    else:
        if args.family is None or args.axis is None:
            parser.error('generation requires --family and --axis')
        result = cases(args.family, args.axis)
    print(json.dumps(result, indent=2)+'\n', end='')


if __name__ == '__main__':
    main()
