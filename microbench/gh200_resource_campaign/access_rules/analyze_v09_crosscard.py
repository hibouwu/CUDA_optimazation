#!/usr/bin/env python3
"""Retrospective matched-reference cross-card comparison; no refit or pass gate."""
import argparse
import gzip
import json
from pathlib import Path
import statistics as st

from analyze_r13_sm import select_attempts
from analyze_r18 import replay
from v06_run import sha


METADATA = {'id', 'set', 'kind', 'group', 'purpose', 'original_experiment', 'variants'}


def compare_static(left, right):
    keys = (set(left) | set(right))-METADATA
    return {k: [left.get(k), right.get(k)] for k in sorted(keys) if left.get(k) != right.get(k)}


def compare(left, right):
    roots = [left, right]
    rows = [{r['id']: r for r in json.loads((p/'cases.json').read_text())} for p in roots]
    setups = [{r['case']: r['setup'] for r in json.loads((p/'static_setup.json').read_text())} for p in roots]
    provenance = {}
    cases = []
    for root in roots:
        for name in ('cases.json', 'static_setup.json', 'environment.json'):
            provenance[str(root/name)] = sha(root/name)
    for name in sorted(set(rows[0]) & set(rows[1])):
        delta = compare_static(rows[0][name], rows[1][name])
        setup_delta = {k: [setups[0][name].get(k), setups[1][name].get(k)]
                       for k in sorted(set(setups[0][name]) | set(setups[1][name]))
                       if k != 'gpu_uuid' and setups[0][name].get(k) != setups[1][name].get(k)}
        if delta or setup_delta:
            cases.append(dict(case=name, compatible=False, input_differences=delta, setup_differences=setup_delta))
            continue
        gpu = [s[name]['gpu_uuid'] for s in setups]
        if gpu[0] == gpu[1]:
            raise ValueError('same GPU is not cross-card evidence')
        cfg = rows[0][name]['config']
        identity = {}
        for variant in ('plain', 'dual'):
            for suffix in ('', '.sass'):
                relative = 'build/'+cfg+'_'+variant+suffix
                hashes = [sha(root/relative) for root in roots]
                for root, digest in zip(roots, hashes):
                    provenance[str(root/relative)] = digest
                identity[relative] = dict(hashes=hashes, identical=hashes[0] == hashes[1])
        if not all(v['identical'] for v in identity.values()):
            cases.append(dict(case=name, compatible=False, binary_identity=identity))
            continue
        observations = []
        failures = []
        for root, table in zip(roots, rows):
            selected, failed, _ = select_attempts(root, name, ('plain', 'dual'))
            failures.append(failed)
            records = []
            for item in selected:
                record = item['record']
                provenance[str(root/item['path'])] = sha(root/item['path'])
                observed = replay(root, record, table[name])
                with gzip.open(root/record['raw'], 'rt') as stream:
                    raw_setup = next(json.loads(line) for line in stream if line.strip() and json.loads(line)['event'] == 'setup')
                expected_gpu = setups[roots.index(root)][name]['gpu_uuid']
                if raw_setup.get('gpu_uuid') != expected_gpu:
                    raise ValueError('raw process GPU differs from static identity')
                if Path(record['command'][0]).name != cfg+'_'+record['variant']:
                    raise ValueError('sample command differs from compared binary')
                provenance[str(root/record['raw'])] = record['raw_sha256']
                value = dict(variant=record['variant'], trial=record['trial'], elapsed_us=observed['elapsed_us'],
                             warmup_calls=len(record['warmup']), gpu_uuid=raw_setup['gpu_uuid'])
                if record['variant'] == 'dual':
                    active = [c for c in observed['ctas'] if c['tiles_ns']]
                    value['envelope_ns'] = max(c['end_ns'] for c in active)-min(c['entry_ns'] for c in observed['ctas'])
                    value['max_cta_cycles'] = max(c['end_c']-c['entry_c'] for c in active)
                    value['median_cta_effective_ghz'] = st.median(
                        (c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns']) for c in active)
                records.append(value)
            for variant in ('plain', 'dual'):
                if sorted(r['trial'] for r in records if r['variant'] == variant) != list(range(10)):
                    raise ValueError(f'{name} {variant}: incomplete independent process set')
            observations.append(records)
        metrics = {}
        for variant, key in [('plain', 'elapsed_us'), ('dual', 'elapsed_us'), ('dual', 'envelope_ns'),
                             ('dual', 'max_cta_cycles'), ('dual', 'median_cta_effective_ghz')]:
            values = [[r[key] for r in records if r['variant'] == variant] for records in observations]
            centers = [st.median(v) for v in values]
            metrics[variant+'/'+key] = dict(medians=centers, target_over_source_percent=100*(centers[1]/centers[0]-1),
                ranges=[[min(v), max(v)] for v in values],
                empirical_ratio_range_percent=[100*(min(values[1])/max(values[0])-1),100*(max(values[1])/min(values[0])-1)])
        cases.append(dict(case=name, compatible=True, config=cfg, gpu=gpu, binary_identity=identity,
                          observations=observations, failed_attempts=failures, metrics=metrics))
    return dict(scope='Retrospective shared reference conditions only. No prospective holdout, causal card-only attribution, '
                'parameter recalibration or acceptance decision. Effective GHz is a CTA-window cycle/ns ratio.',
                roots=[str(p.resolve()) for p in roots], provenance=provenance, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.source, args.target)):
        raise ValueError('do not write into historical input runs')
    result = compare(args.source, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(shared_conditions=len(result['cases']), compatible=sum(c['compatible'] for c in result['cases']))))


if __name__ == '__main__':
    main()
