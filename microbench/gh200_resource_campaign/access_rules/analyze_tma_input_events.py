#!/usr/bin/env python3
"""Offline development comparison; does not launch CUDA or change old evidence."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import least_squares
from tma_input_events import Costs, simulate

ROOT = Path(__file__).resolve().parents[3]
SUITE = Path('results/gh200_resource_campaign/20261001-resource-suite-v2')
S14 = SUITE / 'parallel/tma_cluster/analysis/s14-formal-733392-qualified'
S16 = SUITE / 'implementation/s16-main-execution/published-v1'
R05 = Path('results/gh200_resource_campaign/access_rules/Q07-smem/R03-access-demand/20261006-c-job735059/r05-formal-v2')


def load_conditions():
    identities = {}

    def read(path):
        blob = (ROOT / path).read_bytes()
        identities[str(path)] = hashlib.sha256(blob).hexdigest()
        return blob.decode()

    def js(path):
        return json.loads(read(path))

    def rows(path):
        return list(csv.DictReader(read(path).splitlines()))

    records = []
    for row in rows(S14 / 'cases.csv'):
        if row['case_id'] != 'gmem_to_smem_16kib_one_cta':
            continue
        assert row['eligible_after_independent_C'] == 'True'
        records.append(dict(family='s14', case_id=row['case_id'], stages=1,
                            requests=1, consumer=0, items=int(row['iterations']),
                            observed=16384 / float(row['median']), samples=int(row['samples']),
                            statistic='inverse_published_median_rate', role='aggregate_reference_only',
                            source_union_bytes=32*16384, physical_slots=1,
                            gpu='GPU-ec947ba1-3e15-9860-163d-330d6c66d1b4',
                            cv=float(row['cv']), resources={'dynamic_smem_bytes': 16384+32}))
    js(S14 / 'parameter-candidates.json')
    publication = js(S16 / 'publication-final-state.json')
    assert publication['publication_state'] == 'published_after_independent_bridge_review'
    params = js(S16 / 'qualified-parameters.json')['parameters']
    assert identities[str(S16 / 'qualified-parameters.json')] == publication['parameters']['sha256']
    samples = rows(S16 / 'samples.csv')
    for row in params:
        c = row['conditions']
        if c['direction'] != 'gmem_to_smem' or c['blocks'] != 1:
            continue
        s, r = c['software_stages'], c['requests_per_item']
        selected = [a for a in samples if a['case_id'] == row['case_id']]
        values = []
        for a in selected:
            assert int(a['work_count_bytes']) == c['iterations'] * r * 16384
            assert abs(float(a['value']) - int(a['work_count_bytes']) / float(a['denominator'])) < 1e-8
            values.append(float(a['denominator']) / (c['iterations'] * r))
        assert len(values) == 10
        records.append(dict(family='s16', case_id=row['case_id'], stages=s,
                            requests=r, consumer=0, items=c['iterations'],
                            observed=statistics.median(values), samples=len(values),
                            statistic='median_process_cycles_per_request',
                            role='development_transfer' if s == 4 else 'development_fit',
                            source_union_bytes=32*r*16384, physical_slots=s*r,
                            gpu=row['device_uuid'], cv=row['cv'], resources=c['resources']))
    hashes = js(R05 / 'source_hashes.json')
    for leaf in ['probes/r05.cu', 'probes/r00_common.hpp']:
        path = R05 / 'source' / leaf
        read(path)
        assert identities[str(path)] == hashes[leaf]
    gpu = js(R05 / 'environment.json')['gpu_uuid']
    for row in rows(R05 / 'cases.csv'):
        if not row['id'].startswith('B_tin_'):
            continue
        selected = sorted((ROOT / R05 / 'samples' / row['id']).glob('trial-*/result.json'))
        trials = [js(p.relative_to(ROOT)) for p in selected]
        assert len(trials) == 10
        first = trials[0]
        keys = ['iterations', 'stages', 'consumer', 'blocks', 'threads', 'slots', 'capture', 'unit']
        for t in trials:
            assert all(t[k] == first[k] for k in keys)
            assert t['blocks'] == 1 and t['threads'] == 128 and t['slots'] == 8 and not t['capture']
            assert t['unit'] == 'clock64_cycle/CTA' and t['status'] == 'measured'
            assert t['work_bytes'] == t['iterations'] * 8 * 16384
            assert t['stamps'][0][1] - t['stamps'][0][0] == t['elapsed']
        mean = statistics.mean(t['elapsed'] for t in trials)
        assert abs(mean - float(row['mean_window'])) < .02
        records.append(dict(family='r05', case_id=row['id'], stages=first['stages'],
                            requests=1, consumer=first['consumer'], items=first['iterations']*8,
                            observed=mean/(first['iterations']*8), samples=len(trials),
                            statistic='mean_process_cycles_per_request',
                            role='development_transfer' if first['stages'] == 4 else 'development_fit',
                            source_union_bytes=first['global_allocation_bytes'], physical_slots=4,
                            gpu=gpu, cv=float(row['cv']),
                            resources={k: first[k] for k in ['registers_per_thread','local_bytes_per_thread',
                                      'static_smem_bytes','dynamic_smem_bytes','occupancy_limit_ctas_per_sm']}))
    # Source snapshots inspected to construct the software schedules. Only the R05
    # pair above is independently matched here against its frozen source manifest.
    for p in ['microbench/gh200_resource_campaign/probes/tma_bulk.cu',
              'microbench/gh200_resource_campaign/families/s16/microbench/gh200_resource_campaign/probes/tma_stage_request.cu']:
        read(Path(p))
    return records, identities


def prediction(row, costs, cap, horizon=None):
    # R05 drains all requests every eight: repeating the same memoryless batch
    # is exact for this candidate. This does not assert real cache stationarity.
    n = 8 if row['family'] == 'r05' else (horizon or row['items'])
    return simulate(row['family'], n, row['stages'], row['requests'], costs,
                    cap, row['consumer'])['cycles_per_request']


def compare(records, horizon):
    fits, comparisons, traces = [], [], {}
    starts = [[200, 600, 20, 180, 4], [450, 250, 100, 300, 5], [50, 900, 200, 80, 3]]
    for family in ('s16', 'r05'):
        cases = [r for r in records if r['family'] == family]
        train = [r for r in cases if r['role'] == 'development_fit']
        dimension = 4 if family == 's16' else 5
        for cap in (None, 1, 2, 4, 8):
            label = f'{family}_credits_{cap if cap is not None else "unlimited"}'

            def residual(x):
                costs = Costs(*x)
                return np.array([prediction(r, costs, cap, horizon)/r['observed']-1 for r in train])

            attempts = []
            for start in starts:
                result = least_squares(residual, start[:dimension], bounds=(0, 4000),
                                       max_nfev=150, diff_step=1e-4,
                                       ftol=1e-9, xtol=1e-9, gtol=1e-9)
                attempts.append(dict(parameters=result.x.tolist(), loss=float(np.sum(result.fun**2)),
                                     optimizer_success=bool(result.success), evaluations=result.nfev))
            chosen = min(attempts, key=lambda a: a['loss'])
            costs = Costs(*chosen['parameters'])
            local = []
            for row in cases:
                pred = prediction(row, costs, cap)
                approx = prediction(row, costs, cap, horizon)
                full = simulate(family, row['items'], row['stages'], row['requests'],
                                costs, cap, row['consumer'])['cycles_per_request']
                repeat_delta = 100 * (pred / full - 1)
                assert abs(repeat_delta) < 1e-7
                entry = dict(candidate=label, case_id=row['case_id'], family=family,
                             role=row['role'], observed=row['observed'], predicted=pred,
                             error_percent=100*(pred/row['observed']-1),
                             horizon_delta_percent=100*(approx/pred-1),
                             batch_repeat_delta_percent=repeat_delta)
                local.append(entry)
                comparisons.append(entry)
            fits.append(dict(candidate=label, family=family, credits=cap, costs=asdict(costs),
                             attempts=attempts,
                             fit_max_abs_percent=max(abs(e['error_percent']) for e in local if e['role']=='development_fit'),
                             transfer_max_abs_percent=max(abs(e['error_percent']) for e in local if e['role']=='development_transfer'),
                             max_horizon_delta_percent=max(abs(e['horizon_delta_percent']) for e in local)))
            traces[label] = simulate(family, 8, 2, 2 if family == 's16' else 1,
                                     costs, cap, 16 if family == 'r05' else 0, trace=True)
    reference = next(r for r in records if r['family'] == 's14')
    # Observationally identical examples, not identified hardware parameters.
    for fraction in (0.25, 0.75):
        total = reference['observed']
        traces[f's14_aggregate_example_{fraction}'] = simulate(
            's14', 4, 1, 1, Costs(total*fraction, total*(1-fraction)), trace=True)
    return fits, comparisons, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit-items', type=int, default=512)
    args = parser.parse_args()
    if args.fit_items < 32:
        parser.error('--fit-items must be at least 32')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be new or empty; preserve previous results')
    records, identities = load_conditions()
    fits, comparisons, traces = compare(records, args.fit_items)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(schema_version=1, status='development_only_not_hardware_qualification',
                    fit_items=args.fit_items, family_parameters_shared=False,
                    physical_path='GPU-local cudaMalloc global -> issuing CTA local SMEM; actual L2/HBM traffic unresolved',
                    credit_release='candidate_ready', parameter_identifiability='not_established',
                    units='clock64 cycles per logical 16 KiB request, includes software sequence',
                    excluded=['all_gpu', 'smem_to_global', 'S4R4_capacity_rejections', 'R05_A_invalid_sass'],
                    limitations=['All historical data were previously seen; transfer is development only.',
                                 'No initial/final one-time timer or synchronization cost parameter.',
                                 'S14 only constrains an aggregate, no multi-parameter fit.',
                                 'Control costs constant within family; S16 splits issue/retire equally.',
                                 'Finite credits block software issue until modeled completion.',
                                 'Bounded multistart search is not a proof of global optimum or unique parameters.'],
                    conditions=records, input_sha256=identities, fits=fits)
    for path in [Path(__file__), Path(__file__).with_name('tma_input_events.py')]:
        metadata.setdefault('code_sha256', {})[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'analysis.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (args.output/'traces.json').write_text(json.dumps(traces, indent=2)+'\n')
    with (args.output/'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    print(json.dumps([{k:v for k,v in fit.items() if k not in ('attempts','costs')} for fit in fits], indent=2))


if __name__ == '__main__':
    main()
