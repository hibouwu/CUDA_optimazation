#!/usr/bin/env python3
"""Read the frozen failed pilot; write a new, explicitly unqualified diagnosis."""
import argparse
from array import array
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    return dict(count=len(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values))


def analyze(run, output):
    if output.exists():
        raise ValueError('output must be a new directory')
    manifest = json.loads((run/'manifest.json').read_text())
    frozen = json.loads((run/'freeze.json').read_text())
    inputs = {str(run/'freeze.json'): sha(run/'freeze.json')}
    for name, digest in frozen['files'].items():
        for path in (run/name, run/'returned'/name):
            assert sha(path) == digest, path
            inputs[str(path)] = digest
    for name, digest in manifest['analysis_sha256'].items():
        path = run/'analysis_snapshot'/name
        assert sha(path) == digest, path
        inputs[str(path)] = digest
    sys.path.insert(0, str(run/'analysis_snapshot'))
    from decode_r18_first_output import decode
    from analyze_r18_first_output import SELECTED_CTAS
    from analyze_r18 import reference
    from v08_model import scheduled_work
    work = scheduled_work('cfg_a', 960, 2816, [132, 1, 1], 1)
    assert [len(row) for row in work] == [2]*44 + [1]*88
    records_path = run/'returned/records.json'
    records = json.loads(records_path.read_text())
    inputs[str(records_path)] = sha(records_path)
    assert [{k:row[k] for k in ('id', 'trial', 'flag')} for row in records] == manifest['samples']
    assert json.loads((run/'returned/complete.json').read_text()) == dict(
        manifest_sha256=sha(run/'manifest.json'), records=20)
    original_path = run/'analysis.json'
    original = json.loads(original_path.read_text())
    inputs[str(original_path)] = sha(original_path)
    assert original['errors'] == [] and not original['common_L_control_passed']
    processes = []
    for row in records:
        path = run/'returned'/row['raw']
        assert not row['returncode'] and sha(path) == row['sha256']
        inputs[str(path)] = sha(path)
        with gzip.open(path, 'rt') as stream:
            events = [json.loads(line) for line in stream]
        assert len(events) == 4 and len({e['event'] for e in events}) == 4
        events = {e['event']: e for e in events}
        setup, call = events['setup'], events['call']
        assert setup['gpu_uuid'] == manifest['gpu_uuid']
        assert setup['first_output_observer'] == row['flag']
        assert setup['input_map_m'] == 1024
        decode(setup, call, row['id'])
        check = events['check']
        assert check['samples'] == 4096 and not check['nonfinite'] and not check['padding_errors']
        assert len(check['checked_indices']) == len(check['checked_values']) == 4096
        for index, value in zip(check['checked_indices'], check['checked_values']):
            assert value == reference(index//2816, index%2816, 4096, 17)
        assert events['input_zero_storage']['nonzero'] == 0
        trace = call['trace']
        assert len(trace) == 132*1318
        local = []
        for cta in range(132):
            base = cta*1318
            assert trace[base+2] > 0 and trace[base+10] == 0
            for role in (1, 2):
                count = len(work[cta])
                assert trace[base+3*role+3] == count
                assert trace[base+10+role] == trace[base+2]
                group = 'selected_T2' if cta in SELECTED_CTAS else ('other_T2' if count == 2 else 'other_T1')
                stamps = []
                for tile, coord in enumerate(work[cta]):
                    head = base+16+(role-1)*640+tile*10
                    assert trace[head+4:head+6] == [v+1 for v in coord]
                    cy = trace[head:head+4]
                    ns = trace[head+6:head+10]
                    assert all(v > 0 for v in cy+ns)
                    assert cy == sorted(cy) and ns == sorted(ns)
                    stamps.append(dict(cycles=cy, ns=ns))
                final = dict(cycles=trace[base+3*role+1], ns=trace[base+3*role+2])
                intervals = []
                for tile, stamp in enumerate(stamps):
                    for stage, index in (('L', 0), ('W', 1), ('E', 2)):
                        intervals.append(dict(stage=('first_' if tile == 0 else 'later_')+stage,
                            tile=tile, **{unit:stamp[unit][index+1]-stamp[unit][index] for unit in ('cycles', 'ns')}))
                    if tile+1 < count:
                        intervals.append(dict(stage='between_outputs', tile=tile,
                            **{unit:stamps[tile+1][unit][0]-stamp[unit][3] for unit in ('cycles', 'ns')}))
                intervals.append(dict(stage='source_tail', tile=count-1,
                    **{unit:final[unit]-stamps[-1][unit][3] for unit in ('cycles', 'ns')}))
                whole = {unit:final[unit]-stamps[0][unit][0] for unit in ('cycles', 'ns')}
                assert all(v >= 0 for item in intervals for v in (item['cycles'], item['ns']))
                assert all(sum(item[unit] for item in intervals) == whole[unit] for unit in ('cycles', 'ns'))
                intervals.append(dict(stage='first_to_final', tile=0, **whole))
                local.append(dict(cta=cta, role=role, sm=trace[base+2]-1, group=group,
                                  intervals=intervals))
        processes.append(dict(id=row['id'], trial=row['trial'], flag=row['flag'],
                              elapsed_us=call['elapsed_us'], local=local))
    paired = []
    summaries = []
    for trial in range(10):
        off = next(p for p in processes if p['trial'] == trial and p['flag'] == 0)
        on = next(p for p in processes if p['trial'] == trial and p['flag'] == 1)
        deltas = []
        for left, right in zip(off['local'], on['local']):
            assert (left['cta'], left['role'], left['sm']) == (right['cta'], right['role'], right['sm'])
            for a, b in zip(left['intervals'], right['intervals']):
                assert (a['stage'], a['tile']) == (b['stage'], b['tile'])
                deltas.append(dict(cta=left['cta'], role=left['role'], group=left['group'],
                    stage=a['stage'], tile=a['tile'], off={u:a[u] for u in ('cycles','ns')},
                    on={u:b[u] for u in ('cycles','ns')},
                    **{unit:b[unit]-a[unit] for unit in ('cycles','ns')}))
        paired.append(dict(trial=trial, deltas=deltas))
        for group in ('selected_T2', 'other_T2', 'other_T1'):
            stages = sorted({d['stage'] for d in deltas if d['group'] == group})
            for stage in stages:
                rows = [d for d in deltas if d['group'] == group and d['stage'] == stage]
                summaries.append(dict(trial=trial, group=group, stage=stage,
                    **{unit:summarize([d[unit] for d in rows]) for unit in ('cycles','ns')}))
    # Reproduce the frozen gate without cancellation, pooling, centering or refitting.
    gates = []
    for cta in SELECTED_CTAS:
        for role in (1, 2):
            values = [next(d['ns'] for d in p['deltas'] if (d['cta'],d['role'],d['stage']) == (cta,role,'first_L')) for p in paired]
            absolute = list(map(abs, values))
            median, maximum = statistics.median(absolute), max(absolute)
            passed = median <= manifest['gates']['per_role_L_median_absolute_ns'] and maximum <= manifest['gates']['per_role_L_maximum_absolute_ns']
            original_gate = next(x for x in original['local'] if (x['cta'],x['role']) == (cta,role))
            assert (median, maximum, passed, values) == (original_gate['median_absolute'], original_gate['maximum_absolute'], original_gate['passed'], original_gate['signed'])
            gates.append(dict(cta=cta,role=role,median_absolute_ns=median,maximum_absolute_ns=maximum,passed=passed))
    assert not any(g['passed'] for g in gates)
    # Independent period-17 closed form for the two full-D numerical smoke dumps.
    correctness = run.parent/'20261011-R18-first-output-correctness'
    table = []
    for row in range(17):
        values = []
        for col in range(17):
            products = [((row*7+t*13)%17-8)*((t*5+col*11)%17-8) for t in range(17)]
            values.append((sum(products)*(4096//17)+sum(products[:4096%17]))/1024)
        table.append(values)
    numerical = []
    for extent in (960,1024):
        path = correctness/f'returned/full-d/map{extent}.bin'
        data = array('f')
        data.frombytes(path.read_bytes())
        if sys.byteorder != 'little': data.byteswap()
        assert len(data) == 960*2816
        for row in range(960):
            expected = array('f', (table[row%17][col%17] for col in range(2816)))
            assert data[row*2816:(row+1)*2816] == expected, (extent,row)
        inputs[str(path)] = sha(path)
        numerical.append(dict(map_m=extent, values=len(data), exact=True, sha256=sha(path)))
    assert numerical[0]['sha256'] == numerical[1]['sha256']
    result = dict(input_sha256=inputs, analyzer_sha256=sha(Path(__file__)), processes=processes,
        paired=paired, summaries=summaries, original_gates=gates, numerical=numerical,
        checks=dict(processes=20, raw_numeric_values=20*4096, physical_output_tiles=176,
            same_SM_role_pairs=2640, exact_telescoping_role_paths=5280),
        common_L_control_passed=False, internal_endpoints_qualified=False, observation_qualified=False,
        limits=['Posthoc diagnosis of the original failed map1024 pilot; no target-map timing or new holdout.',
            'Compare durations formed inside one invocation/CTA/role; no cross-CTA native clock subtraction.',
            'Group medians need not telescope; only individual role paths telescope.',
            'Unselected CTAs are not isolated controls. No spatial subtraction or gate repair.',
            'Source tail is DONE_last to original FINAL; FINAL precedes the newly deferred observer flush.',
            'No cycle/ns conversion, physical frequency inference, service fitting or causal identification.'])
    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(output/'diagnosis.json.gz','wt') as stream:
        json.dump(result, stream, indent=2)
    with (output/'stage-summary.csv').open('x', newline='') as stream:
        fields = ['trial','group','stage','count','median_ns','min_ns','max_ns','median_cycles','min_cycles','max_cycles']
        writer=csv.DictWriter(stream, fieldnames=fields);writer.writeheader()
        for row in summaries:
            writer.writerow(dict(trial=row['trial'],group=row['group'],stage=row['stage'],count=row['ns']['count'],
                median_ns=row['ns']['median'],min_ns=row['ns']['minimum'],max_ns=row['ns']['maximum'],
                median_cycles=row['cycles']['median'],min_cycles=row['cycles']['minimum'],max_cycles=row['cycles']['maximum']))
    print(json.dumps(dict(checks=result['checks'], numerical_values=sum(x['values'] for x in numerical),
        failed_roles=len(gates), median_abs_range_ns=[min(g['median_absolute_ns'] for g in gates),max(g['median_absolute_ns'] for g in gates)],
        maximum_abs_range_ns=[min(g['maximum_absolute_ns'] for g in gates),max(g['maximum_absolute_ns'] for g in gates)])))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    analyze(args.input.resolve(), args.output.resolve())
