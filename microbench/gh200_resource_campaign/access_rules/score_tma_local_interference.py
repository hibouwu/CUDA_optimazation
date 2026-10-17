"""固定其他 SM 负载，评分目标 SM WGMMA 的冻结周期响应。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics as st

from analyze_tma_compute import control_partition


def score(root):
    plan = json.loads((root/'plan.json').read_text())
    binding = json.loads((root/'binding.json').read_text())
    for name, digest in binding['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    frozen_ns = int(datetime.fromisoformat(binding['frozen_at']).timestamp()*1e9)
    calls = json.loads((root/'returned/invocations.json').read_text())
    assert len(calls) == plan['sampling_processes']
    records = {case['id']: [] for case in plan['conditions']}
    for call in calls:
        path = root/'returned/samples'/f"r{call['repeat']:02}-{call['id']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == call['sha256']
        assert call['returncode'] == 0 and call['started_ns'] > frozen_ns
        sample = json.loads(path.read_text())
        assert sample['gpu_uuid'] == plan['gpu_uuid']
        assert sample['errors'] == sample['background_errors'] == 0
        assert sample['background_overlap'] == 1 and sample['tma_ctas'] == 1
        target = sample['ctas_data'][0]
        local = 1-sample['background_skip_targets']
        assert (target['background_rounds'] > 0) == bool(local)
        assert all(c['background_rounds'] > 0 for c in sample['ctas_data'][1:])
        dc = target['end'][0]-target['begin'][0]
        dn = target['end'][1]-target['begin'][1]
        row = dict(repeat=call['repeat'], sm=target['smid'], cycles=dc, ns=dn,
                   frequency_ghz=dc/dn)
        if sample['trace']:
            selected = dict(sample, ctas_data=[target],
                group_events=[e for e in sample['group_events'] if e['cta'] == 0])
            row['parts'] = {unit: control_partition(selected, i)
                            for i, unit in enumerate(('cycles', 'ns'))}
        records[call['id']].append(row)
    summaries = {}
    for name, rows in records.items():
        assert sorted(r['repeat'] for r in rows) == list(range(plan['processes']))
        summaries[name] = {key: st.median(r[key] for r in rows)
                           for key in ('cycles', 'ns', 'frequency_ghz')}
        summaries[name]['sms'] = sorted({r['sm'] for r in rows})
    responses = []
    for q, ring in ((1, 128), (4, 256)):
        for trace in (0, 1):
            names = [f'q{q}-r{ring}-local{local}-t{trace}' for local in (0, 1)]
            before, after = [summaries[n] for n in names]
            response = dict(active=q, ring=ring, trace=trace,
                **{key: after[key]/before[key]-1 for key in ('cycles', 'ns', 'frequency_ghz')},
                target_sms=[before['sms'], after['sms']])
            if trace:
                response['mean_control_delta'] = {unit: {
                    part: st.fmean(r['parts'][unit][part] for r in records[names[1]])
                         -st.fmean(r['parts'][unit][part] for r in records[names[0]])
                    for part in records[names[0]][0]['parts'][unit]}
                    for unit in ('cycles', 'ns')}
            responses.append(response)
    candidates = []
    for name, parameters in plan['predictions'].items():
        checks = []
        for response in responses:
            if response['trace']:
                continue
            p = parameters[f"q{response['active']}"] if name == 'local_factor' else parameters
            checks.append(dict(active=response['active'], observed=response['cycles'],
                predicted=p['cycle_response'], tolerance=p['tolerance'],
                passed=abs(response['cycles']-p['cycle_response']) <= p['tolerance']))
        candidates.append(dict(candidate=name, passed=all(c['passed'] for c in checks), checks=checks))
    supported = candidates[0]['passed'] and not candidates[1]['passed']
    return dict(frozen_at=binding['frozen_at'], local_factor_supported=supported,
        candidates=candidates, responses=responses, summaries=summaries, processes=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = score(args.root)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(report['candidates'])
    for response in report['responses']:
        print(response)
