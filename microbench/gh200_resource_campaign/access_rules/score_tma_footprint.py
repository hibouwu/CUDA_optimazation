"""冻结工作集对照：只在两个程序共同的 g4..g11 前缀评分 C2。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics as st

from analyze_tma_tensor_probe import analyze


def score(root):
    binding = json.loads((root/'binding.json').read_text())
    for name, expected in binding['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    plan = json.loads((root/'plan.json').read_text())
    invocations = json.loads((root/'invocations.json').read_text())
    frozen_ns = int(datetime.fromisoformat(binding['frozen_at']).timestamp()*1e9)
    assert min(row['started_ns'] for row in invocations) > frozen_ns
    observations = analyze(root)
    rows = {row['id']: row for row in observations['rows']}
    for row in rows.values():
        if not row['arguments']['trace']:
            continue
        for process in row['processes']:
            sample = json.loads((root/'samples'/process['file']).read_text())
            for unit, index in [('cycles', 0), ('ns', 1)]:
                values = {cta['cta']: [] for cta in sample['ctas_data']}
                for event in sample['group_events']:
                    if 4 <= event['group'] < 12:
                        p = event['points']
                        values[event['cta']].append(p[3][index]-p[0][index])
                assert all(len(v) == 8 for v in values.values())
                process['common_prefix_'+unit] = st.median(st.median(v) for v in values.values())
        for unit in ('cycles', 'ns'):
            values = [p['common_prefix_'+unit] for p in row['processes']]
            row['summary']['common_prefix_'+unit] = dict(median=st.median(values), min=min(values), max=max(values))
    criteria = plan['C2']
    before, after = [rows[criteria[name]+'-t1'] for name in ('before', 'after')]
    response = {unit: after['summary']['common_prefix_'+unit]['median']/before['summary']['common_prefix_'+unit]['median']-1
                for unit in ('cycles', 'ns')}
    frequency = after['summary']['effective_ghz']['median']/before['summary']['effective_ghz']['median']-1
    plain_before, plain_after = [rows[criteria[name]+'-t0'] for name in ('before', 'after')]
    rates = [row['summary']['median_cta_ns']['median']/row['arguments']['groups'] for row in (plain_before, plain_after)]
    bridge = rates[1]/rates[0]-1
    candidate = all(abs(value-criteria['candidate_response']) <= criteria['tolerance'] for value in response.values())
    null = abs(response['cycles']-criteria['null_response']) <= criteria['null_tolerance']
    stable = abs(frequency) <= criteria['frequency_max_relative_change']
    if 'untraced_per_group_min_response' in criteria:
        bridge_pass = bridge >= criteria['untraced_per_group_min_response']
    else:
        bridge_pass = bridge <= criteria['untraced_per_group_max_response']
    conclusion = ('通过' if candidate and not null and stable and bridge_pass else
                  '被反证' if not candidate and stable else '信息不足')
    return dict(schema='tma-footprint-score/1', frozen_at=binding['frozen_at'],
        C2=dict(response=response, predicted_response=criteria['candidate_response'],
            tolerance=criteria['tolerance'], candidate_pass=candidate, null_pass=null,
            frequency_response=frequency, frequency_stable=stable,
            untraced_per_group_response=bridge, trace_bridge_pass=bridge_pass, conclusion=conclusion),
        observations=observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = score(args.root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result['C2'], indent=2, ensure_ascii=False))
    for row in result['observations']['rows']:
        if row['arguments']['trace']:
            print(row['id'], row['summary']['common_prefix_ns'])
