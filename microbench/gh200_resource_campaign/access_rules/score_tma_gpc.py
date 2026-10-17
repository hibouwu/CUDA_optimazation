"""按预冻结的共享组窗口检验有限并行源服务候选。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics as st

from analyze_tma_tensor_probe import analyze


def score(root):
    binding = json.loads((root/'binding.json').read_text())
    for name, digest in binding['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    invocations = json.loads((root/'invocations.json').read_text())
    frozen_ns = int(datetime.fromisoformat(binding['frozen_at']).timestamp()*1e9)
    assert min(r['started_ns'] for r in invocations) > frozen_ns
    plan = json.loads((root/'plan.json').read_text())
    predictions = json.loads((root/'predictions.json').read_text())['candidates']
    observations = analyze(root)
    rows = {row['id']: row for row in observations['rows']}
    mapping_equal = True
    for row in rows.values():
        for process in row['processes']:
            sample = json.loads((root/'samples'/process['file']).read_text())
            mapping = [c['smid'] for c in sample['ctas_data']]
            mapping_equal &= mapping == plan['expected_smids']
            selected = [c for c in sample['ctas_data'] if c['cta'] in plan['group0_ctas']]
            process['group0_full_ns'] = st.median(c['end'][1]-c['begin'][1] for c in selected)
            if sample['trace']:
                for u, unit in enumerate(('cycles', 'ns')):
                    by = {c['cta']: [] for c in selected}
                    for event in sample['group_events']:
                        if event['cta'] in by and 8 <= event['group'] < 16:
                            by[event['cta']].append(event['points'][3][u]-event['points'][0][u])
                    assert all(len(v) == 8 for v in by.values())
                    process['group0_window_'+unit] = st.median(st.median(v) for v in by.values())
        row['group0_full_ns'] = st.median(p['group0_full_ns'] for p in row['processes'])
        if row['arguments']['trace']:
            for unit in ('cycles', 'ns'):
                row['group0_window_'+unit] = st.median(p['group0_window_'+unit] for p in row['processes'])
    bridge_pairs = plan.get('bridge_pairs', [('skew16-t0', 'skew0-t0'), ('skew32-t0', 'skew0-t0')])
    bridge = all(rows[after]['group0_full_ns']/rows[before]['group0_full_ns'] <= 1.05
                 for after, before in bridge_pairs)
    frequencies = [r['summary']['effective_ghz']['median'] for r in rows.values()]
    clock_valid = all(abs(f/plan['predicted_frequency_ghz']-1) <= .03 for f in frequencies)
    results = []
    for candidate in predictions:
        errors = []
        for predicted in candidate['conditions']:
            condition = predicted.get('condition_id', f"skew{predicted['skew_ns']}")
            row = rows[condition+'-t1']
            errors.append(dict(condition_id=condition, skew_ns=predicted['skew_ns'],
                predicted_ns=predicted['group0_window_ns'], observed_ns=row['group0_window_ns'],
                ns_error=predicted['group0_window_ns']/row['group0_window_ns']-1,
                predicted_cycles=predicted['group0_window_ns']*plan['predicted_frequency_ghz'],
                observed_cycles=row['group0_window_cycles'],
                cycle_error=predicted['group0_window_ns']*plan['predicted_frequency_ghz']/row['group0_window_cycles']-1))
        response_pass = all(abs(e[k]) <= plan['maximum_relative_error'] for e in errors for k in ('ns_error', 'cycle_error'))
        results.append(dict(capacity=candidate['capacity'], errors=errors, response_pass=response_pass,
            passed=mapping_equal and clock_valid and bridge and response_pass))
    supported = [r['capacity'] for r in results if r['passed']]
    usable = mapping_equal and clock_valid
    return dict(schema='tma-gpc-heldout-score/1', frozen_at=binding['frozen_at'],
        C2=dict(conclusion='通过' if supported else '被反证' if usable else '信息不足',
            supported_capacities=supported, mapping_equal=mapping_equal,
            clock_valid=clock_valid, untraced_direction_pass=bridge, candidates=results),
        observations=observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = score(args.root)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report['C2'], indent=2, ensure_ascii=False))
