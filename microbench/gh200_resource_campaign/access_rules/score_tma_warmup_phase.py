"""分别干预暖机与正式调用的起步次序，评分两种状态候选。"""
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
    criteria = json.loads((root/'plan.json').read_text())['C2']
    observations = analyze(root)
    rows = {row['id']: row for row in observations['rows']}
    mappings = []
    for row in rows.values():
        for process in row['processes']:
            sample = json.loads((root/'samples'/process['file']).read_text())
            assert sample['warmup_skew_ns'] == row['arguments']['warmup-skew-ns']
            assert sample['launch_skew_ns'] == row['arguments']['launch-skew-ns']
            mappings.append({c['cta']: c['smid'] for c in sample['ctas_data']})
            profile = {c['cta']: c['end'][0]-c['begin'][0] for c in sample['ctas_data']}
            if sample['trace']:
                windows = {c: [] for c in profile}
                begin, end = criteria['group_range']
                for event in sample['group_events']:
                    if begin <= event['group'] < end:
                        windows[event['cta']].append(event['points'][3][0]-event['points'][0][0])
                assert all(len(v) == end-begin for v in windows.values())
                profile = {c: st.median(v) for c, v in windows.items()}
            process['cohort_contrast_cycles'] = (st.fmean(profile[c] for c in criteria['slow_ctas'])
                - st.fmean(profile[c] for c in criteria['fast_ctas']))
    contrasts = {}
    for trace in (0, 1):
        base = rows[f'w0-m0-t{trace}']
        before = {p['repeat']: p['cohort_contrast_cycles'] for p in base['processes']}
        contrasts[str(trace)] = {}
        for condition in ('w0-m0', 'w64-m0', 'w0-m64', 'w64-m64'):
            row = rows[f'{condition}-t{trace}']
            contrasts[str(trace)][condition] = dict(
                contrast_cycles=st.median(p['cohort_contrast_cycles'] for p in row['processes']),
                paired_delta_cycles=st.median(p['cohort_contrast_cycles']-before[p['repeat']] for p in row['processes']),
                frequency_response=row['summary']['effective_ghz']['median']/base['summary']['effective_ghz']['median']-1)
    traced, plain = contrasts['1'], contrasts['0']
    mapping_equal = all(m == mappings[0] for m in mappings)
    usable = (mapping_equal
        and criteria['baseline_min_cycles'] <= traced['w0-m0']['contrast_cycles'] <= criteria['baseline_max_cycles']
        and abs(traced['w64-m64']['paired_delta_cycles']-criteria['joint_delta_cycles']) <= criteria['tolerance_cycles']
        and plain['w0-m0']['contrast_cycles'] >= criteria['untraced_baseline_min_cycles']
        and all(abs(v['frequency_response']) <= criteria['frequency_max_relative_change'] for group in contrasts.values() for v in group.values()))
    results = []
    for hypothesis in criteria['hypotheses']:
        response_pass = all(abs(traced[name]['paired_delta_cycles']-expected) <= criteria['tolerance_cycles']
            for name, expected in hypothesis['deltas_cycles'].items())
        bridge_pass = all(abs(plain[name]['paired_delta_cycles']/plain['w0-m0']['contrast_cycles']-expected)
            <= criteria['untraced_tolerance_fraction'] for name, expected in hypothesis['untraced_deltas_fraction'].items())
        results.append(dict(**hypothesis, response_pass=response_pass, bridge_pass=bridge_pass,
            passed=usable and response_pass and bridge_pass))
    passed = [r['name'] for r in results if r['passed']]
    supported = passed[0] if len(passed) == 1 else None
    return dict(schema='tensor-warmup-phase-score/1', frozen_at=binding['frozen_at'],
        C2=dict(conclusion='通过' if supported else '被反证' if usable else '信息不足',
            supported_candidate=supported, usable=usable, mapping_equal=mapping_equal,
            candidates=results, contrasts=contrasts), observations=observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = score(args.root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result['C2'], indent=2, ensure_ascii=False))
