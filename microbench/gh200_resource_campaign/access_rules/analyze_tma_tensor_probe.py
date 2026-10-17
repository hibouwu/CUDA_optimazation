"""汇总 tensor 探针逐进程端点；保留 cycle/ns 与 trace 开关。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st


def endpoints(sample):
    ctas = sample['ctas_data']
    durations = [[c['end'][u]-c['begin'][u] for c in ctas] for u in (0, 1)]
    result = dict(median_cta_cycles=st.median(durations[0]), median_cta_ns=st.median(durations[1]),
        max_cta_ns=max(durations[1]), min_cta_ns=min(durations[1]),
        envelope_ns=max(c['end'][1] for c in ctas)-min(c['begin'][1] for c in ctas),
        start_span_ns=max(c['begin'][1] for c in ctas)-min(c['begin'][1] for c in ctas),
        end_span_ns=max(c['end'][1] for c in ctas)-min(c['end'][1] for c in ctas),
        effective_ghz=st.median(a/b for a,b in zip(*durations)))
    if sample['trace']:
        events = {(e['cta'],e['group']):e['points'] for e in sample['group_events']}
        q, groups = sample['active_groups'], sample['groups']
        for u, unit in enumerate(('cycles','ns')):
            values = dict(issue=[], wait=[], consumer=[], release_to_issue=[], issue_to_wait=[])
            for c in ctas:
                for g in range(q, groups-q):
                    p = events[c['cta'],g]
                    values['issue'].append(p[1][u]-p[0][u])
                    values['wait'].append(p[3][u]-p[2][u])
                    values['consumer'].append(p[4][u]-p[3][u])
                    values['release_to_issue'].append(p[0][u]-events[c['cta'],g-q][4][u])
                    values['issue_to_wait'].append(p[2][u]-events[c['cta'],g+q-1][1][u])
            for name, value in values.items():
                result[name+'_'+unit] = st.median(value)
            result['prefix_'+unit] = st.median(events[c['cta'],0][0][u]-c['begin'][u] for c in ctas)
            result['tail_'+unit] = st.median(c['end'][u]-events[c['cta'],groups-1][4][u] for c in ctas)
            result['release_window_'+unit] = st.median(events[c['cta'],groups-1][4][u]-c['begin'][u] for c in ctas)
            result['prefill_issue_'+unit] = st.median(events[c['cta'],g][1][u]-events[c['cta'],g][0][u]
                for c in ctas for g in range(q))
            if q > 1:
                result['prefill_gap_'+unit] = st.median(events[c['cta'],g][0][u]-events[c['cta'],g-1][1][u]
                    for c in ctas for g in range(1,q))
            result['first_wait_'+unit] = st.median(events[c['cta'],0][2][u]-events[c['cta'],q-1][1][u]
                for c in ctas)
            result['first_wait_duration_'+unit] = st.median(events[c['cta'],0][3][u]-events[c['cta'],0][2][u]
                for c in ctas)
            result['first_wait_return_'+unit] = st.median(events[c['cta'],0][3][u]-c['begin'][u]
                for c in ctas)
            if q > 1:
                result['drain_gap_'+unit] = st.median(events[c['cta'],g][2][u]-events[c['cta'],g-1][4][u]
                    for c in ctas for g in range(groups-q+1,groups))
    return result


def analyze(root):
    plan = json.loads((root/'plan.json').read_text())
    invocations = json.loads((root/'invocations.json').read_text())
    assert len(invocations) == len(plan['conditions'])*plan['processes']
    observations = {case['id']: [] for case in plan['conditions']}
    for invocation in invocations:
        name = f"r{invocation['repeat']:02d}-{invocation['id']}.json"
        path = root/'samples'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == invocation['sha256'], name
        sample = json.loads(path.read_text())
        assert sample['gpu_uuid'] == plan['gpu_uuid'] and invocation['returncode'] == 0
        assert sample['errors'] == 0 and sample['checked'] == 0 and sample['verify'] == 0
        assert len({c['smid'] for c in sample['ctas_data']}) == sample['ctas']
        observations[invocation['id']].append(dict(file=name, repeat=invocation['repeat'], **endpoints(sample)))
    rows = []
    for case in plan['conditions']:
        runs = observations[case['id']]
        assert {r['repeat'] for r in runs} == set(range(plan['processes']))
        summary = {}
        for key in runs[0]:
            if key not in ('file','repeat'):
                values = [r[key] for r in runs]
                summary[key] = dict(median=st.median(values), min=min(values), max=max(values))
        rows.append(dict(**case, summary=summary, processes=runs))
    return dict(schema='tensor-probe-summary/1', purpose=plan['purpose'],
        plan_sha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(), rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for row in report['rows']:
        s = row['summary']
        print(row['id'], 'CTA ns', s['median_cta_ns'], 'GHz', round(s['effective_ghz']['median'],4))
