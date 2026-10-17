"""核对 cfg_b 源复用候选的冻结端点，写入前后状态另列为开发观测。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st


def score(root):
    plan = json.loads((root/'plan.json').read_text())
    binding = json.loads((root/'binding.json').read_text())
    for name, digest in binding['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    returned = root/'returned'
    invocations = json.loads((returned/'invocations.json').read_text())
    assert len(invocations) == len(plan['conditions'])*plan['processes']
    frozen_ns = int(__import__('datetime').datetime.fromisoformat(binding['frozen_at']).timestamp()*1e9)
    group_of = {sm:g for g,sms in enumerate(plan['groups']) for sm in sms}
    records = {c['id']: [] for c in plan['conditions']}
    for invocation in invocations:
        path = returned/'samples'/f"r{invocation['repeat']:02d}-{invocation['id']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == invocation['sha256']
        assert invocation['returncode'] == 0 and invocation['started_ns'] > frozen_ns
        sample = json.loads(path.read_text())
        assert sample['gpu_uuid'] == plan['gpu_uuid'] and sample['errors'] == sample['pollution_errors'] == 0
        assert [c['tile'] for c in sample['ctas_data']] == plan['coordinates']
        groups = [group_of[c['smid']] for c in sample['ctas_data']]
        endpoints = []
        if sample['trace']:
            events = {(e['cta'],e['group']): e['points'] for e in sample['group_events']}
            for g in range(len(plan['groups'])):
                actors = [c for c,x in enumerate(groups) if x == g]
                for start, end in plan['windows']:
                    endpoints.append(dict(group=g, range=[start,end], **{
                        'window_'+unit: st.median(st.median(events[c,k][3][u]-events[c,k][0][u]
                            for k in range(start,end)) for c in actors)
                        for u,unit in enumerate(('cycles','ns'))}))
        durations = [c['end'][1]-c['begin'][1] for c in sample['ctas_data']]
        f = st.median((c['end'][0]-c['begin'][0])/d for c,d in zip(sample['ctas_data'],durations))
        records[invocation['id']].append(dict(repeat=invocation['repeat'], file=str(path.relative_to(root)),
            effective_ghz=f, mapping_ok=groups == plan['cta_groups'], endpoints=endpoints,
            group_durations_ns=[st.median(d for d,g in zip(durations,groups) if g == i)
                                for i in range(len(plan['groups']))]))
    summaries = {}
    for case in plan['conditions']:
        rows = records[case['id']]
        assert {r['repeat'] for r in rows} == set(range(plan['processes']))
        summaries[case['id']] = dict(
            effective_ghz=st.median(r['effective_ghz'] for r in rows),
            mapping_ok=all(r['mapping_ok'] for r in rows),
            endpoints=[dict(group=e['group'],range=e['range'], **{
                k:st.median(r['endpoints'][i][k] for r in rows) for k in ('window_cycles','window_ns')})
                for i,e in enumerate(rows[0]['endpoints'])],
            group_durations_ns=[st.median(r['group_durations_ns'][i] for r in rows)
                for i in range(len(plan['groups']))], processes=rows)
    predictions = json.loads((root/'predictions.json').read_text())['rows']
    results = []
    for candidate in plan['candidates']:
        comparisons = []
        for prediction in (p for p in predictions if p['candidate'] == candidate):
            actual = summaries[f"q{prediction['active']}-s{prediction['skew']}-d0-t1"]
            for predicted, observed in zip(prediction['endpoints'], actual['endpoints']):
                assert (predicted['group'],predicted['range']) == (observed['group'],observed['range'])
                errors = dict(ns=predicted['window_ns']/observed['window_ns']-1,
                    cycles=predicted['window_ns']*plan['frequency_ghz']/observed['window_cycles']-1)
                comparisons.append(dict(active=prediction['active'],skew=prediction['skew'],
                    group=predicted['group'],range=predicted['range'],predicted_ns=predicted['window_ns'],
                    observed=observed,errors=errors))
        numeric = all(abs(v) <= plan['maximum_error'] for c in comparisons for v in c['errors'].values())
        results.append(dict(candidate=candidate, numeric_pass=numeric,
            maximum_ns_error=max(abs(c['errors']['ns']) for c in comparisons),
            maximum_cycle_error=max(abs(c['errors']['cycles']) for c in comparisons), comparisons=comparisons))
    primary = [s for name,s in summaries.items() if '-d0-' in name]
    mapping_ok = all(s['mapping_ok'] for s in primary)
    frequency_ok = all(abs(s['effective_ghz']/plan['frequency_ghz']-1) <= .03 for s in primary)
    bridge = []
    for q in (1,2,4):
        a,b = [summaries[f'q{q}-s{skew}-d0-t0']['group_durations_ns'] for skew in (0,64)]
        bridge.append(dict(active=q,response=[y/x-1 for x,y in zip(a,b)]))
    bridge_ok = all(x <= .05 for row in bridge for x in row['response'])
    for r in results:
        r['passed'] = r['numeric_pass'] and mapping_ok and frequency_ok and bridge_ok
        r['conclusion'] = '获得支持' if r['passed'] else '被反证' if mapping_ok and frequency_ok else '信息不足'
    state = []
    for q in (1,2,4):
        for skew in (0,64):
            a,b = [summaries[f'q{q}-s{skew}-d{d}-t1'] for d in (0,16896)]
            state.append(dict(active=q,skew=skew,frequency_response=b['effective_ghz']/a['effective_ghz']-1,
                window_response=[dict(group=x['group'],range=x['range'],ns=y['window_ns']/x['window_ns']-1,
                    cycles=y['window_cycles']/x['window_cycles']-1) for x,y in zip(a['endpoints'],b['endpoints'])]))
    return dict(qualification=plan['qualification'],mapping_ok=mapping_ok,frequency_ok=frequency_ok,
        bridge_ok=bridge_ok,bridge=bridge,candidates=results,summaries=summaries,
        prior_write_development=state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    report = score(args.root)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for r in report['candidates']:
        print(r['candidate'],r['conclusion'],'max ns/cycle error',r['maximum_ns_error'],r['maximum_cycle_error'])
    print('mapping/frequency/bridge',report['mapping_ok'],report['frequency_ok'],report['bridge_ok'])
