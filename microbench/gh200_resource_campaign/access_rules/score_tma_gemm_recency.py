"""相同正式源请求、交换暖机工作顺序的冻结响应评分。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics as st


def score(root):
    plan = json.loads((root/'plan.json').read_text())
    frozen = json.loads((root/'binding.json').read_text())
    for name,h in frozen['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == h, name
    started = int(datetime.fromisoformat(frozen['frozen_at']).timestamp()*1e9)
    invocations = json.loads((root/'returned/invocations.json').read_text())
    assert len(invocations) == len(plan['conditions'])*plan['processes']
    owner = {sm:g for g,sms in enumerate(plan['groups']) for sm in sms}
    records = {c['id']:[] for c in plan['conditions']}
    for call in invocations:
        path = root/'returned/samples'/f"r{call['repeat']:02d}-{call['id']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == call['sha256']
        assert call['returncode'] == 0 and call['started_ns'] > started
        s = json.loads(path.read_text())
        assert s['gpu_uuid'] == plan['gpu_uuid'] and s['errors'] == 0 and s['pollution_errors'] == 0
        groups = [owner[c['smid']] for c in s['ctas_data']]
        assert [c['tile'] for c in s['ctas_data']] == plan['coordinates'][s['work_index']]
        events = {(e['cta'],e['group']):e['points'] for e in s['group_events']}
        durations = [c['end'][1]-c['begin'][1] for c in s['ctas_data']]
        frequency = st.median((c['end'][0]-c['begin'][0])/v for c,v in zip(s['ctas_data'],durations))
        observed=[]
        for group in range(len(plan['groups'])):
            ct=[c for c,g in enumerate(groups) if g == group]
            row=dict(group=group,duration_ns=st.median(durations[c] for c in ct))
            if s['trace']:
                for u,unit in enumerate(('cycles','ns')):
                    for phase,(a,b) in plan['windows'].items():
                        row[phase+'_'+unit]=st.median(st.median(events[c,k][3][u]-events[c,k][0][u]
                            for k in range(a,b)) for c in ct)
            observed.append(row)
        records[call['id']].append(dict(repeat=call['repeat'],frequency_ghz=frequency,
            mapping_ok=groups == plan['cta_groups'],groups=observed))
    summary={}
    for name,rows in records.items():
        assert {r['repeat'] for r in rows} == set(range(plan['processes']))
        summary[name]=dict(frequency_ghz=st.median(r['frequency_ghz'] for r in rows),
            mapping_ok=all(r['mapping_ok'] for r in rows),groups=[dict(group=g,**{
                k:st.median(r['groups'][g][k] for r in rows) for k in rows[0]['groups'][g] if k != 'group'})
                for g in range(len(plan['groups']))])
    responses=[]
    for group in plan['primary_groups']:
        old,new=[summary[f'j0-last{last}-t1']['groups'][group] for last in (1,0)]
        a,b=[summary[f'j0-last{last}-t0']['groups'][group]['duration_ns'] for last in (1,0)]
        responses.append(dict(group=group,ns=new['early_ns']/old['early_ns']-1,
            cycles=new['early_cycles']/old['early_cycles']-1,untraced=b/a-1))
    f=[summary[f'j0-last{last}-t1']['frequency_ghz'] for last in (1,0)]
    frequency_ok=abs(f[1]/f[0]-1) <= .03
    mapping_ok=all(summary[f'j0-last{last}-t{trace}']['mapping_ok'] for last in (0,1) for trace in (0,1))
    candidates=[]
    for candidate in plan['predictions']:
        tests=[]
        for prediction,actual in zip(candidate['responses'],responses):
            assert prediction['group'] == actual['group']
            tests.append(dict(group=actual['group'],ns_ok=abs(actual['ns']-prediction['center']) <= candidate['tolerance'],
                cycles_ok=abs(actual['cycles']-prediction['center']) <= candidate['tolerance'],
                untraced_ok=actual['untraced'] <= -.10 if candidate['name']=='recent_source_state'
                    else abs(actual['untraced']) <= .05))
        passed=mapping_ok and frequency_ok and all(all(t[k] for k in ('ns_ok','cycles_ok','untraced_ok')) for t in tests)
        candidates.append(dict(candidate=candidate['name'],passed=passed,checks=tests,
            conclusion='获得支持' if passed else '被反证' if mapping_ok and frequency_ok else '信息不足'))
    return dict(mapping_ok=mapping_ok,frequency_ok=frequency_ok,frequency_response=f[1]/f[0]-1,
        primary_responses=responses,candidates=candidates,summaries=summary,processes=records)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=score(a.root);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(r['primary_responses']);print(r['candidates']);print('mapping/frequency',r['mapping_ok'],r['frequency_ok'])
