"""相同输出集合、按存储域重排 CTA 工作后的暖机响应评分。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics as st


def score(root):
    plan=json.loads((root/'plan.json').read_text())
    frozen=json.loads((root/'binding.json').read_text())
    for name,digest in frozen['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    started=int(datetime.fromisoformat(frozen['frozen_at']).timestamp()*1e9)
    calls=json.loads((root/'returned/invocations.json').read_text())
    assert len(calls)==len(plan['conditions'])*plan['processes']
    group_of={sm:g for g,sms in enumerate(plan['groups']) for sm in sms}
    conditions={c['id']:c for c in plan['conditions']}
    records={name:[] for name in conditions}
    for call in calls:
        path=root/'returned/samples'/f"r{call['repeat']:02d}-{call['id']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==call['sha256']
        assert call['returncode']==0 and call['started_ns']>started
        sample=json.loads(path.read_text());case=conditions[call['id']]
        assert sample['gpu_uuid']==plan['gpu_uuid'] and sample['errors']==sample['pollution_errors']==0
        assert [c['tile'] for c in sample['ctas_data']]==plan['coordinates'][case['layout']][0]
        groups=[group_of[c['smid']] for c in sample['ctas_data']]
        events={(e['cta'],e['group']):e['points'] for e in sample['group_events']}
        durations=[c['end'][1]-c['begin'][1] for c in sample['ctas_data']]
        frequency=st.median((c['end'][0]-c['begin'][0])/d for c,d in zip(sample['ctas_data'],durations))
        observed=[]
        for group in range(8):
            actors=[c for c,g in enumerate(groups) if g==group]
            row=dict(group=group,duration_ns=st.median(durations[c] for c in actors))
            if sample['trace']:
                for u,unit in enumerate(('cycles','ns')):
                    for phase,(a,b) in plan['windows'].items():
                        row[phase+'_'+unit]=st.median(st.median(events[c,k][3][u]-events[c,k][0][u]
                            for k in range(a,b)) for c in actors)
            observed.append(row)
        records[call['id']].append(dict(repeat=call['repeat'],frequency_ghz=frequency,
            mapping_ok=groups==plan['cta_groups'],groups=observed))
    summaries={}
    for name,rows in records.items():
        assert {r['repeat'] for r in rows}==set(range(plan['processes']))
        summaries[name]=dict(frequency_ghz=st.median(r['frequency_ghz'] for r in rows),
            mapping_ok=all(r['mapping_ok'] for r in rows),groups=[dict(group=g,**{
                k:st.median(r['groups'][g][k] for r in rows) for k in rows[0]['groups'][g] if k!='group'})
                for g in range(8)])
    responses=[];frequencies=[]
    for layout in ('original','domain_local'):
        for trace in (0,1):
            old,new=[summaries[f'{layout}-last{last}-t{trace}'] for last in (1,0)]
            frequencies.append(dict(layout=layout,trace=trace,response=new['frequency_ghz']/old['frequency_ghz']-1))
        for group in plan['primary_groups']:
            old,new=[summaries[f'{layout}-last{last}-t1']['groups'][group] for last in (1,0)]
            old_plain,new_plain=[summaries[f'{layout}-last{last}-t0']['groups'][group] for last in (1,0)]
            responses.append(dict(layout=layout,group=group,ns=new['early_ns']/old['early_ns']-1,
                cycles=new['early_cycles']/old['early_cycles']-1,
                untraced=new_plain['duration_ns']/old_plain['duration_ns']-1))
    mapping_ok=all(s['mapping_ok'] for s in summaries.values())
    frequency_ok=all(abs(f['response'])<=plan['frequency_tolerance'] for f in frequencies)
    candidates=[]
    for prediction in plan['predictions']:
        checks=[]
        for p in prediction['responses']:
            observed=next(r for r in responses if (r['layout'],r['group'])==(p['layout'],p['group']))
            local=prediction['name']=='domain_replica_capacity' and p['layout']=='domain_local'
            checks.append(dict(layout=p['layout'],group=p['group'],predicted=p['center'],
                ns_ok=abs(observed['ns']-p['center'])<=p['tolerance'],
                cycles_ok=abs(observed['cycles']-p['center'])<=p['tolerance'],
                untraced_ok=abs(observed['untraced'])<=plan['local_untraced_tolerance'] if local
                    else observed['untraced']<=-plan['minimum_untraced_reduction']))
        passed=mapping_ok and frequency_ok and all(c['ns_ok'] and c['cycles_ok'] and c['untraced_ok'] for c in checks)
        candidates.append(dict(candidate=prediction['name'],passed=passed,checks=checks,
            conclusion='获得支持' if passed else '被反证' if mapping_ok and frequency_ok else '信息不足'))
    return dict(mapping_ok=mapping_ok,frequency_ok=frequency_ok,frequency_responses=frequencies,
        primary_responses=responses,candidates=candidates,summaries=summaries,processes=records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=score(args.root)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['primary_responses']);print(report['candidates'])
    print('mapping/frequency',report['mapping_ok'],report['frequency_ok'])
