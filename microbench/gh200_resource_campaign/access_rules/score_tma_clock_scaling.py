"""独立驻留源条件上，比较固定 SM 周期与固定 ns 的冻结响应。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics as st


def score(root):
    plan=json.loads((root/'plan.json').read_text())
    binding=json.loads((root/'binding.json').read_text())
    for name,digest in binding['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    freeze_ns=int(datetime.fromisoformat(binding['frozen_at']).timestamp()*1e9)
    calls=json.loads((root/'returned/invocations.json').read_text())
    assert len(calls)==plan['processes']*len(plan['conditions'])
    records={c['id']:[] for c in plan['conditions']}
    for call in calls:
        p=root/'returned/samples'/f"r{call['repeat']:02d}-{call['id']}.json"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==call['sha256']
        assert call['returncode']==0 and call['started_ns']>freeze_ns
        s=json.loads(p.read_text());target=s['ctas_data'][0]
        assert s['gpu_uuid']==plan['gpu_uuid'] and s['errors']==s['background_errors']==0
        assert s['background_overlap']==1 and s['tma_ctas']==1 and target['background_rounds']==0
        if s['background']==2:assert all(c['background_rounds']>0 for c in s['ctas_data'][1:])
        duration_cycles=target['end'][0]-target['begin'][0]
        duration_ns=target['end'][1]-target['begin'][1]
        row=dict(repeat=call['repeat'],sm=target['smid'],frequency_ghz=duration_cycles/duration_ns,
            duration_cycles=duration_cycles,duration_ns=duration_ns)
        if s['trace']:
            a,b=plan['primary_window'];events={e['group']:e['points'] for e in s['group_events']}
            for u,unit in enumerate(('cycles','ns')):
                row['window_'+unit]=st.mean(events[k][3][u]-events[k][0][u] for k in range(a,b))
                row['issue_'+unit]=st.mean(events[k][1][u]-events[k][0][u] for k in range(a,b))
                row['consumer_'+unit]=st.mean(events[k][4][u]-events[k][3][u] for k in range(a,b))
                row['to_wait_'+unit]=st.mean(events[k][2][u]-events[k][1][u] for k in range(a,b))
        records[call['id']].append(row)
    summaries={}
    for name,rows in records.items():
        assert {r['repeat'] for r in rows}==set(range(plan['processes']))
        summaries[name]={k:st.median(r[k] for r in rows) for k in rows[0] if k not in ('repeat','sm')}
        summaries[name]['sms']=sorted({r['sm'] for r in rows})
    responses=[]
    for q,ring in ((1,64),(2,64),(4,128)):
        for trace in (0,1):
            old,new=[summaries[f'q{q}-r{ring}-bg{bg}-t{trace}'] for bg in (0,2)]
            metric='window' if trace else 'duration'
            responses.append(dict(active=q,ring=ring,trace=trace,
                frequency=new['frequency_ghz']/old['frequency_ghz']-1,
                cycles=new[metric+'_cycles']/old[metric+'_cycles']-1,
                ns=new[metric+'_ns']/old[metric+'_ns']-1,
                target_sms=[old['sms'],new['sms']]))
    frequency_ok=all(abs(r['frequency']-plan['frequency_response'])<=plan['frequency_tolerance'] for r in responses)
    candidates=[]
    for p in plan['predictions']:
        checks=[dict(active=r['active'],ring=r['ring'],trace=r['trace'],
            cycles_ok=abs(r['cycles']-p['cycle_response'])<=plan['response_tolerance'],
            ns_ok=abs(r['ns']-p['ns_response'])<=plan['response_tolerance']) for r in responses]
        passed=frequency_ok and all(c['cycles_ok'] and c['ns_ok'] for c in checks)
        candidates.append(dict(candidate=p['name'],passed=passed,checks=checks,
            conclusion='获得支持' if passed else '被反证' if frequency_ok else '信息不足'))
    return dict(frequency_ok=frequency_ok,responses=responses,candidates=candidates,
        summaries=summaries,processes=records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=score(args.root)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['responses']);print(report['candidates'])
    print('frequency',report['frequency_ok'])
