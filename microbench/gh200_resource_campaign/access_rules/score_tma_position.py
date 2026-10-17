"""交换源地址或反转起步次序，评分预先冻结的 CTA 响应轮廓。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics as st

from analyze_tma_tensor_probe import analyze


def score(root):
    binding=json.loads((root/'binding.json').read_text())
    for name,digest in binding['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    frozen_ns=int(datetime.fromisoformat(binding['frozen_at']).timestamp()*1e9)
    invocations=json.loads((root/'invocations.json').read_text())
    assert min(row['started_ns'] for row in invocations)>frozen_ns
    plan=json.loads((root/'plan.json').read_text());criteria=plan['C2']
    group_begin,group_end=criteria.get('group_range',[4,12])
    observations=analyze(root)
    rows={row['id']:row for row in observations['rows']}
    mappings=[]
    for row in rows.values():
        for process in row['processes']:
            sample=json.loads((root/'samples'/process['file']).read_text())
            mappings.append({c['cta']:c['smid'] for c in sample['ctas_data']})
            profile={c['cta']:c['end'][0]-c['begin'][0] for c in sample['ctas_data']}
            if row['arguments']['trace']:
                windows={c:[] for c in profile}
                for event in sample['group_events']:
                    if group_begin<=event['group']<group_end:
                        windows[event['cta']].append(event['points'][3][0]-event['points'][0][0])
                assert all(len(values)==group_end-group_begin for values in windows.values())
                profile={c:st.median(values) for c,values in windows.items()}
                starts={e['cta']:e['points'][0][1] for e in sample['group_events'] if e['group']==0}
                origin=min(starts.values())
                process['start_contrast_ns']=st.fmean(starts[c]-origin for c in criteria['slow_ctas'])-st.fmean(starts[c]-origin for c in criteria['fast_ctas'])
            process['position_profile_cycles']=profile
            process['contrast_cycles']=st.fmean(profile[c] for c in criteria['slow_ctas'])-st.fmean(profile[c] for c in criteria['fast_ctas'])
    mapping_equal=all(mapping==mappings[0] for mapping in mappings)
    contrasts={}
    for trace in (0,1):
        prefixes=criteria.get('condition_prefixes',['n32-shift0','n32-shift16'])
        before,after=[rows[f'{prefix}-t{trace}'] for prefix in prefixes]
        values={name:st.median(p['contrast_cycles'] for p in row['processes']) for name,row in [('before',before),('after',after)]}
        values['paired_delta']=st.median(p['contrast_cycles']-next(q['contrast_cycles'] for q in before['processes'] if q['repeat']==p['repeat']) for p in after['processes'])
        values['frequency_response']=after['summary']['effective_ghz']['median']/before['summary']['effective_ghz']['median']-1
        if trace:
            values['start_contrast_ns']={name:st.median(p['start_contrast_ns'] for p in row['processes']) for name,row in [('before',before),('after',after)]}
        contrasts[str(trace)]=values
    traced,plain=contrasts['1'],contrasts['0']
    usable=mapping_equal and criteria['baseline_min_cycles']<=traced['before']<=criteria['baseline_max_cycles'] and plain['before']>=criteria['untraced_baseline_min_cycles'] and all(abs(c['frequency_response'])<=criteria['frequency_max_relative_change'] for c in contrasts.values())
    order_checked=None
    if criteria.get('reverse_start_order'):
        order_checked=all(p['start_contrast_ns']<0 for p in rows[prefixes[0]+'-t1']['processes']) and all(p['start_contrast_ns']>0 for p in rows[prefixes[1]+'-t1']['processes'])
        usable=usable and order_checked
    hypotheses=criteria.get('hypotheses')
    if hypotheses is None:
        hypotheses=[dict(name='address',delta_cycles=criteria['address_delta_cycles'],bridge='reverse'),
                    dict(name='position_or_state',delta_cycles=criteria['position_delta_cycles'],bridge='same')]
    results=[]
    for hypothesis in hypotheses:
        response_pass=abs(traced['paired_delta']-hypothesis['delta_cycles'])<=criteria['tolerance_cycles']
        bridge_pass=plain['after']<=-.5*plain['before'] if hypothesis['bridge']=='reverse' else plain['after']>=.5*plain['before']
        results.append(dict(**hypothesis,response_pass=response_pass,bridge_pass=bridge_pass,passed=usable and response_pass and bridge_pass))
    passed=[r['name'] for r in results if r['passed']]
    supported=passed[0] if len(passed)==1 else None
    conclusion='通过' if supported else '被反证' if usable else '信息不足'
    return dict(schema='tensor-position-score/2',frozen_at=binding['frozen_at'],C2=dict(conclusion=conclusion,supported_candidate=supported,usable=usable,mapping_equal=mapping_equal,start_order_checked=order_checked,candidates=results,contrasts=contrasts),observations=observations)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=score(args.root);args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result['C2'],indent=2,ensure_ascii=False))
