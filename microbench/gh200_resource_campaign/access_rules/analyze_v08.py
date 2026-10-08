#!/usr/bin/env python3
"""Score frozen V08 predictions on every held-out case; report controls. No refit."""
import argparse,json,statistics
from pathlib import Path
import v06_run as common
import v08_model as model

med=statistics.median


def summ(v):
    v=[abs(x) for x in v];return dict(median=med(v),maximum=max(v),n=len(v)) if v else None


def analyze(root,predictions,output):
    common.verify(root);frozen=json.loads(predictions.read_text())
    binding=json.loads((root/'prediction_binding.json').read_text())
    if common.sha(predictions)!=binding['sha256'] or not frozen['frozen_unix_ns']<binding['first_sample_unix_ns']:raise ValueError('freeze binding')
    rows={r['id']:r for r in json.loads((root/'cases.json').read_text())}
    setups={s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
    first=min(json.loads(p.read_text())['host_start_ns'] for cid in frozen['predictions'] for p in (root/'samples'/cid).glob('*.json'))
    if first<=frozen['frozen_unix_ns']:raise ValueError('measurement predates freeze')
    cases=[]
    for cid,p in frozen['predictions'].items():
        m=model.summarize_case(root,rows[cid],setups[cid]);r=rows[cid]
        per=m['per_cta_stamped'];slow=max(range(len(per)),key=per.__getitem__)
        cases.append(dict(case=cid,config=r['config'],kind=r['kind'],m=r['m'],n=r['n'],k=r['k'],swizzle=r['swizzle'],T=m['T'],q=m['q'],fp=m['fp'],
            predicted_us=p['predicted_us'],plain_us=m['plain_us'],error=p['predicted_us']/m['plain_us']-1,plain_cv=m['plain_cv'],
            perturbation=m['perturbation'],ends_perturbation=m['ends_perturbation'],
            predicted_cycles=p['critical_cycles'],c_max_stamped=m['c_max_stamped'],c_max_ends=m['c_max_ends'],
            cycle_error_stamped=p['critical_cycles']/m['c_max_stamped']-1,
            converted_cycle_error_ends=p['converted_cycles']/m['c_max_ends']-1,
            predicted_ghz=p['clock_ghz'],ghz_ends=m['ghz_ends'],clock_error=p['clock_ghz']/m['ghz_ends']-1,
            predicted_supply=p['supply'],measured_supply=m['intervals'].get('P0',0)+m['intervals']['S'],
            predicted_last_epilogue=p['stage_cycles']['last_epilogue'],measured_last_epilogue=m['E_last'],
            predicted_critical_cta=p['critical_cta'],slowest_cta=slow,
            predicted_cta_measured=per[p['critical_cta']],slowest_measured=per[slow],boundary_delta=p['boundary_delta']))
    e=[c['error'] for c in cases]
    result=dict(prediction_sha256=common.sha(predictions),frozen_unix_ns=frozen['frozen_unix_ns'],first_target_process_unix_ns=first,
        complete_time=summ(e),positive=sum(x>0 for x in e),passed=summ(e)['median']<=.05 and summ(e)['maximum']<=.10 and all(c['plain_cv']<=.05 for c in cases),
        per_config={cfg:dict(**summ([c['error'] for c in cases if c['config']==cfg]),positive=sum(c['error']>0 for c in cases if c['config']==cfg)) for cfg in ['cfg_a','cfg_b','cfg_c']},
        cycles_vs_stamped=summ([c['cycle_error_stamped'] for c in cases]),cycles_vs_ends=summ([c['converted_cycle_error_ends'] for c in cases]),
        clock=summ([c['clock_error'] for c in cases]),
        supply=summ([c['predicted_supply']/c['measured_supply']-1 for c in cases]),
        last_epilogue=summ([c['predicted_last_epilogue']/c['measured_last_epilogue']-1 for c in cases]),
        cta_locating_gap=summ([c['slowest_measured']/c['predicted_cta_measured']-1 for c in cases]),
        groups={g:summ([c['error'] for c in cases if f(c)]) for g,f in [('swizzle8',lambda c:c['swizzle']==8),('k_beyond_calibration',lambda c:c['k']>16384),
                ('cfg_a_boundary',lambda c:c['config']=='cfg_a' and c['kind'] in ('oob','partial_odd')),('single_round',lambda c:c['T']==1)]},
        cases=cases)
    summary=json.loads((root/'derived/summary.json').read_text())
    ctrl=[]
    for cid,c in summary.items():
        if c['set']!='ctrl':continue
        warm=summary[cid.replace('_evict','')]
        ctrl.append(dict(case=cid,plain_warm=warm['plain_us'],plain_evict=c['plain_us'],S_warm=warm['intervals']['S'],S_evict=c['intervals']['S'],
                         E_last_warm=warm['E_last'],E_last_evict=c['E_last'],c_max_warm=warm['c_max_stamped'],c_max_evict=c['c_max_stamped'],fp=c['fp']))
    result['l2_control']=ctrl
    output.mkdir(parents=True,exist_ok=False);common.write_json(output/'validation.json',result)
    print('V08 complete time',result['complete_time'],'positive',result['positive'],'passed',result['passed'])
    for k in ['per_config','cycles_vs_stamped','cycles_vs_ends','clock','supply','last_epilogue','cta_locating_gap','groups']:print(k,json.dumps(result[k]))
    for c in cases:print(f"{c['case']:14s} {c['kind']:12s} T{c['T']:3d} pred {c['predicted_us']:9.2f} plain {c['plain_us']:9.2f} e {c['error']*100:+6.2f} cyc {c['cycle_error_stamped']*100:+6.2f} clk {c['clock_error']*100:+6.2f}")
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--predictions',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();analyze(a.run.resolve(),a.predictions.resolve(),a.output.resolve())
