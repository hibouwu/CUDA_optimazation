#!/usr/bin/env python3
"""Rebuild calibration and all errors from raw; test prediction input boundaries."""
from fractions import Fraction as F
import copy
import gzip
import hashlib
import json
from pathlib import Path
import statistics


HERE=Path(__file__).resolve().parent
BASE=Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules')
def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def native(trace):
    rows=[]
    for cta in range(132):
        start=cta*1296; count=trace[start+6]
        assert count==trace[start+9] and count in (1,2)
        for ordinal in range(count):
            a=start+16+10*ordinal; b=a+640
            starts=[trace[a],trace[b]]; ends=[trace[a+1],trace[b+1]]
            first=sorted(range(2),key=lambda i:(starts[i],i))[0]
            last=sorted(range(2),key=lambda i:(-ends[i],i))[0]
            assert trace[a+4:a+6]==trace[b+4:b+6]
            c=ends[last]-starts[first]
            n=trace[[a,b][last]+7]-trace[[a,b][first]+6]
            assert min(c,n)>0
            rows.append(dict(cta=cta,ordinal=ordinal,sm=trace[start+2]-1,cycles=c,ns=n,
                first_wg=first,last_wg=last,work=[trace[a+4]-1,trace[a+5]-1]))
    assert len(rows)==216
    return rows


def raw_calls(name,flag=None):
    root=BASE/name; manifest=read(root/'manifest.json');records=read(root/'returned/records.json')
    assert len(records)==len(manifest['samples'])==20
    result=[]
    for spec,record in zip(manifest['samples'],records):
        assert {key:record[key] for key in spec}==spec and record['returncode']==0
        if flag is not None and record['flag']!=flag: continue
        path=root/'returned'/record['raw'];assert digest(path)==record['sha256']
        with gzip.open(path,'rt') as stream: setup,call,check=map(json.loads,stream)
        assert [setup['event'],call['event'],check['event']]==['setup','call','check']
        assert check['status']=='ok' and check['samples']==4096
        assert setup['producer_tail_observer']==call['producer_tail_observer']==record['flag']==0
        assert call['warmup_calls']==len(call['warmup_us'])==16
        assert call['target_calls']==1 and call['preceding_target_calls']==0
        supplied=dict(setup,binary_sha256=manifest['binary_sha256'],
            prefix_same_a=call['target_a_ptr']==call['prefix_a_ptr'],
            prefix_same_b=call['target_b_ptr']==call['prefix_b_ptr'],
            prefix_calls=call['warmup_calls'],target_calls=call['target_calls'],
            preceding_target_calls=call['preceding_target_calls'])
        result.append((record,supplied,call))
    return result


def summary(values):
    values=list(values)
    return dict(n=len(values),median=float(statistics.median(values)),
        median_absolute=float(statistics.median(abs(v) for v in values)),
        maximum_absolute=float(max(abs(v) for v in values)),minimum=float(min(values)),maximum=float(max(values)),
        rms=(sum(float(v)**2 for v in values)/len(values))**0.5)


def check_row(row,actual,cycles,predicted_rate):
    for key,value in actual.items(): assert row[key]==value
    e_c=cycles/actual['cycles']-1
    e_f=F(actual['cycles'],actual['ns'])/predicted_rate-1
    e_n=cycles/predicted_rate/actual['ns']-1
    assert F(row['cycle_error'])==e_c
    assert F(row['conversion_error'])==e_f
    assert F(row['free_error'])==e_n
    assert F(row['prediction_ns'])==cycles/predicted_rate
    assert (1+e_c)*(1+e_f)==1+e_n


def main():
    contract=read(HERE/'contract.json');calibration=read(HERE/'calibration.json');report=read(HERE/'report.json')
    for name,value in read(HERE/'freeze.json')['files'].items(): assert digest(HERE/name)==value
    assert report['freeze_sha256']==digest(HERE/'freeze.json')
    for name,value in report['input_sha256'].items(): assert digest(Path(name))==value
    assert not any(contract['evaluation_run']+'/' in name for name in calibration['input_sha256'])
    train=raw_calls(contract['calibration_run'],flag=0); assert len(train)==10
    centers=[];expected_setup=calibration['setup']
    for record,setup,call in train:
        assert {key:setup[key] for key in expected_setup}==expected_setup
        prefix=F(str(call['warmup_us'][-1]))*1000;windows=native(call['trace'])
        for ordinal in (0,1):
            values=sorted(F(w['cycles'],w['ns']) for w in windows if w['ordinal']==ordinal)
            middle=(values[(len(values)-1)//2]+values[len(values)//2])/2
            centers.append(dict(pair=record['pair'],ordinal=ordinal,windows=len(values),prefix_ns=str(prefix),
                median_rate=str(middle),kappa_cycles=str(middle*prefix)))
    assert centers==calibration['process_centers']
    kappas=[statistics.median(F(c['kappa_cycles']) for c in centers if c['ordinal']==o) for o in (0,1)]
    common=statistics.median(F(c['kappa_cycles']) for c in centers)
    assert list(map(str,kappas))==calibration['kappa_cycles'] and str(common)==calibration['common_kappa_cycles']
    reference=BASE/contract['reference_run']/'returned';long=[]
    for record in read(reference/'records.json'):
        if record['iterations']!=1024: continue
        path=reference/'samples'/record['id']/'stdout.json'
        value=read(path); assert digest(path)==record['files'][str(path.relative_to(reference))]
        assert value['work_flop']==4294967296 and value['gpu_uuid']==expected_setup['gpu_uuid']
        assert value['unit']=='clock64_cycle/CTA' and value['stamps'][0][4]==124
        long.append(value['elapsed'])
    assert len(long)==3
    static_cycles=F(statistics.median(long)*2*256*128*4096,4294967296)
    assert static_cycles==F(report['static_cycles'])==F(1049379,16)
    with gzip.open(HERE/'window-errors.json.gz','rt') as stream: rows=json.load(stream)
    index={(r['pair'],r['arm'],r['cta'],r['ordinal'],r['method']):r for r in rows}
    forecasts={(r['pair'],r['arm']):r for r in report['forecasts']}
    targets=raw_calls(contract['evaluation_run']);assert len(targets)==20 and len(index)==len(rows)==8640
    event_errors=[];checked=0
    for record,setup,call in targets:
        assert {key:setup[key] for key in expected_setup}==expected_setup
        prefix=F(str(call['warmup_us'][-1])); pair=record['pair'];arm=record['arm']
        rates=[k/(prefix*1000) for k in kappas];common_rate=common/(prefix*1000)
        forecast=forecasts[pair,arm]
        assert forecast['last_prefix_event_us']==str(prefix)
        assert forecast['ordinal_rates']==list(map(str,rates))
        assert forecast['common_rates']==[str(common_rate)]*2
        assert forecast['ordinal_prediction_ns']==[str(static_cycles/r) for r in rates]
        assert forecast['common_prediction_ns']==[str(static_cycles/common_rate)]*2
        event_errors.append(prefix/F(str(call['elapsed_us']))-1)
        for window in native(call['trace']):
            for method,rate in (('common',common_rate),('ordinal',rates[window['ordinal']])):
                row=index[pair,arm,window['cta'],window['ordinal'],method]
                check_row(row,window,static_cycles,rate);checked+=1
    for item in report['summaries']:
        selected=[r for r in rows if r['method']==item['method'] and r['ordinal']==item['ordinal']
            and (item['scope']=='all_SM' or r['sm']==124)]
        for field in ('cycle_error','conversion_error','free_error'): assert summary(F(r[field]) for r in selected)==item[field]
    for item in report['process_summaries']:
        selected=[r for r in rows if all(r[k]==item[k] for k in ('method','pair','arm','ordinal'))]
        for field in ('cycle_error','conversion_error','free_error'): assert summary(F(r[field]) for r in selected)==item[field]
    assert summary(event_errors)==report['event_baseline']
    # Test the callable separately. Unexpected setup reads fail immediately.
    from r09_prefix_ordinal_clock import SETUP_FIELDS,predict_rates
    class RestrictedSetup(dict):
        def __getitem__(self,key):
            if key not in SETUP_FIELDS: raise AssertionError('undeclared prediction input: '+key)
            return super().__getitem__(key)
        def get(self,*args): raise AssertionError('unrestricted setup lookup')
    supplied=RestrictedSetup(expected_setup)
    for field in ('trace','target_cycles','target_ns','elapsed_us','sm_assignment','paired_arm','residual'):
        supplied[field]=object()
    actual=predict_rates(supplied,F(100),calibration)
    assert actual==[k/100000 for k in kappas]
    assert predict_rates(supplied,F(200),calibration)==[v/2 for v in actual]
    assert predict_rates(supplied,F(100),calibration,ordinal_profile=False)==[common/100000]*2
    rejected=[]
    for key,value in [('gpu_uuid','other-card'),('k',1024),('stages',6),('seed',18),
            ('prefix_calls',15),('prefix_same_a',False),('producer_tail_observer',1),('swizzle',8)]:
        bad=dict(expected_setup);bad[key]=value
        try: predict_rates(bad,100,calibration)
        except ValueError: rejected.append(key)
        else: raise AssertionError('unsupported setup accepted: '+key)
    for value in (0,-1,float('nan'),float('inf')):
        try: predict_rates(expected_setup,value,calibration)
        except ValueError: pass
        else: raise AssertionError('invalid prefix state accepted')
    bad=copy.deepcopy(rows[0]);bad['conversion_error'],bad['cycle_error']=bad['cycle_error'],bad['conversion_error']
    assert (1+F(bad['cycle_error']))*(1+F(bad['conversion_error']))==1+F(bad['free_error'])
    try:
        actual={k:rows[0][k] for k in ('cta','ordinal','sm','cycles','ns','first_wg','last_wg','work')}
        prefix=F(forecasts[bad['pair'],bad['arm']]['last_prefix_event_us'])*1000
        check_row(bad,actual,static_cycles,common/prefix)
    except AssertionError: pass
    else: raise AssertionError('swapped closed terms accepted')
    verification=dict(passed=True,input_hashes=len(report['input_sha256']),calibration_processes=10,
        calibration_native_windows=2160,evaluation_processes=20,evaluation_native_windows=4320,
        exact_labeled_errors=checked,process_summary_groups=len(report['process_summaries']),
        prediction_input_boundary='Only frozen static setup, one completed prefix host event and calibration are accessed.',
        unsupported_setup_rejections=rejected,invalid_prefix_rejections=4,
        closed_but_swapped_terms_rejected=True,prospective_holdout=False)
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__': main()
