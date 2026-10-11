#!/usr/bin/env python3
"""Calibrate with one old batch, freeze, then score separate revealed processes."""
import argparse
from collections import defaultdict
from fractions import Fraction as Q
import gzip
import hashlib
import json
from pathlib import Path
import statistics as st

from r09_prefix_ordinal_clock import SETUP_FIELDS, predict_rates


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def common_windows(trace):
    assert len(trace) == 132*1296
    result=[]
    for cta in range(132):
        offset=cta*1296; head=trace[offset:offset+16]
        assert head[6]==head[9] and head[6] in (1,2)
        assert head[10]==0 and head[11]==head[12]==head[2] and 1<=head[2]<=132
        for ordinal in range(head[6]):
            slots=[trace[offset+16+wg*640+ordinal*10:offset+16+wg*640+ordinal*10+10] for wg in (0,1)]
            assert slots[0][4:6]==slots[1][4:6]
            first=min(range(2),key=lambda wg:slots[wg][0])
            last=max(range(2),key=lambda wg:slots[wg][1])
            cycles=slots[last][1]-slots[first][0]
            ns=slots[last][7]-slots[first][6]
            assert cycles>0 and ns>0 and all(v>0 for slot in slots for v in slot)
            result.append(dict(cta=cta,sm=head[2]-1,ordinal=ordinal,cycles=cycles,ns=ns,
                first_wg=first,last_wg=last,work=[v-1 for v in slots[0][4:6]]))
    assert len(result)==216 and sum(r['ordinal']==0 for r in result)==132
    return result


def setup_input(setup,call,binary_sha):
    extra=dict(binary_sha256=binary_sha,prefix_same_a=call['prefix_a_ptr']==call['target_a_ptr'],
        prefix_same_b=call['prefix_b_ptr']==call['target_b_ptr'],prefix_calls=call['warmup_calls'],
        target_calls=call['target_calls'],preceding_target_calls=call['preceding_target_calls'])
    merged=dict(setup,**extra)
    return {key:merged[key] for key in SETUP_FIELDS}


def load_run(base,name,provenance,flag=None):
    root=base/name; returned=root/'returned'; manifest=read(root/'manifest.json')
    assert manifest['status']=='frozen' and manifest['gpu_uuid']=='GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'
    for relative,digest in read(root/'freeze.json')['files'].items():
        for path in (root/relative,returned/relative):
            assert sha(path)==digest; provenance[str(path)]=digest
    receipt=read(root/'receipt.json')
    assert receipt['returncode']==0 and sha(root/'output.tar.gz')==receipt['sha256']
    for path in (root/'freeze.json',root/'receipt.json',root/'output.tar.gz',returned/'records.json',root/'summary.json'):
        provenance[str(path)]=sha(path)
    source=base/'20261011-cfg-c-producer-tail-check-build/source/probes/r18.cu'
    source_manifest=root/'origin-source_hashes.json'
    assert sha(source)==read(source_manifest)['source/probes/r18.cu']
    provenance[str(source)]=sha(source)
    text=source.read_text()
    # Per-call event is returned before prefix insertion and the target launch.
    synchronize=text.index('cudaEventSynchronize(call_end)')
    elapsed=text.index('cudaEventElapsedTime(&ms, call_begin, call_end)',synchronize)
    returned_ms=text.index('return ms;',elapsed)
    common=text.index('if (common_history) {',returned_ms)
    prefix_insert=text.index('warm.push_back(timed(warmup_launch_params));',common)
    target=text.index('float ms = timed(launch_params);',prefix_insert)
    assert synchronize<elapsed<returned_ms<common<prefix_insert<target
    records=read(returned/'records.json'); assert len(records)==20
    selected=[]
    for sample,record in zip(manifest['samples'],records):
        assert {key:record[key] for key in sample}==sample and record['returncode']==0
        if flag is not None and record['flag']!=flag: continue
        path=returned/record['raw']; assert sha(path)==record['sha256'];provenance[str(path)]=sha(path)
        with gzip.open(path,'rt') as stream: packets=[json.loads(line) for line in stream if line.strip()]
        assert [p['event'] for p in packets]==['setup','call','check']
        setup,call,check=packets
        assert check['status']=='ok' and check['samples']==4096 and check['nonfinite']==check['padding_errors']==0
        assert setup['producer_tail_observer']==call['producer_tail_observer']==record['flag']==0
        assert call['warmup_calls']==len(call['warmup_us'])==16 and call['producer_tail_prefix_flag']==0
        assert setup['gpu_uuid']==manifest['gpu_uuid'] and setup['config']=='cfg_c'
        selected.append(dict(pair=record['pair'],arm=record.get('arm',record['flag']),raw=str(path),
            setup=setup_input(setup,call,manifest['binary_sha256']),call=call))
    assert len(selected)==(10 if flag is not None else 20)
    return selected


def stats(values):
    values=list(values)
    return dict(n=len(values),median=float(st.median(values)),median_absolute=float(st.median(map(abs,values))),
        maximum_absolute=float(max(map(abs,values))),minimum=float(min(values)),maximum=float(max(values)),
        rms=(sum(float(v)**2 for v in values)/len(values))**0.5)


def calibrate(base,out,contract):
    assert not (out/'freeze.json').exists()
    provenance={}; calls=load_run(base,contract['calibration_run'],provenance,flag=0)
    setup=calls[0]['setup']; centers=[]
    for item in calls:
        assert item['setup']==setup
        prefix=Q(str(item['call']['warmup_us'][-1]))*1000
        rows=common_windows(item['call']['trace'])
        for ordinal in (0,1):
            rates=[Q(r['cycles'],r['ns']) for r in rows if r['ordinal']==ordinal]
            centers.append(dict(pair=item['pair'],ordinal=ordinal,windows=len(rates),prefix_ns=str(prefix),
                median_rate=str(st.median(rates)),kappa_cycles=str(prefix*st.median(rates))))
    calibration=dict(schema='r09-prefix-ordinal-clock-calibration/1',setup=setup,
        kappa_cycles=[str(st.median(Q(r['kappa_cycles']) for r in centers if r['ordinal']==o)) for o in (0,1)],
        common_kappa_cycles=str(st.median(Q(r['kappa_cycles']) for r in centers)),
        process_centers=centers,input_sha256=provenance,calibration_processes=10,
        observation_qualified=False,prediction_qualified=False)
    write(out/'calibration.json',calibration)
    names=['contract.json','calibration.json','r09_prefix_ordinal_clock.py','analyze.py']
    write(out/'freeze.json',dict(files={name:sha(out/name) for name in names},
        status='development scoring freeze only; not a new independently acquired holdout'))
    print(json.dumps(dict(kappa_cycles=calibration['kappa_cycles'],common_kappa_cycles=calibration['common_kappa_cycles'],
        calibration_processes=10,input_hashes=len(provenance)),indent=2))


def evaluate(base,out,contract):
    assert not (out/'report.json').exists()
    for name,digest in read(out/'freeze.json')['files'].items(): assert sha(out/name)==digest
    calibration=read(out/'calibration.json'); provenance=dict(calibration['input_sha256'])
    reference=base/contract['reference_run']/'reanalysis/A-cycle-time-transfer-v1/transfer.json'
    transfer=read(reference)
    for name,digest in transfer['input_sha256'].items():
        assert sha(Path(name))==digest;provenance[name]=digest
    provenance[str(reference)]=sha(reference)
    static_cycles=Q(1049379,16)
    calls=load_run(base,contract['evaluation_run'],provenance)
    rows=[];event_rows=[];forecasts=[]
    for item in calls:
        prefix=Q(str(item['call']['warmup_us'][-1]))
        # Forecast both ordinals before decoding any target trace or scoring event.
        ordinal_rates=predict_rates(item['setup'],prefix,calibration)
        common_rates=predict_rates(item['setup'],prefix,calibration,ordinal_profile=False)
        forecasts.append(dict(pair=item['pair'],arm=item['arm'],last_prefix_event_us=str(prefix),
            ordinal_rates=[str(v) for v in ordinal_rates],common_rates=[str(v) for v in common_rates],
            ordinal_prediction_ns=[str(static_cycles/v) for v in ordinal_rates],
            common_prediction_ns=[str(static_cycles/v) for v in common_rates]))
        event_rows.append(dict(pair=item['pair'],arm=item['arm'],last_prefix_event_us=str(prefix),
            target_event_us=str(item['call']['elapsed_us']),error=str(prefix/Q(str(item['call']['elapsed_us']))-1)))
        for native in common_windows(item['call']['trace']):
            for method,rates in (('common',common_rates),('ordinal',ordinal_rates)):
                rate=rates[native['ordinal']]; cycle_error=static_cycles/native['cycles']-1
                conversion_error=Q(native['cycles'],native['ns'])/rate-1
                free_error=static_cycles/rate/native['ns']-1
                assert (1+cycle_error)*(1+conversion_error)==1+free_error
                rows.append(dict(pair=item['pair'],arm=item['arm'],method=method,**native,
                    cycle_error=str(cycle_error),conversion_error=str(conversion_error),free_error=str(free_error),
                    prediction_ns=str(static_cycles/rate)))
    groups=defaultdict(list)
    for row in rows:
        for scope in ('all_SM', 'SM124') if row['sm']==124 else ('all_SM',):
            groups[row['method'],scope,row['ordinal']].append(row)
    summaries=[dict(method=method,scope=scope,ordinal=ordinal,
        **{field:stats(Q(r[field]) for r in values) for field in ('cycle_error','conversion_error','free_error')})
        for (method,scope,ordinal),values in sorted(groups.items())]
    process_summaries=[]
    for method in ('common','ordinal'):
        for pair in range(10):
            for arm in (0,1):
                for ordinal in (0,1):
                    values=[r for r in rows if (r['method'],r['pair'],r['arm'],r['ordinal'])==(method,pair,arm,ordinal)]
                    process_summaries.append(dict(method=method,pair=pair,arm=arm,ordinal=ordinal,
                        **{field:stats(Q(r[field]) for r in values) for field in ('cycle_error','conversion_error','free_error')}))
    result=dict(schema='r09-prefix-ordinal-clock-evaluation/1',input_sha256=provenance,
        freeze_sha256=sha(out/'freeze.json'),reference_transfer_sha256=sha(reference),
        static_cycles=str(static_cycles),calibration_processes=10,evaluation_processes=20,
        native_windows=4320,error_rows=len(rows),summaries=summaries,process_summaries=process_summaries,
        event_baseline=stats(Q(r['error']) for r in event_rows),events=event_rows,forecasts=forecasts,
        observation_qualified=False,prediction_qualified=False,prospective_holdout=False,
        limits=['Only fixed cfg_c/K4096/S4/compact dyadic seed17/sw1/common16 observer-off/same-card/same-binary protocol.',
            'Both batches and the later reference were already known. Frozen scoring separation does not create prospective independence.',
            'No current target cycle, ns, event, SM assignment, residual, or paired arm is a predictor input.',
            'Target cycle is used only in labeled conditional error; the free ns forecast uses independent static native cycles.',
            'The original local observer qualification failed. This profile cannot lift it or prove physical-rate identification.',
            'No full-event composition, other configuration/history/K, cross-card support or GPU authorization follows.'])
    write(out/'report.json',result)
    with gzip.open(out/'window-errors.json.gz','wt') as stream: json.dump(rows,stream)
    print(json.dumps(dict(summaries=summaries,event_baseline=result['event_baseline']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['calibrate','evaluate'])
    parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args();output=Path(__file__).resolve().parent
    contract=read(output/'contract.json')
    (calibrate if args.phase=='calibrate' else evaluate)(args.base.resolve(),output,contract)
