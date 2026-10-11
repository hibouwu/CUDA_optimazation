#!/usr/bin/env python3
"""Retrospective transfer of the unchanged prior R09 event multiplier."""
import argparse
from collections import defaultdict
from fractions import Fraction
import gzip,hashlib,json,math
from pathlib import Path
import statistics as st


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def metrics(values):
    return dict(n=len(values),median=st.median(values),median_absolute=st.median(abs(x) for x in values),
                maximum_absolute=max(abs(x) for x in values),rms=math.sqrt(st.fmean(x*x for x in values)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();base=a.base.resolve();out=a.output.resolve();out.mkdir(exist_ok=True)
    if (out/'transfer.json').exists(): raise ValueError('new output directory required')
    source=base/'20261010-R09-history-series/reanalysis/A-static-history-equivalence-v2'
    coefficient=read(source/'equivalence.json')
    ratio=Fraction(coefficient['first_switch_oracle']['fitted_static_ratio'])
    provenance={str(p):sha(p) for p in (source/'equivalence.json',source/'event-identity.json.gz')}
    root=base/'20261010-R09-common-history';history=base/'20261010-R09-history-series'
    summary_file=base/'20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-common-history.json'
    summary=read(summary_file);provenance[str(summary_file)]=sha(summary_file)
    cases={r['id']:r for r in read(root/'cases.json')}
    setups={s['case']:s['setup'] for s in read(root/'static_setup.json')}
    for path in (root/'cases.json',root/'static_setup.json',root/'environment.json'):
        provenance[str(path)]=sha(path)
    code=root/'source/probes/r18.cu';source_manifest=root/'source_hashes.json'
    if sha(code)!=read(source_manifest)['source/probes/r18.cu']: raise ValueError('runtime source mismatch')
    text=code.read_text()
    assert text.index('return ms;')<text.index('for (int i = 0; i < 16; ++i) warm.push_back(timed(warmup_launch_params));')<text.index('float ms = timed(launch_params);')
    for path in (code,source_manifest): provenance[str(path)]=sha(path)
    binary_identity={};sass_identity={}
    for run in (root,history):
        binary=run/'build/cfg_b_ends';sass=run/'build/cfg_b_ends.sass'
        bm=run/'build/binary_hashes.json';sm=run/'build/sass_hashes.json'
        if sha(binary)!=read(bm)['build/cfg_b_ends'] or sha(sass)!=read(sm)['cfg_b_ends.sass']:
            raise ValueError('binary/SASS archive mismatch')
        binary_identity[run.name]=sha(binary);sass_identity[run.name]=sha(sass)
        for path in (binary,sass,bm,sm):provenance[str(path)]=sha(path)
    if len(set(sass_identity.values()))!=1: raise ValueError('device code changed')
    groups=defaultdict(list);rows=[]
    def score(case,trial,index,dataset,previous_swizzle,current_swizzle,previous_us,actual_us,mode):
        if (previous_swizzle,current_swizzle) not in ((1,1),(1,8),(8,8)):
            raise ValueError('candidate does not define this transition')
        multiplier=ratio if (previous_swizzle,current_swizzle)==(1,8) else Fraction(1)
        checkpoint=Fraction(str(previous_us))
        prediction=multiplier*checkpoint
        target=Fraction(str(actual_us));error=prediction/target-1
        row=dict(case=case,trial=trial,target_index=index,dataset=dataset,mode=mode,
            previous_swizzle=previous_swizzle,current_swizzle=current_swizzle,
            previous_event_us=str(checkpoint),actual_event_us=str(target),multiplier=str(multiplier),
            prediction_us=str(prediction),error=str(error),error_float=float(error),
            previous_call_error=float(checkpoint/target-1))
        rows.append(row);groups[dataset,case].append(row)
    with gzip.open(source/'event-identity.json.gz','rt') as stream:old=json.load(stream)
    for p in old:
        current=p['swizzle'];previous=1 if p['target_index']==0 else current
        score(p['case'],p['trial'],p['target_index'],'prior_development_run',previous,current,
              p['prediction_us'],p['actual_us'],p['mode'])
    raw_count=0;common_sw8=[]
    for p in summary['processes']:
        case=cases[p['case']];setup=setups[p['case']]
        if (case['config'],case['m'],case['n'],case['k'],setup['stages'],setup['grid'],setup['gpu_uuid'])!=(
            'cfg_b',6144,8192,32768,6,[132,1,1],'GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114'):
            raise ValueError('transfer support changed')
        path=root/'samples'/p['case']/('ends-%02d.json'%p['trial']);rec=read(path);raw=root/rec['raw']
        if sha(raw)!=rec['raw_sha256'] or rec['raw_sha256']!=p['raw_sha256']:
            raise ValueError('raw identity mismatch')
        with gzip.open(raw,'rt') as stream:events={q['event']:q for line in stream if line.strip() for q in [json.loads(line)]}
        call=events['call'];check=events['check']
        if rec['warmup']!=call['warmup_us'] or rec['elapsed_us']!=call['elapsed_us'] or p['event_us']!=call['elapsed_us']:
            raise ValueError('raw/host event mismatch')
        if check['status']!='ok' or check['samples']!=4096 or check['nonfinite'] or check['padding_errors']:
            raise ValueError('original numerical check did not pass')
        common=p['policy']=='common-sw1-16'
        if common and (len(rec['warmup'])!=16 or call['warmup_swizzle']!=1):
            raise ValueError('common history changed')
        previous=1 if common else p['swizzle']
        score(p['case'],p['trial'],0,'cross_run_development',previous,p['swizzle'],
              rec['warmup'][-1],rec['elapsed_us'],'single')
        if common and p['swizzle']==8:common_sw8.append(rows[-1])
        for x in (path,raw):provenance[str(x)]=sha(x)
        raw_count+=1
    if (len(rows),raw_count,len(common_sw8))!=(1200,40,10): raise ValueError('complete sample count changed')
    output=dict(schema='gh200-r09-switch-multiplier-transfer/1',
        status='failed_development_transfer_not_adopted',
        coefficient=dict(value=str(ratio),origin=str(source/'equivalence.json'),training_first_switch_targets=20,
            origin_document_commit='9faf43aa79a159b2b4d6787e90270f34f85894d4',changed=False),
        availability=dict(checkpoint='Previous completed CUDA event; returned before target launch.',
            current_target_W_f_used=False,role_checkpoints_used=False),
        support='Diagnostic only: cfg_b G6 on GPU-ef, original ends event, known (1,1)/(1,8)/(8,8) transitions. Other configurations, unknown history and plain/cross-card prediction remain unsupported.',
        verification=dict(prior_target_scores=1160,cross_run_raw_hashes=40,cross_run_original_sample_checks=40,
            cross_run_targets=40,all_target_scores=len(rows),device_sass_identical=True,
            host_binary_identical=len(set(binary_identity.values()))==1),
        binary_sha256=binary_identity,sass_sha256=sass_identity,
        groups=[dict(dataset=d,case=c,unmodified_previous_call=metrics([r['previous_call_error'] for r in values]),
                     multiplier_rule=metrics([r['error_float'] for r in values])) for (d,c),values in groups.items()],
        common_sw8_cross_run=metrics([r['error_float'] for r in common_sw8]),
        common_sw8_worst=max(common_sw8,key=lambda r:abs(Fraction(r['error']))),
        input_sha256=provenance,analysis_sha256=sha(Path(__file__)),
        limits=['All input timing data were previously available: this is retrospective transfer, never a new independent frozen holdout.',
            'The multiplier was unchanged from the prior turn, but no prospective validation claim follows.',
            'Host binaries/protocols differ while archived device SASS is byte-identical; no unique cause is assigned to the cross-run residual.',
            'The original 1160 scores retain sw1, same/mirror, first/continuation and single controls. Good first-switch training scores do not qualify the whole history family.',
            'No B source definition/price, public model, runner, GPU allocation or previous qualification changed.'])
    (out/'transfer.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    with gzip.open(out/'event-scores.json.gz','wt') as stream:json.dump(rows,stream,allow_nan=False)
    print(json.dumps(dict(verification=output['verification'],cross_run_common_sw8=output['common_sw8_cross_run'],worst=output['common_sw8_worst']),indent=2))


if __name__=='__main__':main()
