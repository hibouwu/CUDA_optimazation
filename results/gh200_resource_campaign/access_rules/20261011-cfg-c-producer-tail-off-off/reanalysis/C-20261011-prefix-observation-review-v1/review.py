#!/usr/bin/env python3
"""Offline acceptance review. Preserve old gates; never launch or refit anything."""
from fractions import Fraction as Q
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics as st
import sys

sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent
run=out.parents[1]
base=run.parent
candidate=run/'reanalysis/main-prefix-ordinal-clock-v1'
inputs={}

def sha(path):
    digest=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path)]=digest;return digest

def read(path):
    sha(path);return json.loads(path.read_text())

def stats(values):
    values=list(values)
    return dict(count=len(values),median=float(st.median(values)),
        median_absolute=float(st.median(map(abs,values))),maximum_absolute=float(max(map(abs,values))),
        minimum=float(min(values)),maximum=float(max(values)))

freeze=read(candidate/'freeze.json')
for name,digest in freeze['files'].items():assert sha(candidate/name)==digest
calibration=read(candidate/'calibration.json')
contract=read(candidate/'contract.json')
main_report=read(candidate/'report.json')
assert main_report['observation_qualified'] is main_report['prediction_qualified'] is main_report['prospective_holdout'] is False
assert contract['calibration_run']=='20261011-cfg-c-producer-tail-control'
assert contract['evaluation_run']==run.name
assert contract['native_cycles'].find('1049379/16')>=0
K=Q(calibration['common_kappa_cycles']);C=Q(1049379,16)
assert main_report['static_cycles']==str(C)
spec=importlib.util.spec_from_file_location('pinned_candidate',candidate/'r09_prefix_ordinal_clock.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

source=base/'20261011-cfg-c-producer-tail-check-build/source/probes/r18.cu'
source_manifest=read(base/'20261011-cfg-c-producer-tail-check-build/source_hashes.json')
assert sha(source)==source_manifest['source/probes/r18.cu']
text=source.read_text()
sync=text.index('cudaEventSynchronize(call_end)')
elapsed=text.index('cudaEventElapsedTime(&ms, call_begin, call_end)',sync)
returned=text.index('return ms;',elapsed)
common=text.index('if (common_history) {',returned)
prefix=text.index('warm.push_back(timed(warmup_launch_params));',common)
target=text.index('float ms = timed(launch_params);',prefix)
report_call=text.index('report("warm", ms, warm);',target)
assert sync<elapsed<returned<common<prefix<target<report_call
assert 'std::cout' not in text[common:report_call] and 'std::cin' not in text[common:report_call]
availability=read(run/'reanalysis/main-prefix-availability-v1/verification.json')
assert availability['source_sha256']==sha(source)
assert availability['prefix_values_in_host_before_target'] is True
assert availability['prefix_values_exported_before_target'] is False
assert availability['existing_binary_generates_numeric_prediction'] is False

manifest=read(run/'manifest.json')
records=read(run/'returned/records.json')
assert len(records)==20 and manifest['comparison']=='off_off'
assert [{k:r[k] for k in ('id','pair','flag','command','arm')} for r in records]==manifest['samples']
original=read(run/'analysis.json')
assert original['baseline_consistency_passed'] is original['observation_qualified'] is False
assert sum(not r['passed'] for r in original['local'])==4884
assert sum(not r['passed'] for r in original['local'] if r['series'].endswith('/L'))==432
calibration_snapshot=out/'calibration.json'
calibration_snapshot.write_bytes((candidate/'calibration.json').read_bytes())
assert sha(calibration_snapshot)==sha(candidate/'calibration.json')
processes=[];free_errors=[];strict_selection_disagreements=0;poison_checks=0
for row in records:
    raw=run/'returned'/row['raw'];assert row['returncode']==0 and sha(raw)==row['sha256']
    with gzip.open(raw,'rt') as stream:packets={p['event']:p for p in map(json.loads,stream)}
    setup,call=packets['setup'],packets['call']
    derived=dict(binary_sha256=manifest['binary_sha256'],prefix_same_a=call['prefix_a_ptr']==call['target_a_ptr'],
        prefix_same_b=call['prefix_b_ptr']==call['target_b_ptr'],prefix_calls=call['warmup_calls'],
        target_calls=call['target_calls'],preceding_target_calls=call['preceding_target_calls'])
    static={key:derived[key] if key in derived else setup[key] for key in module.SETUP_FIELDS}
    past=Q(str(call['warmup_us'][-1]));rate=module.predict_rates(static,past,calibration,ordinal_profile=False)
    assert rate==[K/(past*1000)]*2
    forecast=C*past*1000/K
    assert forecast==C/rate[0]
    # Future measurement keys are unreadable and absent from the allowed input.
    class Restricted(dict):
        def __getitem__(self,key):
            assert key in module.SETUP_FIELDS
            return super().__getitem__(key)
    poisoned=Restricted(static)
    for key in ('trace','elapsed_us','target_cycle','target_ns','residual','sm_assignment'):
        poisoned[key]=object()
    assert module.predict_rates(poisoned,past,calibration,ordinal_profile=False)==rate
    poison_checks+=1
    windows={};trace=call['trace']
    assert len(trace)==132*1296 and call['producer_tail_observer']==row['flag']==0
    for cta in range(132):
        h=trace[cta*1296:(cta+1)*1296]
        assert h[6]==h[9] and h[6] in (1,2) and h[2]==h[11]==h[12]
        for ordinal in range(h[6]):
            tiles=[h[16+640*wg+10*ordinal:26+640*wg+10*ordinal] for wg in range(2)]
            assert tiles[0][4:6]==tiles[1][4:6]
            first=min(range(2),key=lambda wg:tiles[wg][0])
            last=max(range(2),key=lambda wg:tiles[wg][1])
            native=tiles[last][1]-tiles[first][0]
            ns=tiles[last][7]-tiles[first][6]
            direct=max(t[7] for t in tiles)-min(t[6] for t in tiles)
            strict_selection_disagreements+=direct!=ns
            key=f'{cta}/{ordinal}/{tiles[0][4]-1},{tiles[0][5]-1}'
            windows[key]=dict(cycles=native,ns=ns,ordinal=ordinal)
            free_errors.append(forecast/ns-1)
    assert len(windows)==216
    processes.append(dict(pair=row['pair'],arm=row['arm'],forecast=forecast,windows=windows,
        prefix_event_us=str(past),raw=str(raw)))
assert strict_selection_disagreements==0 and len(free_errors)==4320
main_forecasts={(x['pair'],x['arm']):x for x in main_report['forecasts']}
for p in processes:
    assert main_forecasts[p['pair'],p['arm']]['common_prediction_ns']==[str(p['forecast'])]*2

series=[]
for key in sorted(processes[0]['windows']):
    observed=[];predicted=[];residual=[]
    for pair in range(10):
        a,b=sorted(processes[2*pair:2*pair+2],key=lambda p:p['arm'])
        assert a['pair']==b['pair']==pair and a['arm']==0 and b['arm']==1
        delta=Q(b['windows'][key]['ns']-a['windows'][key]['ns'])
        forecast=b['forecast']-a['forecast']
        observed.append(delta);predicted.append(forecast);residual.append(delta-forecast)
    def gate(values):
        median=st.median(map(abs,values));maximum=max(map(abs,values))
        return dict(median_absolute_ns=float(median),maximum_absolute_ns=float(maximum),
            diagnostic_within_64_256=median<=64 and maximum<=256)
    series.append(dict(series=key,observed_signed_ns=[str(x) for x in observed],
        forecast_signed_ns=[str(x) for x in predicted],residual_signed_ns=[str(x) for x in residual],
        raw=gate(observed),residual=gate(residual)))
assert len(series)==216
counts={kind:sum(not row[kind]['diagnostic_within_64_256'] for row in series) for kind in ('raw','residual')}
assert counts==dict(raw=215,residual=216)
summaries={kind:dict(series=216,failed=counts[kind],
    median_abs_range_ns=[min(s[kind]['median_absolute_ns'] for s in series),max(s[kind]['median_absolute_ns'] for s in series)],
    maximum_abs_range_ns=[min(s[kind]['maximum_absolute_ns'] for s in series),max(s[kind]['maximum_absolute_ns'] for s in series)])
    for kind in ('raw','residual')}
payload=json.dumps(series,indent=2,allow_nan=False).encode()+b'\n'
(out/'series.json.gz').write_bytes(gzip.compress(payload,mtime=0))
report=dict(input_sha256=inputs,candidate='frozen common kappa only; no ordinal alternative fitted or selected',
    formula='N_hat=C_static*last_prefix_event_ns/kappa_common; residual=(N_arm1-N_arm0)-(N_hat_arm1-N_hat_arm0)',
    static_cycles=str(C),common_kappa_cycles=str(K),prefix_only_input_checks=poison_checks,
    scored_common_windows=4320,matched_common_series=216,diagnostic_summaries=summaries,
    common_free_prediction_error=stats(free_errors),strict_cycle_vs_ns_extrema_disagreements=0,
    original_roles_L_failed=432,original_all_series_failed=4884,original_gate_rejudged=False,
    numeric_predictions_committed_before_target=False,prefix_host_available_before_target=True,
    prospective_holdout=False,observation_qualified=False,prediction_qualified=False,new_gpu_runs_by_C=0,
    conclusions=['Input isolation and static source availability are established; numerical precommit before target was not performed.',
        'Pre-freezing this deterministic rule can support a new test of prediction of the recorded common window, conditional on this exact protocol.',
        'It cannot certify local marker accuracy, base observer perturbation, output subparts, critical CTA, complete-event composition or cross-card transfer.',
        'On revealed off/off data, all 216 common-window residual series exceed the unchanged absolute diagnostic thresholds; the prefix predictor has not controlled baseline variation at that scale.',
        'These residuals are a labeled diagnostic, not a corrected observation, new acceptance gate or recovery of any old failure.',
        'Minimum missing boundary evidence is a same-call per-endpoint read-time bracket independent of the fitted conversion; see protocol.txt.'])
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','conclusions')},indent=2))
