#!/usr/bin/env python3
"""Matched CTA and envelope accounting, with explicitly conditioned diagnostics.

No fitting, candidate adoption, target-input prediction, or GPU execution.
Direct globaltimer intervals are retained as the original observer's evidence.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics as st

from analyze_r13_sm import select_attempts
from analyze_r18 import replay
import v06_run
import v09_long_k_rejected as v09_model


def read(path):
    return json.loads(path.read_text())


def partition(times, entry, end):
    """An exact signed partition; inter-tile gaps may encode overlap."""
    if not times:
        raise ValueError('inactive CTA has no partition')
    parts = dict(prefix=times[0][0]-entry,
                 first_L=times[0][1]-times[0][0],
                 later_L=sum(t[1]-t[0] for t in times[1:]),
                 permits=sum(t[2]-t[1] for t in times),
                 epilogues=sum(t[3]-t[2] for t in times),
                 signed_intertile_gaps=sum(b[0]-a[3] for a, b in zip(times,times[1:])),
                 tail=end-times[-1][3])
    if abs(sum(parts.values())-(end-entry)) > 1e-5:
        raise ValueError('CTA partition identity failed')
    return parts


def series(values):
    if not values:
        return dict(count=0, median=None, minimum=None, maximum=None)
    if any(not math.isfinite(v) for v in values):
        raise ValueError('nonfinite diagnostic')
    return dict(count=len(values), median=st.median(values),
                minimum=min(values), maximum=max(values))


def condition_frequency(calibration, frequency):
    """An observation-conditioned diagnostic, never a deployable candidate."""
    cal = copy.deepcopy(calibration)
    cal['clock']['coefficients'] = [frequency]+[0.0]*6
    return cal


def predict_partition(prediction, index):
    return partition(prediction['cta_events'][index]['events'], 0,
                     prediction['cta_duration_ns'][index])


def process_accounting(ctas, prediction, conditioned):
    if len(ctas) != len(prediction['cta_duration_ns']):
        raise ValueError('CTA count mismatch')
    active = [c for c in ctas if c['tiles_ns']]
    if any(c['cta'] != i for i,c in enumerate(ctas)) or not active:
        raise ValueError('CTA identities or active work invalid')
    earliest = min(c['entry_ns'] for c in ctas)
    latest = max(active, key=lambda c:c['end_ns'])
    longest = max(active, key=lambda c:c['end_ns']-c['entry_ns'])
    envelope = latest['end_ns']-earliest
    maximum_tiles = max(len(c['tiles_ns']) for c in active)
    cohort = [c for c in active if len(c['tiles_ns']) == maximum_tiles]
    tied = [c for c in active if c['cta'] in prediction['critical_candidates']]
    selected = ctas[prediction['critical_cta']]
    observed_parts = {c['cta']:partition(c['tiles_ns'],c['entry_ns'],c['end_ns']) for c in active}
    actual_durations = {c['cta']:c['end_ns']-c['entry_ns'] for c in active}
    actual_frequency = {c['cta']:(c['end_c']-c['entry_c'])/actual_durations[c['cta']] for c in active}
    versions = {}
    for name, model in (('r2_free',prediction),('measured_frequency_conditioned',conditioned)):
        model_envelope = model['window_us']*1000
        predicted_parts = {c['cta']:predict_partition(model,c['cta']) for c in active}
        details = {}
        for label, cta in (('representative',selected),('observed_last',latest),('longest_duration',longest)):
            index = cta['cta']
            residual = {k:observed_parts[index][k]-predicted_parts[index][k] for k in observed_parts[index]}
            local_residual = actual_durations[index]-model['cta_duration_ns'][index]
            if abs(sum(residual.values())-local_residual) > 1e-5:
                raise ValueError('residual partition identity failed')
            details[label] = dict(cta=index, sm=cta['sm'], tiles=len(cta['tiles_ns']),
                native_cycles=cta['end_c']-cta['entry_c'], interval_ghz=actual_frequency[index],
                actual_duration_ns=actual_durations[index], predicted_duration_ns=model['cta_duration_ns'][index],
                entry_offset_ns=cta['entry_ns']-earliest, end_regret_ns=latest['end_ns']-cta['end_ns'],
                observed_parts_ns=observed_parts[index], predicted_parts_ns=predicted_parts[index],
                signed_observed_minus_predicted_parts_ns=residual)
        index = latest['cta']
        decomposition = dict(local_duration_residual_ns=actual_durations[index]-model['cta_duration_ns'][index],
            entry_offset_ns=latest['entry_ns']-earliest,
            predicted_selected_minus_envelope_ns=model['cta_duration_ns'][index]-model_envelope)
        if abs(sum(decomposition.values())-(envelope-model_envelope)) > 1e-5:
            raise ValueError('last-CTA envelope decomposition failed')
        groups = {}
        for label, group in (('all_active',active),('max_tile_count',cohort),('predicted_max_set',tied)):
            groups[label] = dict(ctas=len(group),
                duration_ns=series([actual_durations[c['cta']] for c in group]),
                interval_ghz=series([actual_frequency[c['cta']] for c in group]),
                native_cycles=series([c['end_c']-c['entry_c'] for c in group]),
                signed_parts_residual_ns={k:series([observed_parts[c['cta']][k]-predicted_parts[c['cta']][k] for c in group])
                    for k in observed_parts[active[0]['cta']]})
        versions[name] = dict(frequency_ghz=model['frequency_ghz'],envelope_ns=model_envelope,
            plain_us=model['plain_us'], envelope_error_pct=100*(model_envelope/envelope-1),
            observed_last_decomposition=decomposition, details=details, groups=groups)
    return dict(observed_envelope_ns=envelope,
        observed_maximum_cta_duration_ns=actual_durations[longest['cta']],
        entry_envelope_extra_ns=envelope-actual_durations[longest['cta']],
        observed_last=latest['cta'], longest_duration_cta=longest['cta'],
        entry_spread_ns=max(c['entry_ns'] for c in ctas)-earliest,
        observed_last_in_predicted_max_set=latest['cta'] in prediction['critical_candidates'],
        representative_end_regret_ns=latest['end_ns']-selected['end_ns'],
        best_predicted_set_end_regret_ns=latest['end_ns']-max(c['end_ns'] for c in tied),
        versions=versions)


def analyze_run(run, rows, calibration, source_label, provenance):
    setups = {s['case']:s['setup'] for s in read(run/'static_setup.json')}
    for name in ('cases.json','static_setup.json','environment.json'):
        provenance[str(run/name)] = v06_run.sha(run/name)
    cases = []
    for row in rows:
        prediction = v09_model.predict_components(row,setups[row['id']],calibration)
        static_work = v09_model.events.scheduled_work(row['config'],row['m'],row['n'],setups[row['id']]['grid'],row['swizzle'])
        selected,failed,_ = select_attempts(run,row['id'],('plain','dual'))
        plain = [i for i in selected if i['record']['variant']=='plain']
        dual = [i for i in selected if i['record']['variant']=='dual']
        if len(plain) != 10 or len(dual) != 10:
            raise ValueError('ten complete plain and dual processes required: '+row['id'])
        for item in plain:
            replay(run,item['record'],row)
            provenance[str(run/item['path'])] = v06_run.sha(run/item['path'])
            provenance[str(run/item['record']['raw'])] = item['record']['raw_sha256']
        processes = []
        for item in dual:
            record = item['record'];ctas = replay(run,record,row)['ctas']
            if [c['work'] for c in ctas] != static_work:
                raise ValueError('static schedule differs from recorded work')
            maximum_tiles = max(len(c['tiles_ns']) for c in ctas)
            frequency = st.median((c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns'])
                for c in ctas if len(c['tiles_ns'])==maximum_tiles)
            conditioned = v09_model.predict_components(row,setups[row['id']],condition_frequency(calibration,frequency))
            processes.append(dict(trial=record['trial'],dual_event_us=record['elapsed_us'],
                **process_accounting(ctas,prediction,conditioned)))
            provenance[str(run/item['path'])] = v06_run.sha(run/item['path'])
            provenance[str(run/record['raw'])] = record['raw_sha256']
        actual_plain = st.median(i['record']['elapsed_us'] for i in plain)
        actual_envelope = st.median(p['observed_envelope_ns'] for p in processes)/1000
        transfer = calibration['plain_transfer'][row['config']]
        actual_envelope_transfer = transfer['F_us']+transfer['kappa']*actual_envelope
        summary = dict(plain_median_us=actual_plain,plain_cv_pct=100*st.pstdev(i['record']['elapsed_us'] for i in plain)/st.mean(i['record']['elapsed_us'] for i in plain),
            envelope_median_us=actual_envelope,
            entry_envelope_extra_ns=series([p['entry_envelope_extra_ns'] for p in processes]),
            representative_end_regret_ns=series([p['representative_end_regret_ns'] for p in processes]),
            best_predicted_set_end_regret_ns=series([p['best_predicted_set_end_regret_ns'] for p in processes]),
            last_in_predicted_max_set_count=sum(p['observed_last_in_predicted_max_set'] for p in processes),
            observed_envelope_through_frozen_plain_map_us=actual_envelope_transfer,
            observed_envelope_through_frozen_plain_map_error_pct=100*(actual_envelope_transfer/actual_plain-1),versions={})
        for version in ('r2_free','measured_frequency_conditioned'):
            values=[p['versions'][version] for p in processes]
            critical_parts = {k:series([v['details']['observed_last']['signed_observed_minus_predicted_parts_ns'][k] for v in values])
                for k in values[0]['details']['observed_last']['observed_parts_ns']}
            summary['versions'][version] = dict(envelope_error_pct=series([v['envelope_error_pct'] for v in values]),
                plain_error_pct=100*(st.median(v['plain_us'] for v in values)/actual_plain-1),
                frequency_ghz=series([v['frequency_ghz'] for v in values]),
                observed_last_parts_residual_ns=critical_parts,
                observed_last_decomposition_ns={k:series([v['observed_last_decomposition'][k] for v in values])
                    for k in values[0]['observed_last_decomposition']})
        cases.append(dict(case=row['id'],row=row,source=source_label,setup=setups[row['id']],
            all_supply_supported=prediction['all_supply_supported'],failed_attempts_retained=failed,
            summary=summary,processes=processes))
        print(row['id'], 'entry extra ns',summary['entry_envelope_extra_ns']['median'],
            'conditioned envelope error %',summary['versions']['measured_frequency_conditioned']['envelope_error_pct']['median'],flush=True)
    return cases


def run(base, output):
    freeze=base/'Q15-full-time/V09-component-validation/20261009-V09-freeze-job738972-r2'
    observed=base/'Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011'
    calibration=read(freeze/'frozen/calibration.json')
    provenance={str(freeze/'frozen/calibration.json'):v06_run.sha(freeze/'frozen/calibration.json')}
    references=[r for r in read(freeze/'cases.json') if r['set']=='calib']
    targets=[r for r in read(observed/'cases.json') if r['set']=='heldout']
    if len(references)!=6 or len(targets)!=30:
        raise ValueError('fixed six calibration and thirty development identities changed')
    output.mkdir(parents=True,exist_ok=False)
    cases=analyze_run(freeze,references,calibration,'same-protocol reference',provenance)
    cases+=analyze_run(observed,targets,calibration,'V09 development including original rejection and range-out conditions',provenance)
    result=dict(scope='Development attribution; no fit or adoption. Measured-frequency substitution is conditional diagnostic, never a formal prediction input. '
        'Local pieces use the original merged dual globaltimer observer, not independent hardware service. Signed intertile gaps retain overlap. '
        'Per-process identities hold; separate medians need not add. Plain and dual observations are distinct calls.',
        gpu_jobs=0, cases=cases,input_sha256=provenance,
        analysis_sha256=v06_run.sha(Path(__file__)),
        implementation_sha256={str(Path(module.__file__).resolve()):v06_run.sha(Path(module.__file__))
            for module in (v09_model,v09_model.events,v09_model.events.base,v09_model.clock,v09_model.supply)},
        selector_invariant='The envelope callback uses max of all CTA durations; representative selection happens afterwards. Tie-only changes cannot alter frequency, envelope or plain time.')
    (output/'attribution.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'source').mkdir()
    (output/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('Completed',len(cases),'cases and',sum(len(c['processes']) for c in cases),'dual processes')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.base.resolve(),args.output.resolve())
