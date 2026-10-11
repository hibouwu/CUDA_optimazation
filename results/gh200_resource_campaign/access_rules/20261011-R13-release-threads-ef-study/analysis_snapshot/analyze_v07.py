#!/usr/bin/env python3
"""Score every frozen V07 case; never refit or remove a valid slow observation."""
import argparse,json,statistics
from collections import Counter
from pathlib import Path
import v06_fit
import v06_model as base
import v06_run as common
import v07_model
from analyze_r18 import replay


def error(pred,meas):return (pred-meas)/abs(meas) if meas else None

def summary(values):
    values=[abs(v) for v in values if v is not None]
    return dict(median=statistics.median(values),maximum=max(values)) if values else None

def analyze(root,predictions,output):
    common.verify(root);frozen=json.loads(predictions.read_text());binding=json.loads((root/'prediction_binding.json').read_text())
    if common.sha(predictions)!=binding['sha256']:raise ValueError('prediction binding changed')
    rows={r['id']:r for r in json.loads((root/'cases.json').read_text())};results=[];first=None;checked=0;rejected=[]
    for cid,pred in frozen['predictions'].items():
        row=rows[cid];observed=[]
        for path in sorted((root/'samples'/cid).glob('*.json')):
            record=json.loads(path.read_text())
            first=record['host_start_ns'] if first is None else min(first,record['host_start_ns'])
            if record['returncode']:
                if 'last five warmups CV exceeds 2%' in record['stderr']:rejected.append(dict(path=str(path.relative_to(root)),record=record));continue
                raise ValueError('unresolved numeric/execution failure')
            rec=replay(root,record,row);checked+=rec['checked_values'];observed.append(rec)
        plain=[r['elapsed_us'] for r in observed if r['variant']=='plain'];trace=[r['elapsed_us'] for r in observed if r['variant']=='stamped']
        if len(plain)<10 or len(trace)<10:raise ValueError('missing V07 sample')
        measured=statistics.median(plain);perturbation=statistics.median(trace)/measured-1
        pcv=statistics.pstdev(plain)/statistics.mean(plain);tcv=statistics.pstdev(trace)/statistics.mean(trace)
        parts={k:[] for k in ['supply','mainloop','last_epilogue','tail','handoff_interval']};maxima=[];entry=[];selected_durations=[];max_ids=[];last_ids=[]
        for rec in observed:
            if rec['variant']!='stamped':continue
            ctas=rec['ctas'];critical=ctas[pred['critical_cta']];tiles=critical['tiles']
            if [c['work'] for c in ctas]!=[[tuple(w) for w in ws] for ws in pred['work']]:raise ValueError('frozen scheduler work changed')
            active=[c for c in ctas if c['tiles']]
            longest=max(active,key=lambda c:c['end_c']-c['entry_c']);latest=max(active,key=lambda c:c['end_ns'])
            maxima.append(longest['end_c']-longest['entry_c']);max_ids.append(longest['cta']);last_ids.append(latest['cta'])
            selected_durations.append(critical['end_c']-critical['entry_c'])
            entry.append(max(c['entry_ns'] for c in ctas)-min(c['entry_ns'] for c in ctas))
            parts['supply'].append(tiles[0][0]-critical['entry_c'])
            parts['mainloop'].append(sum(t[1]-t[0] for t in tiles))
            parts['last_epilogue'].append(tiles[-1][3]-tiles[-1][2])
            parts['tail'].append(critical['end_c']-max(t[3] for t in tiles))
            portable=dict(ctas=[dict(c,smid=c['sm']) for c in ctas])
            iv=v06_fit.intervals(portable,base.CONFIGS[row['config']]['schedule'])[0]
            if 'h' in iv:parts['handoff_interval'].append(iv['h'])
        components={}
        for key,values in parts.items():
            if not values:components[key]=dict(status='not_observed');continue
            meas=statistics.median(values)
            predicted=frozen['calibration']['params'][row['config']]['h'] if key=='handoff_interval' else pred['stage_cycles'][key]
            components[key]=dict(predicted=predicted,measured=meas,absolute_error_cycles=predicted-meas,
                 relative_error=error(predicted,meas),status='quantitative' if abs(meas)>=512 else 'short_interval_boundary_evidence_required')
        cmax=statistics.median(maxima)
        results.append(dict(case=cid,config=row['config'],kind=row['kind'],predicted_us=pred['predicted_us'],measured_us=measured,
          us_error=error(pred['predicted_us'],measured),plain_processes=len(plain),trace_processes=len(trace),plain_cv=pcv,trace_cv=tcv,
          perturbation=perturbation,plain_qualified=pcv<=.05,trace_qualified=pcv<=.05 and tcv<=.05 and abs(perturbation)<=.05,
          predicted_critical_cycles=pred['critical_cycles'],measured_max_cta_cycles=cmax,cycle_error=error(pred['critical_cycles'],cmax),
          frozen_critical_cta=pred['critical_cta'],measured_frozen_cta_cycles=statistics.median(selected_durations),
          selection_gap_cycles=cmax-statistics.median(selected_durations),actual_longest_cta_counts=dict(Counter(max_ids)),actual_latest_cta_counts=dict(Counter(last_ids)),entry_spread_ns=statistics.median(entry),components=components,
          boundary_unknown_cells=pred['boundary_unknown_cells']))
    if first<=frozen['frozen_unix_ns']:raise ValueError('measurement predates freeze')
    time_errors=summary(r['us_error'] for r in results)
    by_phase={}
    for name in ['supply','mainloop','last_epilogue','tail','handoff_interval']:
        values=[r['components'][name]['relative_error'] for r in results if r['trace_qualified'] and r['components'][name]['status']=='quantitative']
        stat=summary(values);by_phase[name]=dict(statistics=stat,quantitatively_scored=len(values),
            passed=bool(stat and stat['median']<=.10 and stat['maximum']<=.20),
            short_or_unqualified=24-len(values))
    result=dict(cases=results,checked_values=checked,rejected_processes=rejected,prediction_sha256=common.sha(predictions),
        frozen_unix_ns=frozen['frozen_unix_ns'],first_target_process_unix_ns=first,freeze_before_measurement=True,
        complete_time=time_errors,complete_time_passed=all(r['plain_qualified'] for r in results) and time_errors['median']<=.05 and time_errors['maximum']<=.10,
        critical_cycles=summary(r['cycle_error'] for r in results if r['trace_qualified']),component_assessment=by_phase,
        per_config={cfg:summary(r['us_error'] for r in results if r['config']==cfg) for cfg in ['cfg_a','cfg_b','cfg_c']},
        component_scope='aggregate supply/mainloop/last epilogue/tail for the frozen predicted critical CTA; max CTA cycles scored separately; handoff h matches calibrated interval',
        qualification='small intervals retain raw errors and event evidence; no automatic pass from a small denominator or component cancellation')
    output.mkdir(parents=True,exist_ok=True);common.write_json(output/'validation.json',result)
    print('V07 complete time',time_errors,'passed',result['complete_time_passed'])
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--predictions',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();analyze(a.run.resolve(),a.predictions,a.output)
