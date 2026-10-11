#!/usr/bin/env python3
"""Offline R09 history floors and time-error decomposition, without fitting.

Source work and the V09 calibration remain fixed. Empirical oracle floors are
descriptive bounds for these observations, never hardware noise limits.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st

import clock_model
import r09_input_source_clock as r09
from analyze_r13_sm import select_attempts
from analyze_r18 import replay
import v06_run as common
import v08_model


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def stats(values):
    values = list(values)
    return dict(n=len(values), median=st.median(values), mean=st.fmean(values),
                minimum=min(values), maximum=max(values),
                rms=math.sqrt(st.fmean(v*v for v in values)))


def relative_point_floor(values):
    """Best constant under maximum and RMS relative loss on positive samples."""
    values = list(values)
    if not values or any(not math.isfinite(y) or y <= 0 for y in values):
        raise ValueError('positive finite observations required')
    lo, hi = min(values), max(values)
    minimax = 2*lo*hi/(lo+hi)
    rms_point = sum(1/y for y in values)/sum(1/y**2 for y in values)
    return dict(n=len(values), minimax_point=minimax,
                maximum_relative_floor=(hi-lo)/(hi+lo),
                rms_point=rms_point,
                rms_relative_floor=math.sqrt(st.fmean((rms_point/y-1)**2 for y in values)))


def stratified_floor(groups):
    """Separate repeat spread from changes of declared history centers.

    The between/within split is an exact descriptive ANOVA in native units.
    Within-stratum variation includes hidden state and measurement effects;
    neither part is asserted to be independent random noise.
    """
    centers = [st.median(v) for v in groups.values()]
    all_values = [y for v in groups.values() for y in v]
    grand = st.fmean(all_values)
    total = sum((y-grand)**2 for y in all_values)
    within = sum(sum((y-st.fmean(v))**2 for y in v) for v in groups.values())
    return dict(history_centers=relative_point_floor(centers),
                all_observations=relative_point_floor(all_values),
                between_stratum_fraction=1-within/total if total else 0,
                repeat_cv={str(k):st.pstdev(v)/st.fmean(v) for k,v in groups.items()},
                centers={str(k):st.median(v) for k,v in groups.items()})


def cycle_rate_split(ca, na, cb, nb):
    """Exact symmetric decomposition of nb-na for matched role endpoints."""
    if min(ca, na, cb, nb) <= 0:
        raise ValueError('nonpositive matched cycle/ns window')
    ia, ib = na/ca, nb/cb
    cycle = (cb-ca)*(ia+ib)/2
    rate = (ca+cb)*(ib-ia)/2
    if not math.isclose(cycle+rate, nb-na, abs_tol=1e-7):
        raise ValueError('cycle/rate decomposition did not close')
    return dict(delta_ns=nb-na, cycle_term_ns=cycle, inverse_rate_term_ns=rate,
                cycle_change=cb/ca-1, rate_change=(cb/nb)/(ca/na)-1,
                time_change=nb/na-1)


def constant_rate_slack(windows):
    """Smallest native-ns residual for N_i=C_i/f, with a constant positive f.

    With exact cycle differences and independently bounded timer endpoint
    errors, each interval may differ by at most 2*epsilon. This returns the
    necessary epsilon. It is conditional on that error model, not a measured
    timer accuracy or a test of an arbitrary varying common f(t).
    """
    if not windows or any(not all(math.isfinite(v) and v>0 for v in (c,n)) for c,n in windows):
        raise ValueError('positive cycle/ns windows required')
    radius,pair = 0.0,(0,0)
    for i,(ca,na) in enumerate(windows):
        for j,(cb,nb) in enumerate(windows[:i]):
            value = abs(na*cb-nb*ca)/(ca+cb)
            if value>radius:
                radius,pair = value,(i,j)
    lo = max((n-radius)/c for c,n in windows)
    hi = min((n+radius)/c for c,n in windows)
    if lo>hi+1e-9:
        raise ValueError('constant-rate interval feasibility did not close')
    inverse_rate = (lo+hi)/2
    if inverse_rate<=0:
        raise ValueError('positive constant rate not identified')
    return dict(interval_slack_ns=radius,necessary_endpoint_slack_ns=radius/2,
                best_constant_ghz=1/inverse_rate,witness_indices=pair)


def stage_endpoint_contract(base,out):
    root = base/'20261010-R09-R13-R18-failure-controls'
    common.verify(root)
    processes,provenance = [],{}
    for row in read(root/'cases.json'):
        if row['id'] not in ('r13_stage_k4096_s4','r13_stage_k4096_s6'):
            continue
        selected,failed,_ = select_attempts(root,row['id'],('dual',))
        if failed or len(selected)!=10:
            raise ValueError('all declared stage processes, including failures, must be reviewed')
        for item in selected:
            rec = item['record']
            observed = replay(root,rec,row,retain_roles=True)
            windows,labels,local = [],[],[]
            for cta in observed['ctas']:
                same_cta = []
                for role,r in enumerate(cta['roles']):
                    if len(r['tiles'])!=1:
                        raise ValueError('two-output pingpong role mapping changed')
                    c,n = r['tiles'][0],r['tiles_ns'][0]
                    window = (c[1]-c[0],n[1]-n[0])
                    windows.append(window)
                    labels.append(dict(cta=cta['cta'],sm=cta['sm'],role=role,cycles=window[0],ns=window[1]))
                    same_cta.append(window)
                local.append(dict(cta=cta['cta'],**constant_rate_slack(same_cta)))
            common_rate = constant_rate_slack(windows)
            processes.append(dict(case=row['id'],trial=rec['trial'],stage=row.get('stages',6),
                raw_sha256=rec['raw_sha256'],common_constant_rate=common_rate,
                witness=[labels[i] for i in common_rate['witness_indices']],
                same_cta_necessary_endpoint_slack_ns=stats(x['necessary_endpoint_slack_ns'] for x in local),
                per_cta=local))
            provenance[str(root/rec['raw'])] = rec['raw_sha256']
    if len(processes)!=20:
        raise ValueError('twenty stage processes required')
    result = dict(scope='Parameter-free endpoint/constant-rate counterexample; no frequency fit or GPU sampling.',
        assumptions=['Both L windows retain the same CTA, original consumer role and integer endpoint differences.',
            'The candidate being falsified uses one constant positive cycle/ns conversion within the call.',
            'Necessary endpoint slack assumes cycle differences exact and per-globaltimer endpoint errors bounded by epsilon.',
            'Real within-call rate variation, clock read delays, and cross-SM effects remain alternative explanations.',
            'Slack is not globaltimer resolution or accuracy; all windows exceed 3.2 us.'],
        processes=processes,summary={str(stage):dict(
            common_constant_rate_endpoint_slack_ns=stats(p['common_constant_rate']['necessary_endpoint_slack_ns'] for p in processes if p['stage']==stage),
            same_cta_endpoint_slack_median_ns=stats(p['same_cta_necessary_endpoint_slack_ns']['median'] for p in processes if p['stage']==stage))
            for stage in (4,6)},input_sha256=provenance,analysis_sha256=sha(Path(__file__)))
    write(out/'stage-endpoint-contract.json',result)
    print(json.dumps(result['summary'],indent=2))


def replay_histories(root, provenance):
    """Replay all series calls; keep single controls and all failed attempts."""
    common.verify(root)
    rows = read(root/'cases.json')
    setups = {x['case']:x['setup'] for x in read(root/'static_setup.json')}
    processes, failures = [], []
    for name in ('cases.json','static_setup.json','environment.json','source_hashes.json','build/binary_hashes.json'):
        provenance[str(root/name)] = sha(root/name)
    for row in rows:
        selected, failed, _ = select_attempts(root,row['id'],('ends',))
        failures.extend(failed)
        if len(selected) != 10:
            raise ValueError('ten successful processes required, without CV filtering')
        for item in selected:
            rec = item['record']
            observed = replay(root,rec,row) if row['history_mode']=='series' else v08_model.observe(root,rec,row,setups[row['id']])
            with gzip.open(root/rec['raw'],'rt') as stream:
                call = next(e for line in stream if line.strip() for e in [json.loads(line)] if e['event']=='call')
            p = dict(case=row['id'],trial=rec['trial'],swizzle=row['swizzle'],
                     mode=row['history_mode'],source=row.get('history_source','same'),
                     slot=row.get('mirror_target_slot'),protocol=call['history_protocol'],
                     raw_sha256=rec['raw_sha256'],prefix_event_us=call['history_prefix_event_us'])
            if row['history_mode']=='series':
                p['calls'] = r09.history_call_metrics(observed['history_series'],call['history_inter_call_event_us'])
                if len(p['calls']) != 35:
                    raise ValueError('complete prefix16-target19 required')
            else:
                p['target'] = dict(event_us=observed['elapsed_us'],**r09.control_window(observed))
            processes.append(p)
            provenance[str(root/item['path'])] = sha(root/item['path'])
            provenance[str(root/rec['raw'])] = rec['raw_sha256']
    return dict(run=str(root),environment=read(root/'environment.json'),processes=processes,failures=failures)


def history_floors(common_report, series, mirror):
    fields = ('frequency_ghz','max_cta_cycles','window_us','event_us')
    result = dict(common_adaptive={},series={},mirror={})
    for sw in (1,8):
        group = [p for p in common_report['processes'] if p['swizzle']==sw]
        result['common_adaptive'][str(sw)] = {key:stratified_floor({policy:[p[key] for p in group if p['policy']==policy]
            for policy in ('adaptive','common-sw1-16')}) for key in fields}
    trajectory_fields = ('frequency_ghz','median_full_cta_cycles','median_full_cta_window_us','window_us','event_us')
    for sw in (1,8):
        group = [p for p in series['processes'] if p['mode']=='series' and p['swizzle']==sw]
        # For the sw8 group, prefix and target have different static orders.
        indices = dict(prefix=range(16),target=range(16,35))
        if sw==1:
            indices['all_same_sw1'] = range(35)
        result['series'][str(sw)] = {segment:{key:stratified_floor({j:[p['calls'][j][key] for p in group] for j in ids})
            for key in trajectory_fields} for segment,ids in indices.items()}
    for source in ('same','mirror'):
        for slot in (0,1):
            group = [p for p in mirror['processes'] if p['source']==source and p['slot']==slot]
            # Mirror prefix uses another allocation: keep target-only floors.
            result['mirror'][source+str(slot)] = {key:stratified_floor({j:[p['calls'][j][key] for p in group] for j in range(16,35)})
                for key in trajectory_fields}
    return result


def history_splits(report):
    rows = []
    segments = dict(prefix_early=(0,1,2),prefix_late=(13,14,15),
                    target_early=(16,17,18),target_late=(32,33,34))
    for p in report['processes']:
        if p['mode']!='series':
            continue
        for left,right in (('prefix_early','prefix_late'),('prefix_late','target_early'),('target_early','target_late')):
            # Decompose each role before aggregation. Component medians need
            # not sum to the median total; means and every role do close.
            samples = []
            maps = {name:[{(r[0],r[1]):r for r in p['calls'][j]['role_windows']} for j in ids]
                    for name,ids in segments.items() if name in (left,right)}
            keys = set(maps[left][0])
            if any(set(m)!=keys for v in maps.values() for m in v):
                raise ValueError('matched roles/work set changed across history')
            for key in sorted(keys):
                a,b = ([m[key] for m in maps[name]] for name in (left,right))
                if len({r[2] for r in a+b})!=1:
                    raise ValueError('matched role tile count changed')
                samples.append(cycle_rate_split(st.median(r[4] for r in a),st.median(r[5] for r in a),
                                                st.median(r[4] for r in b),st.median(r[5] for r in b)))
            row = dict(case=p['case'],trial=p['trial'],swizzle=p['swizzle'],source=p['source'],slot=p['slot'],
                       contrast=left+'__'+right,roles=len(samples),
                       same_static_order=(left!='prefix_late' or p['swizzle']==1),
                       same_source_allocation=(left!='prefix_late' or p['source']=='same'))
            for key in samples[0]:
                row[key+'_mean'] = st.fmean(s[key] for s in samples)
                row[key+'_median'] = st.median(s[key] for s in samples)
            if abs(row['delta_ns_mean']-row['cycle_term_ns_mean']-row['inverse_rate_term_ns_mean'])>1e-7:
                raise ValueError('process mean decomposition did not close')
            rows.append(row)
    return rows


def conditional_controls(base, report):
    """Frozen r2 transfer diagnostics on the other GPU, never validation."""
    cal = read(base/'20261009-V09-freeze-job738972-r2/frozen/calibration.json')
    root = Path(report['run'])
    rows = {r['id']:r for r in read(root/'cases.json')}
    setups = {s['case']:s['setup'] for s in read(root/'static_setup.json')}
    points = {}
    for cid,row in rows.items():
        if not cid.startswith('r09_protocol'):
            continue
        if row['config']!='cfg_b' or setups[cid]['stages']!=6:
            raise ValueError('frozen cfg_b nominal Q/S scope changed')
        work = v08_model.scheduled_work(row['config'],row['m'],row['n'],setups[cid]['grid'],row['swizzle'])
        # Exactly the cfg_b frozen-r2 definition, without changing a source
        # quantity or bypassing the supply model's GPU-identity rejection.
        count = sum(map(len,work))*(row['k']//64)
        points[cid] = dict(input_mode=row['input_mode'],tensor_mean_cycles=count*512/132,
                           source_mean_kib=count*32/132)
    output = []
    for p in report['processes']:
        if p['variant']=='plain':
            continue
        f = p['frequency_ghz']
        point = points[p['case']]
        conditional = clock_model.source_frequency(point,p['window_us'],cal['clock'])['frequency_ghz']
        output.append(dict(case=p['case'],trial=p['trial'],variant=p['variant'],k=p['k'],
            scope='Cross-card frozen-r2 transfer diagnostic; calibration GPU differs from observations.',
            observed_frequency_ghz=f,observed_window_us=p['window_us'],observed_event_us=p['event_us'],
            static_clock_work=point,conditional_frequency_ghz=conditional,
            conditional_frequency_error=conditional/f-1,
            conditional_time_error=f/conditional-1 if conditional>0 else None,
            event_minus_window_us=p['event_minus_window_us'],
            free_composition_status='unsupported: frozen-r2 supply GPU identity differs; no card guard bypass',
            free_envelope_error=None,free_plain_error=None))
    return output


def archived_v09(base, provenance):
    """Preserve all 300 original diagnostics, including the rejected condition.

    This pass verifies the archive and every referenced raw payload hash. It
    consumes the prior replay's derived component errors rather than rerunning
    the complete shared V09 analyzer or refitting any model.
    """
    root = base/'20261009-V09-heldout-job739011'
    path = root/'reanalysis/failure-decomposition-v1/A-clock-failure.json'
    data = read(path)
    provenance[str(path)] = sha(path)
    processes = []
    indexed = {(p['case'],p['trial']):p for p in data['processes']}
    if len(indexed)!=300:
        raise ValueError('all thirty V09 conditions required')
    case_ids = {cid for cid,trial in indexed}
    for row in read(root/'cases.json'):
        if row['id'] not in case_ids:
            continue
        selected, _, _ = select_attempts(root,row['id'],('dual',))
        for item in selected:
            rec = item['record']
            p = indexed[row['id'],rec['trial']]
            if p['raw_sha256']!=rec['raw_sha256'] or sha(root/rec['raw'])!=rec['raw_sha256']:
                raise ValueError('archived V09 diagnostic identity changed')
            provenance[str(root/rec['raw'])] = rec['raw_sha256']
            f = p['conditional_frequency_ghz']
            processes.append(dict(p,conditional_time_error=p['observed_frequency_ghz']/f-1 if f>0 else None))
    if len(processes)!=300:
        raise ValueError('V09 process selection incomplete')
    return dict(scope='Existing V09 development replay with raw-hash verification; no new validation.',processes=processes)


def summary_groups(processes, keys):
    groups = defaultdict(list)
    for p in processes:
        groups[tuple(p[k] for k in keys)].append(p)
    return [{'group':dict(zip(keys,key)),'metrics':{field:stats(p[field] for p in group if p.get(field) is not None)
        for field in group[0] if field.endswith('_error') and any(p.get(field) is not None for p in group)}}
        for key,group in groups.items()]


def write_csv(path, rows):
    fields = [k for k,v in rows[0].items() if not isinstance(v,(dict,list))]
    with path.open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-incomplete',action='store_true',help='Retry this analyzer own failed output; a completed report remains protected')
    parser.add_argument('--stage-endpoints',action='store_true',help='Only replay the two-output endpoint counterexample')
    args = parser.parse_args()
    base,out = args.base.resolve(),args.output.resolve()
    # Resume only an explicitly recorded failure of this analysis. Never
    # overwrite a completed report or the shared development directory.
    if out.exists():
        if not args.resume_incomplete or (out/'time-contract.json').exists() or not (out/'analysis-attempts.json').exists():
            raise ValueError('output exists without a resumable incomplete R09 analysis')
    else:
        out.mkdir(parents=True,exist_ok=False)
    if args.stage_endpoints:
        stage_endpoint_contract(base,out)
        return
    provenance = {}
    common_report = r09.common_history_diagnostics(base/'20261010-R09-common-history',out)
    controls = r09.control_diagnostics(base/'20261010-R09-R13-R18-failure-controls',out)
    for report in (common_report,controls):
        root = Path(report['run'])
        for path,digest in report['input_sha256'].items():
            provenance[str(root/path)] = digest
    series = replay_histories(base/'20261010-R09-history-series',provenance)
    mirror = replay_histories(base/'20261010-R09-mirror-history',provenance)
    floors = history_floors(common_report,series,mirror)
    splits = history_splits(series)+history_splits(mirror)
    transferred = conditional_controls(base,controls)
    v09 = archived_v09(base,provenance)
    # Existing phase-clock evidence keeps its original endpoint support.
    stage_path = base/'20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/A-stage-clock.json'
    stages = read(stage_path)
    provenance[str(stage_path)] = sha(stage_path)
    for p in stages['processes']:
        path = Path(p['source'])
        if sha(path)!=p['source_sha256']:
            raise ValueError('stage evidence raw identity changed')
        provenance[str(path)] = p['source_sha256']
    cal_path = base/'20261009-V09-freeze-job738972-r2/frozen/calibration.json'
    provenance[str(cal_path)] = sha(cal_path)
    result = dict(scope='R09 time-input contract development; no parameter fit, GPU work, or new heldout score.',
        analysis_sha256=sha(Path(__file__)),source_work='Unchanged frozen-r2 static_clock_work: nominal Q/S over all 132 SMs; B owns future revisions.',
        history_floors=floors,history_process_splits=splits,
        controls=transferred,control_summary=summary_groups(transferred,('k','variant')),
        v09_summary=summary_groups(v09['processes'],('case','config','frozen_status')),
        stage_evidence=dict(path=str(stage_path),summary=stages['summary'],containment_witnesses=stages['containment_witnesses']),
        failed_attempts=common_report['failed_attempts']+controls['failed_attempts']+series['failures']+mirror['failures'],
        verification=dict(common_history_processes=len(common_report['processes']),control_processes=len(controls['processes']),
            history_processes=len(series['processes'])+len(mirror['processes']),
            series_calls=sum(len(p.get('calls',[])) for p in series['processes']),
            mirror_calls=sum(len(p.get('calls',[])) for p in mirror['processes']),
            v09_hash_verified_processes=len(v09['processes']),stage_hash_verified_processes=len(stages['processes']),
            role_split_closure_max_ns=max(abs(r['delta_ns_mean']-r['cycle_term_ns_mean']-r['inverse_rate_term_ns_mean']) for r in splits)),
        limits=['Empirical fixed-point floors depend on the observed strata, not population confidence bounds or unavoidable hardware noise.',
                'Within-stratum repeat spread includes unknown state and endpoint effects. It is not identified as random measurement noise.',
                'Conditional time holds the observed cycle interval fixed and substitutes f(W_observed); it is not free prediction.',
                'Observed-f recursion errors contain service/structure and endpoint differences; they are not pure hardware cycle-service errors.',
                'Plain/ends/dual are sequential processes; their trial-labelled differences are not same-call causal endpoint decompositions.',
                'GPU-ef8692f3 control absolute errors are cross-card transfer diagnostics; r2 calibration belongs to GPU-43269fbc.',
                'Same-role cycle/rate splits are algebraic. The inverse-rate term contains timer endpoint effects and cannot be called physical DVFS.',
                'The failed single-pole state model remains unadopted; this pass fits no new dynamic clock model.'])
    write(out/'time-contract.json',result)
    write(out/'series-replay.json',series)
    write(out/'mirror-replay.json',mirror)
    write(out/'v09-processes.json',v09)
    write(out/'input-sha256.json',provenance)
    write_csv(out/'history-process-splits.csv',splits)
    write_csv(out/'control-process-errors.csv',transferred)
    write_csv(out/'v09-process-errors.csv',v09['processes'])
    print(json.dumps(dict(output=str(out),verification=result['verification']),indent=2))


if __name__=='__main__':
    main()
