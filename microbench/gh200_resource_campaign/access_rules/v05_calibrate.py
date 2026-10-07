#!/usr/bin/env python3
"""Prepare V05 calibration parameters. Never freezes or measures. Writes one parameter file to the requested new path.

Example:
  python /tmp/v05-calibration-candidate.py \
    --r14 main RUN_MAIN ANALYSIS_MAIN --r14 output RUN_OUTPUT ANALYSIS_OUTPUT \
    --r14 supply RUN_SUPPLY ANALYSIS_SUPPLY --r14 critical RUN_CRIT ANALYSIS_CRIT \
    --r10-summary R10_ANALYSIS/summary.json --output /tmp/v05-calibration-candidate.json

No data is required: absent/unqualified points are explicitly reported as missing.
Status is 'calibrated' when nothing is missing; v05_predict.py refuses anything else.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
import statistics as stats

ANALYZER = Path(__file__).resolve().parent/'analyze_r14.py'
CONFIGS = ('baseline', 'cfg_a', 'cfg_b')
PHASES = ('supply', 'mainloop', 'output', 'handoff', 'epi_permission_wait')
EVENTS = {
    'supply': ['first_output_tile_producer_work', 'first_output_tile_first_mma'],
    'mainloop': ['first_mma', 'main_end'],
    'output': ['final_tile_epi_permit', 'final_tile_source_tail_return'],
    'handoff': ['tile_main_end', 'next_tile_first_mma'],
    'epi_permission_wait': ['final_tile_main_end', 'final_tile_epi_permit'],
}
ASSUMPTIONS = [
    'CTA0 fine-stage sampling transfers to the critical CTA of the same configuration.',
    'Four-CTA calibration transfers to the full-card scheduler without added contention.',
    'Clock frequency is treated as homogeneous across CTAs for secondary microsecond conversion.',
    'Median tile stages and median adjacent handoff approximate repeated-tile stationarity.',
    'Final tile W/E use only terminal-tile same-run endpoints; intermediate delayed acquire guarantees are not added as E.',
    'Independent observation pairs supply scalar parameters, never a stitched absolute timeline.',
    'Source release is not complete global writeback; critical cycle endpoint is source release.',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def median(values):
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('missing/nonfinite measurement; not replaced with zero')
    return stats.median(values)


def cv(values):
    return stats.pstdev(values)/stats.mean(values)


def group_tiles(rows):
    tiles = defaultdict(list)
    for row in rows:
        if row['role']:
            tiles[(row['cta'], row['m_tile'], row['n_tile'], row['l_tile'])].append(row)
    return tiles


def stage_samples(pair, rows):
    """All differences come from one process and one CTA; merge cooperative roles by tile."""
    samples = defaultdict(list)
    terminals = {}
    if pair == 'critical':
        return samples, terminals
    if len({r['sm'] for r in rows}) != 1:
        raise ValueError('fine-stage source spans multiple SMs')
    tiles = group_tiles(rows)
    if pair == 'main':
        timeline = []
        for key, group in tiles.items():
            begin = min(r['first_mma_cycle'] for r in group)
            end = max(r['main_end_cycle'] for r in group)
            samples['mainloop'].append(end-begin)
            timeline.append((begin, end, key))
        timeline.sort()
        for left, right in zip(timeline, timeline[1:]):
            samples['handoff'].append(right[0]-left[1])  # preserve negative overlap
        terminals['first_mainloop'] = timeline[0][1]-timeline[0][0]
        terminals['last_mainloop'] = timeline[-1][1]-timeline[-1][0]
    elif pair == 'output':
        timeline = []
        last_producer=max((r for r in rows if r['role']==0), key=lambda r:r['seq'])
        terminal_key=(last_producer['cta'],last_producer['m_tile'],last_producer['n_tile'],last_producer['l_tile'])
        for key, group in tiles.items():
            end = max(r['main_end_cycle'] for r in group)
            permit = max(r['epi_permit_cycle'] for r in group)
            releasing = [r for r in group if r['source_release_cycle'] is not None]
            if len(releasing) != 1:
                raise ValueError('expected exactly one issuing role per output tile')
            # Match analyze_v05.measures: merged max permission anchors both W and E.
            issuer = releasing[0]
            wait = permit-end
            output = issuer['source_release_cycle']-permit
            if key==terminal_key:
                samples['epi_permission_wait']=[wait]
                samples['output']=[output]
            timeline.append((end, wait, output))
        timeline.sort()
        terminals.update(final_W=samples['epi_permission_wait'][0], final_E=samples['output'][0],
            output_scope='last output tile in producer scheduler order; intermediate acquire release is not service latency')
    elif pair == 'supply':
        producers = [r for r in rows if r['role'] == 0]
        first = min(producers, key=lambda r:r['seq'])
        key = (first['cta'], first['m_tile'], first['n_tile'], first['l_tile'])
        consumers = tiles.get(key)
        if not consumers:
            raise ValueError('first producer tile has no matching consumer')
        start = first['work_cycle']
        first_mma = min(r['first_mma_cycle'] for r in consumers)
        samples['supply'] = [first_mma-start]
    for phase, values in samples.items():
        if phase != 'handoff' and min(values) < 0:
            raise ValueError('negative duration for '+phase)
    return samples, terminals


def gpu_identity(environment):
    value = environment.get('gpu_uuid') or environment.get('sample_gpu') or environment.get('gpu')
    found = re.search(r'GPU-[0-9a-fA-F-]+', value or '')
    if not found:
        raise ValueError('missing concrete calibration GPU UUID')
    return found[0]


def timing_sample(rows, event_us):
    latest = max(r['source_release_ns'] for r in rows)
    critical = max((r for r in rows if r['source_release_ns'] == latest),
                   key=lambda r:r['source_release_cycle']-r['work_cycle'])
    dc = critical['source_release_cycle']-critical['work_cycle']
    dn = critical['source_release_ns']-critical['work_ns']
    if dc <= 0 or dn <= 0:
        raise ValueError('invalid same-CTA clock calibration')
    span = latest-min(r['work_ns'] for r in rows)
    return dict(critical_cta=critical['cta'], cycles=dc, ns=dn,
                frequency_mhz=1000*dc/dn, fixed_us=event_us-span/1000,
                observed_source_span_ns=span, event_us=event_us)


class Extraction:
    def __init__(self, decoder):
        self.decoder = decoder
        self.evidence = {}
        self.records = {}
        self.rejections = []
        self.environments = {}

    def evidence_file(self, path):
        path = Path(path).resolve()
        self.evidence[str(path)] = sha(path)
        return str(path)

    def extract(self, pair, run, analysis):
        run, analysis = Path(run).resolve(), Path(analysis).resolve()
        try:
            summary = read_json(analysis/'summary.json')
            if summary.get('set') != 'main' or summary.get('trace_pair') != pair:
                raise ValueError('requires formal main summary for the selected observation pair')
            manifest = self.evidence_file(run/'source_hashes.json')
            if summary.get('source_manifest_sha256') != self.evidence[manifest]:
                raise ValueError('analysis source manifest identity mismatch')
            self.evidence_file(analysis/'summary.json')
            envpath = self.evidence_file(run/'environment.json')
            env = read_json(envpath)
            gpu_identity(env)
            self.environments[pair] = env
            if env.get('trace_pair') != pair:
                raise ValueError('environment observation pair mismatch')
            for filename in ('tile_roles.csv', 'tile_intervals.csv', 'critical_endpoints.csv'):
                if (analysis/filename).exists():
                    self.evidence_file(analysis/filename)
        except (ValueError, KeyError, OSError) as error:
            self.rejections.append(dict(pair=pair, run=str(run), reason=str(error)))
            return
        for case in summary.get('cases', []):
            case_id = case.get('case_id', '')
            if not (case.get('trace_usable') is True and case.get('stable') is True
                    and case.get('perturbation_ok') is True and case.get('processes') == 10):
                self.rejections.append(dict(pair=pair, case=case_id,
                                            reason='not trace_usable/stable/complete ten pairs'))
                continue
            try:
                config, remainder = case_id.rsplit('_k', 1)
                ktext, ttext = remainder.split('_t')
                K, tiles = int(ktext), int(ttext)
                if config not in CONFIGS or K not in (128,512,4096) or tiles not in (1,8):
                    raise ValueError('outside R14 calibration matrix')
                if '735985-v6-main' in run.name and config == 'cfg_a' and K == 128:
                    raise ValueError('explicit v6 cfg_a K128 negative result excluded')
                failures = summary.get('failures', [])
                for failure in failures:
                    if (failure.get('case') == case_id
                            or case_id in failure.get('path', '')):
                        raise ValueError('analysis failure attached to this case')
                for kind in ('plain','trace'):
                    candidates = list((run/'samples'/f'{kind}_{case_id}').glob('main-*/result.json'))
                    if len(candidates) != 10:
                        raise ValueError('requires exactly ten saved trials, no extra retry selection')
                per_trial, plain_times, trace_times, sources = [], [], [], set()
                for trial in range(10):
                    calls, setups, trace_rows = {}, {}, None
                    for kind in ('plain','trace'):
                        folder = run/'samples'/f'{kind}_{case_id}'/f'main-{trial:02}'
                        result_path = folder/'result.json'
                        result = read_json(result_path)
                        if (result['status'] != 'measured' or result['trial'] != trial
                                or result['set'] != 'main'):
                            raise ValueError('sample failed or is not a formal trial')
                        raw_path = folder/'stdout.txt.gz'
                        if result.get('record_format') == 'r14_compact_v1':
                            meta = result['raw']
                            if (meta['path'] != raw_path.name or meta['bytes'] != raw_path.stat().st_size
                                    or meta['sha256'] != sha(raw_path)):
                                raise ValueError('raw identity mismatch')
                        with gzip.open(raw_path,'rt') as stream:
                            events = [json.loads(line) for line in stream if line.strip()]
                        setup = next(v for v in events if v.get('event') == 'setup')
                        call = next(v for v in events if v.get('event') == 'call')
                        check = next(v for v in events if v.get('event') == 'check')
                        if (setup['config'] != config or setup['k'] != K
                                or setup['tiles_per_cta'] != tiles or setup['trace_pair'] != pair
                                or setup['traced'] != (kind == 'trace')):
                            raise ValueError('sample configuration/pair mismatch')
                        if check['status'] != 'ok' or check['samples'] != setup['m']*setup['n']:
                            raise ValueError('not a complete successful numeric check')
                        warm = call['warmup_us']
                        if not 8 <= len(warm) <= 30 or cv(warm[-5:]) > .02:
                            raise ValueError('warmup failed; do not silently drop a trial')
                        calls[kind], setups[kind] = call, setup
                        if kind == 'trace':
                            trace_rows = self.decoder.decode(setup,call['trace'])
                        sources.add(self.evidence_file(result_path))
                        sources.add(self.evidence_file(raw_path))
                    plain_times.append(calls['plain']['elapsed_us'])
                    trace_times.append(calls['trace']['elapsed_us'])
                    values, diagnostics = stage_samples(pair,trace_rows)
                    item = dict(trial=trial, phases={p:median(v) for p,v in values.items()},
                                within_process_counts={p:len(v) for p,v in values.items()},
                                terminal_diagnostics=diagnostics)
                    if pair == 'critical':
                        item['timing'] = timing_sample(trace_rows,calls['trace']['elapsed_us'])
                    per_trial.append(item)
                perturbations = [(t/p-1)*100 for p,t in zip(plain_times,trace_times)]
                if cv(plain_times)>.05 or cv(trace_times)>.05 or abs(median(perturbations))>5:
                    raise ValueError('raw statistics fail current R14 qualification policy')
                key = (pair, config, K//64, tiles)
                if key in self.records:
                    raise ValueError('duplicate case origin; choose one run explicitly')
                sources.update((str((analysis/'summary.json').resolve()), manifest, envpath))
                self.records[key] = dict(pair=pair,config=config,ktiles=K//64,tiles=tiles,
                    processes=per_trial,source=sorted(sources),scope=summary['trace_scope'],
                    paired_perturbation_percent=perturbations,
                    per_process_then_across_process_median=True)
            except (ValueError, KeyError, OSError, StopIteration) as error:
                self.rejections.append(dict(pair=pair,case=case_id,reason=str(error)))

    def curve(self, config, phase):
        pair = {'mainloop':'main','handoff':'main','output':'output',
                'epi_permission_wait':'output','supply':'supply'}[phase]
        points, missing, sources, diagnostics = [], [], set(), []
        for P in (2,8,64):
            record = self.records.get((pair,config,P,8))
            if record is None:
                missing.append(dict(ktiles=P,reason='qualified t8 case missing'))
                continue
            values = [p['phases'].get(phase) for p in record['processes']]
            if any(v is None for v in values):
                missing.append(dict(ktiles=P,reason='event difference unobserved'))
                continue
            points.append(dict(ktiles=P,cycles=median(values)))
            diagnostics.append(dict(ktiles=P,process_medians=values,
                                    process_diagnostics=record['processes']))
            sources.update(record['source'])
        return dict(points=points,source=sorted(sources),events=EVENTS[phase],missing=missing,
            status='candidate_complete' if not missing else 'missing',
            aggregation='t8: per-process median over merged output tiles, then median of10 processes; '
                        'supply uses first producer tile only; handoff keeps signed gaps',
            process_evidence=diagnostics,transfer_assumptions=[])

    def timing(self, config, P=8, tiles=8):
        record = self.records.get(('critical',config,P,tiles))
        if record is None:
            return dict(status='missing',frequency_mhz=None,fixed_us=None,source=[],
                        missing=[f'qualified critical {config} P{P} t{tiles} missing'])
        observed = [p['timing'] for p in record['processes']]
        return dict(status='candidate',frequency_mhz=median([x['frequency_mhz'] for x in observed]),
            fixed_us=median([x['fixed_us'] for x in observed]),source=record['source'],
            assumptions=['Same-call same-CTA cycles/ns; no max-cycle/max-ns cross-CTA ratio.',
                         f'Fixed P{P} t{tiles} four-CTA clock and residual transferred to every V05 size.',
                         'Residual includes launch, source-release to full-output and unobserved gaps; '
                         'it is not an isolated host overhead.',
                         'No measured V05 clock or event time may refit this secondary prediction.'],
            per_process=observed)


def candidate_curve_policy(config, phase, curve, allow_single_p64=False):
    """Keep every qualified point; authorize only a reviewable single-P64 E/W hypothesis."""
    points = curve['points']
    distinct = {point['ktiles'] for point in points}
    if len(distinct) != len(points):
        raise ValueError('duplicate qualified calibration P')
    curve['input_point_count'] = len(points)
    curve['numerical_support_domain'] = [min(distinct), max(distinct)] if distinct else None
    curve['kind'] = 'piecewise_linear'
    curve['required_minimum_points'] = 2
    curve['numerically_ready'] = len(points) >= 2
    curve['missing_calibration_points'] = list(curve.get('missing', []))
    if (allow_single_p64 and phase in ('output', 'epi_permission_wait')
            and len(points) == 1 and points[0]['ktiles'] == 64):
        curve['kind'] = 'constant_service'
        curve['required_minimum_points'] = 1
        curve['numerically_ready'] = True
        curve['status'] = 'candidate_single_P64_K_invariance_hypothesis'
        curve['transfer_assumptions'] = list(curve.get('transfer_assumptions', [])) + [
            'Only qualified P64 is observed; unchanged output bytes do not establish K independence.',
            'Assume this E/W value applies at other K; every P other than64 is extrapolated.',
            'Failed/unobserved P2/P8 remain negative/missing evidence, not additional constant points.',
            'Output/permission overlap with computation and schedule can invalidate this assumption.',
        ]
    elif curve['numerically_ready']:
        curve['status'] = 'candidate_domain_ready'
    else:
        curve['status'] = 'candidate_insufficient_points'
    curve['point_interpretation'] = ('transferred model inputs, not cfg_c R14 observations'
                                   if config == 'cfg_c' else 'qualified R14 observations')
    return curve


def cfg_c_proxy(extraction, baseline, r10_summary):
    result = dict(curves={},critical_assumptions=ASSUMPTIONS+[
        'cfg_c handoff/output/supply/permission are baseline proxies, not measured equivalence.',
        'Equal48KiB input and128KiB output does not prove equivalence of A/B split, '
        'MMA N, consumer mapping or cluster orientation.'])
    for phase in PHASES:
        result['curves'][phase] = dict(baseline['curves'][phase])
        result['curves'][phase]['transfer_assumptions'] = [
            'baseline128x256 cluster2x1 -> cfg_c256x128 cluster1x2; unvalidated conditional proxy']
        result['curves'][phase]['status'] = 'transfer_hypothesis'
    curve = result['curves']['mainloop']
    curve.update(points=[],missing=[],source=[],process_evidence=[])
    try:
        if not r10_summary:
            raise ValueError('R10 aligned cfg_c P32 anchor not supplied')
        summary = read_json(r10_summary)
        r10_uuid = gpu_identity(summary['environment'])
        if any(gpu_identity(e) != r10_uuid for e in extraction.environments.values()):
            raise ValueError('R10 anchor and R14 curves have different GPU UUIDs')
        candidates = [r for r in summary['cases'] if r.get('config')=='cfg_c'
                      and r.get('ldb')==3072 and (r.get('m'),r.get('n'),r.get('k'))==(2600,3000,2000)]
        if len(candidates)!=1:
            raise ValueError('exact cfg_c aligned P32 R10 case absent/ambiguous')
        anchor = candidates[0]
        if not (anchor.get('trace_disturbance_pass') and anchor.get('warmup_all_converged')
                and anchor.get('processes_plain')==10 and anchor.get('processes_trace')==10
                and anchor.get('ktile_count')==32 and anchor.get('max_check_error')==0):
            raise ValueError('R10 anchor not qualified')
        # Preserve R10's own pooled CTA aggregation, never relabel as process medians.
        L32 = anchor['mainloop_cycles_per_ktile_median']*32
        points = {p['ktiles']:p['cycles'] for p in baseline['curves']['mainloop']['points']}
        if 8 not in points or 64 not in points:
            raise ValueError('qualified baseline P8/P64 mainloop slope missing')
        slope = (points[64]-points[8])/56
        path = extraction.evidence_file(r10_summary)
        curve['points'] = [dict(ktiles=P,cycles=L32+slope*(P-32)) for P in (2,8,64)]
        if any(p['cycles']<0 for p in curve['points']):
            raise ValueError('proxy generated negative mainloop; reject rather than clamp')
        curve['source'] = sorted(set(baseline['curves']['mainloop']['source']+[path]))
        curve['anchor'] = dict(ktiles=32,cycles=L32,aggregation='R10 pooled selected CTA/group median')
        curve['slope_cycles_per_ktile'] = slope
        curve['transfer_assumptions'] += [
            'L_c(P)=R10_c_aligned_L32+baseline_P8_P64_slope*(P-32).',
            'R10 partial32nd Ktile, all-card/source pattern and last-tile domain transfer to R14 curve.',
            'P2/P8/P64 here are model evaluations, not cfg_c R14 observations.']
    except (ValueError, KeyError, OSError) as error:
        curve.update(points=[],status='missing',missing=[dict(reason=str(error))])
    result['microseconds'] = dict(baseline['microseconds'])
    result['microseconds']['assumptions'] = list(baseline['microseconds'].get('assumptions',[]))+[
        'baseline clock/residual transferred to cfg_c; no same-byte equivalence claimed.']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r14',nargs=3,action='append',default=[],metavar=('PAIR','RUN','ANALYSIS'))
    parser.add_argument('--r14-analyzer',type=Path,default=ANALYZER)
    parser.add_argument('--r10-summary',type=Path)
    parser.add_argument('--allow-single-p64-output-service',action='store_true',
                        help='optional E/W constant hypothesis only if the sole qualified point is P64')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    destination=args.output.resolve()
    if destination.exists():
        parser.error('choose a new candidate output; never overwrite evidence')
    spec=importlib.util.spec_from_file_location('r14_schema_readonly',args.r14_analyzer)
    decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    extraction=Extraction(decoder);extraction.evidence_file(args.r14_analyzer)
    for pair,run,analysis in args.r14:
        if pair not in ('main','output','supply','critical'):
            parser.error('unknown R14 observation pair '+pair)
        extraction.extract(pair,run,analysis)
    # Do not silently fit parameters pooled over different devices.
    uuids={gpu_identity(e) for e in extraction.environments.values()}
    if len(uuids)>1:
        raise ValueError('calibration observation groups have different GPU identities')
    configs={}
    for config in CONFIGS:
        configs[config]=dict(curves={p:extraction.curve(config,p) for p in PHASES},
                            critical_assumptions=ASSUMPTIONS,
                            microseconds=extraction.timing(config))
    configs['cfg_c']=cfg_c_proxy(extraction,configs['baseline'],args.r10_summary)
    missing=[]
    unobserved=[]
    for config,data in configs.items():
        for phase,curve in data['curves'].items():
            # No point synthesis here. Qualified extraction always precedes model policy.
            candidate_curve_policy(config,phase,curve,args.allow_single_p64_output_service)
            expected_pair = {'mainloop':'main','handoff':'main','output':'output',
                             'epi_permission_wait':'output','supply':'supply'}[phase]
            curve['negative_evidence'] = [r for r in extraction.rejections
                if r.get('pair') == expected_pair and
                r.get('case','').startswith(config+'_k') and r.get('case','').endswith('_t8')]
            if curve.get('missing'):
                unobserved.append(dict(config=config,phase=phase,details=curve['missing']))
            if not curve['numerically_ready']:
                missing.append(dict(config=config,phase=phase,
                    available_points=len(curve['points']),required_points=curve['required_minimum_points'],
                    details=curve.get('missing',[])))
        if data['microseconds'].get('status')=='missing':
            missing.append(dict(config=config,phase='microseconds',details=data['microseconds']['missing']))
    result=dict(status='candidate_missing' if missing else 'calibrated',
        never_freeze=True,configs=configs,
        sources=[dict(path=p,sha256=h) for p,h in sorted(extraction.evidence.items())],
        missing=missing,unobserved_calibration_points=unobserved,
        single_P64_EW_hypotheses_enabled=args.allow_single_p64_output_service,
        rejected=extraction.rejections,
        r14_observations=[dict(key=list(key),**value) for key,value in extraction.records.items()],
        extraction_policy='only formal main cases with trace_usable, complete ten pairs and raw warm checks; '
                          't1 retained as diagnostics; curves use t8; no missing-value zero fill',
        candidate_equation='S + tiles*L + (tiles-1)*G + W + E',
        additional_limits=['R13/R15 are provenance/consistency constraints, not added whole-window costs.',
                           'Stage6 vs R13 stage1/2/4 remains extrapolation.',
                           'R15 read-completion and full-output gap not identified by R14 source release.',
                           'R14 four CTA -> full card and cfg_c proxies remain hypotheses.',
                           'V05 measured data must not enter this extractor.'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(destination),status=result['status'],
                          accepted_r14_cases=len(extraction.records),missing=len(missing),
                          rejected=len(extraction.rejections)),ensure_ascii=False))


if __name__=='__main__':
    main()
