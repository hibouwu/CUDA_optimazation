#!/usr/bin/env python3
"""B-only offline ledger/lifecycle audit. Existing reports and samples are read-only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from analyze_r13_supply import (a_source_rotation_diagnostics,
    local_epilogue_exclusion_diagnostics, stage_source_candidate_diagnostics,
    stage_source_identity)
from supply_model import input_slot_state, shared_stage_events, supply_work_ledger


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def audit_events(model, kt, stages, parameters):
    """Check the implemented graph, without interpreting its ns as hardware costs."""
    eps = 1e-7
    failures = []
    checks = 0

    def require(condition, label, cta, group):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(dict(check=label, cta=cta, group=group))

    for cta, actor in enumerate(model['ctas']):
        for i in range(2*kt):
            state = input_slot_state(i, stages)
            require(actor['ready_count'][i] == 3, 'all three destination atoms', cta, i)
            require(actor['starts'][i]+eps >= actor['ready'][i], 'full before consumption', cta, i)
            require(actor['commits'][i]+eps >= actor['starts'][i], 'begin before commit', cta, i)
            require(actor['released'][i]+eps >= actor['completed'][i], 'safe release after retirement', cta, i)
            previous = state['previous_group']
            if previous is not None:
                require(actor['issued_a'][i]+eps >= actor['released'][previous]+parameters.get('reuse_delay_ns', 0.),
                    'prior slot generation released before reuse', cta, i)
            if i % kt != kt-1:
                require(actor['released'][i] == actor['wait1_return'][i+1],
                    'wait1 releases previous group', cta, i)
            else:
                require(actor['released'][i] == actor['main_ends'][i//kt],
                    'final group released through tail', cta, i)
        for out, epi in enumerate(actor['outputs']):
            require(epi['begin']+eps >= actor['main_ends'][out], 'tail before epilogue', cta, out)
            for j in range(8):
                require(epi['submitted'][j]+eps >= epi['write_end'][j], 'SMEM written before D submit', cta, j)
                if j >= 2:
                    require(epi['write_begin'][j]+eps >= epi['source_done'][j-2],
                        'D source buffer safe before overwrite', cta, j)
    demands = model['input_demands']
    sources = {s['source_id']: s for s in model['sources']}
    for demand in demands:
        source = sources[demand['source_id']]
        if demand['merged']:
            require(demand['arrival_ns'] < source['join_end_ns'],
                'strict source-generation subscription', demand['cta'], demand['index'])
    require(model['input_source_count']+model['input_merged_count'] == model['input_demand_count'],
        'source plus subscriptions conserve demand', None, None)
    require(model['input_delivery_count'] == model['input_demand_count'],
        'all destinations delivered', None, None)
    require(model['input_delivery_work'] == model['input_demand_work'],
        'destination work is never merged', None, None)
    require(len([r for r in model['requests'] if r['kind']=='store']) == len(model['ctas'])*16,
        'all D source reads retained', None, None)
    if failures:
        raise ValueError(f'event graph violations: {failures[:8]}')
    return dict(checks=checks, failures=failures,
        demand_count=model['input_demand_count'], source_count=model['input_source_count'],
        merged_count=model['input_merged_count'], delivery_count=model['input_delivery_count'],
        source_work_16KiB=model['input_source_work'], destination_work_16KiB=model['input_delivery_work'],
        output_atoms_8KiB=len(model['ctas'])*16,
        limitation='Checks implementation dependencies only; release event currently combines emit/visibility. No hardware acceptance.')


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    previous = root/'Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1'
    rotation = root/'Q02-tma-request/R13-async-retirement/20261011-R13-a-row-rotation'
    inputs = {}
    manifest = read(rotation/'handoff-manifest.json')
    for item in manifest['files']:
        path = Path(manifest['shared_checkout'])/item['path']
        if sha(path) != item['sha256']:
            raise ValueError('handoff input changed: '+str(path))
        inputs[str(path)] = item['sha256']
    replay_dir = output/'replay'
    replay_dir.mkdir()
    for name in ('B-second-prefill.json', 'B-source-candidate-examples.json', 'B-epilogue-order.json'):
        path = previous/name
        inputs[str(path)] = sha(path)
        shutil.copy2(path, replay_dir/name)
    prefill = root/'Q02-tma-request/R13-async-retirement/20261010-R13-second-prefill'
    candidate = stage_source_candidate_diagnostics(prefill, replay_dir)
    exclusion = local_epilogue_exclusion_diagnostics(prefill, replay_dir)
    rotated = a_source_rotation_diagnostics(rotation, replay_dir)
    replay_checks = []
    for name, current, fields in (
        ('B-source-candidate-replay', candidate, ('predictions','processes','summaries','responses','insertion_sensitivity')),
        ('B-local-epilogue-exclusion', exclusion, ('predictions','observations','comparisons','defer2_invariants')),
        ('B-a-source-rotation', rotated, ('processes','pairs','summary','repeatability'))):
        old_path = (rotation/'reanalysis' if name=='B-a-source-rotation' else previous)/(name+'.json')
        old = read(old_path)
        for field in fields:
            if field not in current or field not in old:
                raise ValueError(f'missing replay field: {name}/{field}')
            if current[field] != old[field]:
                raise ValueError(f'replay changed: {name}/{field}')
            replay_checks.append(name+'/'+field)

    ledgers = []
    for run_name in ('Q02-tma-request/R13-async-retirement/20261009-R13-R15-R18-composition-job738496',
        'Q02-tma-request/R18-first-fill/20261009-R18-first-fill-job738707', 'Q02-tma-request/R18-first-fill/20261009-R18-first-fill-job738865',
        'Q03-source-service/R10-layout-cache/20261010-R10-raster-pitch', 'Q03-source-service/R10-layout-cache/20261010-R10-source-pressure',
        'Q02-tma-request/R13-async-retirement/20261010-R13-second-prefill'):
        run = root/run_name
        cases_path, setup_path = run/'cases.json', run/'static_setup.json'
        cases = read(cases_path)
        setups = {v['case']: v['setup'] for v in read(setup_path)}
        inputs[str(cases_path)] = sha(cases_path)
        inputs[str(setup_path)] = sha(setup_path)
        for row in cases:
            ledger = supply_work_ledger(row, setups[row['id']])
            ledgers.append(dict(run=run_name, case=row['id'], ledger=ledger))
    raw = json.dumps(ledgers, separators=(',', ':'), allow_nan=False).encode()
    (output/'B-work-ledgers.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    summaries = [dict(run=r['run'], case=r['case'], config=r['ledger']['config'],
        stages=r['ledger']['stages'], ktiles=r['ledger']['ktiles'], **r['ledger']['totals']) for r in ledgers]
    save(output/'B-work-ledger-summary.json', summaries)

    row = next(r for r in read(prefill/'cases.json') if r.get('set')!='reference')
    setup = next(v['setup'] for v in read(prefill/'static_setup.json') if v['case']==row['id'])
    keys = stage_source_identity(row, setup)['source_keys']
    event_checks = []
    for name, example in read(previous/'B-source-candidate-examples.json')['examples'].items():
        for stages in (4, 6):
            for flag in (False, True):
                model = shared_stage_events(64, stages, 132, epilogue_defer=2,
                    share_inflight_sources=name=='shared', source_keys=keys,
                    second_prefill=flag, trace_requests=True, **example['parameters'])
                event_checks.append(dict(candidate=name, stages=stages, second_prefill=flag,
                    **audit_events(model, 64, stages, example['parameters'])))
    save(output/'B-lifecycle-implementation-checks.json', event_checks)

    # Every listed window/intervention is retained in its native domain. This
    # is an evidence extraction, not a new fit, acceptance test or task ledger.
    fields = {
        'B-stage-events': ['processes', 'paired_summary', 'interval_limits'],
        'B-stage-progress': ['processes', 'paired_summary', 'observer_instruction_evidence', 'interval_limits'],
        'B-stage-handoff-samples': ['processes', 'limitations'],
        'B-epilogue-order': ['deferred_minus_normal', 'progress_minus_plain', 'limitations'],
        'B-stage-background': ['all_minus_cta0', 'paired_diagnostic_subsets', 'limitations'],
        'B-second-prefill': ['on_minus_off', 'prefilled_constant_reuse', 'wide_constant_candidates', 'limitations'],
        'B-source-spatial': ['summaries', 'distinct_work_sm_maps', 'limitations'],
        'B-source-offsets': ['summaries', 'limitations'],
        'B-source-join-boundary': ['observed', 'candidates', 'limitations'],
        'R13-ready-release': ['paired_summaries', 'total_coverage', 'delay_bound_checks', 'startup_shift_summaries', 'interpretation'],
        'R10-raster-pitch': ['summaries', 'interaction_summary', 'current_model_blindness', 'limits'],
        'R10-source-pressure': ['summaries', 'interaction_summary', 'limits'],
        'R10-base-shift': ['pattern_summaries', 'same_position_interactions', 'selected_CTA_interactions', 'interpretation'],
        'R10-work-swap': ['correspondence_summaries', 'unmodified_CTA_summary', 'interpretation'],
        'R10-producer-delay': ['direct_delay_summaries', 'delay_by_pitch_interaction', 'interpretation'],
    }
    constraints = []
    for name, selected in fields.items():
        path = previous/(name+'.json')
        data = read(path)
        inputs[str(path)] = sha(path)
        evidence = {k: data[k] for k in selected}
        constraints.append(dict(report=str(path), sha256=sha(path), scope=data.get('scope'),
            evidence=evidence))
    save(output/'B-existing-constraints.json', constraints)
    code = [Path(__file__), Path(__file__).with_name('supply_model.py'),
        Path(__file__).with_name('analyze_r13_supply.py'), Path(__file__).with_name('v08_model.py')]
    for path in code:
        inputs[str(path.resolve())] = sha(path)
    save(output/'B-verification.json', dict(handoff_files_verified=len(manifest['files']),
        replay_fields_exact=replay_checks, static_cases=len(ledgers),
        event_graphs=len(event_checks), dependency_checks=sum(v['checks'] for v in event_checks),
        input_sha256=inputs, output_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        status='Development audit only. No new fitted price, frozen local holdout or GPU measurement.'))
    print(json.dumps(dict(output=str(output), static_cases=len(ledgers), event_graphs=len(event_checks),
        checks=sum(v['checks'] for v in event_checks), replay_fields=len(replay_checks))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new exclusive B reanalysis directory')
    args = parser.parse_args()
    analyze(args.data_root.resolve(), args.output.resolve())
