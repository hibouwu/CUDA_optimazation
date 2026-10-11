#!/usr/bin/env python3
"""Audit joint R10 delay endpoints and distinguish median estimands."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

from analyze_r18 import replay


FIELDS = ('P0', 'S', 'L', 'S_plus_L', 'entry_to_MAIN')
DOMAINS = ('ns', 'cycles')
ACCEPTED_REPORT_SHA = 'e7483fc0ad707acaedd9a900ddb916b9208ab3a3b9d1bf84a661e85ec25a2c4e'


def aggregation_summary(values):
    """Keep the combined interval as its own estimand."""
    medians = {field: median(row[field] for row in values) for field in FIELDS}
    for row in values:
        if row['S'] + row['L'] != row['S_plus_L']:
            raise ValueError('S + L does not close within the experimental unit')
        if row['P0'] + row['S_plus_L'] != row['entry_to_MAIN']:
            raise ValueError('P0 + S + L does not close within the experimental unit')
    return dict(
        medians=medians,
        sum_segment_medians_minus_joint_median=medians['S'] + medians['L'] - medians['S_plus_L'],
        sum_three_medians_minus_total_median=medians['P0'] + medians['S'] + medians['L'] - medians['entry_to_MAIN'],
        process_range={field: [min(row[field] for row in values), max(row[field] for row in values)] for field in FIELDS},
    )


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def read_json(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    data = root / '20261010-R10-producer-delay'
    accepted_path = root / '20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-producer-delay.json'
    accepted = read_json(accepted_path)
    if inputs[str(accepted_path)] != ACCEPTED_REPORT_SHA:
        raise ValueError('accepted producer-delay report changed')
    rows = {row['id']: row for row in read_json(data / 'cases.json')}
    for filename in ('static_setup.json', 'environment.json', 'source_hashes.json', 'run_config.json'):
        read_json(data / filename)
    names = [f'r10_proddelay_d{delay}_p{pitch}' for delay in (0, 512) for pitch in (0, 32)]
    accepted_processes = {(row['case'], row['trial']): row for row in accepted['processes']}
    vectors = {}
    records = []
    accepted_scalar_checks = 0
    for name in names:
        vectors[name] = {}
        for trial in range(10):
            record_path = data / 'samples' / name / f'dual-{trial:02d}.json'
            record = read_json(record_path)
            if (record['case'], record['variant'], record['trial'], record['returncode']) != (name, 'dual', trial, 0):
                raise ValueError('successful process identity changed')
            raw = data / record['raw']
            inputs[str(raw)] = hashlib.sha256(raw.read_bytes()).hexdigest()
            if inputs[str(raw)] != record['raw_sha256']:
                raise ValueError('raw process SHA changed')
            observed = replay(data, record, rows[name])
            if len(observed['ctas']) != 132:
                raise ValueError('expected all 132 CTA vectors')
            vectors[name][trial] = {}
            for cta in observed['ctas']:
                endpoints = dict(
                    ns=[cta['entry_ns'], cta['prod_ns'], *cta['tiles_ns'][0][:2]],
                    cycles=[cta['entry_c'], cta['prod_c'], *cta['tiles'][0][:2]],
                )
                durations = {}
                for domain, (entry, producer, first, main) in endpoints.items():
                    if not entry <= producer <= first <= main:
                        raise ValueError('first-output endpoints reversed')
                    durations[domain] = dict(P0=producer-entry, S=first-producer, L=main-first,
                        S_plus_L=main-producer, entry_to_MAIN=main-entry)
                value = dict(case=name, trial=trial, cta=cta['cta'], sm=cta['sm'],
                    coord=cta['work'][0], endpoints=endpoints, durations=durations)
                vectors[name][trial][cta['cta']] = value
                if cta['cta'] == 55:
                    previous = accepted_processes[name, trial]
                    for domain in DOMAINS:
                        for field in FIELDS:
                            old_field = 'entry_to_MAIN_END' if field == 'entry_to_MAIN' else field
                            if durations[domain][field] != previous[old_field + '_' + domain]:
                                raise ValueError('target duration differs from accepted report')
                            accepted_scalar_checks += 1
            records.append(dict(case=name, trial=trial, host_start_ns=record['host_start_ns'],
                host_stop_ns=record['host_stop_ns'], metadata=str(record_path), raw=str(raw)))
    # These are independently launched processes; trial labels define the existing
    # scheduling blocks, not a shared hidden hardware state across the four arms.
    blocks = []
    for trial in range(10):
        ordered = sorted((row for row in records if row['trial'] == trial), key=lambda row: row['host_start_ns'])
        if any(a['host_stop_ns'] > b['host_start_ns'] for a, b in zip(ordered, ordered[1:])):
            raise ValueError('factorial processes overlap in host metadata')
        blocks.append(dict(trial=trial, ordered_cases=[row['case'] for row in ordered]))
    ordered_trials = [row['trial'] for row in sorted(records, key=lambda row: row['host_start_ns'])]
    if ordered_trials != [trial for trial in range(10) for _ in range(4)]:
        raise ValueError('existing trial blocks are not contiguous')

    case_statistics = []
    for name in names:
        for cta in range(132):
            for domain in DOMAINS:
                values = [vectors[name][trial][cta]['durations'][domain] for trial in range(10)]
                case_statistics.append(dict(case=name, cta=cta, domain=domain, **aggregation_summary(values)))

    contrasts = [
        ('delay_p0', [(names[2], 1), (names[0], -1)]),
        ('delay_p32', [(names[3], 1), (names[1], -1)]),
        ('pitch_d0', [(names[1], 1), (names[0], -1)]),
        ('pitch_d512', [(names[3], 1), (names[2], -1)]),
        ('delay_by_pitch', [(names[3], 1), (names[2], -1), (names[1], -1), (names[0], 1)]),
    ]
    contrast_statistics = []
    contrast_vectors = []
    for label, terms in contrasts:
        for cta in range(132):
            for domain in DOMAINS:
                values = []
                for trial in range(10):
                    value = {field: sum(sign * vectors[name][trial][cta]['durations'][domain][field]
                        for name, sign in terms) for field in FIELDS}
                    values.append(value)
                    contrast_vectors.append(dict(contrast=label, cta=cta, domain=domain, trial=trial, **value))
                summary = aggregation_summary(values)
                marginal_contrast = {field: sum(sign * median(vectors[name][trial][cta]['durations'][domain][field]
                    for trial in range(10)) for name, sign in terms) for field in FIELDS}
                summary['contrast_of_arm_medians'] = marginal_contrast
                summary['arm_median_contrast_minus_block_contrast_median'] = {
                    field: marginal_contrast[field] - summary['medians'][field] for field in FIELDS}
                contrast_statistics.append(dict(contrast=label, cta=cta, domain=domain, terms=terms, **summary))

    for path, digest in inputs.items():
        if path in accepted['input_sha256'] and digest != accepted['input_sha256'][path]:
            raise ValueError('accepted input SHA changed: ' + path)
    reused = sum(path in accepted['input_sha256'] for path in inputs)
    for filename in ('analyze_r10_paired_estimands.py', 'analyze_r18.py', 'v06_run.py', 'v06_model.py'):
        path = Path(__file__).with_name(filename).resolve()
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    ordered_vectors = [vectors[name][trial][cta] for name in names for trial in range(10) for cta in range(132)]
    for filename, values in [('endpoint-vectors.json.gz', ordered_vectors), ('block-contrasts.json.gz', contrast_vectors)]:
        (output / filename).write_bytes(gzip.compress(json.dumps(values, separators=(',', ':')).encode(), mtime=0))
    report = dict(
        scope='Retrospective estimand audit of existing R10 first-output supply intervals; no new validation, fitting, or uncertainty qualification.',
        definitions=dict(P0='producer-entry', S='FIRST-producer', L='MAIN-FIRST',
            S_plus_L='MAIN-producer', entry_to_MAIN='MAIN-entry'),
        counts=dict(processes=len(records), CTA_vectors=len(ordered_vectors), case_summaries=len(case_statistics),
            contrast_summaries=len(contrast_statistics), block_contrast_vectors=len(contrast_vectors),
            accepted_target_scalar_checks=accepted_scalar_checks, accepted_input_hashes_reused=reused),
        scheduling_blocks=blocks, records=records, case_statistics=case_statistics,
        contrast_statistics=contrast_statistics,
        target_case_statistics=[row for row in case_statistics if row['cta'] == 55],
        target_contrast_statistics=[row for row in contrast_statistics if row['cta'] == 55],
        requirements=[
            'Freeze combined-window medians and segment medians as distinct estimands; a median vector need not close additively.',
            'Retain all endpoints from the same invocation before subtraction, aggregation, or resampling.',
            'Freeze whether the effect is a median of scheduled block contrasts or a contrast of arm medians; they are distinct.',
            'An uncertainty analysis must preserve its declared process/block unit and within-call endpoint covariance; CTAs and segments are not independent process replicas.',
            'The old trial blocks contain distinct sequential processes. They do not establish paired identical initial state or a pathwise intervention counterfactual.',
            'Native-cycle differences are local to each CTA. Do not subtract absolute clock64 readings across SMs.',
            'Median control, pointwise order, critical-CTA selection, and service identification remain separate claims.',
            'This audit does not qualify old controls, provide confidence intervals, identify service prices, or freeze a new holdout.',
        ], input_sha256=inputs,
        output_sha256={filename: hashlib.sha256((output / filename).read_bytes()).hexdigest()
            for filename in ('endpoint-vectors.json.gz', 'block-contrasts.json.gz')},
    )
    (output / 'B-paired-estimands.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(counts=report['counts'], target_contrasts=[dict(
        contrast=row['contrast'], domain=row['domain'], joint_medians=row['medians'],
        segment_gap=row['sum_segment_medians_minus_joint_median'],
        arm_contrast_gap=row['arm_median_contrast_minus_block_contrast_median'])
        for row in report['target_contrast_statistics']]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.output.resolve())
