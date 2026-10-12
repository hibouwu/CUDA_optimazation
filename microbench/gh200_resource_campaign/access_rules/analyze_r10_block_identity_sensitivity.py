#!/usr/bin/env python3
"""Exhaust same-arm process transpositions in the archived R10 block estimands."""
import argparse
from collections import Counter
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import median


FIELDS = ('P0', 'S', 'L', 'S_plus_L', 'entry_to_MAIN')
NAMES = [f'r10_proddelay_d{delay}_p{pitch}' for delay in (0, 512) for pitch in (0, 32)]
CONTRASTS = {
    'delay_p0': [(NAMES[2], 1), (NAMES[0], -1)],
    'delay_p32': [(NAMES[3], 1), (NAMES[1], -1)],
    'pitch_d0': [(NAMES[1], 1), (NAMES[0], -1)],
    'pitch_d512': [(NAMES[3], 1), (NAMES[2], -1)],
    'delay_by_pitch': [(NAMES[3], 1), (NAMES[2], -1), (NAMES[1], -1), (NAMES[0], 1)],
}


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def sha(path):
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs[str(path)] = value
        return value

    previous = root / '20261010-R10-producer-delay/reanalysis/B-20261011-paired-estimands-v1'
    report_path = previous / 'B-paired-estimands.json'
    if sha(report_path) != '5acf156511a9a77b05cf23840a654e2e7704a4d48d498121db681a23fead27ed':
        raise ValueError('accepted joint-estimand report changed')
    report = json.loads(report_path.read_text())
    for path, expected in report['input_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError('accepted input changed: ' + path)
    vector_path = previous / 'endpoint-vectors.json.gz'
    if sha(vector_path) != report['output_sha256'][vector_path.name]:
        raise ValueError('accepted joint endpoint vectors changed')
    vectors = json.loads(gzip.decompress(vector_path.read_bytes()))
    if len(vectors) != 5280:
        raise ValueError('expected forty complete132-CTA process vectors')
    # A transposition relabels whole processes. This projection reads CTA55 only;
    # no endpoint, coordinate, clock domain, or component is independently moved.
    target = {(row['case'], row['trial']): row for row in vectors if row['cta'] == 55}
    if len(target) != 40:
        raise ValueError('expected forty target-CTA vectors')
    baseline = {(row['contrast'], row['domain']): row for row in report['target_contrast_statistics']}
    case_multisets = {}
    for name in NAMES:
        for domain in ('ns', 'cycles'):
            for field in FIELDS:
                case_multisets[name, domain, field] = sorted(
                    target[name, trial]['durations'][domain][field] for trial in range(10))
    results = []
    checks = Counter()
    for changed_case in NAMES:
        for first, second in combinations(range(10), 2):
            mapping = {name: list(range(10)) for name in NAMES}
            mapping[changed_case][first], mapping[changed_case][second] = second, first
            for name in NAMES:
                if sorted(mapping[name]) != list(range(10)):
                    raise ValueError('process mapping is not a permutation')
                for domain in ('ns', 'cycles'):
                    for field in FIELDS:
                        current = sorted(target[name, trial]['durations'][domain][field] for trial in mapping[name])
                        if current != case_multisets[name, domain, field]:
                            raise ValueError('transposition changed an arm marginal')
                        checks['arm_marginal_multisets_unchanged'] += 1
            for label, terms in CONTRASTS.items():
                for domain in ('ns', 'cycles'):
                    medians = {}
                    marginal_contrast = {}
                    for field in FIELDS:
                        paired = [sum(sign * target[name, mapping[name][trial]]['durations'][domain][field]
                            for name, sign in terms) for trial in range(10)]
                        medians[field] = median(paired)
                        marginal_contrast[field] = sum(sign * median(case_multisets[name, domain, field])
                            for name, sign in terms)
                    if marginal_contrast != baseline[label, domain]['contrast_of_arm_medians']:
                        raise ValueError('arm median contrast changed under a permutation')
                    results.append(dict(changed_case=changed_case, swapped_trials=[first, second],
                        contrast=label, domain=domain, block_contrast_medians=medians,
                        change_from_original={field: medians[field] - baseline[label, domain]['medians'][field]
                            for field in FIELDS}, contrast_of_arm_medians=marginal_contrast))
                    checks['arm_median_contrasts_unchanged'] += len(FIELDS)
    summaries = []
    for label in CONTRASTS:
        for domain in ('ns', 'cycles'):
            selected = [row for row in results if (row['contrast'], row['domain']) == (label, domain)]
            if len(selected) != 180:
                raise ValueError('expected all four arms times45 transpositions')
            for field in FIELDS:
                low = min(selected, key=lambda row: row['block_contrast_medians'][field])
                high = max(selected, key=lambda row: row['block_contrast_medians'][field])
                summaries.append(dict(contrast=label, domain=domain, field=field,
                    original_block_median=baseline[label, domain]['medians'][field],
                    invariant_arm_median_contrast=baseline[label, domain]['contrast_of_arm_medians'][field],
                    transposition_median_range=[low['block_contrast_medians'][field], high['block_contrast_medians'][field]],
                    changed_count=sum(row['change_from_original'][field] != 0 for row in selected),
                    lower_witness={key: low[key] for key in ('changed_case', 'swapped_trials')},
                    upper_witness={key: high[key] for key in ('changed_case', 'swapped_trials')}))
    if not any(row['changed_count'] for row in summaries):
        raise ValueError('expected actual-data paired-estimand counterexample')
    filename = 'same-arm-transpositions.json.gz'
    compressed = gzip.compress(json.dumps(results, separators=(',', ':')).encode(), mtime=0)
    (output / filename).write_bytes(compressed)
    sha(Path(__file__).resolve())
    audit = dict(
        scope='Retrospective identity-sensitivity counterexamples for CTA55 in existing R10 data; no claim of actual archive reassignment.',
        counts=dict(original_processes=40, original_CTA_vectors=5280, target_CTA=55,
            transpositions=180, response_domain_rows=len(results), scalar_response_medians=len(results)*len(FIELDS)),
        checks=dict(checks), summaries=summaries,
        conclusions=[
            'Every transposition preserves each whole-process vector and all arm marginal distributions; arm-median contrasts therefore remain identical.',
            'Scheduled-block contrast medians can change solely through process-to-block reassignment.',
            'Raw-byte integrity, complete joint vectors and distinct process hashes do not by themselves prove acquisition ownership or correct cross-process pairing.',
            'Prospective paired supply responses require a verified process-to-acquisition/block binding before statistical projection.',
            'This enumeration is a sensitivity certificate, not a confidence interval, noise estimate, service fit, old-score correction or new observation qualification.',
        ], input_sha256=inputs, output_sha256={filename: hashlib.sha256(compressed).hexdigest()},
        observation_qualified=False, prediction_qualified=False,
    )
    (output / 'B-block-identity-sensitivity.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(counts=audit['counts'], checks=audit['checks'], S_plus_L=[row for row in summaries
        if row['field'] == 'S_plus_L']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.output.resolve())
