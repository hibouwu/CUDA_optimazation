#!/usr/bin/env python3
"""Read-only frozen V09 observation audit. No thresholds, refit, or pass decision."""
import argparse
import json
import math
from pathlib import Path
import statistics as st

from analyze_r13_sm import select_attempts
from analyze_r18 import replay
from v06_run import sha


COMPONENTS = ('prefix', 'prefill', 'first_L', 'later_L', 'last_E', 'tail')


def error_summary(values, invalid=0):
    if not values:
        return dict(windows=0, invalid_windows=invalid, signed_median_pct=None,
                    median_abs_pct=None, max_abs_pct=None)
    return dict(windows=len(values), invalid_windows=invalid,
                signed_median_pct=100*st.median(values),
                median_abs_pct=100*st.median(abs(v) for v in values),
                max_abs_pct=100*max(abs(v) for v in values))


def process_audit(ctas, prediction):
    predicted = prediction['cta_events']
    if len(ctas) != len(predicted):
        raise ValueError('CTA count mismatch; truncation is forbidden')
    errors = {key: [] for key in COMPONENTS}
    invalid = {key: 0 for key in COMPONENTS}
    for index, (cta, p) in enumerate(zip(ctas, predicted)):
        if cta['cta'] != index:
            raise ValueError('CTA identity/order mismatch')
        actual, events = cta['tiles_ns'], p['events']
        if len(actual) != len(events):
            raise ValueError('tile count mismatch; truncation is forbidden')
        if not actual:
            continue
        windows = [
            ('prefix', actual[0][0]-cta['entry_ns'], p['supply']),
            ('prefill', actual[0][0]-cta['prod_ns'], p['prefill']),
            ('first_L', actual[0][1]-actual[0][0], events[0][1]-events[0][0]),
            ('last_E', actual[-1][3]-actual[-1][2], events[-1][3]-events[-1][2]),
            ('tail', cta['end_ns']-max(t[3] for t in actual), p['tail']),
        ]
        windows += [('later_L', a[1]-a[0], b[1]-b[0])
                    for a, b in zip(actual[1:], events[1:])]
        for key, observed, estimate in windows:
            if observed <= 0 or not math.isfinite(observed) or not math.isfinite(estimate):
                invalid[key] += 1
            else:
                errors[key].append(estimate/observed-1)
    chosen = prediction['critical_cta']
    candidates = prediction['critical_candidates']
    if not candidates or len(set(candidates)) != len(candidates) or chosen not in candidates:
        raise ValueError('invalid frozen critical candidates')
    if any(not isinstance(c, int) or c < 0 or c >= len(ctas) for c in candidates):
        raise ValueError('critical CTA out of range')
    active = [c for c in ctas if c['tiles_ns']]
    if not active:
        raise ValueError('no active CTA')
    last_end = max(c['end_ns'] for c in active)
    first_entry = min(c['entry_ns'] for c in ctas)
    observed_last = [c['cta'] for c in active if c['end_ns'] == last_end]
    regret = last_end-ctas[chosen]['end_ns']
    return dict(components={k: error_summary(errors[k], invalid[k]) for k in COMPONENTS},
                active_ctas=len(active), critical=dict(
                    chosen=chosen, frozen_candidate_count=len(candidates),
                    observed_exact_last=observed_last,
                    chosen_is_exact_last=chosen in observed_last,
                    frozen_set_intersects_exact_last=bool(set(candidates)&set(observed_last)),
                    end_regret_ns=regret,
                    end_regret_percent=100*regret/(last_end-first_entry)))


def audit(root):
    frozen_path = root/'frozen/predictions.json'
    frozen = json.loads(frozen_path.read_text())
    rows = json.loads((root/'cases.json').read_text())
    binding = json.loads((root/'prediction_binding.json').read_text())
    if sha(frozen_path) != binding['prediction_sha256']:
        raise ValueError('frozen prediction differs from approved binding')
    if sha(root/'frozen/manifest.json') != binding['manifest_sha256']:
        raise ValueError('frozen manifest differs from approved binding')
    provenance = {str(p.relative_to(root)): sha(p) for p in
                  (frozen_path, root/'frozen/manifest.json', root/'cases.json', root/'prediction_binding.json')}
    cases = []
    for row in rows:
        if row['set'] != 'heldout':
            continue
        prediction = frozen['predictions'][row['id']]
        selected, failed, _ = select_attempts(root, row['id'], ('dual',))
        processes = []
        for item in selected:
            record = item['record']
            provenance[item['path']] = sha(root/item['path'])
            if record['host_start_ns'] < binding['first_sample_unix_ns']:
                raise ValueError('observation predates frozen binding')
            observed = replay(root, record, row)
            provenance[record['raw']] = record['raw_sha256']
            if prediction['status'] == 'predicted':
                processes.append(dict(trial=record['trial'], **process_audit(observed['ctas'], prediction)))
        cases.append(dict(case=row['id'], config=row['config'], status=prediction['status'],
                          selected_dual_processes=len(selected), failed_attempts=failed,
                          processes=processes))
    summary = {}
    for key in COMPONENTS:
        per_case = []
        for case in cases:
            windows = [p['components'][key] for p in case['processes']]
            valid = [w for w in windows if w['windows']]
            if valid:
                per_case.append(dict(case=case['case'],
                    abs_legacy_signed_median_pct=abs(st.median(w['signed_median_pct'] for w in valid)),
                    median_of_process_median_abs_pct=st.median(w['median_abs_pct'] for w in valid),
                    worst_window_abs_pct=max(w['max_abs_pct'] for w in valid),
                    invalid_windows=sum(w['invalid_windows'] for w in windows)))
        summary[key] = per_case
    return dict(scope='Frozen V09 development audit only; no new acceptance thresholds or pass decision. '
                'Absolute errors are taken before CTA aggregation. Critical regret uses common globaltimer endpoints. '
                'Observed exact endpoint ties do not establish timer resolution or observer validity.',
                root=str(root.resolve()), frozen_policy=frozen['policy'], provenance=provenance,
                cases=cases, component_summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('write audit outside the historical run')
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(cases=len(result['cases']), provenance_files=len(result['provenance']),
                         output=str(args.output))))


if __name__ == '__main__':
    main()
