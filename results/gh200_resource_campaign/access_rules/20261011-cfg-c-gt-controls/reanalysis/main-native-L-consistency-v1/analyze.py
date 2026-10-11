"""Raw within-SM L consistency, without a new gate, rate fit or observer correction."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys


def statistics_of(values):
    return dict(minimum=min(values), median=statistics.median(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    def sha(path):
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs[str(path.resolve())] = value
        return value
    def read(path):
        sha(path)
        return json.loads(path.read_text())
    manifest = read(args.run / 'manifest.json')
    previous = read(args.run / 'analysis.json')
    records = read(args.run / 'returned/records.json')
    assert previous['manifest_sha256'] == sha(args.run / 'manifest.json')
    assert previous['observation_qualified'] is False and not previous['record_errors']
    snapshot = args.run / 'analysis_snapshot'
    evaluator = snapshot / 'analyze_cfg_c_tail_control.py'
    assert sha(evaluator) == manifest['analysis_sha256']['analyze_cfg_c_tail_control.py']
    sys.path.insert(0, str(snapshot.resolve()))
    from analyze_cfg_c_tail_control import metric
    processes = {}
    for record in records:
        assert record['returncode'] == 0
        raw_path = args.run / 'returned' / record['raw']
        assert sha(raw_path) == record['sha256']
        with gzip.open(raw_path, 'rt') as stream:
            setup, call, _ = [json.loads(line) for line in stream]
        assert setup['gpu_uuid'] == manifest['gpu_uuid']
        assert setup['trace_version'] == 'cfg-c-gt-bracket-v1'
        assert call['gt_bracket_observer'] == record['flag']
        trace = call['trace']
        assert len(trace) == 132 * 1296
        windows = {}
        for cta in range(132):
            head = cta * 1296
            sm = trace[head + 2] - 1
            for wg in (0, 1):
                count = trace[head + (6 if wg == 0 else 9)]
                assert count in (1, 2)
                for ordinal in range(count):
                    index = head + 16 + wg * 640 + ordinal * 10
                    cycles = trace[index + 1] - trace[index]
                    ns = trace[index + 7] - trace[index + 6]
                    coord = (trace[index + 4] - 1, trace[index + 5] - 1)
                    key = f'{cta}/{wg}/{ordinal}/{coord[0]},{coord[1]}/L'
                    assert cycles > 0 and ns > 0
                    windows[key] = dict(cycles=cycles, ns=ns, sm=sm, wg=wg, ordinal=ordinal)
        assert len(windows) == 432
        processes[record['comparison'], record['pair'], record['arm']] = windows
    assert len(processes) == 40
    all_rows, summaries = [], {}
    for cohort in ('off_on', 'off_off'):
        old_rows = {row['series']: row for row in previous['cohorts'][cohort]['local'] if row['series'].endswith('/L')}
        assert len(old_rows) == 432
        rows = []
        for key, old in sorted(old_rows.items()):
            pairs = []
            for pair in range(10):
                a = processes[cohort, pair, 0][key]
                b = processes[cohort, pair, 1][key]
                assert a['sm'] == b['sm'] and not old['sm_changed'][pair]
                dc, dn = b['cycles'] - a['cycles'], b['ns'] - a['ns']
                assert dc == old['native_cycle_deltas'][pair] and dn == old['signed'][pair]
                pairs.append(dict(pair=pair, arm0=a, arm1=b, delta_cycles=dc, delta_ns=dn,
                    relative_cycle_difference=dc / a['cycles'], relative_ns_difference=dn / a['ns']))
            row = dict(cohort=cohort, series=key, wg=a['wg'], ordinal=a['ordinal'], pairs=pairs,
                original_ns_gate_passed=old['passed'])
            for unit, field in (('cycles', 'delta_cycles'), ('ns', 'delta_ns'),
                                ('relative_cycles', 'relative_cycle_difference'), ('relative_ns', 'relative_ns_difference')):
                values = [p[field] for p in pairs]
                row[unit] = dict(median_signed=statistics.median(values),
                    median_absolute=statistics.median(abs(x) for x in values), maximum_absolute=max(abs(x) for x in values))
            rows.append(row)
        summaries[cohort] = {}
        for label, selected in [('all', rows)] + [(f'wg{wg}_ordinal{ordinal}', [r for r in rows if r['wg'] == wg and r['ordinal'] == ordinal]) for wg in (0, 1) for ordinal in (0, 1)]:
            summaries[cohort][label] = dict(series=len(selected), pairs=10 * len(selected),
                statistics={unit: {stat: statistics_of([row[unit][stat] for row in selected])
                    for stat in ('median_signed', 'median_absolute', 'maximum_absolute')}
                    for unit in ('cycles', 'ns', 'relative_cycles', 'relative_ns')})
        all_rows.extend(rows)

    # These are executed synthetic counterexamples about estimands, not GH200 observations.
    limit = manifest['gates']
    gate = lambda differences: metric(differences, limit['per_common_series_median_absolute_ns'], limit['per_common_series_maximum_absolute_ns'])
    exact_unbiased_variable = gate([512, -512] * 5)
    cancelled_bias = gate([0] * 10)
    off, on = [10000] * 10, [9400] * 6 + [10900] * 4
    assert statistics.mean(off) == statistics.mean(on) and statistics.median(off) != statistics.median(on)
    tail_on = [10000] * 6 + [11000] * 4
    synthetic = dict(
        unbiased_reader_with_execution_variation=dict(true_observer_effect_ns=0, read_error_ns=0, gate=exact_unbiased_variable),
        execution_and_read_error_cancel=dict(true_observer_effect_ns=512, read_error_difference_ns=-512, gate=cancelled_bias),
        equal_mean_different_median=dict(off=off, on=on, mean_difference=0, median_difference=-600, original_gate=gate([b-a for a,b in zip(off,on)])),
        equal_median_different_tail=dict(off=off, on=tail_on, median_difference=0, original_gate=gate([b-a for a,b in zip(off,tail_on)])))
    assert not exact_unbiased_variable['passed'] and cancelled_bias['passed']
    assert not synthetic['equal_mean_different_median']['original_gate']['passed']
    assert not synthetic['equal_median_different_tail']['original_gate']['passed']
    sha(Path(__file__))
    report = dict(scope='All432 original L coordinates in each cohort; raw native cycle windows and recorded GT differences remain separate.',
        input_sha256=inputs, summaries=summaries, raw_processes=40, L_pair_differences_reproduced=8640,
        synthetic_estimand_examples=synthetic, original_ns_gate_unchanged=True,
        original_L_failure_counts={name: sum(not r['original_ns_gate_passed'] for r in all_rows if r['cohort'] == name) for name in summaries},
        limits=['No GHz conversion, common-mode subtraction, between-cohort center subtraction or new passing threshold.',
                'Cycle differences include execution variation and marker effects; they do not isolate observer bias.',
                'Relative values divide each pair difference by its own arm0 window, not by a pooled median.',
                'Synthetic examples test logic only, not physical GPU execution or a new acceptance protocol.',
                'All original failures remain. Mean/median equivalence alone cannot establish physical timestamp accuracy or last-CTA ordering.'],
        observation_qualified=False, prediction_qualified=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'coordinate-pairs.json.gz').write_bytes(gzip.compress(json.dumps(all_rows, separators=(',', ':')).encode(), mtime=0))
    print(json.dumps({name: value['all'] for name, value in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
