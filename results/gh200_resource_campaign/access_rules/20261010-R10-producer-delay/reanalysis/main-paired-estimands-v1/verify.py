#!/usr/bin/env python3
"""Independent direct-word and sorted-order verification; imports no model code."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

output = Path(__file__).resolve().parent
report_path = output / 'B-paired-estimands.json'
report = json.loads(report_path.read_text())
checks = Counter()
for path, expected in report['input_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    checks['input_hashes'] += 1
for filename, expected in report['output_sha256'].items():
    assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    checks['ledger_hashes'] += 1

saved = json.loads(gzip.decompress((output / 'endpoint-vectors.json.gz').read_bytes()))
saved = {(row['case'], row['trial'], row['cta']): row for row in saved}
direct = {}
for record in report['records']:
    events = {}
    for line in gzip.decompress(Path(record['raw']).read_bytes()).decode().splitlines():
        event = json.loads(line)
        assert event['event'] not in events
        events[event['event']] = event
    setup, call = events['setup'], events['call']
    assert setup['trace_version'] == 'r18-dual-clock-events'
    assert setup['trace_tile_words'] == 10 and setup['trace_words'] == 1296
    assert setup['grid'] == [132, 1, 1] and setup['config'] == 'cfg_b'
    words = call['trace']
    assert len(words) == 132 * 1296
    for cta in range(132):
        head = words[cta * 1296:(cta + 1) * 1296]
        # In cfg_b pingpong the first output belongs to role0, slot0.
        endpoints = {'ns': [head[1], head[13], head[22], head[23]],
            'cycles': [head[0], head[3], head[16], head[17]]}
        key = (record['case'], record['trial'], cta)
        row = saved[key]
        assert row['sm'] == head[2] - 1
        assert row['coord'] == [head[20] - 1, head[21] - 1]
        assert row['endpoints'] == endpoints
        direct[key] = {}
        for domain, (entry, producer, first, main) in endpoints.items():
            values = {'P0': producer-entry, 'S': first-producer, 'L': main-first,
                'S_plus_L': main-producer, 'entry_to_MAIN': main-entry}
            assert row['durations'][domain] == values
            direct[key][domain] = values
            assert values['P0'] + values['S'] + values['L'] == values['entry_to_MAIN']
            checks['direct_duration_scalars'] += len(values)
            checks['same_call_closures'] += 1
        checks['direct_raw_CTA_vectors'] += 1


def sorted_midpoint(values):
    ordered = sorted(values)
    assert len(ordered) == 10
    return (ordered[4] + ordered[5]) / 2


def check_summary(summary, values):
    expected = {field: sorted_midpoint([row[field] for row in values]) for field in values[0]}
    assert summary['medians'] == expected
    assert summary['sum_segment_medians_minus_joint_median'] == expected['S'] + expected['L'] - expected['S_plus_L']
    assert summary['sum_three_medians_minus_total_median'] == expected['P0'] + expected['S'] + expected['L'] - expected['entry_to_MAIN']
    for field in expected:
        assert summary['process_range'][field] == [min(row[field] for row in values), max(row[field] for row in values)]
    checks['summary_medians'] += len(expected)


for summary in report['case_statistics']:
    values = [direct[summary['case'], trial, summary['cta']][summary['domain']] for trial in range(10)]
    check_summary(summary, values)
    checks['case_summaries'] += 1

contrast_rows = json.loads(gzip.decompress((output / 'block-contrasts.json.gz').read_bytes()))
contrast_rows = {(row['contrast'], row['cta'], row['domain'], row['trial']): row for row in contrast_rows}
for summary in report['contrast_statistics']:
    values = []
    for trial in range(10):
        expected = {}
        for field in ('P0', 'S', 'L', 'S_plus_L', 'entry_to_MAIN'):
            expected[field] = sum(sign * direct[name, trial, summary['cta']][summary['domain']][field]
                for name, sign in summary['terms'])
        stored = contrast_rows[summary['contrast'], summary['cta'], summary['domain'], trial]
        assert all(stored[field] == value for field, value in expected.items())
        assert expected['P0'] + expected['S'] + expected['L'] == expected['entry_to_MAIN']
        values.append(expected)
        checks['block_contrast_vectors'] += 1
    check_summary(summary, values)
    for field in expected:
        arm_value = sum(sign * sorted_midpoint([direct[name, trial, summary['cta']][summary['domain']][field]
            for trial in range(10)]) for name, sign in summary['terms'])
        assert summary['contrast_of_arm_medians'][field] == arm_value
        assert summary['arm_median_contrast_minus_block_contrast_median'][field] == arm_value - summary['medians'][field]
    checks['contrast_summaries'] += 1

verification = dict(report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
    checks=dict(checks), independent_method='Direct archived trace words and sorted midpoint; no analysis/model imports.',
    qualification=False, uncertainty_intervals_computed=False, new_tests_or_GPU=False)
(output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
print(json.dumps(verification, indent=2))
