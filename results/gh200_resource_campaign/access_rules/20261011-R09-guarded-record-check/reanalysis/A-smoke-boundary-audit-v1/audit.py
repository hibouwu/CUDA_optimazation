#!/usr/bin/env python3
"""A independent read-only replay of 740071 and copies for parser negatives."""
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

out = Path(__file__).resolve().parent
run = out.parents[1]
inputs = {}


def read(path):
    value = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(value).hexdigest()
    return value


manifest = json.loads(read(run / 'manifest.json'))
build = Path(manifest['build_run'])
assert hashlib.sha256(read(build / 'source_hashes.json')).hexdigest() == manifest['source_manifest_sha256']
assert hashlib.sha256(read(build / 'build-audit.json')).hexdigest() == manifest['build_audit_sha256']
for name, digest in json.loads((build / 'source_hashes.json').read_text()).items():
    assert hashlib.sha256(read(build / name)).hexdigest() == digest
for name, digest in json.loads(read(run / 'freeze.json'))['files'].items():
    assert hashlib.sha256(read(run / name)).hexdigest() == digest
    assert hashlib.sha256(read(run / 'returned' / name)).hexdigest() == digest
for name, digest in manifest['binaries'].items():
    for base in (run / 'build', run / 'returned/build', build / 'returned/build'):
        assert hashlib.sha256(read(base / name)).hexdigest() == digest
for name, digest in manifest['analysis_sha256'].items():
    assert hashlib.sha256(read(run / 'analysis_snapshot' / name)).hexdigest() == digest
receipt = json.loads(read(run / 'receipt.json'))
assert not receipt['returncode'] and hashlib.sha256(read(run / 'output.tar.gz')).hexdigest() == receipt['sha256']
sys.path.insert(0, str(run / 'analysis_snapshot'))
from decode_r09_clock_observer import decode
from analyze_r18 import reference

records = json.loads(read(run / 'returned/records.json'))
assert [{key: row[key] for key in ('arm', 'command')} for row in records] == manifest['samples']
original = json.loads(read(run / 'verification.json'))
conditional = json.loads(read(run / 'reanalysis/conditional-brackets-v1/conditional-brackets.json'))
decoded, raw_events, resources = {}, {}, []
for row in records:
    assert not row['returncode']
    path = run / 'returned' / row['raw']
    assert hashlib.sha256(read(path)).hexdigest() == row['sha256']
    with gzip.open(path, 'rt') as stream:
        packets = [json.loads(line) for line in stream]
    events = {packet['event']: packet for packet in packets}
    assert len(events) == len(packets)
    setup, call, check = [events[key] for key in ('setup', 'call', 'check')]
    assert check['status'] == 'ok' and check['samples'] == 4096 and not check['nonfinite'] and not check['padding_errors']
    indices, values = check['checked_indices'], check['checked_values']
    assert len(indices) == len(set(indices)) == len(values) == 4096
    for index, value in zip(indices, values):
        assert type(index) is int and 0 <= index < 2816 * 1536
        assert math.isfinite(value) and value == reference(index // 1536, index % 1536, 4096, 17)
    result = decode(setup, call, row['arm'], setup['gpu_uuid'])
    assert result['record_valid'] and not result['observation_qualified']
    assert json.loads(json.dumps(result)) == next(item['decoded'] for item in original['results'] if item['arm'] == row['arm'])
    if row['arm'] != 'plain':
        trace = call['trace']
        for cta in range(132):
            b = trace[1552 * cta:1552 * (cta + 1)]
            assert 0 < b[2] <= setup['sm_count'] and b[11] == b[12] == b[2]
            assert not b[10] and not any(b[14:16])
            assert 0 < b[0] <= b[3] and 0 < b[1] <= b[13]
            for role in (1, 2):
                offset = 16 + 768 * (role - 1)
                tile = b[offset:offset + 12]
                assert b[0] <= tile[0] and b[1] <= tile[6]
                assert b[4 + 3 * (role - 1)] >= tile[3] and b[5 + 3 * (role - 1)] >= tile[9]
    raw_events[row['arm']] = events
    decoded[row['arm']] = result['records']
    resources.append({key: setup[key] for key in ('smem', 'smem_epilogue', 'max_active_ctas_per_sm', 'sm_count', 'scheduler_sm_count')})
assert len(records) == 3 and len({events['setup']['gpu_uuid'] for events in raw_events.values()}) == 1
assert resources == [dict(smem=215040, smem_epilogue=18432, max_active_ctas_per_sm=1, sm_count=132, scheduler_sm_count=132)] * 3
pairs = []
widths = {'FIRST': [], 'MAIN': []}
for cta in range(132):
    rows = [row for row in decoded['guarded_bracket'] if row['cta'] == cta]
    assert [row['role'] for row in rows] == [1, 2] and rows[0]['sm'] == rows[1]['sm']
    bounds = []
    for row in rows:
        begin, end = row['window']['begin'], row['window']['end']
        lo, hi, ns = max(0, end[0] - begin[2]), end[2] - begin[0], end[1] - begin[1]
        assert ns >= 3200
        bounds.append((lo, hi, ns))
        widths['FIRST'].append(begin[2] - begin[0])
        widths['MAIN'].append(end[2] - end[0])
    lower = max(Fraction(lo, ns + 128) for lo, hi, ns in bounds)
    upper = min(Fraction(hi, ns - 128) for lo, hi, ns in bounds)
    delta = max([Fraction(0)] + [Fraction(na * lob - nb * hia, lob + hia)
                                for loa, hia, na in bounds for lob, hib, nb in bounds if lob])
    assert max(Fraction(lo) / (ns + delta) for lo, hi, ns in bounds) <= min(Fraction(hi) / (ns - delta) for lo, hi, ns in bounds if ns > delta)
    main_pair = conditional['cta_pairs'][cta]
    assert main_pair['cta'] == cta and main_pair['lower_at_128'] == [lower.numerator, lower.denominator]
    assert main_pair['upper_at_128'] == [upper.numerator, upper.denominator]
    assert main_pair['result']['minimum_interval_slack_fraction'] == [delta.numerator, delta.denominator]
    pairs.append(dict(cta=cta, sm=rows[0]['sm'], feasible_at_128=lower <= upper,
                      lower_fraction=[lower.numerator, lower.denominator], upper_fraction=[upper.numerator, upper.denominator],
                      minimum_total_interval_slack_fraction=[delta.numerator, delta.denominator],
                      minimum_total_interval_slack_ns=float(delta)))
negatives = []
mutations = [('entry_c_zero', {0: 0}), ('entry_ns_zero', {1: 0}), ('producer_c_zero', {3: 0}),
             ('producer_ns_zero', {13: 0}), ('out_of_range_sm', {2: 133, 11: 133, 12: 133})]
for arm in ('matched_dual', 'guarded_bracket'):
    setup, original_call = raw_events[arm]['setup'], raw_events[arm]['call']
    for name, changes in mutations:
        call = dict(original_call, trace=list(original_call['trace']))
        for word, value in changes.items():
            call['trace'][word] = value
        result = decode(setup, call, arm, setup['gpu_uuid'])
        negatives.append(dict(arm=arm, mutation=name, cta=0, changed_head_words=changes,
                              record_valid=result['record_valid'], roles=len(result['records']), errors=result['errors'],
                              scope='Synthetic copy only; original raw unchanged.'))
assert all(row['record_valid'] for row in negatives)
stats = lambda values: dict(minimum=min(values), median=statistics.median(values), maximum=max(values))
read(out / 'audit.py')
report = dict(schema='gh200-r09-smoke-boundary-audit/1', input_sha256=inputs,
              numerical_values_verified=12288, actual_raw_complete_heads_valid=True, actual_packets_valid=True,
              resources=resources, bracket_width_cycles={name: stats(values) for name, values in widths.items()},
              cta_pairs=pairs, feasible_at_128=sum(pair['feasible_at_128'] for pair in pairs),
              minimum_total_interval_slack_ns=stats([pair['minimum_total_interval_slack_ns'] for pair in pairs]),
              packet_negative_copies=negatives, parser_complete_head_refusal=False,
              observation_qualified=False, original_phenomenon_inference='inconclusive',
              limits=['One actual call per arm, no independent perturbation distribution or local holdout.',
                      'Rate intersection assumes hardware sampling enclosure; this replay does not establish that assumption.',
                      '128ns remains total interval allowance; minimum slack is diagnostic and never replaces it.',
                      'API maximum-active-CTA input is not execution occupancy sampling.',
                      'Negative copies reveal packet validation gaps but do not change original raw or its numerical checks.',
                      'The snapshot is frozen; future reader repairs must retain this version and its refusal counterexamples.'])
target = out / 'audit.json'
if target.exists():
    assert json.loads(target.read_text()) == json.loads(json.dumps(report))
else:
    target.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ('numerical_values_verified', 'feasible_at_128', 'minimum_total_interval_slack_ns', 'parser_complete_head_refusal', 'observation_qualified')}))
