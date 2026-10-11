#!/usr/bin/env python3
"""A-only native-cycle versus time conversion replay; no fitted coefficients."""
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st

out = Path(__file__).resolve().parent
run = out.parents[1]
main = run / 'reanalysis/main-cfg-c-cross-k-v1'
report = json.loads((main / 'report.json').read_text())
inputs = dict(report['input_sha256'])
for name in ('report.json', 'windows.json.gz'):
    path = main / name
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
for name, digest in inputs.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
with gzip.open(main / 'windows.json.gz', 'rt') as stream:
    original_windows = json.load(stream)
cases = {row['id']: row for row in json.loads((run / 'cases.json').read_text())}


@lru_cache(None)
def reference(i, j, k):
    period = sum(((i * 7 + t * 13 + 51) % 17 - 8) * ((t * 5 + j * 11 + 85) % 17 - 8) for t in range(17))
    tail = sum(((i * 7 + t * 13 + 51) % 17 - 8) * ((t * 5 + j * 11 + 85) % 17 - 8) for t in range(k % 17))
    return (period * (k // 17) + tail) / 1024


processes = {}
conditions = []
for pitch in ('aligned', 'a16', 'b16'):
    for k in (1024, 4096):
        case = f'cfg_c_r13_comp_k{k}_{pitch}'
        row = cases[case]
        condition = dict(case=case, **{key: row[key] for key in ('m', 'n', 'k', 'lda', 'ldb', 'ldd', 'alloc_lda', 'alloc_ldb')})
        warmups = []
        paths = sorted((run / 'samples' / case).glob('dual-*.json'))
        assert len(paths) == 10
        for path in paths:
            record = json.loads(path.read_text())
            assert not record['returncode']
            raw = run / record['raw']
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == record['raw_sha256']
            with gzip.open(raw, 'rt') as stream:
                packets = [json.loads(line) for line in stream]
            events = {packet['event']: packet for packet in packets}
            assert len(events) == len(packets)
            setup, call, check = [events[key] for key in ('setup', 'call', 'check')]
            fixed = dict(config='cfg_c', m=2304, n=3072, k=k, stages=4, cluster=[1, 2], tile=[256, 128, 64],
                         grid=[66, 2, 1], swizzle=1, trace_tile_words=10, trace_words=1296,
                         trace_version='r18-dual-clock-events', input_mode='dyadic', seed=17, zero_m=-1, zero_n=-1)
            assert all(setup[key] == value for key, value in fixed.items())
            assert all(setup[key] == row[key] for key in ('lda', 'ldb', 'ldd', 'alloc_lda', 'alloc_ldb'))
            assert check['status'] == 'ok' and check['samples'] == 4096 and not check['nonfinite'] and not check['padding_errors']
            indices, values = check['checked_indices'], check['checked_values']
            assert len(indices) == len(set(indices)) == len(values) == 4096
            for index, value in zip(indices, values):
                assert type(index) is int and 0 <= index < 2304 * 3072
                i, j = divmod(index, 3072)
                assert math.isfinite(value) and value == reference(i % 17, j % 17, k)
            trace = call['trace']
            assert len(trace) == 132 * 1296 and all(type(value) is int and 0 <= value < 2**64 for value in trace)
            windows = {}
            for cta in range(132):
                b = trace[1296 * cta:1296 * (cta + 1)]
                assert not b[10] and b[2] > 0 and b[11] == b[12] == b[2]
                for role in (1, 2):
                    end_c, end_n, count = b[4 + 3 * (role - 1):7 + 3 * (role - 1)]
                    assert count in (1, 2)
                    head = 16 + (role - 1) * 640
                    assert not any(b[head + count * 10:head + 640])
                    for j in range(count):
                        off = head + 10 * j
                        c, n = b[off:off + 4], b[off + 6:off + 10]
                        assert all(c + n) and c == sorted(c) and n == sorted(n)
                        assert c[3] <= end_c and n[3] <= end_n
                        windows[cta, role, j] = dict(cycles=c[1] - c[0], ns=n[1] - n[0],
                                                    coordinate=[b[off + 4] - 1, b[off + 5] - 1], sm=b[2] - 1, outputs=count)
            warmups.append(call.get('warmup_calls'))
            processes[pitch, k, record['trial']] = dict(windows=windows, host_start_ns=record['host_start_ns'], host_stop_ns=record['host_stop_ns'], gpu=setup['gpu_uuid'])
        condition['observed_warmup_calls'] = sorted(set(warmups))
        conditions.append(condition)
assert len(processes) == 60 and len({p['gpu'] for p in processes.values()}) == 1
terms = []
groups = defaultdict(list)
host_gaps = []
for row in original_windows:
    pitch, trial, cta, role, ordinal = [row[key] for key in ('pitch', 'trial', 'cta', 'role', 'output_ordinal')]
    small, large = [processes[pitch, k, trial]['windows'][cta, role, ordinal] for k in (1024, 4096)]
    assert small['coordinate'] == large['coordinate'] == row['coordinate']
    assert small['outputs'] == large['outputs'] == row['outputs']
    assert (small['sm'] != large['sm']) == row['sm_changed']
    c0, c1, n0, n1 = small['cycles'], large['cycles'], small['ns'], large['ns']
    assert [c0, c1, n0, n1] == [row[key] for key in ('cycles_1024', 'cycles_4096', 'ns_1024', 'ns_4096')]
    assert min(c0, c1, n0, n1) > 0
    ec = Fraction(4 * c0, c1) - 1
    ef = Fraction(c1 * n0, n1 * c0) - 1
    en = Fraction(4 * n0, n1) - 1
    assert (1 + ec) * (1 + ef) == 1 + en
    term = dict(row, cycle_prediction_error=float(ec), conditional_conversion_error=float(ef),
                time_prediction_error=float(en), exact_error_fractions=[[value.numerator, value.denominator] for value in (ec, ef, en)])
    terms.append(term)
    groups[pitch, role, ordinal, row['outputs'], trial].append(term)
assert len(terms) == 12960 and sum(row['sm_changed'] for row in terms) == 64
for pitch in ('aligned', 'a16', 'b16'):
    for trial in range(10):
        a, b = processes[pitch, 1024, trial], processes[pitch, 4096, trial]
        host_gaps.append(dict(pitch=pitch, trial=trial, large_start_minus_small_stop_s=(b['host_start_ns'] - a['host_stop_ns']) / 1e9))
metrics = ('cycle_prediction_error', 'conditional_conversion_error', 'time_prediction_error')
centers = []
for key, rows in sorted(groups.items()):
    pitch, role, ordinal, outputs, trial = key
    centers.append(dict(pitch=pitch, role=role, output_ordinal=ordinal, outputs=outputs, trial=trial, windows=len(rows),
                        **{metric: st.median(row[metric] for row in rows) for metric in metrics}))
summaries = []
for key in sorted({key[:4] for key in groups}):
    rows = [row for row in centers if tuple(row[k] for k in ('pitch', 'role', 'output_ordinal', 'outputs')) == key]
    assert len(rows) == 10
    summaries.append(dict(zip(('pitch', 'role', 'output_ordinal', 'outputs'), key),
                          **{metric: dict(median=st.median(row[metric] for row in rows), minimum=min(row[metric] for row in rows), maximum=max(row[metric] for row in rows)) for metric in metrics}))
pooled = []
for ordinal in (0, 1):
    rows = [row for row in terms if row['output_ordinal'] == ordinal]
    stats = {}
    for metric in metrics:
        values = [row[metric] for row in rows]
        absolute = sorted(map(abs, values))
        stats[metric] = dict(signed_median=st.median(values), rms=math.sqrt(st.fmean(value * value for value in values)),
                             absolute_p95_nearest_rank=absolute[math.ceil(.95 * len(values)) - 1], maximum_absolute=max(absolute))
    pooled.append(dict(output_ordinal=ordinal, windows=len(rows), metrics=stats))
inputs[str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result = dict(schema='gh200-r09-cross-K-conversion/1', input_sha256=inputs, conditions=conditions,
              numerical_values_verified=245760, pairs=len(terms), sm_changes=64, exact_multiplicative_closures=12960,
              definitions='eC=4*C1024/C4096-1; ef=(C4096/N4096)/(C1024/N1024)-1; eN=4*N1024/N4096-1; (1+eC)*(1+ef)=1+eN.',
              process_centers=centers, group_summaries=summaries, pooled_descriptive=pooled, host_separations=host_gaps,
              observation_qualified=False,
              limits=['Historical development repeats aligned by trial ID, not randomized causal pairs or new holdout.',
                      'K changes A lda and allocation; old adaptive warmup counts and process acquisition times are retained.',
                      'Native cycle prediction is a zero-parameter proportional proxy, not an isolated instruction service law.',
                      'Conditional conversion uses observed endpoint ratios; not identified hardware frequency or a free target predictor.',
                      'Separate medians need not multiply to close. Exact closure is verified per original window.',
                      'All64 changed-SM pair flags retained; each native duration is subtracted within its own call/SM.',
                      'Pooled RMS is descriptive only; all18 fixed strata and180 process centers remain available.'])
target = out / 'audit.json.gz'
if target.exists():
    with gzip.open(target, 'rt') as stream:
        assert json.load(stream) == json.loads(json.dumps(result))
else:
    target.write_bytes(gzip.compress((json.dumps(result, indent=2) + '\n').encode(), mtime=0))
print(json.dumps(dict(numerical_values_verified=245760, pairs=len(terms), closures=12960, pooled_descriptive=pooled, conditions=conditions)))
