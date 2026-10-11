#!/usr/bin/env python3
"""Conditional joint clock/timer robustness; existing smoke data only."""
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

from r09_guarded_clock_robustness import minimum_clock_allowance

out = Path(__file__).resolve().parent
run = out.parents[1]
source = run / 'reanalysis/A-guarded-common-curve-v1/audit.json.gz'
with gzip.open(source, 'rt') as stream:
    old = json.load(stream)
inputs = dict(old['input_sha256'])
inputs[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
for name, digest in inputs.items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
        raise ValueError('predecessor identity changed: ' + name)
stamps = {int(sm): points for sm, points in old['original_stamps'].items()}
if len(stamps) != 132 or sum(map(len, stamps.values())) != 528:
    raise ValueError('complete original guarded FIRST/MAIN set required')
cases = []
for epsilon in (0, 16, 32, 64, 128, 160, 176, 177, 186, 187):
    for name, cap in (('uncapped', None), ('capped', 1.98)):
        result = minimum_clock_allowance(stamps, epsilon, cap)
        cases.append(dict(branch=name, **result))
        print(epsilon, name, result['minimum_extra_clock_cycles_fraction'], flush=True)
windows = []
labels = ('WG1_FIRST', 'WG1_MAIN', 'WG2_FIRST', 'WG2_MAIN')
for sm, points in stamps.items():
    # Construct counter intervals, not service windows. Endpoints can belong
    # to different WGs but must share this invocation and SM counter epoch.
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i == j or b[1] < a[1] or b[2] < a[0]:
                continue
            windows.append(dict(sm=sm, start=labels[i], end_label=labels[j], point_indices=[i, j],
                                begin=a, end=b, begin_ns=a[1], end_ns=b[1],
                                cycles_lo=max(0, b[0] - a[2]), cycles_hi=b[2] - a[0]))
contained = []
for inner in windows:
    for outer in windows:
        if inner['sm'] == outer['sm']:
            continue
        begin_margin = inner['begin_ns'] - outer['begin_ns']
        end_margin = outer['end_ns'] - inner['end_ns']
        gap = inner['cycles_lo'] - outer['cycles_hi']
        if min(begin_margin, end_margin) >= 0 and gap > 0:
            contained.append(dict(inner=inner, outer=outer, cycle_gap=gap,
                                  timer_containment_allowance_ns=min(begin_margin, end_margin) / 2,
                                  sufficient_cycle_allowance_open_upper=gap / 4))
frontier = []
for epsilon in (0, 16, 32, 64, 128, 160):
    eligible = [row for row in contained if row['timer_containment_allowance_ns'] >= epsilon]
    selected = max(eligible, key=lambda row: row['cycle_gap'])
    frontier.append(dict(endpoint_allowance_ns=epsilon, contained_pairs=len(eligible), **selected))
for name in ('analyze.py', 'r09_guarded_clock_robustness.py', 'r09_shared_clock.py', 'test_r09_guarded_clock_robustness.py'):
    path = out / name
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
result = dict(schema='gh200-r09-guarded-robustness/1', input_sha256=inputs,
              original_stamps=stamps, cases=cases, constructed_counter_intervals=len(windows), contained_pair_count=len(contained),
              containment_frontier=frontier,
              model='Cb-kappa<=offset_sm+F(t)<=Ca+kappa, t in [N-epsilon,N+epsilon], common nonnegative continuous F; independent SM offsets.',
              scope='Posthoc development identifiability and sensitivity; no new sampling, fit or changed gate.',
              original_phenomenon_inference='inconclusive', observation_qualified=False,
              limits=['epsilon is extra single-endpoint timer shift, not the128ns total-window budget or measured timer precision.',
                      'kappa widens each native CLOCK bracket on both sides. It is not read overhead, measured counter error or a timing correction.',
                      'At fixed epsilon, all graph bounds are affine in kappa; a separating negative cycle and a feasible potential prove an attained exact minimum, with or without a rate cap.',
                      'This differs from varying epsilon without a cap, whose infimum may be unattained.',
                      'Pair selection maximizes a diagnostic contradiction margin after seeing the data. It is not a frozen critical-CTA or holdout selection.',
                      'Constructed intervals may cross original WG endpoints on the SAME SM. They only test counter consistency and are not role-L windows, service times or prices.',
                      'Losing one contained-pair contradiction does not prove full-record compatibility; the complete graph cases give that boundary.',
                      'Do not convert kappa to ns without separately established conversion. No physical SM clock architecture is identified.',
                      'A compatible curve is constructed from target observations and cannot enter a free prediction.'])
target = out / 'robustness.json.gz'
if target.exists():
    with gzip.open(target, 'rt') as stream:
        if json.load(stream) != json.loads(json.dumps(result)):
            raise ValueError('saved output differs; do not overwrite it')
else:
    target.write_bytes(gzip.compress((json.dumps(result, indent=2) + '\n').encode(), mtime=0))
print('20 exact joint-boundary cases;798 counter intervals;2996 contained interval contradictions')
