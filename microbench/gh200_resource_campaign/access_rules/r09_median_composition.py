"""Exact three-state population witnesses for median composition.

Synthetic arithmetic only: identical complete component marginals need not
identify the median of C/f, serial sum or parallel completion max. This does
not estimate GPU timing, change observation gates, or fit any model price.
"""
import argparse
from collections import Counter
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path


def exact(value):
    if type(value) not in (int, str):
        raise ValueError('positive exact integer/rational string required')
    result = Q(value)
    if result <= 0:
        raise ValueError('strictly positive value required')
    return result


def median3(values):
    if len(values) != 3:
        raise ValueError('three equiprobable population states required')
    return sorted(values)[1]


def distribution(values):
    return [dict(value=str(value), probability=str(Q(count, 3)))
            for value, count in sorted(Counter(values).items())]


def evaluate(cases):
    if cases['schema'] != 'r09-synthetic-median-composition/1' or cases['state_probabilities'] != ['1/3'] * 3:
        raise ValueError('known finite-population cases required')
    conversion = []
    for case in cases['conversion']:
        cycles = [exact(v) for v in case['cycles']]
        rates = [exact(v) for v in case['rates']]
        if len(cycles) != 3 or len(rates) != 3 or any(c.denominator != 1 for c in cycles):
            raise ValueError('three integer cycle counts and positive rates required')
        times = [c / f for c, f in zip(cycles, rates)]
        cycle_prediction, rate_prediction = median3(cycles), median3(rates)
        time_prediction = cycle_prediction / rate_prediction
        rows = []
        for state, (c, f, t) in enumerate(zip(cycles, rates, times)):
            e_c = cycle_prediction / c - 1
            e_f = f / rate_prediction - 1
            e_t = time_prediction / t - 1
            if (1 + e_c) * (1 + e_f) != 1 + e_t:
                raise ValueError('same-state error factorization failed')
            rows.append(dict(state=state, cycles=str(c), rate=str(f), time=str(t),
                cycle_error=str(e_c), conversion_error=str(e_f), time_error=str(e_t)))
        conversion.append(dict(name=case['name'], rows=rows,
            cycle_distribution=distribution(cycles), rate_distribution=distribution(rates),
            time_distribution=distribution(times), median_cycles=str(cycle_prediction),
            median_rate=str(rate_prediction), median_time=str(median3(times)),
            ratio_of_component_medians=str(time_prediction),
            signed_error_medians={name: str(median3([Q(r[name]) for r in rows]))
                for name in ('cycle_error', 'conversion_error', 'time_error')},
            same_state_error_factorization=True))
    composition = []
    for case in cases['composition']:
        stages = [[exact(v) for v in row] for row in case['durations']]
        if len(stages) != 3 or any(len(row) != 2 for row in stages):
            raise ValueError('three states with two positive stage durations required')
        medians = [median3([row[j] for row in stages]) for j in (0, 1)]
        serial = [sum(row) for row in stages]
        parallel = [max(row) for row in stages]
        composition.append(dict(name=case['name'], durations=[[str(v) for v in row] for row in stages],
            component_distributions=[distribution([row[j] for row in stages]) for j in (0, 1)],
            component_medians=[str(v) for v in medians],
            serial_times=[str(v) for v in serial], median_serial_time=str(median3(serial)),
            sum_of_component_medians=str(sum(medians)),
            parallel_times=[str(v) for v in parallel], median_parallel_time=str(median3(parallel)),
            max_of_component_medians=str(max(medians)),
            parallel_last_components=[[j for j in (0, 1) if row[j] == max(row)] for row in stages]))
    if len(conversion) != 2 or len(composition) != 2:
        raise ValueError('two explicitly compared worlds required')
    same_conversion_marginals = all(conversion[0][key] == conversion[1][key]
        for key in ('cycle_distribution', 'rate_distribution'))
    same_stage_marginals = composition[0]['component_distributions'] == composition[1]['component_distributions']
    if not same_conversion_marginals or not same_stage_marginals:
        raise ValueError('counterexample worlds must preserve complete component marginals')
    return dict(schema='r09-synthetic-median-composition-report/1',
        scope='Exact known equiprobable populations; no GPU samples, fitted rule or timing score.',
        conversion=conversion, composition=composition,
        complete_component_marginals_identical=True,
        median_time_identified_by_conversion_marginals=(conversion[0]['median_time'] == conversion[1]['median_time']),
        median_serial_identified_by_stage_marginals=(composition[0]['median_serial_time'] == composition[1]['median_serial_time']),
        median_parallel_identified_by_stage_marginals=(composition[0]['median_parallel_time'] == composition[1]['median_parallel_time']),
        observation_qualified=False, prediction_qualified=False, prospective_holdout_qualified=False,
        cross_card_qualified=False,
        consequence='Exact local population medians, even complete marginals, do not replace joint same-execution conversion/phase/CTA evidence and independent complete-time validation.',
        limits='Synthetic nonidentifiability of a general implication; no claim that these populations occur on GH200 or give its error floor. Extra validated dependence/range assumptions can change the inference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--cases-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.cases_sha256:
        raise ValueError('cases differ from caller-pinned identity')
    report = evaluate(json.loads(raw))
    report.update(cases_sha256=args.cases_sha256,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in ('median_time_identified_by_conversion_marginals',
        'median_serial_identified_by_stage_marginals', 'median_parallel_identified_by_stage_marginals')}))


if __name__ == '__main__':
    main()
