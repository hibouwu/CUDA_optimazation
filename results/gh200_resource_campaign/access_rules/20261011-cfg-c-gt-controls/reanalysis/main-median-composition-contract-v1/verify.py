"""Check the known populations by CDFs and enumerate all marginal-preserving permutations."""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction as Q
import hashlib
from itertools import permutations
import json
from pathlib import Path


def require(value):
    if not value:
        raise ValueError('independent median-composition verification failed')


def read(path):
    return json.loads(path.read_text())


def population_median(values):
    # The finite CDF definition, not the analyzer's sorted middle element.
    possible = [v for v in set(values) if sum(x <= v for x in values) >= 2
                and sum(x >= v for x in values) >= 2]
    require(len(possible) == 1)
    return possible[0]


def verify_report(cases, report):
    require(cases['state_probabilities'] == ['1/3'] * 3)
    cycle_columns, rate_columns = [], []
    conversion_medians = []
    for case, result in zip(cases['conversion'], report['conversion']):
        cycles, rates = [Q(v) for v in case['cycles']], [Q(v) for v in case['rates']]
        require(len(cycles) == len(rates) == 3 and result['name'] == case['name'])
        cycle_columns.append(Counter(cycles)); rate_columns.append(Counter(rates))
        med_c, med_f = population_median(cycles), population_median(rates)
        time_prediction = med_c / med_f
        times, error_columns = [], {k: [] for k in ('cycle_error', 'conversion_error', 'time_error')}
        for state, row in enumerate(result['rows']):
            c, f = cycles[state], rates[state]
            t = c / f
            require(row['state'] == state and Q(row['cycles']) == c and Q(row['rate']) == f and Q(row['time']) == t)
            times.append(t)
            expected = {'cycle_error': med_c / c - 1, 'conversion_error': f / med_f - 1,
                        'time_error': time_prediction / t - 1}
            for k, v in expected.items():
                require(Q(row[k]) == v); error_columns[k].append(v)
            # Cross multiplication uses the same labeled state on both sides.
            require((1 + Q(row['cycle_error'])) * (1 + Q(row['conversion_error'])) == 1 + Q(row['time_error']))
        for name, values in [('cycle', cycles), ('rate', rates), ('time', times)]:
            require(result[name + '_distribution'] == [dict(value=str(v), probability=str(Q(n, 3)))
                for v, n in sorted(Counter(values).items())])
        require(Q(result['median_cycles']) == med_c and Q(result['median_rate']) == med_f)
        require(Q(result['ratio_of_component_medians']) == time_prediction)
        med_t = population_median(times); conversion_medians.append(med_t)
        require(Q(result['median_time']) == med_t)
        require(result['signed_error_medians'] == {k: str(population_median(v)) for k, v in error_columns.items()})
    require(cycle_columns[0] == cycle_columns[1] and rate_columns[0] == rate_columns[1])
    require(conversion_medians == [Q(8), Q(20, 3)])
    require(report['conversion'][1]['signed_error_medians'] ==
            dict(cycle_error='0', conversion_error='0', time_error='1/5'))
    stage_columns = []
    serial_medians, parallel_medians = [], []
    for case, result in zip(cases['composition'], report['composition']):
        states = [[Q(v) for v in row] for row in case['durations']]
        require(result['name'] == case['name'] and result['durations'] == [[str(v) for v in row] for row in states])
        stage_columns.append([Counter(row[j] for row in states) for j in (0, 1)])
        medians = [population_median([row[j] for row in states]) for j in (0, 1)]
        sums = [row[0] + row[1] for row in states]
        maxima = [row[0] if row[0] >= row[1] else row[1] for row in states]
        require(result['component_medians'] == [str(v) for v in medians])
        require(result['component_distributions'] == [[dict(value=str(v), probability=str(Q(n, 3)))
            for v, n in sorted(column.items())] for column in stage_columns[-1]])
        require(result['serial_times'] == [str(v) for v in sums] and result['parallel_times'] == [str(v) for v in maxima])
        serial = population_median(sums); parallel = population_median(maxima)
        serial_medians.append(serial); parallel_medians.append(parallel)
        require(Q(result['median_serial_time']) == serial and Q(result['median_parallel_time']) == parallel)
        require(Q(result['sum_of_component_medians']) == sum(medians) and Q(result['max_of_component_medians']) == max(medians))
        last = [[0, 1] if a == b else [0] if a > b else [1] for a, b in states]
        require(result['parallel_last_components'] == last)
    require(stage_columns[0] == stage_columns[1] and serial_medians == [2, 10] and parallel_medians == [1, 9])
    require(report['complete_component_marginals_identical'] is True)
    for key in ('median_time_identified_by_conversion_marginals', 'median_serial_identified_by_stage_marginals',
                'median_parallel_identified_by_stage_marginals', 'observation_qualified', 'prediction_qualified',
                'prospective_holdout_qualified', 'cross_card_qualified'):
        require(report[key] is False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    cases, report = read(args.cases), read(args.report)
    require(hashlib.sha256(args.cases.read_bytes()).hexdigest() == report['cases_sha256'])
    require(hashlib.sha256((args.cases.parent / 'r09_median_composition.py').read_bytes()).hexdigest() == report['source_sha256'])
    verify_report(cases, report)
    # Exhaust all joint assignments with the first column fixed. These keep
    # every component marginal unchanged; the named worlds are not cherry-
    # picked from any measured execution or fitted to a hardware error.
    C = [Q(8), Q(10), Q(12)]
    rate_permutations = set(permutations([Q(1), Q(5, 4), Q(3, 2)]))
    converted = {population_median([c / f for c, f in zip(C, p)]) for p in rate_permutations}
    require(converted == {Q(20, 3), Q(8), Q(48, 5)})
    x = [Q(1), Q(1), Q(9)]
    stage_permutations = set(permutations(x))
    serial = {population_median([a + b for a, b in zip(x, p)]) for p in stage_permutations}
    parallel = {population_median([max(a, b) for a, b in zip(x, p)]) for p in stage_permutations}
    require(serial == {Q(2), Q(10)} and parallel == {Q(1), Q(9)})
    bad_reports = []
    wrong_conversion = deepcopy(report); wrong_conversion['conversion'][1]['median_time'] = '8'
    wrong_error = deepcopy(report); wrong_error['conversion'][1]['signed_error_medians']['time_error'] = '0'
    wrong_sum = deepcopy(report); wrong_sum['composition'][1]['median_serial_time'] = '2'
    wrong_max = deepcopy(report); wrong_max['composition'][1]['median_parallel_time'] = '1'
    bad_reports.extend((wrong_conversion, wrong_error, wrong_sum, wrong_max))
    rejected = 0
    for bad in bad_reports:
        try:
            verify_report(cases, bad)
        except ValueError:
            rejected += 1
        else:
            raise ValueError('incorrect composition claim accepted')
    print(json.dumps(dict(populations_checked=4, same_state_conversion_rows=6,
        marginal_preserving_rate_permutations=len(rate_permutations),
        possible_conversion_medians=[str(v) for v in sorted(converted)],
        marginal_preserving_stage_permutations=len(stage_permutations),
        possible_serial_medians=[str(v) for v in sorted(serial)],
        possible_parallel_medians=[str(v) for v in sorted(parallel)],
        incorrect_claims_rejected=rejected, GPU_evidence=False, prediction_qualified=False),
        indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
