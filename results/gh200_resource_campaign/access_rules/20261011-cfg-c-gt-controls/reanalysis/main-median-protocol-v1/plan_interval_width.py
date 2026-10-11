"""Exact necessary-width probability bound under the explicitly empirical IID scenario.

This is a planning sensitivity calculation, not estimated power for the GPU.
Each historical arm's ten observations define a hypothetical discrete population.
"""
import argparse
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import itertools
import json
from math import comb
from pathlib import Path

from acquisition_identity import verify_acquisition


@lru_cache(None)
def tail(n, required, count, population):
    numerator = sum(comb(n, j) * count**j * (population-count)**(n-j)
                    for j in range(required, n+1))
    return Fraction(numerator, population**n)


def width_bound(values, n, low_rank, high_rank, width):
    # If X_high-X_low <= width, the interval starting at X_low contains
    # at least high-low+1 observations. Union over every possible start.
    counts = [sum(start <= value <= start+width for value in values)
              for start in sorted(set(values))]
    required = high_rank-low_rank+1
    upper = min(Fraction(1), sum((tail(n, required, count, len(values))
                                for count in counts), Fraction(0)))
    return upper, counts


def verify_small_populations():
    checks = []
    for population in ([0, 0, 3], [0, 2, 4], [0, 1, 1, 2]):
        n, low, high = 5, 2, 4
        for width in (0, 1, 2):
            accepted = 0
            for sample in itertools.product(population, repeat=n):
                ordered = sorted(sample)
                if ordered[high-1]-ordered[low-1] <= width:
                    accepted += 1
                    assert any(sum(start <= x <= start+width for x in sample) >= high-low+1
                               for start in sorted(set(population)))
            exact = Fraction(accepted, len(population)**n)
            bound, _ = width_bound(population, n, low, high, width)
            assert exact <= bound
            checks.append(dict(population=population, width=width,
                               exact=str(exact), upper=str(bound)))
    # For singleton-width clusters, the tail count is independently enumerable.
    for population in (2, 3, 4):
        for count in range(population+1):
            successful = sum(sum(x < count for x in sample) >= 3
                             for sample in itertools.product(range(population), repeat=5))
            assert tail(5, 3, count, population) == Fraction(successful, population**5)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output required')
    package = Path(__file__).resolve().parent
    archive_sha = '4e372a3dbe50e33657fd57a0c265faba2b7ba857eb2ea34e78803b60435d0b04'
    _, _, hashes = verify_acquisition(args.run.resolve(), archive_sha)
    protocol_path = package/'data/protocol-proposal.json'
    receipt_path = package/'joint-replay-v4/receipt.json'
    records_path = package/'joint-replay-v4/joint-process-records.json.gz'
    protocol = json.loads(protocol_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    assert receipt['acquisition_archive_sha256'] == archive_sha
    assert all(receipt['input_sha256'][name] == digest for name, digest in hashes.items())
    assert hashlib.sha256(records_path.read_bytes()).hexdigest() == receipt['output_sha256'][records_path.name]
    records = json.loads(gzip.decompress(records_path.read_bytes()))
    n = protocol['repeats_per_arm']
    family = 4*len(protocol['series'])
    alpha = Fraction(protocol['family_failure_probability'])
    candidates = [k for k in range(1, (n+1)//2+1)
                  if family*Fraction(2*sum(comb(n,j) for j in range(k)), 2**n) <= alpha]
    low, high = max(candidates), n-max(candidates)+1
    assert (n, family, low, high) == (64, 25488, 14, 51)
    # Equivalence requires (B_hi-B_lo)+(A_hi-A_lo) <= 2*margin.
    # Therefore each individual width must also be <= 2*margin, without
    # needing independence between arms or coordinates.
    width = 2*protocol['median_difference_margin_ns']
    names = [name for name in protocol['series'] if name.endswith('/L')]
    assert len(names) == 432
    rows = []
    for cohort in ('off_on','off_off'):
        for arm in (0,1):
            selected = sorted((r for r in records if r['cohort']==cohort and r['arm']==arm),
                              key=lambda r:r['block'])
            assert [r['block'] for r in selected] == list(range(10))
            for name in names:
                values = [r['series'][name][0] for r in selected]
                probability, counts = width_bound(values,n,low,high,width)
                rows.append(dict(cohort=cohort, arm=arm, series=name,
                    empirical_values_ns=values, possible_interval_counts=counts,
                    necessary_width_probability_upper_exact=str(probability),
                    necessary_width_probability_upper=float(probability)))
    strongest = min(rows, key=lambda r:Fraction(r['necessary_width_probability_upper_exact']))
    summary = dict(schema='gh200-empirical-interval-planning/1',
        scenario='Each arm and coordinate is IID from its own ten historical empirical values; this is unverified for future GPU acquisitions.',
        prospective_n=n, simultaneous_arm_medians=family, ranks=[low,high],
        margin_ns=width/2, necessary_individual_width_ns=width,
        necessary_sample_count_in_width=high-low+1,
        arm_coordinate_checks=len(rows), strongest_necessary_constraint=strongest,
        full_family_equivalence_probability_upper_under_scenario=strongest['necessary_width_probability_upper'],
        per_cohort={cohort: min(r['necessary_width_probability_upper'] for r in rows if r['cohort']==cohort)
                    for cohort in ('off_on','off_off')},
        no_arm_or_coordinate_independence_required=True,
        exact_small_population_checks=verify_small_populations(),
        rows=rows,
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (protocol_path,receipt_path,records_path)},
        acquisition_archive_sha256=archive_sha,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Not a confidence statement about the unknown GPU population or future power.',
                'Ten old samples and sequential cohorts do not establish IID or a stable population.',
                'This necessary-width union bound ignores centering; actual equivalence may be less likely.',
                'No threshold, sample count, frozen protocol, failure or prediction has changed.'],
        observation_qualified=False,prediction_qualified=False)
    with args.output.open('x') as stream:
        json.dump(summary,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','input_sha256','exact_small_population_checks')},indent=2))


if __name__ == '__main__':main()
