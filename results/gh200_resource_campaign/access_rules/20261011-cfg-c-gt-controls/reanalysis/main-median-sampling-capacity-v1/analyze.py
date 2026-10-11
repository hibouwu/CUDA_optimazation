"""Conditional median sampling capacity; no change to any observation gate."""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from math import comb
from pathlib import Path
import statistics


def failure_probability(n, k):
    return Fraction(2*sum(comb(n,j) for j in range(k)), 2**n)


def rank_pair(n, family):
    candidates = [k for k in range(1,(n+1)//2+1)
                  if family*failure_probability(n,k) <= Fraction(1,20)]
    if not candidates:
        return None
    k = max(candidates)
    return dict(lower_rank=k,upper_rank=n-k+1,
                simultaneous_coverage_lower_bound=str(1-family*failure_probability(n,k)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    base = args.native_analysis
    report = json.loads((base/'data/report.json').read_text())
    for name,digest in report['input_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    rows = json.loads(gzip.decompress((base/'data/coordinate-pairs.json.gz').read_bytes()))
    assert len(rows)==864 and report['original_ns_gate_unchanged']
    assert report['observation_qualified'] is False
    assert report['original_L_failure_counts']=={'off_on':432,'off_off':432}
    # Enumerate all 1024 sign vectors, independently checking the binomial sum.
    enumeration=[]
    for k in range(1,6):
        success=sum(k<=mask.bit_count()<=10-k for mask in range(2**10))
        assert Fraction(success,2**10)==1-failure_probability(10,k)
        enumeration.append(dict(n=10,k=k,sign_patterns=1024,covered_patterns=success))
    families={
        'one_coordinate_one_cohort':2,
        'one_coordinate_two_cohorts':4,
        'all432_L_one_cohort':864,
        'all432_L_two_cohorts':1728,
        'all6372_series_one_cohort':12744,
        'all6372_series_two_cohorts':25488}
    capacity={}
    for name,family in families.items():
        n=1
        while family*failure_probability(n,1)>Fraction(1,20):
            n+=1
        assert family*failure_probability(n,1)<=Fraction(1,20)
        assert n==1 or family*failure_probability(n-1,1)>Fraction(1,20)
        capacity[name]=dict(arm_medians=family,minimum_n_per_arm_for_finite_equal_tail_interval=n,
            n10=rank_pair(10,family),minimum_n_ranks=rank_pair(n,family),
            ranks_by_n={str(j):rank_pair(j,family) for j in (10,16,17,20,32,64)})
    selected=rank_pair(10,2)
    assert (selected['lower_rank'],selected['upper_rank'])==(2,9)
    k=selected['lower_rank'];intervals=[]
    for row in rows:
        assert len(row['pairs'])==10 and row['original_ns_gate_passed'] is False
        entry=dict(cohort=row['cohort'],series=row['series'])
        for unit in ('ns','cycles'):
            a=sorted(pair['arm0'][unit] for pair in row['pairs'])
            b=sorted(pair['arm1'][unit] for pair in row['pairs'])
            # Difference of two population medians, not median paired difference.
            lo,hi=b[k-1]-a[-k],b[-k]-a[k-1]
            delta=statistics.median(b)-statistics.median(a)
            paired=statistics.median(pair['arm1'][unit]-pair['arm0'][unit] for pair in row['pairs'])
            assert lo<=delta<=hi
            entry[unit]=dict(marginal_difference_interval=[lo,hi],
                sample_median_difference=delta,sample_median_paired_difference=paired,
                contains_zero=lo<=0<=hi,width=hi-lo)
        intervals.append(entry)
    summaries={}
    for cohort in ('off_on','off_off'):
        selected_rows=[row for row in intervals if row['cohort']==cohort]
        assert len(selected_rows)==432
        summaries[cohort]={}
        for unit in ('ns','cycles'):
            summaries[cohort][unit]=dict(
                marginal_intervals_containing_zero=sum(row[unit]['contains_zero'] for row in selected_rows),
                median_difference_disagrees_with_median_paired_difference=sum(row[unit]['sample_median_difference']!=row[unit]['sample_median_paired_difference'] for row in selected_rows),
                interval_width_median=statistics.median(row[unit]['width'] for row in selected_rows),
                interval_width_max=max(row[unit]['width'] for row in selected_rows))
    result=dict(
        method='Noninterpolated equal-tail order-statistic intervals; exact binomial tails and Bonferroni union bound. 95% is an illustrative design level, not a newly adopted gate.',
        assumptions='For each coordinate/arm, independent identically distributed process repeats from a fixed population with a median. No independence between CTA coordinates or between arms is needed for the union bound. These sampling assumptions have not been established for the sequential GPU records.',
        formula='P(median outside [X_(k),X_(n-k+1)]) <= 2*sum_{j=0}^{k-1} choose(n,j)/2^n. family_failure <= number_of_arm_medians * marginal_failure.',
        ties='Equality at a median may make coverage conservative; no interpolation or jitter is added to quantized readings.',
        capacity=capacity,sign_enumeration=enumeration,
        marginal_difference_coverage_lower_bound=str(1-2*failure_probability(10,2)),
        descriptive_conditional_interval_summaries=summaries,
        current_n=10,original_L_failure_counts=report['original_L_failure_counts'],
        input_sha256={str(base/name):hashlib.sha256((base/name).read_bytes()).hexdigest()
                      for name in ('analyze.py','data/report.json','data/coordinate-pairs.json.gz','verification.json')},
        source_inputs_verified=len(report['input_sha256']),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['The minimum n applies only to this equal-tail Bonferroni construction, not every possible simultaneous method.',
                'Finite endpoints do not ensure a narrow interval, equivalence, observer accuracy, or stable process populations.',
                '432 within-process CTA coordinates do not supply432 independent process repetitions.',
                'Containing zero is not evidence of equivalence; no equivalence margin or pass count is defined.',
                'Old pointwise64/256ns failures remain; no resampling of old data, cross-cohort center subtraction, new GPU or critical-CTA qualification.'],
        observation_qualified=False,prediction_qualified=False)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'conditional-marginal-intervals.json.gz').write_bytes(gzip.compress(json.dumps(intervals,separators=(',',':')).encode(),mtime=0))
    print(json.dumps(dict(capacity={name:row['minimum_n_per_arm_for_finite_equal_tail_interval'] for name,row in capacity.items()},summaries=summaries),indent=2))


if __name__=='__main__':
    main()
