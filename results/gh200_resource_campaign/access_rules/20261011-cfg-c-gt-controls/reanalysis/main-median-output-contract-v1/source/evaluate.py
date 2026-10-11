"""Candidate simultaneous median equivalence evaluator; never grants observation qualification."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics


def evaluate(protocol, samples):
    if protocol['schema'] != 'gh200-median-equivalence-proposal/1':
        raise ValueError('unknown protocol')
    series = protocol['series']
    if not series or len(set(series)) != len(series):
        raise ValueError('expected unique, nonempty fixed series')
    cohorts = ('off_on','off_off')
    n = protocol['repeats_per_arm']
    if type(n) is not int or n < 1:
        raise ValueError('positive integer repetitions required')
    alpha = Fraction(protocol['family_failure_probability'])
    margin = protocol['median_difference_margin_ns']
    if not 0 < alpha < 1 or type(margin) not in (int,float) or not math.isfinite(margin) or margin < 0:
        raise ValueError('invalid confidence or equivalence margin')
    if set(samples) != set(cohorts):
        raise ValueError('both original cohorts required')
    family = len(cohorts)*2*len(series)
    ranks = [k for k in range(1,(n+1)//2+1)
             if family*Fraction(2*sum(math.comb(n,j) for j in range(k)),2**n) <= alpha]
    k = max(ranks) if ranks else None
    results=[]
    for cohort in cohorts:
        if set(samples[cohort]) != set(series):
            raise ValueError('missing or extra coordinate; family cannot shrink')
        for name in series:
            arms=samples[cohort][name]
            if set(arms) != {'arm0','arm1'}:
                raise ValueError('exactly two arms required')
            for values in arms.values():
                if not isinstance(values,list) or len(values) != n:
                    raise ValueError('fixed sample count; missing/extra repeats rejected')
                if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
                    raise ValueError('finite numeric ns values required')
            a,b=arms['arm0'],arms['arm1']
            paired=[y-x for x,y in zip(a,b)]
            old=dict(median_absolute_ns=statistics.median(abs(x) for x in paired),
                     maximum_absolute_ns=max(abs(x) for x in paired))
            old['passed']=(old['median_absolute_ns']<=64 and old['maximum_absolute_ns']<=256)
            interval=None
            status='insufficient simultaneous coverage'
            if k is not None:
                ordered_a,ordered_b=sorted(a),sorted(b)
                low,high=ordered_b[k-1]-ordered_a[-k],ordered_b[-k]-ordered_a[k-1]
                interval=[low,high]
                if low >= -margin and high <= margin:
                    status='conditional equivalence'
                elif low > margin or high < -margin:
                    status='conditional non-equivalence'
                else:
                    status='inconclusive'
            results.append(dict(cohort=cohort,series=name,status=status,
                median_difference_interval_ns=interval,
                sample_median_difference_ns=statistics.median(b)-statistics.median(a),
                original_point_control=old))
    counts={status:sum(row['status']==status for row in results) for status in
            ('conditional equivalence','conditional non-equivalence','inconclusive','insufficient simultaneous coverage')}
    return dict(schema='gh200-median-equivalence-diagnostic/1',
        simultaneous_arm_medians=family,repeats_per_arm=n,
        order_statistic_ranks=[k,n-k+1] if k is not None else None,
        coverage_lower_bound=str(1-family*Fraction(2*sum(math.comb(n,j) for j in range(k)),2**n)) if k else None,
        counts=counts,series_results=results,
        original_point_failures={cohort:sum(not r['original_point_control']['passed'] for r in results if r['cohort']==cohort) for cohort in cohorts},
        all_statistical_intervals_equivalent=counts['conditional equivalence']==len(results),
        sampling_assumptions_verified=False,physical_timestamp_accuracy_verified=False,
        critical_cta_ordering_verified=False,observation_qualified=False,prediction_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--expected-protocol-sha256',required=True)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    digest=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    if digest != args.expected_protocol_sha256:
        raise ValueError('protocol identity mismatch')
    result=evaluate(json.loads(args.protocol.read_text()),json.loads(args.samples.read_text()))
    result.update(protocol_sha256=digest,samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
