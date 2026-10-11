"""Independent algebra checks for the saved C median-composition witnesses."""
import csv
from fractions import Fraction
import gzip
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parent
DATA=BASE/'data-v1'
report=json.load(open(DATA/'report.json'))
samples=json.load(gzip.open(DATA/'synthetic-samples.json.gz','rt'))
result=json.load(gzip.open(DATA/'synthetic-evaluator-result.json.gz','rt'))
assert result['order_statistic_ranks']==[14,51]
assert result['counts']['conditional equivalence']==12744
assert all(row['median_difference_interval_ns']==[0,0] for row in result['series_results'])
assert result['observation_qualified'] is False and result['prediction_qualified'] is False
identities=0
marginal_equal=0
for cohort,series in samples.items():
    for name,arms in series.items():
        assert sorted(arms['arm0'])==sorted(arms['arm1'])
        marginal_equal+=1
    finals=[name for name in series if name.endswith('/GLOBAL_FINAL')]
    for name in finals:
        prefix=name[:-len('GLOBAL_FINAL')]
        cta=prefix.split('/')[0]
        outputs=[key[:-len('GLOBAL_DONE')] for key in series if key.startswith(prefix) and key.endswith('/GLOBAL_DONE')]
        last=max(outputs,key=lambda key:int(key.split('/')[2]))
        for arm in ['arm0','arm1']:
            for i in range(64):
                final=series[name][arm][i]
                origin=series[cta+'/GLOBAL_ENTRY'][arm][i]
                done=series[last+'GLOBAL_DONE'][arm][i]
                assert final-origin==series[prefix+'FINAL'][arm][i]
                assert final-done==series[prefix+'TAIL'][arm][i]
                assert final>=done
                identities+=1
for file,key in [('role-composition.csv','real_role_sum_median_gaps'),('merged-composition.csv','real_merged_sum_median_gaps')]:
    with (DATA/file).open(newline='') as stream:rows=list(csv.DictReader(stream))
    gaps=[float(row['median_composed'])-float(row['composed_medians']) for row in rows]
    assert gaps==[float(row['median_composed_minus_composed_medians']) for row in rows]
    assert len(rows)==report[key]['series']
    assert sum(g!=0 for g in gaps)==report[key]['nonzero']
    assert min(gaps)==report[key]['minimum'] and max(gaps)==report[key]['maximum']
    assert statistics.median(abs(g) for g in gaps)==report[key]['median_absolute_gap']
# These two exact joint populations may be sampled IID across calls. Their
# marginal distributions are identical; their dependence across roles differs.
H=1000
common=[([0]*8,Fraction(7,8)),([H]*8,Fraction(1,8))]
dispersed=[([H if i==j else 0 for i in range(8)],Fraction(1,8)) for j in range(8)]
assert sum(p for _,p in common)==sum(p for _,p in dispersed)==1
for j in range(8):
    for value in [0,H]:
        assert sum(p for row,p in common if row[j]==value)==sum(p for row,p in dispersed if row[j]==value)
common_low=sum(p for row,p in common if max(row)==0)
dispersed_low=sum(p for row,p in dispersed if max(row)==0)
assert common_low>Fraction(1,2) and dispersed_low==0
arm0,arm1=[100,200,300],[200,300,100]
paired=[b-a for a,b in zip(arm0,arm1)]
assert statistics.median(arm1)-statistics.median(arm0)==0
assert statistics.median(paired)==100
verification={'schema':'C-median-composition-verification/1',
 'full_family_equal_sorted_arm_marginals':marginal_equal,
 'synthetic_FINAL_and_TAIL_same_call_identities':identities,
 'all_statistical_difference_intervals_zero':True,
 'joint_population_witness':{'selected_roles':8,'high_amount_symbolic':H,
   'each_role_marginal_high_probability':'1/8',
   'common_joint':{'all_low_probability':'7/8','all_high_probability':'1/8'},
   'dispersed_joint':'Exactly one of8 roles high, each with probability1/8.',
   'common_max_population_median':0,'dispersed_max_population_median':H,
   'IID_across_calls_mathematically_possible':True,
   'GPU_sampling_assumptions_proved':False},
 'paired_estimand_witness':{'arm0':arm0,'arm1':arm1,'paired_differences':paired,
   'difference_of_arm_medians':0,'median_of_paired_differences':100},
 'saved_arithmetic_summaries_verified':True,
 'physical_or_prediction_qualification':False}
with (BASE/'verification.json').open('x') as stream:
    stream.write(json.dumps(verification,indent=2,sort_keys=True)+'\n')
print(json.dumps(verification,indent=2))
