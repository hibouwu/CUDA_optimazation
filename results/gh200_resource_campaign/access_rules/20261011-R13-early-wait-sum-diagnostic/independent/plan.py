#!/usr/bin/env python3
"""Fixed four-arm diagnostic proposal; CPU arithmetic only, no acquisition."""
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
import random
import statistics

ROOT = Path('/home/jianyeshi/Note/CUDA/CUDA_optimazation')
RUN = ROOT / 'results/gh200_resource_campaign/access_rules/20261011-R13-early-wait-sum-correctness'
OUTPUT = Path(__file__).resolve().parent
N = 16
ARMS = {'s4_off':(4,0), 's4_on':(4,1), 's6_off':(6,0), 's6_on':(6,1)}
SERIES = ['delta_D_on','delta_Q','delta_R','delta_Q_minus_R','delta_D_off',
          'delta_D_on_minus_off','s4_on_minus_off','s6_on_minus_off']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_interval(values):
    assert len(values) == N
    ordered = sorted(values)
    return [ordered[2],ordered[13]]


def contrasts(records):
    """Consume strict upstream decoded records; this is not a raw verifier."""
    if len(records) != 4*N:
        raise ValueError('expected all 64 declared processes; no replacement or filtering')
    grouped = {block:{} for block in range(N)}
    for row in records:
        block, arm = row['block'], row['arm']
        if type(block) is not int or block not in grouped or arm not in ARMS or arm in grouped[block]:
            raise ValueError('unknown or duplicated block/arm')
        decoded = row['decoded']
        stage, flag = ARMS[arm]
        if decoded['stages'] != stage or decoded['observer'] != flag:
            raise ValueError('decoded stage/observer does not match assigned arm')
        D, Q, R = (decoded[name] for name in ['early_cycles','full_wait_cycles','outside_wait_cycles'])
        if type(D) is not int or D <= 0:
            raise ValueError('D must be a positive native-cycle integer')
        if flag:
            if type(Q) is not int or type(R) is not int or min(Q,R)<0 or Q+R!=D:
                raise ValueError('enabled same-call partition does not close')
        elif Q is not None or R is not None:
            raise ValueError('disabled Q/R must remain unobserved')
        grouped[block][arm] = (D,Q,R)
    rows=[]
    for block, arms in grouped.items():
        if set(arms) != set(ARMS):
            raise ValueError('missing arm')
        D4off,_,_ = arms['s4_off']; D4on,Q4,R4 = arms['s4_on']
        D6off,_,_ = arms['s6_off']; D6on,Q6,R6 = arms['s6_on']
        a, b, c = D4on-D6on, Q4-Q6, R4-R6
        e = D4off-D6off
        g, h = D4on-D4off, D6on-D6off
        values=dict(delta_D_on=a,delta_Q=b,delta_R=c,delta_Q_minus_R=b-c,
            delta_D_off=e,delta_D_on_minus_off=a-e,s4_on_minus_off=g,s6_on_minus_off=h)
        assert values['delta_D_on'] == values['delta_Q']+values['delta_R']
        assert values['delta_D_on_minus_off'] == values['s4_on_minus_off']-values['s6_on_minus_off']
        rows.append(dict(block=block,**values))
    summaries={}
    for name in SERIES:
        values=[row[name] for row in rows]
        low, high=rank_interval(values)
        sign='strictly_positive' if low>0 else 'strictly_negative' if high<0 else 'zero_not_excluded'
        summaries[name]=dict(median=statistics.median(values), conditional_median_interval=[low,high],
            ranks_one_based=[3,14], sign=sign, equivalence_claim=False)
    return dict(rows=rows,series=summaries, statistic='median of declared block contrasts',
        upstream_raw_identity_verified_here=False, IID_block_assumption_verified=False,
        intervals_valid_only_under_declared_block_sampling_assumptions=True,
        observation_qualified=False, uninstrumented_service_identified=False, prediction_qualified=False)


def cpu_checks():
    # These are coherent synthetic partitions, not GPU samples or a power estimate.
    records=[]
    x=[0]*9+[100]*7
    y=[0]*2+[100]*7+[0]*7
    for block in range(N):
        for arm,(stage,flag) in ARMS.items():
            Q=50+(x[block] if stage==4 else 0)
            R=50+(y[block] if stage==4 else 0)
            decoded=dict(stages=stage,observer=flag,early_cycles=Q+R,
                full_wait_cycles=Q if flag else None,outside_wait_cycles=R if flag else None)
            records.append(dict(block=block,arm=arm,decoded=decoded))
    result=contrasts(records)
    med={name:row['median'] for name,row in result['series'].items()}
    assert med['delta_D_on']==100 and med['delta_Q']==med['delta_R']==0
    assert all(row['delta_D_on']==row['delta_Q']+row['delta_R'] for row in result['rows'])
    assert all(row['delta_D_on_minus_off']==row['s4_on_minus_off']-row['s6_on_minus_off'] for row in result['rows'])
    for changed in [records[:-1],records+[records[0]],records[:-1]+[records[0]]]:
        try: contrasts(changed)
        except ValueError: pass
        else: raise AssertionError('incomplete/duplicate records accepted')
    bad=[dict(row,decoded=dict(row['decoded'])) for row in records]
    bad[0]['decoded']['full_wait_cycles']=0
    try: contrasts(bad)
    except ValueError: pass
    else: raise AssertionError('off Q=0 accepted')
    assert rank_interval(list(range(16))) == [2,13]
    return dict(coherent_synthetic_blocks=16, median_nonadditivity_witness=med,
        closed_block_partitions=16, closed_interactions=16,
        rejected_missing_duplicate_extra_and_off_zero=True, rank_endpoints_verified=True)


def main():
    checks=cpu_checks()
    failure=Fraction(2*sum(comb(N,j) for j in range(3)),2**N)
    next_failure=Fraction(2*sum(comb(N,j) for j in range(4)),2**N)
    assert failure <= Fraction(1,20*len(SERIES)) < next_failure
    manifest_path=RUN/'manifest.json'
    old=json.loads(manifest_path.read_text())
    by_identity={(sample['stage'],sample['flag']):sample for sample in old['samples']}
    binaries=old['binary_sha256']
    for name, digest in binaries.items():
        assert sha(RUN/'build'/name) == digest
    # Independent uniform arm permutations, precomputed from a fixed seed.
    # No after-the-draw balancing or selection based on timing outcomes.
    rng=random.Random(20261011)
    schedule=[]
    position_counts={arm:[0]*4 for arm in ARMS}
    for block in range(N):
        order=list(ARMS);rng.shuffle(order)
        for position, arm in enumerate(order):
            stage,flag=ARMS[arm]
            position_counts[arm][position]+=1
            schedule.append(dict(process_id=f'b{block:02d}_{arm}',block=block,arm=arm,
                schedule_index=len(schedule),position=position,stage=stage,flag=flag,
                command=by_identity[stage,flag]['command']))
    definitions=dict(
        delta_D_on='D4_on-D6_on',delta_Q='Q4_on-Q6_on',delta_R='R4_on-R6_on',
        delta_Q_minus_R='(Q4_on-Q6_on)-(R4_on-R6_on)',delta_D_off='D4_off-D6_off',
        delta_D_on_minus_off='(D4_on-D6_on)-(D4_off-D6_off)',
        s4_on_minus_off='D4_on-D4_off',s6_on_minus_off='D6_on-D6_off')
    proposal=dict(status='bounded development diagnostic proposal; not frozen or executed by B',
        purpose='Partition the stage contrast in the current instrumented protocol and test its D-response interaction with the runtime observer switch.',
        fixed_scope=dict(config='cfg_b',shape=[2816,1536,4096],stages=[4,6],CTA=0,thread=256,
            CTA_j=1,WG_local_j=0,groups=list(range(2,17)),grid=[1,132,1],
            raster='heuristic resolved along_n',swizzle=1,input='dyadic seed17',
            GPU_uuid=old['gpu_uuid'],new_source_or_probe_required=False,
            binaries=binaries,reference_manifest_sha256=sha(manifest_path)),
        schedule=dict(blocks=N,arms=ARMS,processes=4*N,seed=20261011,
            assignment='uniform independent permutation within each block; fixed before execution',
            actual_position_counts=position_counts, fresh_process_per_target=True,
            target_calls_per_process=1,prefix_calls_per_process=16,prefix_observer=0,
            same_buffers_within_process=True,common_cross_stage_history_claim=False,
            serial_stream=True, schedule_file='schedule-proposal.json'),
        records=dict(preserve=['block/arm/process/actual order','pinned raw SHA and collection archive',
            'full setup and decoder/binary identities','all native clocks and actual SMID',
            '16 prefix event durations/protocol','target event duration','D/Q/R'],
            off_Q_R='None',units='native same-thread duration cycles; component ns stays None',
            cross_process_absolute_clock_subtraction=False,SM_filtering_after_measurement=False),
        estimands=definitions, exact_block_closures=['delta_D_on=delta_Q+delta_R',
            'delta_D_on_minus_off=s4_on_minus_off-s6_on_minus_off'],
        statistical_plan=dict(point='median of each declared block contrast separately',series_count=8,
            candidate_family_confidence='at least 95 percent conditional on appropriate IID blocks',
            interval_ranks=[3,14], per_coordinate_tail_exact=str(failure),
            family_union_failure_exact=str(8*failure),family_union_failure_decimal=float(8*failure),
            next_tighter_symmetric_rank_pair=[4,13],next_pair_tail_exact=str(next_failure),
            no_between_series_independence_required=True, IID_blocks_not_established_by_randomization=True,
            no_precision_or_power_guarantee_from_sample_count=True,
            zero_inside_interval='zero not excluded; not equivalence or absence of perturbation'),
        interpretation=[
            'delta_Q/delta_R and delta_Q_minus_R describe signed stage contrasts under the current observer; they do not recover off-path Q.',
            'delta_D_on_minus_off is a block interaction. A median of interactions is not a difference of separately computed median effects.',
            'A detected D interaction establishes a recorded stage-dependent observer response. Failure to detect it does not establish no Q redistribution.',
            'A positive Q contrast with a negative R contrast suggests examining waiting relocation in the instrumented path; medians do not sum to D.',
            'Use outcomes to choose the next mechanism or disprove a candidate. Do not fit new supply constants from this diagnostic.'
        ],
        stopping='Exactly 64 new processes once. Stop/preserve on identity, device, process, numeric or trace failure. No replacement, filtering, extra samples, or timing-based early success stop.',
        uninstrumented_model_evidence=dict(current_diagnostic_sufficient=False,
            required='An explicit model must account for the observer nodes and independently predict the corresponding no-added-observer/progress endpoints on new frozen data. D compatibility alone cannot transfer Q to the uninstrumented path.',
            original_progress='Historical progress C1/C16 remain a distinct binary/protocol/development reference; do not pool them into the new off arm or relabel cross-batch compatibility as equivalence.'),
        final_goal_scope='All final predeclared cfg_a/b/c acceptance conditions and old failures remain unchanged; this narrow diagnostic grants no family qualification.',
        source_note='NIST binomial-CDF confidence-limit inversion is the statistical reference; the 16/8 ranks and bounds here are independently derived integer binomial calculations.',
        source_url='https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm',
        observation_qualified=False, service_identified=False, prediction_qualified=False)
    (OUTPUT/'proposal.json').write_text(json.dumps(proposal,indent=2)+'\n')
    (OUTPUT/'schedule-proposal.json').write_text(json.dumps(schedule,indent=2)+'\n')
    (OUTPUT/'cpu-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (OUTPUT/'README.md').write_text('''固定现有 cfg_b 两个 binary，16 block，每 block s4/off、s4/on、s6/off、s6/on 各一个独立进程，共64新目标；每进程先16次同buffer、同stage binary的observer0前序。单流串行，block内四臂按预存seed20261011独立打散，不按结果重排、删样或追加。复用当前源码与严格解码器，不新建观测矩阵。

每block定义：`a=D4on−D6on`、`b=Q4on−Q6on`、`c=R4on−R6on`、`d=b−c`、`e=D4off−D6off`、`f=a−e`、`g=D4on−D4off`、`h=D6on−D6off`。先逐调用计算D/Q/R，再逐block核`a=b+c`及`f=g−h`，保留同一完整记录包。off Q/R是None。各量分别取中位数，不相加、不把交互中位数换成两个中位数之差。

8量同时95%条件区间取第3～14次序量：每坐标尾界137/32768，8量并集界137/4096=0.033447265625。第4～13秩不满足本族的Bonferroni界。依据二项CDF反演，[NIST说明](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)；本目录用整数二项系数独立计算具体秩。只在适当IID-block假设下解释区间；随机顺序不证明GPU样本IID，16个block不保证区间足够窄或功效足够。

区间全正/全负只定位本观察路径的有符号对比。含0表示未排除0，不能判D等价或无扰动；`b−c`比较有符号贡献，不是等待占比。若Q差正而R差负，应优先检查带观察路径中的等待重分配。原progress C1/C16是不同binary/协议的开发参照，不并入当前off臂。

本批可以得到instrumented stage差分解与D开关交互，不能恢复off Q；即使D精确等价，等待内外仍可能重分配。用于无新增观察模型还需要显式核算observer节点，并在新的冻结数据中独立预测相应无新增观察/progress端点。该要求保持后续候选检验，不把它扩大成当前跨probe资格矩阵。

此次仅有界机制诊断提案，未由B冻结或执行；CPU核对包含逐block闭合、16个相干合成记录的中位数不可加反例、缺项/重复/额外记录和off Q=0拒绝。plan.py的contrasts是数值计算入口，不代替主线raw身份/数值验证。任何输出的观察/服务/预测资格均false。最终cfg_a/b/c范围与旧失败保持，不追加供给拟合、GPU或重复raw审查。
''')
    receipt=dict(proposal_sha256=sha(OUTPUT/'proposal.json'),schedule_sha256=sha(OUTPUT/'schedule-proposal.json'),
        script_sha256=sha(Path(__file__)),processes=len(schedule),blocks=N,series_count=8,
        interval_ranks=[3,14],family_union_failure_exact=str(8*failure),CPU_checks_passed=True,
        B_GPU_executed=False)
    (OUTPUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':main()
