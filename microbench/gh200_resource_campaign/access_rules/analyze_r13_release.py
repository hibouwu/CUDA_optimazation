#!/usr/bin/env python3
"""Same-call cfg_b release prerequisites and a bounded observation request.

Existing single-lane clocks cannot locate the last of 128 empty arrivals.
Witnesses below are post-observation feasible decompositions, never prices
or inputs to a prediction. No GPU operations or old-report mutations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from analyze_r18 import replay
from supply_model import input_slot_state


def release_arrival_bounds(before, after, acquire_end):
    """Bounds in one native SM clock domain, given all four warp brackets.

    Each bracket must enclose that warp's original empty arrival instruction;
    caller must verify the CTA, group/slot/phase and unchanged 128 arrivals.
    The post-arrival interval still contains processing, polling, scheduling,
    expect_tx and the acquire-end clock path. It is not barrier latency.
    """
    if len(before)!=4 or len(after)!=4 or any(b>e for b,e in zip(before,after)):
        raise ValueError('four ordered warp arrival brackets required')
    lower = max(before)
    upper = min(max(after), acquire_end)
    if lower > upper:
        raise ValueError('producer returned before a required warp could arrive')
    return dict(last_arrival_emit=dict(lower=lower, upper=upper),
        other_warp_skew=dict(lower=max(0, lower-after[0]), upper=upper-before[0]),
        after_last_emit_to_acquire_end=dict(lower=acquire_end-upper, upper=acquire_end-lower),
        leader_pre_to_acquire_end=acquire_end-before[0],
        maximum_bracket_width=max(e-b for b,e in zip(before,after)))


def request_spec():
    return dict(status='Proposed development observation; not an independent local holdout or a runnable runner config.',
        question='Does the cfg_b release-to-acquire interval mainly precede the last required warp arrival or follow it?',
        competing_explanations={
            'late_required_arrivals': 'All four warp arrivals may be spread out; after the last arrival emit, acquire-end follows within 64 cycles.',
            'post_arrival_path': 'The last required warp arrival emits within 64 cycles of thread256 actual arrival emit; most remaining time follows it.'},
        diagnostic_scale_cycles=64,
        scale_reason='Existing single-warp release bracket is 20..23 cycles, acquire-to-three-input-issues is 60..69; 64 is a declared resolution band, not a hardware constant.',
        conditions=dict(config='cfg_b', m=2816, n=1536, k=4096, stages=[4,6],
            cluster=[1,1], ctas=132, raster_order='heuristic', resolved_raster_order='along_n', swizzle=1,
            input_mode='dyadic', seed=17, lda=4096, ldb=1536, ldd=1536,
            alloc_lda=4104, alloc_ldb=1536, epilogue_defer=0,
            target_cta=0, target_output=1, release_group=80, release_at_local_ktile=17,
            refill_groups={'4':84,'6':86},
            observer_flags=[0,1], study_processes=40, repetitions_per_arm=10),
        implementation=dict(
            base='20261010-R13-stage-events sparse protocol; same binary per stage with runtime flag',
            off='Existing thread256 release bracket and original producer/full/commit/wait1/MAIN endpoints.',
            on='Also bracket the same consumer_release instruction in thread288/320/352, using clock64 only. All 128 original arrival calls remain exactly once.',
            no_extra_dependencies='No named barrier, warpgroup sync, arrival, expect_tx or producer wait added. No independent polling warp is needed.',
            record_fields=['cta','sm','output','global_group','slot','consumer_phase',
                'warp_threads[256,288,320,352]','release_pre_cycles[4]','release_post_cycles[4]',
                'producer_refill_begin','producer_refill_end','producer_refill_issue_end',
                'full_wait_begin16','full_wait_end16','commit16','wait1_end16',
                'FIRST','mma_return','MAIN','event_us','observer_flag'],
            warmup='Both arms use the same existing warmup/preparation with additional warp observation disabled; arm flag changes only for target sampling.'),
        valid_before_interpretation=dict(
            source_and_sass='Verify original consumer_release/wait1/slot logic, clock brackets around the actual arrival in each warp, unchanged GMMA/TMA counts, and static resource/spill report. Same binary within each on/off pair.',
            numeric='Full D off/on versus the same uninstrumented result, unchanged tolerance; nonuniform dyadic input; no new relaxed check.',
            identity='Actual allocated GPU UUID, work mapping and target SM must agree within paired arms; all four brackets present and same group/slot/phase; producer refill waiting starts before the required release.',
            brackets='Each warp release bracket width <=32 cycles; producer refill-end >= max(release-pre[4]); failures remain invalid/censored.',
            local_observer_gate='Same-stage on-minus-off leader release-pre→refill-end: absolute paired median <=32 cycles and largest absolute pair <=128 cycles.',
            upstream_observer_gate='Same-stage on-minus-off H16→wait1-return16: absolute paired median <=64 cycles and largest absolute pair <=192 cycles.',
            whole_observer_gate='Incremental on/off CUDA-event perturbation absolute paired median <=2%, maximum <=5%; this does not prove local endpoints unperturbed.'),
        interpretation=dict(
            all_windows='Report every process, both stages, all warp brackets, last-emit/skew/post-arrival intervals and paired observer windows. No filtering by favorable mechanism classification.',
            late_required_arrivals='Compatible only if the post-last-emit upper bound <=64; excluded if its lower bound >64; otherwise unresolved.',
            post_arrival_path='Compatible only if the other-warp-skew upper bound <=64; excluded if its lower bound >64; otherwise unresolved.',
            mixed='Both mechanisms may contribute. Neither is adopted if both fail, intervals straddle the bands, or observer validity fails.',
            limits='No measurement here identifies bare notification latency, physical transport bandwidth or a new supply price. No old V09 score changes.'),
        resources=dict(gpu_uuid='GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114',
            scheduling='Main integration only, valid Slurm allocation and one serial timing stream; actual AllocTRES recorded there.',
            study='40 study processes plus the main integration prelisted same-card references; two stage binaries; full correctness outside timing.',
            no_matrix='No new geometry, pitch, A rotation, full V10 or broad cutoff sweep.'),
        freeze='Main integration must save final source/binary identities, numeric gates, complete conditions and this comparison before sampling. A source/endpoint change requires a new identity.')


def analyze(run, output):
    output.mkdir(parents=True, exist_ok=False)
    cases = {r['id']:r for r in json.loads((run/'cases.json').read_text())}
    setups = {s['case']:s['setup'] for s in json.loads((run/'static_setup.json').read_text())}
    inputs = {}
    def fingerprint(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (run/'cases.json',run/'static_setup.json',Path(__file__),Path(__file__).with_name('supply_model.py')):
        fingerprint(path)
    processes = []
    for stages in (4,6):
        name = f'r13_stage_k4096_s{stages}'
        for trial in range(10):
            record_path = run/f'samples/{name}/stamped-{trial:02d}.json'
            record = json.loads(record_path.read_text())
            raw = run/record['raw']
            fingerprint(record_path);fingerprint(raw)
            if inputs[str(raw)] != record['raw_sha256'] or record['returncode']:
                raise ValueError('source identity/failure: '+str(record_path))
            sparse = replay(run, record, cases[name])['sparse']
            c = sparse['clocks']
            state = input_slot_state(80, stages)
            if (sparse['slot'], sparse['phase']) != (state['slot'], state['empty_wait_phase']):
                raise ValueError('producer slot/generation phase mismatch')
            if not c['refill_begin'] < c['release_begin'] < c['release_end'] <= c['refill_end']:
                raise ValueError('expected same-generation waiter/leader ordering is absent')
            leader = (c['release_begin']+c['release_end'])//2
            # These two diagnostic points reproduce the exact same observations.
            # Neither hidden last-arrival value is used to make a prediction.
            witnesses = []
            for label, last in (('late_required_arrivals', c['refill_end']-32),
                                ('post_arrival_path', leader+16)):
                if not leader <= last <= c['refill_end']:
                    raise ValueError('declared diagnostic witness is outside the feasible interval')
                witnesses.append(dict(explanation=label, leader_emit=leader,
                    other_warp_last_emit=last, warp_skew=last-leader,
                    after_last_emit=c['refill_end']-last,
                    exact_acquire_end=leader+(last-leader)+(c['refill_end']-last)))
            processes.append(dict(stages=stages, trial=trial, sm=sparse['sm'],
                slot=state['slot'], generation=state['generation'], refill_group=80+stages,
                release_begin=c['release_begin'],release_end=c['release_end'],refill_end=c['refill_end'],
                leader_pre_to_refill_end=c['refill_end']-c['release_begin'],
                leader_post_to_refill_end=c['refill_end']-c['release_end'],
                leader_bracket=c['release_end']-c['release_begin'],
                possible_last_emit=dict(lower=c['release_begin'],upper=c['refill_end']),
                witnesses=witnesses,record=str(record_path),raw=str(raw)))
    source_paths = [run/'source/overlay/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp',
        run/'source/overlay/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp',
        run/'source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp',
        run/'build/cfg_b_s4_stamped.sass',run/'build/cfg_b_stamped.sass']
    for path in source_paths:fingerprint(path)
    def span(values):
        return dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
    summary = [dict(stages=s,**{k:span([p[k] for p in processes if p['stages']==s])
        for k in ('leader_pre_to_refill_end','leader_post_to_refill_end','leader_bracket')}) for s in (4,6)]
    report = dict(scope='Existing same-SM cycle clocks; hidden-arrival feasibility only, not a fit or physical delay inference.',
        source_facts=dict(empty_arrivals=128,required_warps=4,observed_release_thread=256,
            kernel='num_consumers=NumThreadsPerWarpGroup',
            pipeline='cluster_size==1 initializes empty count=params.num_consumers; is_signaling_thread=true',
            collective='K_PIPE_MMAS=1; Ktile17 wait1 precedes release of Ktile16; only thread256 stamped',
            sass={'stage4':['2190 pre clock','21f0 SYNCS.ARRIVE','2200 post clock'],
                  'stage6':['2220 pre clock','2280 SYNCS.ARRIVE','2290 post clock']}),
        processes=processes,summary=summary,
        conclusion='Both late-required-arrival and post-arrival explanations have exact nonnegative witnesses in all 20 calls. Current clocks cannot locate the last of 128 arrivals.',
        input_sha256=inputs)
    (output/'B-release-prerequisites.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'B-minimal-release-observation.json').write_text(json.dumps(request_spec(),indent=2,allow_nan=False)+'\n')
    lines=['# cfg_b empty释放前驱的离线检验','',
        '同一次调用的thread256释放标记只括住一个warp的arrival；cluster1的empty barrier需要四个warp共128次arrival。现有记录未给其它三个warp的发出时刻。',
        '全部20次producer在thread256释放前已开始等同slot的下一代；但标记后至acquire返回不能独占归因通知/轮询。','',
        '| stage | leader pre→acquire后，cycle | leader post→acquire后，cycle | leader bracket，cycle |',
        '|---|---|---|---|']
    for s in summary:
        values=[]
        for field in ('leader_pre_to_refill_end','leader_post_to_refill_end','leader_bracket'):
            v=s[field];values.append(f'{v["median"]} [{v["minimum"]},{v["maximum"]}]')
        lines.append('| '+str(s['stages'])+' | '+' | '.join(values)+' |')
    lines+=['','两组明确的非负分解逐调用重现同一组端点：其一最后warp较晚、之后32cycle；其二其它warp只晚16cycle、剩余时间在最后arrival之后。16/32仅为诊断见证点，不是定出的硬件参数，不能用于正式预测。',
        '最小观察只补thread288/320/352对原arrival的clock64前后界，同stage同binary以runtime开关匹配thread256-only协议。无需新增polling warp、named barrier或arrival。冻结请求见JSON，只有主集成能实现公共probe并安排GPU。',
        '新的区间会分别报告last arrival之前的warp分散与之后的处理/轮询/调度/expect_tx合并路径；后者仍不是裸barrier延迟。所有原失败和冻结分数保留。','']
    (output/'B-release-prerequisites.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),processes=len(processes),summary=summary)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    analyze(args.run.resolve(),args.output.resolve())
