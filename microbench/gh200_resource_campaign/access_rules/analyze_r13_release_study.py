#!/usr/bin/env python3
"""Read-only B replay of a frozen release-thread study and H16 decomposition."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys


def span(values):
    return dict(median=statistics.median(values),minimum=min(values),maximum=max(values))


def analyze(run, output):
    output.mkdir(parents=True,exist_ok=False)
    inputs = {}
    def sha(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        name = str(path.resolve())
        if name in inputs and inputs[name]!=digest:
            raise ValueError('input changed: '+name)
        inputs[name] = digest
        return digest
    def read(path):
        sha(path)
        return json.loads(path.read_text())
    manifest,freeze = read(run/'manifest.json'),read(run/'freeze.json')
    for name,digest in freeze['files'].items():
        if sha(run/name)!=digest or sha(run/'returned'/name)!=digest:
            raise ValueError('execution freeze mismatch: '+name)
    for name,digest in manifest['analysis_sha256'].items():
        if sha(run/'analysis_snapshot'/name)!=digest:
            raise ValueError('analysis snapshot mismatch: '+name)
    for name,digest in manifest['binary_sha256'].items():
        if sha(run/'build'/name)!=digest:
            raise ValueError('compiled binary mismatch: '+name)
    receipt = read(run/'receipt.json')
    if receipt['returncode'] or sha(run/'output.tar.gz')!=receipt['sha256']:
        raise ValueError('incomplete or changed returned archive')
    expected_gates = dict(each_thread_bracket_cycles=32,local_median_cycles=32,local_max_cycles=128,
        upstream_median_cycles=64,upstream_max_cycles=192,event_median_ratio=.02,event_max_ratio=.05,
        classification_scale_cycles=64)
    if manifest['gates']!=expected_gates or manifest['median_convention']!='median(abs(delta))':
        raise ValueError('frozen release gate contract differs')
    if manifest['study_processes']!=40 or manifest['reference_processes']!=20:
        raise ValueError('unexpected frozen study scope')
    # A fresh process resolves every replay dependency from the frozen snapshot.
    sys.path.insert(0,str(run/'analysis_snapshot'))
    from analyze_r18 import replay
    from analyze_r13_release_interpretation import release_thread_process,paired_observer_gates
    data = run/'returned'
    complete = read(data/'collection-complete.json')
    if complete['manifest_sha256']!=sha(run/'manifest.json'):
        raise ValueError('collector manifest identity differs')
    rows = {r['id']:r for r in read(run/'cases.json')}
    records = read(data/'records.json')
    if [r['id'] for r in records]!=[s['id'] for s in manifest['samples']]:
        raise ValueError('missing, extra or reordered frozen samples')
    processes,references,components = [],[],[]
    for execution_index,(record,expected) in enumerate(zip(records,manifest['samples'])):
        if any(record[k]!=v for k,v in expected.items()) or record['returncode']:
            raise ValueError('sample execution identity/failure: '+record['id'])
        raw = data/record['raw']
        if sha(raw)!=record['raw_sha256']:
            raise ValueError('raw process identity differs')
        result = replay(data,record,rows[record['case']])
        events = {v['event']:v for line in gzip.open(raw,'rt') for v in [json.loads(line)]}
        setup = events['setup']
        if (setup['gpu_uuid']!=manifest['gpu_uuid'] or setup['stages']!=record['stage']
                or setup['grid']!=[1,132,1] or setup['resolved_raster_order']!='along_n'):
            raise ValueError('actual GPU/stage/grid/raster differs')
        process = dict(stages=record['stage'],observer=record['flag'],trial=record['trial'],
            id=record['id'],raw=record['raw'],gpu_uuid=setup['gpu_uuid'],
            elapsed_us=result['elapsed_us'],checked_values=result['checked_values'])
        if record['kind']=='reference':
            process['position'] = record['position']
            references.append(process)
            continue
        if setup['trace_version']!=manifest['protocol']:
            raise ValueError('actual source protocol differs')
        sparse = result['sparse']
        process.update(sm=sparse['sm'],work=sparse['work'],slot=sparse['slot'],phase=sparse['phase'],
            sparse=sparse,diagnostic=release_thread_process(sparse))
        processes.append(process)
        c = sparse['clocks']
        full_wait = c['full_wait_end']-c['full_wait_begin']
        issue_commit = c['commit_end']-c['full_wait_end']
        wait1 = c['wait1_end']-c['commit_end']
        H = c['wait1_end']-c['full_wait_begin']
        if H!=full_wait+issue_commit+wait1:
            raise ValueError('H16 components do not close')
        post_marker_wait = max(0,c['full_wait_end']-max(c['full_wait_begin'],c['issue_end']))
        component = dict(stages=record['stage'],observer=record['flag'],trial=record['trial'],
            execution_index=execution_index,
            H16=H,full_wait=full_wait,issue_commit=issue_commit,wait1=wait1,
            local_release_to_acquire=c['refill_end']-c['release_begin'],
            producer_issue_lead_at_wait=c['full_wait_begin']-c['issue_end'],
            producer_acquire=c['acquire_end']-c['acquire_begin'],
            producer_after_acquire_to_issue=c['issue_end']-c['acquire_end'],
            wait_after_issue_marker_lower_bound=post_marker_wait,
            wait_before_issue_marker=full_wait-post_marker_wait,
            elapsed_us=result['elapsed_us'])
        if record['flag']:
            component['earliest_extra_release_pre_minus_H16_end'] = min(sparse['release_threads']['pre'])-c['wait1_end']
        components.append(component)
    gates = paired_observer_gates(processes)
    reference_gates = []
    for stage in (4,6):
        selected = [r for r in references if r['stages']==stage]
        times = [r['elapsed_us'] for r in selected]
        pre = [r['elapsed_us'] for r in selected if r['position']=='pre']
        post = [r['elapsed_us'] for r in selected if r['position']=='post']
        off = [p['elapsed_us'] for p in processes if p['stages']==stage and not p['observer']]
        cv = statistics.pstdev(times)/statistics.fmean(times)
        reference_gates.append(dict(stage=stage,n=len(times),CV=cv,
            passed=len(times)==10 and cv<=manifest['reference_cv_limit'],
            pre_post_median_ratio=statistics.median(post)/statistics.median(pre)-1,
            off_plain_median_ratio=statistics.median(off)/statistics.median(times)-1))
    numeric = read(data/'full-D-checks.json')
    full_D = (len(numeric)==4 and {r['binary'] for r in numeric}==set(manifest['binary_sha256'])
        and all(r['elements']==4325376 for r in numeric) and len({r['D_sha256'] for r in numeric})==1)
    qualified = full_D and all(g['gate_passed'] for g in gates) and all(r['passed'] for r in reference_gates)
    replayed = dict(manifest_sha256=sha(run/'manifest.json'),processes=processes,references=references,
        errors=[],missing=[],full_D_passed=full_D,paired_observer_gates=gates,
        reference_gates=reference_gates,observation_qualified=qualified)
    original = read(run/'study-analysis.json')
    for field,value in replayed.items():
        if original[field]!=value:
            raise ValueError('independent replay changed frozen result: '+field)
    summaries,paired = [],[]
    fields = ('H16','full_wait','issue_commit','wait1','local_release_to_acquire',
        'producer_issue_lead_at_wait','wait_after_issue_marker_lower_bound','elapsed_us')
    for stage in (4,6):
        for flag in (0,1):
            group = [v for v in components if (v['stages'],v['observer'])==(stage,flag)]
            within_arm_H_differences = [abs(a['H16']-b['H16'])
                for i,a in enumerate(group) for b in group[:i]]
            summaries.append(dict(stages=stage,observer=flag,n=len(group),
                windows={k:span([p[k] for p in group]) for k in fields},
                issue_marker_before_wait_count=sum(p['producer_issue_lead_at_wait']>=0 for p in group),
                within_arm_all45_absolute_H_differences=span(within_arm_H_differences)))
        for trial in range(10):
            off = next(p for p in components if (p['stages'],p['observer'],p['trial'])==(stage,0,trial))
            on = next(p for p in components if (p['stages'],p['observer'],p['trial'])==(stage,1,trial))
            delta = {k:on[k]-off[k] for k in fields}
            if delta['H16']!=delta['full_wait']+delta['issue_commit']+delta['wait1']:
                raise ValueError('paired H16 components do not close')
            paired.append(dict(stages=stage,trial=trial,on_minus_off=delta,
                event_ratio=delta['elapsed_us']/off['elapsed_us'],
                exceeds_upstream_maximum=abs(delta['H16'])>192))
    largest = [max((v for v in components if v['stages']==s),key=lambda p:p['H16']) for s in (4,6)]
    sha(Path(__file__))
    extra_clock_gaps = [p['earliest_extra_release_pre_minus_H16_end'] for p in components if p['observer']]
    report = dict(scope='Frozen development observation replay; no refit or adopted mechanism.',
        execution_freeze_revision=freeze.get('revision'),execution_previous_attempt=freeze.get('previous_attempt'),
        replayed=replayed,all_components=components,all_pairs=paired,by_arm=summaries,largest_H16=largest,
        earliest_extra_clock_after_H16_end=span(extra_clock_gaps),
        reference_and_observer_qualification=qualified,exact_main_fields=list(replayed),input_sha256=inputs,
        limits=['All 60 frozen records retained; no new pairing, filtering, gate relaxation or replacement.',
            'Full D entries prove equality between archived binary checks; 60x4096 values are independently numerically replayed.',
            'Producer issue_end is a marker after all three TMA issues, not source completion or physical visibility.',
            'Wait after that marker excludes still-unissued requests in that bounded interval; it is not bare TMA latency.',
            'A failed on/off gate does not by itself identify observer causality, TC retirement or a memory resource.',
            'Extra release clocks may lie outside H16 without excluding prior control/history/scheduling differences.',
            'Conditional all-thread intervals remain unadopted while observation qualification fails.'])
    (output/'B-release-study-replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# 128线程冻结研究：独立复放与上游分解', '',
        '40 study与20 plain原记录、冻结源码/分析/二进制身份重新核对；与主线全部结果字段精确一致。',
        '两stage均因上游门槛失败，观察资格false。全部记录保留，不将条件分类采用为价格。', '',
        '| stage / 开关 | H16中位[范围] | full_wait中位[范围] | issue→commit中位[范围] | wait1中位[范围] |',
        '|---|---|---|---|---|']
    for s in summaries:
        cells = [f'{s["stages"]}/{s["observer"]}']
        for field in ('H16','full_wait','issue_commit','wait1'):
            v = s['windows'][field]
            cells.append(f'{v["median"]} [{v["minimum"]},{v["maximum"]}]')
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '两个最大H16窗口：']
    for p in largest:
        lines.append(f'- stage{p["stages"]} trial{p["trial"]} flag{p["observer"]}：{p["H16"]} = '
            f'{p["full_wait"]} full_wait + {p["issue_commit"]} issue/commit + {p["wait1"]} wait1；'
            f'producer issue_end早于wait_begin {p["producer_issue_lead_at_wait"]}cycle。')
    lines += ['', '超过冻结上游最大门槛的原配对全部保留：', '',
        '| stage / trial | ΔH16 | Δfull_wait | Δissue/commit | Δwait1 | Δlocal | event变化 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in paired:
        if p['exceeds_upstream_maximum']:
            d = p['on_minus_off']
            lines.append(f'| {p["stages"]}/{p["trial"]} | {d["H16"]} | {d["full_wait"]} | '
                f'{d["issue_commit"]} | {d["wait1"]} | {d["local_release_to_acquire"]} | {100*p["event_ratio"]:.4f}% |')
    before_count = sum(p['producer_issue_lead_at_wait']>=0 for p in components)
    lines += ['', f'{before_count}/40次issue_end在full_wait开始前，这部分记录的整段等待均在三次请求发出标记之后。',
        f'20个on记录中，最早新增release前戳仍晚于H16结束至少{min(extra_clock_gaps)}cycle。',
        '新增读取不属于该H16计时区间，但这不排除此前控制、执行历史或调度的间接差异。',
        '同臂45对差只描述原样本内部波动，不重新配对或替换冻结on/off评分。',
        'issue_end是三次请求发出后的标记，后续等待仍可包含排队、源/目的进度、通知和调度；不能命名为裸TMA延迟。',
        '全部on/off及off内部窗口、每对分解和producer前后界在JSON；未剔除大值，也未用其它窗口抵消门槛失败。', '']
    (output/'B-release-study-replay.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),study=len(processes),references=len(references),
        exact_main_fields=len(replayed),qualified=qualified,largest_H16=largest)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new exclusive B reanalysis directory')
    args = parser.parse_args()
    analyze(args.run.resolve(),args.output.resolve())
