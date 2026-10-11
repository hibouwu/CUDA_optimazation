#!/usr/bin/env python3
"""B interpretation of four-warp smoke traces under the original release gates.

Smoke is not the ten-process/stage/arm study. Native-cycle interval labels are
conditional diagnostics; they never become a price or an accepted mechanism.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

from analyze_r13_release import release_arrival_bounds, thread_release_arrival_bounds


def interval_classification(interval, scale=64):
    low, high = interval['lower'], interval['upper']
    if not 0<=low<=high:
        raise ValueError('ordered nonnegative interval required')
    return 'compatible' if high<=scale else 'excluded' if low>scale else 'unresolved'


def release_process(sparse):
    if 'release_threads' in sparse:
        raise ValueError('release_threads requires release_thread_process; four-leader interpretation would drop required arrivals')
    clocks = sparse['clocks']
    warp = sparse['release_warps']
    local = clocks['refill_end']-clocks['release_begin']
    upstream = clocks['wait1_end']-clocks['full_wait_begin']
    reasons = []
    if not clocks['refill_begin']<clocks['release_begin']:
        reasons.append('producer_not_waiting_before_leader_release')
    result = dict(leader_pre_to_acquire_end_cycles=local,
        H16_to_wait1_return_cycles=upstream, reasons=reasons)
    if not warp['enabled']:
        if warp['post'][0]-warp['pre'][0]>32:
            reasons.append('warp_bracket_exceeds_32_cycles')
        result.update(four_warp_brackets=False, conditional_classification=None)
        return result
    bounds = release_arrival_bounds(warp['pre'],warp['post'],clocks['refill_end'])
    if bounds['maximum_bracket_width']>32:
        reasons.append('warp_bracket_exceeds_32_cycles')
    result.update(four_warp_brackets=True, bounds=bounds,
        conditional_classification={
            'late_required_arrivals':interval_classification(bounds['after_last_emit_to_acquire_end']),
            'post_arrival_path':interval_classification(bounds['other_warp_skew'])})
    return result


def release_thread_process(sparse):
    """v2 per-thread decoder interface; never fall back to four compatibility fields."""
    threads = sparse['release_threads']
    if (threads.get('requires_warp_simultaneity') is not False
            or threads.get('conditional_on_sass_instruction_order') is not True):
        raise ValueError('per-thread arrival coverage and SASS-order contract required')
    clocks = sparse['clocks']
    reasons = []
    if not clocks['refill_begin']<clocks['release_begin']:
        reasons.append('producer_not_waiting_before_leader_release')
    result = dict(protocol='r13-release-threads-v2',
        leader_pre_to_acquire_end_cycles=clocks['refill_end']-clocks['release_begin'],
        H16_to_wait1_return_cycles=clocks['wait1_end']-clocks['full_wait_begin'],reasons=reasons)
    if not threads['enabled']:
        if clocks['release_end']-clocks['release_begin']>32:
            reasons.append('warp_bracket_exceeds_32_cycles')
        result.update(required_thread_brackets=False,conditional_classification=None)
        return result
    bounds = thread_release_arrival_bounds(threads['physical_threads'],
        threads['pre'],threads['post'],clocks['refill_end'])
    leader = threads['physical_threads'].index(256)
    if (threads['pre'][leader],threads['post'][leader])!=(clocks['release_begin'],clocks['release_end']):
        raise ValueError('per-thread leader differs from original sparse bracket')
    if bounds['maximum_bracket_width']>32:
        reasons.append('required_thread_bracket_exceeds_32_cycles')
    result.update(required_thread_brackets=True,bounds=bounds,
        conditional_classification={
            'late_required_arrivals':interval_classification(bounds['after_last_emit_to_acquire_end']),
            'post_arrival_path':interval_classification(bounds['other_thread_skew'])})
    return result


def paired_observer_gates(processes):
    """Pair first, then median(abs(delta)); signed cancellation cannot pass.

    Limits are the original 32/128 cycle, 64/192 cycle and 2/5 percent gates.
    Main integration confirmed this pre-sampling convention. abs(median(delta))
    remains diagnostic only. Neither endpoint distribution may be filtered.
    """
    grouped = {}
    for p in processes:
        if (type(p['stages']) is not int or p['stages'] not in (4,6)
                or type(p['observer']) is not int or p['observer'] not in (0,1)
                or type(p['trial']) is not int or p['trial']<0):
            raise ValueError('unexpected stage/trial/observer identity')
        key = p['stages'],p['trial'],p['observer']
        if key in grouped:
            raise ValueError('duplicate stage/trial/observer process')
        grouped[key] = p
    summaries = []
    limits = {'local_cycles':(32,128),'upstream_cycles':(64,192),'event_ratio':(.02,.05)}
    for stage in (4,6):
        trials = sorted({t for s,t,f in grouped if s==stage})
        pairs, reasons = [], []
        if len(trials)!=10 or set(trials)!=set(range(10)):
            reasons.append('not_ten_prelisted_trials_per_arm')
        for trial in trials:
            off, on = grouped.get((stage,trial,0)),grouped.get((stage,trial,1))
            if off is None or on is None:
                pairs.append(dict(trial=trial,reasons=['missing_paired_arm']))
                continue
            pair_reasons = []
            for field in ('gpu_uuid','sm','work','slot','phase'):
                if off.get(field)!=on.get(field):
                    pair_reasons.append('paired_identity_differs_'+field)
            if 'decode_error' in off or 'decode_error' in on:
                pair_reasons.append('decode_or_numeric_failure')
            if pair_reasons:
                pairs.append(dict(trial=trial,reasons=pair_reasons))
                continue
            for p in (off,on):
                pair_reasons.extend(p['diagnostic']['reasons'])
                if type(p['elapsed_us']) not in (int,float) or not math.isfinite(p['elapsed_us']) or p['elapsed_us']<=0:
                    raise ValueError('positive finite event time required')
            pairs.append(dict(trial=trial,reasons=pair_reasons,
                local_cycles=on['diagnostic']['leader_pre_to_acquire_end_cycles']-off['diagnostic']['leader_pre_to_acquire_end_cycles'],
                upstream_cycles=on['diagnostic']['H16_to_wait1_return_cycles']-off['diagnostic']['H16_to_wait1_return_cycles'],
                event_ratio=(on['elapsed_us']-off['elapsed_us'])/off['elapsed_us']))
        metrics = {}
        for field, (median_limit,maximum_limit) in limits.items():
            values = [p[field] for p in pairs if field in p]
            if len(values)!=len(trials) or not values:
                metrics[field] = dict(status='incomplete')
                reasons.append('incomplete_pairs_'+field)
                continue
            signed = statistics.median(values)
            median_absolute = statistics.median(abs(v) for v in values)
            absolute_median = abs(signed)
            max_absolute = max(abs(v) for v in values)
            passed = median_absolute<=median_limit and max_absolute<=maximum_limit
            status = 'pass' if passed else 'fail'
            metrics[field] = dict(status=status,signed_median=signed,
                median_absolute=median_absolute,absolute_median=absolute_median,
                maximum_absolute=max_absolute,median_limit=median_limit,maximum_limit=maximum_limit)
            if status!='pass':
                reasons.append('observer_gate_'+status+'_'+field)
        if any(p['reasons'] for p in pairs):
            reasons.append('one_or_more_pairs_invalid')
        summaries.append(dict(stages=stage,pairs=pairs,metrics=metrics,
            median_convention='median_absolute',gate_passed=not reasons,reasons=reasons))
    return summaries


def analyze_smoke(run, decoder_path, request_path, output):
    output.mkdir(parents=True,exist_ok=False)
    inputs = {}
    def fingerprint(path):
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        key = str(path.resolve())
        if key in inputs and inputs[key]!=value:
            raise ValueError('input changed during analysis: '+key)
        inputs[key] = value
        return value
    for path in (decoder_path,request_path,Path(__file__),Path(__file__).with_name('analyze_r13_release.py')):
        fingerprint(path)
    request = json.loads(request_path.read_text())
    if request['diagnostic_scale_cycles']!=64 or request['conditions']['repetitions_per_arm']!=10:
        raise ValueError('original release request scale or sample count differs')
    spec = importlib.util.spec_from_file_location('release_C_main_decoder',decoder_path)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    fingerprint(run/'cases.json')
    # cases.json is the build matrix. The separate correctness script explicitly
    # runs its four smoke calls with common-sw1-16, unlike the adaptive study rows.
    fingerprint(run/'run.sh')
    fingerprint(run/'returned/run.sh')
    commands = (run/'run.sh').read_text()
    if ((run/'returned/run.sh').read_text()!=commands
            or commands.count('--mode warm --warmup-policy common-sw1-16')!=4):
        raise ValueError('smoke command warmup identity differs')
    rows = json.loads((run/'cases.json').read_text())
    processes = []
    for stage in (4,6):
        for flag in (0,1):
            row = next(r for r in rows if r['stages']==stage and r['release_warp_observer']==flag)
            raw = run/f'returned/raw/stage{stage}_flag{flag}_smoke.jsonl.gz'
            raw_sha = fingerprint(raw)
            events = {e['event']:e for line in gzip.open(raw,'rt') if line.strip() for e in [json.loads(line)]}
            setup,call = events['setup'],events['call']
            process = dict(stages=stage,observer=flag,trial=0,gpu_uuid=setup['gpu_uuid'],protocol=setup['trace_version'],
                elapsed_us=call['elapsed_us'],raw=str(raw),raw_sha256=raw_sha)
            try:
                if setup['stages']!=stage:
                    raise ValueError('smoke stage identity differs from filename')
                record = dict(case=row['id'],variant='stamped',trial=0,
                    raw=str(raw.relative_to(run)),raw_sha256=raw_sha)
                replayed = decoder.replay(run,record,dict(row,warmup_policy='common-sw1-16'))
                sparse = replayed['sparse']
                process.update(sm=sparse['sm'],work=sparse['work'],slot=sparse['slot'],phase=sparse['phase'],
                    sampled_D_values_checked=replayed['checked_values'],
                    native_clocks=sparse['clocks'],warp_brackets=sparse['release_warps'],
                    diagnostic=(release_thread_process(sparse) if 'release_threads' in sparse else release_process(sparse)))
                if 'release_threads' in sparse:
                    process['thread_brackets'] = sparse['release_threads']
            except ValueError as error:
                process['decode_error'] = str(error)
            processes.append(process)
    summaries = paired_observer_gates(processes)
    protocols = sorted({p['protocol'] for p in processes})
    if protocols==['r13-release-warps-v1']:
        build_record = 'kernel-identity.json'
    elif protocols==['r13-release-threads-v2']:
        build_record = 'build-audit.json'
    else:
        raise ValueError('mixed or unsupported smoke protocols')
    for name in ('receipt.json',build_record,'source_hashes.json','returned/source_hashes.json',
            'returned/correctness-result.txt','returned/full-d-sha256.txt','returned/allocation.txt','returned/gpu.txt'):
        fingerprint(run/name)
    fingerprint(decoder_path)
    expected_gpu = request['resources']['gpu_uuid']
    reasons = ['correctness_smoke_only_not_prelisted_40_process_study',
        'source_SASS_active_mask_and_full_D_qualification_not_independently_proved_by_this_analyzer']
    if any(p['gpu_uuid']!=expected_gpu for p in processes):
        reasons.append('smoke_GPU_differs_from_original_request_'+expected_gpu)
    if protocols!=['r13-release-warps-v1']:
        reasons.append('new_protocol_identity_requires_its_own_freeze_not_original_four_leader_request')
    if any(not s['gate_passed'] for s in summaries):
        reasons.append('paired_observer_gates_not_qualified')
    labels = Counter(tuple(sorted(p['diagnostic']['conditional_classification'].items()))
        for p in processes if 'diagnostic' in p and p['diagnostic']['conditional_classification'])
    report = dict(schema='gh200-r13-release-interpretation/1',protocols=protocols,processes=processes,
        smoke_preparation='common-sw1-16 explicitly declared by the archived run.sh; build rows are not smoke command records',
        paired_observer_gates=summaries,conditional_label_counts=[dict(labels=dict(k),count=v) for k,v in labels.items()],
        interpretation_qualified=False,unqualified_reasons=reasons,original_request=request,
        limits=['Intervals are native same-SM cycles, conditional on brackets enclosing all required arrivals.',
            'No cross-process absolute clock subtraction; pairs compare within-call durations.',
            'Conditional compatible/excluded labels are not adopted mechanisms when validity fails.',
            'A smoke failure is a preparation diagnostic, not a formal-study or heldout score.',
            'No source price, bare barrier latency, physical traffic or frequency conversion.'],
        input_sha256=inputs)
    (output/'B-release-warp-smoke.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    checked = sum('decode_error' not in p for p in processes)
    lines = ['# release smoke：条件区间与观察扰动', '',
        f'{checked}/4份原始 smoke 通过严格 replay 与每进程4096个 D 抽样值复核。每stage各一对，',
        '不是预列40进程研究；所有机制解释保持不合格。',
        '实际协议：'+', '.join(protocols)+'。',
        '实际GPU：'+', '.join(sorted({p['gpu_uuid'] for p in processes}))+'。', '',
        'smoke暖机按归档run.sh中预先声明的common-sw1-16绑定；cases.json是构建矩阵，',
        '不能把其默认adaptive身份当作这四次实际命令。两份run.sh逐字一致并保存SHA。', '',
        '| stage | local on−off cycle | H16→wait1 on−off cycle | event on/off |',
        '|---|---:|---:|---:|']
    for s in summaries:
        p = s['pairs'][0]
        if 'local_cycles' in p:
            lines.append(f'| {s["stages"]} | {p["local_cycles"]} | {p["upstream_cycles"]} | {100*p["event_ratio"]:.4f}% |')
    lines += ['', '| stage | post-last-emit→acquire后 cycle | 相对leader的必要arrival分散 cycle |',
        '|---|---|---|']
    for p in processes:
        if p['observer'] and 'diagnostic' in p:
            b = p['diagnostic']['bounds']
            a = b['after_last_emit_to_acquire_end']
            c = b['other_thread_skew'] if 'other_thread_skew' in b else b['other_warp_skew']
            lines.append(f'| {p["stages"]} | [{a["lower"]},{a["upper"]}] | [{c["lower"]},{c["upper"]}] |')
    lines += ['', '每stage仅有一对smoke。上表报告实际差值；不足十次，不能判正式扰动门槛通过或正式研究失败。',
        '局部差值是否落在数值门槛内，不提升整批观察资格；不选择性采用条件区间。',
        '主集成已在正式采样前明确 median(abs(delta)) 口径；abs(median(delta))仅列诊断，',
        '不参与通过判定。最大绝对单对及其它阈值均不变。', '']
    (output/'B-release-warp-smoke.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),processes=len(processes),interpretation_qualified=False,
        observer_summaries=summaries)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-run',type=Path,required=True)
    parser.add_argument('--decoder',type=Path,required=True,help='read-only main analyze_r18.py')
    parser.add_argument('--request',type=Path,required=True,help='original B minimal observation JSON')
    parser.add_argument('--output',type=Path,required=True,help='new B-only reanalysis directory')
    args = parser.parse_args()
    analyze_smoke(args.smoke_run.resolve(),args.decoder.resolve(),args.request.resolve(),args.output.resolve())
