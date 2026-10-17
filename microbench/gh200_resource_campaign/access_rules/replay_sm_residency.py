#!/usr/bin/env python3
"""完整回放固定 R06 51记录和 R16驻留144记录；不执行GPU或拟合。"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys


BASE = Path('gh200_resource_campaign/access_rules')
R06 = BASE / '20261006-r01r06-unrolled/r06'
R16 = BASE / '20261008-R16-residency-job736989-v1'
ANCHORS = {
    R06: {
        'source_hashes.json': '38f30ff4b0b97732339a66af03a91b156b31f09698d83ef08ce3f0c7e25d2e2b',
        'build/r06': '33a46464c1866ac48b7d92539f681fc9050d572cbfacb1f86f13a230660b9ebc',
        'build/sass.txt': '0def023634960a81f806534b120c8f44db1719d838a709c1a8824dd136bca33f',
        'rules.json': 'f874f2c36b6a1025f3ef30021962e1d7b2128d08c943fa792447702965244402',
        'sass_check.json': 'ae0ff730b9fe313e40de49ef6e7c052a96461619150ebc8455bdf7fe2f3b44da',
        'sm_intervals.json': '2be42f14ed22036cab03b289dbd794f3a2378c5a61710df4b7ad8c35fad028e2',
        'cpu_check.json': 'a8468ee0aabd71648fcb3f80c95e702cd49fffea489063bf28566ace7419a84b',
        'environment.json': '3672926881b0b9b196dea25e80337940731c44d65ab804e64b3bb41648bffd15',
        'protocol.json': '6af28187b910912bbb36994f29b83c7bfed5654128f881fec5efaebd3652a92a',
    },
    R16: {
        'source_hashes.json': '486d9b91de9feef5d8e3782ebd2a0bca22d45fd3fcfd1fb587f380cc307ae69e',
        'build/r16': '2fd5ba7669519c4f4c47de623af31dcf37016b38891c332fd885c32cf96e00f3',
        'build/sass.txt': 'f3c41fcde356f6b4b7e745fb003ebb0c826e07834ef2e5b93c294af50e5a4c6e',
        'analysis/rules.json': '51b4ad85405b448347a29402cef25524869cdb36f5628e4d3c0b9059e28a6561',
        'analysis/checks.json': '71e3ec2114bfab0343cc3b31fe9f9efd024b6dae6ab6cc986b17ef14d74b9d33',
        'analysis/sass-check.json': '6f989ba5e071b88157185f23190fd4d2b94aee362368f353ba455ce14fb87aff',
    },
}
KNOWN_CV_DIFFERENCES = {
    'tc_s1_independent_large_resident2': (0.01312041764516119, 0.013120417645161189),
    'tc_s2_independent_large_resident2': (0.012422167694311505, 0.012422167694311504),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(results, relative):
    root = results / relative
    for name, expected in ANCHORS[relative].items():
        require(sha(root / name) == expected, '原身份不符: ' + str(root / name))
    sources = read(root / 'source_hashes.json')
    for name, expected in sources.items():
        path = root / ('source/' + name if relative == R06 else name)
        require(path.resolve().is_relative_to(root) and sha(path) == expected,
                '冻结源/依赖身份不符: ' + str(path))
    return dict(root=str(relative), anchors=ANCHORS[relative], source_files=sources,
                environment=read(root / 'environment.json'))


def process_files(path):
    return {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file()}


def replay_r06(results, output):
    root = results / R06
    bound = identity(results, R06)
    old_module = sys.modules.get('analyze_r01')
    scalar = load_module(root / 'source/analyze_r01.py', 'frozen_residency_r01')
    sys.modules['analyze_r01'] = scalar
    try:
        helper = load_module(root / 'source/analyze_r06.py', 'frozen_residency_r06')
    finally:
        if old_module is None:
            sys.modules.pop('analyze_r01', None)
        else:
            sys.modules['analyze_r01'] = old_module
    sass = []
    for name, instructions in scalar.sass_functions((root / 'build/sass.txt').read_text()).items():
        if 'resource_probe' not in name:
            continue
        window, body = scalar.timed_loop(instructions)
        info = dict(function=name, **scalar.loop_summary(window, body), status='passed')
        require(info['loop_opcodes']['FFMA'] == 256 and info['branch_compare_per_iteration'] == 2,
                'R06固定256FFMA循环不符')
        sass.append(info)
    require(len(sass) == 3 and sass == read(root / 'sass_check.json')['functions'], 'R06原SASS评分不符')
    records, checks, timeline, groups = [], [], {}, collections.defaultdict(list)
    checked = 0
    for path in sorted((root / 'samples').glob('*/*/result.json')):
        row = read(path)
        require(row['case_id'] == path.parent.parent.name and row['status'] == 'measured' and
                row['gpu_uuid'] == bound['environment']['gpu_uuid'] and row['threads'] == 128 and
                row['chains'] == 8 and row['unroll'] == 32 and row['iterations'] == row['steps'] // 32 and
                row['local_bytes_per_thread'] == 0, 'R06原条件/正确性状态不符')
        full = row['scope'] == 'all_gpu'
        require(row['blocks'] == (528 if full else 1) and row['sm_count'] == 132 and
                row['work_flop'] == 2 * row['blocks'] * 128 * 8 * row['steps'] and
                row['ffma_per_lane'] == 8 * row['steps'], 'R06工作量不符')
        require(read(path.parent / 'command.json')[1:] == ['--registers', str(row['register_target']),
                '--scope', row['scope'], '--steps', str(row['steps'])], 'R06原命令不符')
        require(all(row[key] == value for key, value in read(path.parent / 'stdout.json').items()) and
                (path.parent / 'stderr.log').read_text() == '', 'R06 stdout/result不符')
        stamps = row['stamps']
        require(len(stamps) == row['blocks'] and all(s[4] == s[5] and s[3] > s[2] and s[1] > s[0]
                for s in stamps), 'R06正窗口/首末SMID不符')
        elapsed = max(s[1] for s in stamps) - min(s[0] for s in stamps) if full else stamps[0][3] - stamps[0][2]
        require(elapsed == row['elapsed'], 'R06原窗口不符')
        count = helper.replay(path.parent, row)
        checked += count
        row['record_path'] = path.relative_to(root).as_posix()
        records.append(row)
        checks.append(dict(record_path=row['record_path'], trial_role=row['trial_role'], checked_values=count,
            elapsed=row['elapsed'], unit=row['unit'], warmup_converged=row['warmup_converged'],
            smids=sorted({s[4] for s in stamps}), file_hashes=process_files(path.parent)))
        if row['trial_role'] == 'formal':
            require(row['steps'] == 2048 and row['warmup_converged'], 'R06正式点资格不符')
            groups[row['case_id']].append(row)
            if full:
                timeline[row['record_path']] = helper.intervals(row)
        else:
            require(row['trial_role'] == 'check' and row['steps'] in (32, 64), 'R06有限短检查不符')
        if len(records) % 10 == 0 or len(records) == 51:
            print(f'R06 {len(records)}/51 records; {checked} values', flush=True)
    require(len(records) == 51 and checked == 21914496, 'R06固定完整计数不符')
    cpu_check = dict(records=len(records), checked_elements=checked, status='passed')
    require(cpu_check == read(root / 'cpu_check.json') and
            json.loads(json.dumps(timeline)) == read(root / 'sm_intervals.json'), 'R06原数值/SM区间评分不符')
    rows = []
    for case_id, items in groups.items():
        row = items[0]
        stats = scalar.stats([r['elapsed'] for r in items])
        full = row['scope'] == 'all_gpu'
        rate = row['work_flop'] / stats['mean'] / (1000 if full else 1)
        clock = statistics.mean((s[3] - s[2]) / (s[1] - s[0]) * 1000 for r in items for s in r['stamps'])
        ceiling = 256 * (row['sm_count'] * clock * 1e-6 if full else 1)
        peaks = collections.Counter(v['window_peak_overlap'] for r in items
                                    for v in timeline.get(r['record_path'], {}).values())
        starts, gap = helper.stagger(items, timeline) if full else (None, None)
        rows.append(dict(case=case_id, scope=row['scope'], register_target=row['register_target'],
            registers=row['registers_per_thread'], local_bytes=row['local_bytes_per_thread'],
            occupancy_limit_ctas_per_sm=row['occupancy_limit_ctas_per_sm'], blocks=row['blocks'],
            ffma_per_lane=row['ffma_per_lane'], unroll=row['unroll'], **stats, work_flop=row['work_flop'],
            work_rate=rate, unit='TFLOP/s' if full else 'FLOP/cycle', window_clock_mhz=clock,
            fraction_of_ceiling=rate/ceiling, per_sm_peak_overlap=' '.join(f'{k}:{v}' for k,v in sorted(peaks.items())),
            median_window_start_2_3_4_us=starts, median_start3_minus_first_end_us=gap,
            warmup_all_converged=all(r['warmup_converged'] for r in items)))
    rows.sort(key=lambda row: (row['scope'] != 'one_cta', row['registers']))
    require(rows == read(root / 'rules.json')['cases'] and sum(row['count'] for row in rows) == 39,
            'R06原正式评分不符')
    failed = [row['record_path'] for row in records if not row['warmup_converged']]
    require(failed == ['samples/check_ffma_r64_all_gpu_64/check/result.json'], 'R06旧短检查不收敛身份不符')
    save(output / 'r06-sm-intervals.json', timeline)
    save(output / 'r06.json', dict(identity=bound, original_scores_exact=True, profiles=rows, sass=sass,
        cpu_check=cpu_check, raw_checks=checks, short_warmup_nonconvergence=failed,
        CTA_intervals=sum(len(row['stamps']) for row in records),
        timing='单CTA clock64；整卡globaltimer包络；含FFMA链、最终shared消费者与barrier，额外live值消费在窗外',
        boundary='固定128线程/8链资源对照；无IMAD、1/4warp或展开长度扫描；窗口重叠不是完整CTA驻留'))
    return helper, rows


def compare_r16_scores(original, replayed):
    """只承认两项已核定的具体CV差异，其他字段严格相等。"""
    adjusted = copy.deepcopy(replayed)
    diagnostics = []
    require(len(original['conditions']) == len(adjusted['conditions']) == 12, 'R16十二条件不符')
    for old, new in zip(original['conditions'], adjusted['conditions']):
        require(old['case'] == new['case'], 'R16条件顺序不符')
        if old['cv'] == new['cv']:
            continue
        require(KNOWN_CV_DIFFERENCES.get(old['case']) == (old['cv'], new['cv']),
                '出现已核两项之外的CV差异: ' + old['case'])
        diagnostics.append(dict(case=old['case'], field='cv', old=old['cv'], new=new['cv'],
            old_hex=old['cv'].hex(), new_hex=new['cv'].hex(), ulp_difference=1,
            old_qualified=old['qualified'], new_qualified=new['qualified'],
            old_threshold_pass=old['cv'] <= .05, new_threshold_pass=new['cv'] <= .05))
        new['cv'] = old['cv']
    require(adjusted == original, 'R16除已核CV显示差异之外的原评分不符')
    return diagnostics


def replay_r16(results, output, interval_helper):
    root = results / R16
    bound = identity(results, R16)
    helper = load_module(root / 'source/analyze_r16.py', 'frozen_residency_r16')
    original_replay = helper.replay
    checks, timeline, overlap_histogram = [], {}, collections.Counter()

    def tracked_replay(path, row):
        require(row['case_id'] == path.parent.name and row['label'] == path.name and
                row['returncode'] == row['event_errors'] == row['local_bytes'] == 0 and
                row['registers'] == 58 and row['threads'] == 256 and row['blocks'] == 264 and
                row['stage'] in (1, 2, 4) and row['resident'] in (1, 2) and row['large'] in (0, 1) and
                row['occupancy_limit'] == row['resident'] and
                row['reserved_smem'] == (131072 if row['resident'] == 1 else 98304), 'R16原条件/资源不符')
        formal = row['label'].startswith('formal-')
        require(row['tiles'] == 32 if formal else row['label'].startswith('smoke-') and row['tiles'] in (1, 2),
                'R16有限正式/短检查不符')
        require(row['source_bytes'] == (264 * row['tiles'] * 16384 if row['large'] else 16384) and
                row['warmup_converged'], 'R16供给/暖机资格不符')
        require(read(path / 'command.json')[1:] == ['--stage', str(row['stage']), '--resident', str(row['resident']),
                '--tiles', str(row['tiles']), '--large', str(row['large'])], 'R16原命令不符')
        require(all(row[key] == value for key, value in read(path / 'stdout.json').items()) and
                (path / 'stderr.log').read_text() == '', 'R16 stdout/result不符')
        result = original_replay(path, row)
        normalized = dict(stamps=[(s[2], s[3], s[0], s[1], s[4], s[5]) for s in row['stamps']])
        intervals = interval_helper.intervals(normalized)
        require(len(row['stamps']) == 264 and len(intervals) == 132 and
                all(sm['blocks'] == 2 for sm in intervals.values()), 'R16实际CTA/SM分布不符')
        relative = (path / 'result.json').relative_to(root).as_posix()
        timeline[relative] = intervals
        if formal:
            overlap_histogram.update((row['resident'], sm['window_peak_overlap']) for sm in intervals.values())
        checks.append(dict(record_path=relative, label=row['label'], checked_values=result['words'],
            elapsed_ns=row['elapsed_ns'], warmup_converged=row['warmup_converged'],
            observed_sms=result['observed_sms'], window_overlap_max=result['window_overlap_max'],
            file_hashes=process_files(path)))
        if len(checks) % 12 == 0 or len(checks) == 144:
            print(f'R16 {len(checks)}/144 records; {sum(r["checked_values"] for r in checks)} values', flush=True)
        return result

    helper.replay = tracked_replay
    helper.analyze(root, output=output / 'r16-analysis')
    original, replayed = read(root / 'analysis/rules.json'), read(output / 'r16-analysis/rules.json')
    diagnostics = compare_r16_scores(original, replayed)
    require(read(output / 'r16-analysis/checks.json') == read(root / 'analysis/checks.json') ==
            dict(processes=144, checked_values=467140608), 'R16完整数值计数不符')
    require(read(output / 'r16-analysis/sass-check.json') == read(root / 'analysis/sass-check.json'),
            'R16原SASS评分不符')
    require(len(checks) == 144 and sum(row['label'].startswith('formal-') for row in checks) == 120 and
            overlap_histogram == {(1, 1): 7920, (2, 2): 7920}, 'R16有限进程/动态窗口重叠不符')
    save(output / 'r16-sm-intervals.json', timeline)
    save(output / 'r16.json', dict(identity=bound, original_rules=original, cv_diagnostics=diagnostics,
        all_original_fields_bitwise_exact=not diagnostics, other_original_fields_exact=True,
        generic_float_tolerance_added=False, raw_checks=checks, CTA_intervals=144 * 264,
        formal_SM_window_overlap_histogram={str(key):value for key,value in sorted(overlap_histogram.items())},
        timing='globaltimer整卡包络；含bulk TMA、ready等待、4 WGMMA/tile、commit/wait0、槽释放、最终归约与global输出；末槽保存在窗外',
        boundary='固定64x64x64、256线程、32Ktile；同stage容量1/2只改128/96KiB动态SMEM预留；不识别驻留仲裁或通用occupancy时间函数'))
    return original['conditions'], diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not sys.flags.optimize and sys.byteorder == 'little', '要求普通模式及little endian见证重放')
    sys.dont_write_bytecode = True
    results, output = args.results_root.resolve(strict=True), args.output.resolve()
    require(all(not output.is_relative_to(results / root) for root in (R06, R16)), '不能输出到旧归档')
    output.mkdir(parents=True, exist_ok=False)
    try:
        interval_helper, r06 = replay_r06(results, output)
        r16, diagnostics = replay_r16(results, output, interval_helper)
        full = [row for row in r06 if row['scope'] == 'all_gpu']
        counterexamples = [dict(family='R06', capacity_from=8, capacity_to=4,
            time_from_ns=full[0]['mean'], time_to_ns=full[2]['mean'],
            relative_time_change=full[2]['mean']/full[0]['mean']-1)]
        for stage in (1, 2, 4):
            for large in (0, 1):
                rows = [row for row in r16 if row['stage'] == stage and row['large'] == large]
                require([row['resident'] for row in rows] == [1, 2], 'R16容量对照缺失')
                counterexamples.append(dict(family='R16', stage=stage, independent_large=bool(large),
                    capacity_from=1, capacity_to=2, time_from_ns=rows[0]['ns'], time_to_ns=rows[1]['ns'],
                    relative_time_change=rows[1]['ns']/rows[0]['ns']-1,
                    original_CV=[row['cv'] for row in rows]))
        save(output / 'capacity-responses.json', dict(responses=counterexamples,
            conclusion='容量是准入上限；窗口重叠是执行观测；供给/stage改变容量收益，不能按occupancy固定比例缩放时间'))
        save(output / 'summary.json', dict(schema='gh200-sm-residency-replay/1', status='verified',
            results_root=str(results), replay_source_sha256=sha(Path(__file__)),
            R06=dict(conditions=6, formal_processes=39, short_checks=12, processes=51, checked_values=21914496),
            R16=dict(conditions=12, formal_processes=120, short_checks=24, processes=144, checked_values=467140608),
            total_conditions=18, total_formal_processes=159, total_processes=195, total_checked_values=489055104,
            cv_diagnostics=diagnostics, new_GPU_executions=0, new_fits=0, R16_quota_repeated=False,
            generic_occupancy_time_multiplier_exported=False, full_CTA_residency_observed=False,
            artifacts={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    except Exception as error:
        save(output / 'failure.json', dict(status='failed', error=str(error)))
        raise
    print(f'已完整回放18条件195记录489055104值：{output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
