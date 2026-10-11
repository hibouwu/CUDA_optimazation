#!/usr/bin/env python3
"""R06 offline analysis: CPU replay, SASS loop check, per-SM intervals, report."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

from analyze_r01 import (collect, load_witness, loop_summary, sass_functions, stats, table,
                         timed_loop, write_json)


def replay(directory, row):
    values = load_witness(directory, row, 'output.bin', 'f')
    extra = load_witness(directory, row, 'extra.bin', 'f')
    if len(values) != row['blocks'] * 128 * 8 or len(extra) != row['blocks'] * 128:
        raise ValueError('truncated resource witness')
    for q, value in enumerate(values):
        t, c = (q // 8) % 128, q % 8
        if value != (t + 1) / 1024 + c / 16 + row['steps'] / 1024:
            raise ValueError(f'FFMA CPU mismatch: {directory}')
    for q, value in enumerate(extra):
        if value != sum((q % 128 + 1 + i) / 1024 for i in range(row['live_extra'])):
            raise ValueError(f'live-register consumer mismatch: {directory}')
    if any(s[4] != s[5] for s in row['stamps']):
        raise ValueError('CTA changed SM inside the window')
    return len(values) + len(extra)


def sass_check(path, unroll):
    result = []
    for name, instructions in sass_functions((path / 'build/sass.txt').read_text()).items():
        if 'resource_probe' not in name:
            continue
        window, body = timed_loop(instructions)
        info = dict(function=name, **loop_summary(window, body))
        ffma = sum(1 for x in body if x['op'] == 'FFMA')
        if ffma != 8 * unroll or info['branch_compare_per_iteration'] != 2:
            raise ValueError(f'{name}: {ffma} FFMA per iteration, expected {8 * unroll}')
        info['status'] = 'passed'
        result.append(info)
    if len(result) != 3:
        raise ValueError('expected three register tiers')
    write_json(path / 'sass_check.json', dict(
        status='passed', unroll=unroll, functions=result,
        scope='timed loop = single backward branch between the clock reads; 8 x unroll FFMA '
              'and one compare + branch per iteration'))


def intervals(row):
    """Per SM: CTAs, peak overlap of their timed windows, covered span."""
    by_sm = collections.defaultdict(list)
    for block, s in enumerate(row['stamps']):
        by_sm[s[4]].append((s[0], s[1], block))
    out = {}
    for sm, spans in sorted(by_sm.items()):
        events = sorted([(a, 1) for a, _, _ in spans] + [(b, -1) for _, b, _ in spans])
        active = peak = 0
        for _, d in events:
            active += d
            peak = max(peak, active)
        span = max(b for _, b, _ in spans) - min(a for a, _, _ in spans)
        out[str(sm)] = dict(blocks=len(spans), window_peak_overlap=peak, covered_span_ns=span,
                            intervals=spans)
    return out


def stagger(items, timeline):
    """Median over SMs (all processes) of window starts 2-4 relative to the SM's first start,
    and of start 3 minus the earliest end among windows 1-2 (µs)."""
    starts, gaps = [[], [], []], []
    for x in items:
        for v in timeline[x['record_path']].values():
            spans = sorted(v['intervals'])
            if len(spans) != 4:
                continue
            for k in range(3):
                starts[k].append((spans[k + 1][0] - spans[0][0]) / 1000)
            gaps.append((spans[2][0] - min(spans[0][1], spans[1][1])) / 1000)
    return [statistics.median(s) for s in starts], statistics.median(gaps)


def plot(path, rows, records):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    (path / 'plots').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, scope, label in zip(axes, ('one_cta', 'all_gpu'),
                                ('single CTA FLOP/cycle', 'whole GPU TFLOP/s')):
        x = sorted([r for r in rows if r['scope'] == scope], key=lambda r: r['registers'])
        ax.plot([r['registers'] for r in x], [r['work_rate'] for r in x], 'o-')
        ax.set_xlabel('actual registers / thread')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / 'plots/resource_windows.svg')
    fig.savefig(path / 'plots/resource_windows.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, target in zip(axes, (64, 96, 128)):
        row = next(r for r in records if r['trial_role'] == 'formal' and r['scope'] == 'all_gpu'
                   and r['register_target'] == target)
        start = min(s[0] for s in row['stamps'])
        for s in row['stamps']:
            ax.broken_barh([((s[0] - start) / 1000, (s[1] - s[0]) / 1000)], (s[4] - 0.4, 0.8),
                           facecolors='tab:blue', alpha=0.25)
        ax.set_title(f"{row['registers_per_thread']} registers/thread: timed CTA windows")
        ax.set_ylabel('SM id')
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel('globaltimer µs from first CTA start')
    fig.tight_layout()
    fig.savefig(path / 'plots/sm_intervals.svg')
    fig.savefig(path / 'plots/sm_intervals.png', dpi=150)
    plt.close(fig)
    return True


def analyze(path):
    env = json.loads((path / 'environment.json').read_text())
    unroll = env['unroll']
    sass_check(path, unroll)
    records, checked = collect(path, replay)
    write_json(path / 'cpu_check.json', dict(records=len(records), checked_elements=checked,
                                             status='passed'))
    if env['mode'] != 'formal':
        print(f'smoke: CPU checked {len(records)} processes / {checked} values; SASS passed')
        return
    groups = collections.defaultdict(list)
    timeline = {}
    for r in records:
        if r['trial_role'] == 'formal':
            groups[r['case_id']].append(r)
            if r['scope'] == 'all_gpu':
                timeline[r['record_path']] = intervals(r)
    rows = []
    for key, items in groups.items():
        r = items[0]
        s = stats([x['elapsed'] for x in items])
        full = r['scope'] == 'all_gpu'
        # One CTA: clock64 cycles -> FLOP/cycle; whole GPU: globaltimer ns -> TFLOP/s.
        rate = r['work_flop'] / s['mean'] / (1000 if full else 1)
        # SM clock inside the timed windows = d(clock64) / d(globaltimer), mean over all CTAs.
        clock = statistics.mean((st[3] - st[2]) / (st[1] - st[0]) * 1000
                                for x in items for st in x['stamps'])
        # Ceiling: 4 SMSP x 32 lanes x 2 FLOP = 256 FLOP/cycle/SM at the measured clock.
        ceiling = 256 * (r['sm_count'] * clock * 1e-6 if full else 1)
        peaks = collections.Counter(v['window_peak_overlap'] for x in items
                                    for v in timeline.get(x['record_path'], {}).values())
        starts, gap = stagger(items, timeline) if full else (None, None)
        rows.append(dict(case=key, scope=r['scope'], register_target=r['register_target'],
                         registers=r['registers_per_thread'],
                         local_bytes=r['local_bytes_per_thread'],
                         occupancy_limit_ctas_per_sm=r['occupancy_limit_ctas_per_sm'],
                         blocks=r['blocks'], ffma_per_lane=r['ffma_per_lane'], unroll=r['unroll'],
                         **s, work_flop=r['work_flop'], work_rate=rate,
                         unit='TFLOP/s' if full else 'FLOP/cycle', window_clock_mhz=clock,
                         fraction_of_ceiling=rate / ceiling,
                         per_sm_peak_overlap=' '.join(f'{k}:{v}' for k, v in sorted(peaks.items())),
                         median_window_start_2_3_4_us=starts, median_start3_minus_first_end_us=gap,
                         warmup_all_converged=all(x['warmup_converged'] for x in items)))
    rows.sort(key=lambda x: (x['scope'] != 'one_cta', x['registers']))
    with (path / 'cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(path / 'sm_intervals.json', timeline)
    write_json(path / 'rules.json', dict(
        environment=env, unroll=unroll, cases=rows, cpu_checked_elements=checked,
        scope='128-thread FFMA, 8 chains, 16384 FFMA per lane; interval overlap is a lower '
              'bound on concurrency, not residency'))
    plotted = plot(path, rows, records)

    one = next(r for r in rows if r['scope'] == 'one_cta' and r['register_target'] == 64)
    raw = sorted(groups[one['case']], key=lambda r: r['record_path'])[0]
    s0 = raw['stamps'][0]
    full_raw = sorted(groups['ffma_r64_all_gpu'], key=lambda r: r['record_path'])[0]
    first = min(s[0] for s in full_raw['stamps'])
    last = max(s[1] for s in full_raw['stamps'])
    hand = [
        f"单 CTA `{raw['record_path']}`：clock64 {s0[2]} → {s0[3]}，窗口 {s0[3] - s0[2]} cycle；"
        f"工作量 2 × 128 线程 × 8 链 × 2048 = {raw['work_flop']} FLOP，"
        f"{raw['work_flop'] / raw['elapsed']:.1f} FLOP/cycle（理论上限 4 SMSP × 32 lane × 2 = 256）。"
        f"同一窗口 globaltimer {s0[1] - s0[0]} ns，折合 SM 时钟 "
        f"{(s0[3] - s0[2]) / (s0[1] - s0[0]) * 1000:.0f} MHz。",
        f"整卡 `{full_raw['record_path']}`：{full_raw['blocks']} 个 CTA 的 globaltimer 包络 "
        f"{first} → {last} = {last - first} ns；{full_raw['work_flop']} FLOP / {last - first} ns = "
        f"{full_raw['work_flop'] / (last - first) / 1000:.2f} TFLOP/s。"]
    fmt = lambda v, d=1: f'{v:.{d}f}'
    lines = [
        '# R06 资源与并发：实测（循环展开版）', '',
        f"GPU {env['gpu_uuid']}，{env['hostname']}，作业 {env['job']}，CUDA 12.9，sm_90a。"
        f"128 线程，每 lane 8 条独立 FFMA 链 × 2048 步 = 16384 FFMA；每次循环迭代 8×{unroll} FFMA。"
        f"CPU 重算 {len(records)} 个进程、{checked} 个结果通过；SASS 检查见 `sass_check.json`。", '']
    lines += table(['配置', '实际寄存器/thread', 'API 上限 CTA/SM', '进程', '均值', 'CV', '工作率',
                    '窗口内 SM 时钟 MHz', '占 256 FLOP/cycle/SM 上限', '每 SM 区间最大重叠: SM 数'], [
        [r['case'], r['registers'], r['occupancy_limit_ctas_per_sm'], r['count'],
         f"{fmt(r['mean'])} {'ns' if r['scope'] == 'all_gpu' else 'cycle'}", f"{r['cv']:.2%}",
         f"{r['work_rate']:.4g} {r['unit']}", fmt(r['window_clock_mhz'], 0),
         f"{r['fraction_of_ceiling']:.1%}", r['per_sm_peak_overlap'] or '—'] for r in rows])
    lines += ['', '整卡每 SM 4 个 CTA 的计时窗口分两批（相对该 SM 第一个窗口起点，全部整卡进程的 SM 中位数）：', '']
    lines += table(['配置', '第 2/3/4 个窗口起点 µs', '第 3 个起点 − 前两个最早结束 µs'], [
        [r['case'], ' / '.join(fmt(v, 2) for v in r['median_window_start_2_3_4_us']),
         fmt(r['median_start3_minus_first_end_us'], 2)] for r in rows if r['scope'] == 'all_gpu'])
    lines += ['', '探针只记录窗口起止，不能区分后两个 CTA 是尚未驻留，还是已驻留但计时前未获发射；'
              '单 CTA（每 SMSP 1 warp）已达上限的 94%，整卡速率不受这一点限制。']
    lines += ['', '## 手算核对', ''] + hand
    lines += [
        '', '## 边界', '',
        '- 单 CTA：本 CTA clock64 窗口；整卡：4×SM 数个 CTA 的 globaltimer 包络（含发射偏斜）。'
        '窗口止于最终 shared 消费者与 barrier；额外活跃值在窗口外准备和消费。',
        '- 寄存器档位取 ptxas/API 实际值；local 0 B（无 spill）。API 上限是资源上限，'
        '每 SM 计时区间重叠是观测到的并发下界，不等于完整驻留。',
        '- 采样：单 CTA 3 进程（CV>1% 补 10），整卡 10 进程；每进程 8–30 次预热（末 5 次 CV≤2%）后 1 个正式窗口。']
    if plotted:
        lines += ['', '![资源窗口](plots/resource_windows.svg)', '',
                  '![逐SM计时区间](plots/sm_intervals.svg)']
    (path / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'CPU checked {len(records)} processes / {checked} values; report {path / "report.md"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    analyze(parser.parse_args().input.resolve())
