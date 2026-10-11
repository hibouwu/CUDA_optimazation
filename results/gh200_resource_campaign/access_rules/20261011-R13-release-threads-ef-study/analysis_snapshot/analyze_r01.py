#!/usr/bin/env python3
"""R01 offline analysis: CPU replay of saved witnesses, SASS loop check, fits, report."""
import argparse
import array
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

SCALAR_OPS = ('ffma', 'add', 'shared', 'global_small', 'global_large')


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_witness(directory, row, name, code):
    raw = gzip.open(directory / (name + '.gz'), 'rb').read()
    info = row['files'][name]
    if len(raw) != info['bytes'] or hashlib.sha256(raw).hexdigest() != info['sha256']:
        raise ValueError(f'witness hash mismatch: {directory / name}')
    values = array.array(code)
    values.frombytes(raw)
    return values


def wgmma_reference(n, steps):
    """FP32 accumulator fragment of m64nNk16: thread t, register j -> (row, col)."""
    expected = []
    for t in range(128):
        for j in range(n // 2):
            row = (t // 32) * 16 + (t % 32) // 4 + ((j // 2) % 2) * 8
            col = (t % 4) * 2 + j % 2 + (j // 4) * 8
            dot = sum((1 + (row + 2 * k) % 7) / 16 * (1 + (col + 3 * k) % 11) / 32
                      for k in range(16))
            expected.append(1 / 64 + steps * dot)
    return expected


def replay(directory, row):
    """Recompute every saved lane/chain result on the CPU; return the element count."""
    op = row['op']
    steps = 0 if row['configuration'].get('control') else row['steps']
    if op == 'wgmma':
        values = load_witness(directory, row, 'output.bin', 'f')
        expected = wgmma_reference(row['n'], steps)
    elif op == 'ldmatrix':
        # Three-block ring: after N steps every d register and the offset equal 512*(N%3).
        values = load_witness(directory, row, 'output.bin', 'I')
        expected = [512 * (steps % 3)] * (32 * 5)
    else:
        values = load_witness(directory, row, 'output.bin', 'f' if op == 'ffma' else 'I')
        starts = load_witness(directory, row, 'starts.u32', 'I')
        edges = load_witness(directory, row, 'edges.u32', 'I')
        ring = dict(zip(edges[0::2], edges[1::2]))
        if len(ring) * 2 != len(edges) or len(set(ring.values())) != len(ring):
            raise ValueError('edge witness is not a partial permutation')
        streams = row['streams']
        single = row['warps'] == 1 and op not in ('ffma', 'add')
        if len(values) != row['warps'] * 32 * streams or len(starts) != len(values):
            raise ValueError('truncated scalar witness')
        expected = []
        for q, start in enumerate(starts):
            t, chain = divmod(q, streams)
            count = steps if (not single or t == 0) else 0
            if op == 'ffma':
                x = (t + 1) / 1024 + chain / 16 + count / 1024
            elif op == 'add':  # x_{n+1} = x_n + x_{n-1} mod 2^32, x_{-1} = 17, x_0 = start
                prev, x = 17, start
                for _ in range(count):
                    prev, x = x, (x + prev) & 0xffffffff
            else:
                x = start
                for _ in range(count):
                    x = ring[x]
            expected.append(x)
    if list(values) != expected:
        raise ValueError(f'CPU replay mismatch: {directory}')
    if any(s[4] != s[5] for s in row['stamps']):
        raise ValueError('CTA changed SM inside the window')
    return len(values)


def collect(path, replay_fn):
    records, checked = [], 0
    for file in sorted((path / 'samples').glob('*/*/result.json')):
        row = json.loads(file.read_text())
        checked += replay_fn(file.parent, row)
        row['record_path'] = str(file.relative_to(path))
        records.append(row)
    if not records:
        raise ValueError('no samples')
    env = json.loads((path / 'environment.json').read_text())
    if any(r['gpu_uuid'] != env['gpu_uuid'] for r in records):
        raise ValueError('mixed GPU UUID')
    if env['mode'] == 'formal':
        protocol = json.loads((path / 'protocol.json').read_text())
        planned = json.loads((path / 'cases.json').read_text())
        actual = collections.Counter(r['case_id'] for r in records if r['trial_role'] == 'formal')
        if dict(actual) != {c['id']: protocol['process_counts'][c['id']] for c in planned}:
            raise ValueError('formal archive incomplete or has extra processes')
    return records, checked


def stats(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return dict(count=len(values), mean=mean, sd=sd, cv=sd / mean if mean else 0.0,
                minimum=min(values), maximum=max(values))


# ---------------------------------------------------------------------------------------
# SASS: locate the timed loop (single backward branch between the two clock reads).
SASS_LINE = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);')
REG = re.compile(r'\bR(\d+)\b')


def sass_functions(text):
    functions = {}
    for block in text.split('Function : ')[1:]:
        name = block.splitlines()[0].strip()
        instructions = []
        for line in block.splitlines():
            m = SASS_LINE.search(line)
            if m:
                instructions.append(dict(addr=int(m[1], 16), pred=(m[2] or '').strip(),
                                         op=m[3], args=m[4].strip()))
        functions[name] = instructions
    return functions


def timed_loop(instructions):
    clocks = [i for i, x in enumerate(instructions) if 'SR_CLOCKLO' in x['args']]
    if len(clocks) != 2:
        raise ValueError('expected two clock reads')
    loops = []
    for i in range(clocks[0], clocks[1]):
        x = instructions[i]
        if x['op'] == 'BRA' and x['args'].startswith('0x') and int(x['args'], 16) < x['addr']:
            start = next(j for j, y in enumerate(instructions) if y['addr'] == int(x['args'], 16))
            loops.append((start, i))
    if len(loops) != 1:
        raise ValueError(f'expected one timed loop, found {len(loops)}')
    window = instructions[clocks[0]:clocks[1] + 1]
    return window, instructions[loops[0][0]:loops[0][1] + 1]


def operands(x):
    """(destination register, source registers) of one SASS instruction."""
    parts = [p.strip() for p in x['args'].split(',')]
    dest = REG.search(parts[0])
    sources = {int(r) for p in parts[1:] for r in REG.findall(p)}
    return (int(dest[1]) if dest else None), sources


def check_chain(body, ops):
    """Each instruction of the chain must read the register written by the previous one."""
    chain = [x for x in body if x['op'].split('.')[0] in ops]
    for previous, current in zip(chain, chain[1:]):
        dest, _ = operands(previous)
        _, sources = operands(current)
        if dest not in sources:
            raise ValueError(f"broken chain at {current['addr']:04x}: "
                             f"{current['op']} {current['args']}")
    return len(chain)


def loop_summary(window, body):
    opcodes = collections.Counter(x['op'] for x in body)
    window_ops = collections.Counter(x['op'].split('.')[0] for x in window)
    if window_ops['LDL'] or window_ops['STL']:
        raise ValueError('local-memory traffic inside window')
    if not window_ops['STS'] or not window_ops['BAR']:
        raise ValueError('final consumer or barrier missing')
    control = sum(n for op, n in opcodes.items() if op.split('.')[0] in ('BRA', 'ISETP', 'UISETP'))
    return dict(loop_instructions=len(body), loop_opcodes=dict(opcodes),
                branch_compare_per_iteration=control)


def sass_check(path, unroll):
    functions = sass_functions((path / 'build/sass.txt').read_text())
    result = []
    for name, instructions in functions.items():
        if not any(k in name for k in ('scalar_chain', 'matrix_chain', 'wgmma_chain')):
            continue
        window, body = timed_loop(instructions)
        info = dict(function=name, **loop_summary(window, body))
        count = lambda *ops: sum(1 for x in body if x['op'].split('.')[0] in ops)
        if 'wgmma_chain' in name:
            if count('HGMMA') != unroll or sum('DEPBAR.LE gsb0, 0x0' in (x['op'] + ' ' + x['args'])
                                               for x in body) != unroll:
                raise ValueError('WGMMA/wait0 count per iteration differs')
            info['chain_links'] = count('HGMMA')
        elif 'matrix_chain' in name:
            if count('LDSM') != unroll:
                raise ValueError('ldmatrix count per iteration differs')
            info['chain_links'] = check_chain(body, ('LDSM', 'IADD3', 'IMAD'))
        else:
            kind, streams, single = re.search(r'scalar_chainILi(\d)ELi(\d)ELb(\d)', name).groups()
            kind, streams = int(kind), int(streams)
            ops = {0: ('FFMA',), 1: ('IADD3', 'IMAD', 'VIADD', 'LEA'), 2: ('LDS',),
                   3: ('LDG',), 4: ('LDG',)}[kind]
            if count(*ops) != unroll * streams:
                raise ValueError(f'{name}: {count(*ops)} chain ops per iteration, '
                                 f'expected {unroll * streams}')
            if streams == 1:
                chain_ops = ops + {2: ('LEA', 'IMAD'), 3: ('IMAD',), 4: ('IMAD',)}.get(kind, ())
                info['chain_links'] = check_chain(body, chain_ops)
        if info['branch_compare_per_iteration'] != 2:
            raise ValueError(f'{name}: loop control is not one compare + one branch')
        info['status'] = 'passed'
        result.append(info)
    if len(result) != 19:  # 15 scalar specializations + ldmatrix + 3 WGMMA shapes
        raise ValueError(f'expected 19 timed kernels, found {len(result)}')
    write_json(path / 'sass_check.json', dict(
        status='passed', unroll=unroll, functions=result,
        scope='timed loop = single backward branch between the clock reads; counts per '
              'iteration; single-chain kernels: each chain instruction reads the previous '
              'chain result'))
    return result


# ---------------------------------------------------------------------------------------
def fit(items):
    """Least-squares T(N) = a + b N over the per-length process means."""
    items = sorted(items, key=lambda r: r['steps'])
    xs = [r['steps'] for r in items]
    ys = [r['mean'] for r in items]
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
    a = ym - b * xm
    residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
    # Pre-declared empirical check, not a confidence interval.
    linear = all(abs(e) <= max(64, 3 * r['sd'], 0.01 * r['mean'])
                 for r, e in zip(items, residuals))
    return dict(steps=xs, mean_cycles=ys, counts=[r['count'] for r in items], intercept_cycles=a,
                cycles_per_step=b, residual_cycles=residuals,
                status='observed_linear' if linear else 'lookup_non_linear')


def plot(path, fits):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    (path / 'plots').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, f in fits.items():
        ax = axes[1] if key.startswith('wgmma') or key == 'ldmatrix' else axes[0]
        ax.plot(f['steps'], [y / x for x, y in zip(f['steps'], f['mean_cycles'])], 'o-',
                label=f"{key}: b={f['cycles_per_step']:.2f}")
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xlabel('dependent steps N per chain')
        ax.set_ylabel('window / N  (clock64 cycles per step)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_yscale('log')
    fig.tight_layout()
    fig.savefig(path / 'plots/dependency_windows.svg')
    fig.savefig(path / 'plots/dependency_windows.png', dpi=150)
    plt.close(fig)
    return True


def table(header, rows):
    lines = ['|' + '|'.join(header) + '|', '|' + '|'.join('---' if i == 0 else '---:'
                                                        for i in range(len(header))) + '|']
    return lines + ['|' + '|'.join(str(c) for c in row) + '|' for row in rows]


def analyze(path):
    env = json.loads((path / 'environment.json').read_text())
    unroll = env['unroll']
    sass = sass_check(path, unroll)
    records, checked = collect(path, replay)
    write_json(path / 'cpu_check.json', dict(records=len(records), checked_elements=checked,
                                             status='passed'))
    if env['mode'] != 'formal':
        print(f'smoke: CPU checked {len(records)} processes / {checked} values; SASS passed')
        return
    groups = collections.defaultdict(list)
    for r in records:
        if r['trial_role'] == 'formal':
            groups[r['case_id']].append(r)
    rows = []
    for key, items in groups.items():
        r = items[0]
        s = stats([x['elapsed'] for x in items])
        work = r['work_flop'] or r['work_op'] or r['requested_bytes']
        unit = 'FLOP/cycle' if r['work_flop'] else 'OP/cycle' if r['work_op'] else \
            'requested_B/cycle'
        rows.append(dict(case=key, op=r['op'], n=r.get('n', 0), warps=r['warps'],
                         streams=r['streams'], steps=r['steps'], unroll=r['unroll'], **s,
                         cycles_per_step=s['mean'] / r['steps'], work=work,
                         work_per_cycle=work / s['mean'], unit=unit,
                         warmup_all_converged=all(x['warmup_converged'] for x in items),
                         registers=r['registers_per_thread'],
                         local_bytes=r['local_bytes_per_thread']))
    rows.sort(key=lambda x: x['case'])
    with (path / 'cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_path = collections.defaultdict(list)
    for row in rows:
        if row['warps'] == 1 or row['op'] == 'wgmma':
            by_path[row['op'] + (f"_n{row['n']}" if row['op'] == 'wgmma' else '')].append(row)
    order = ['ffma', 'add', 'shared', 'global_small', 'global_large', 'ldmatrix',
             'wgmma_n64', 'wgmma_n128', 'wgmma_n256']
    fits = {k: fit(by_path[k]) for k in order if len(by_path[k]) == 3}

    pairs = []
    for op in SCALAR_OPS:
        a, b = groups[f'{op}_w4_s1_2048'], groups[f'{op}_w4_s4_2048']
        ta = {Path(r['record_path']).parent.name: r['elapsed'] for r in a}
        tb = {Path(r['record_path']).parent.name: r['elapsed'] for r in b}
        common = sorted(ta.keys() & tb.keys())
        deltas = [tb[k] - ta[k] for k in common]
        d = stats(deltas)
        s1, s4 = statistics.mean(ta.values()), statistics.mean(tb.values())
        pairs.append(dict(op=op, matched_processes=len(common), s1_mean_cycle=s1,
                          s4_mean_cycle=s4, mean_delta_cycle=d['mean'],
                          standard_error_delta_cycle=d['sd'] / len(common) ** 0.5,
                          work_rate_gain=4 * s1 / s4, raw_deltas_cycle=deltas))
    controls = [dict(case=r['case_id'], control=r['configuration']['control'],
                     elapsed=r['elapsed'], unit=r['unit'])
                for r in records if r['trial_role'] == 'control']
    # Optional U=32 reference (FFMA/add, same source built with -DR01_UNROLL=32): the slope
    # difference to U=64 bounds the per-iteration loop cost left in the U=64 slope.
    reference = []
    ref_dir = path / 'unroll32_reference'
    if ref_dir.is_dir():
        for op in ('ffma', 'add'):
            items = []
            for steps in (128, 512, 2048):
                files = sorted(ref_dir.glob(f'{op}_{steps}_*/stdout.json'))
                runs = [json.loads(f.read_text()) for f in files]
                if any(x['status'] != 'measured' or x['unroll'] != 32 for x in runs):
                    raise ValueError('U=32 reference not measured')
                items.append(dict(steps=steps, **stats([x['elapsed'] for x in runs])))
            b32 = fit(items)['cycles_per_step']
            b64 = fits[op]['cycles_per_step']
            reference.append(dict(op=op, slope_u32=b32, slope_u64=b64,
                                  loop_cycles_per_iteration=64 * (b32 - b64),
                                  slope_without_loop=b64 - (b32 - b64)))

    plotted = plot(path, fits)
    loop_mix = {s['function']: s['loop_opcodes'] for s in sass}
    write_json(path / 'rules.json', dict(
        environment=env, unroll=unroll, fits=fits, cases=rows, independent_stream_pairs=pairs,
        controls=controls, unroll32_reference=reference, cpu_checked_elements=checked,
        plot_generated=plotted,
        loop_opcodes=loop_mix,
        scope='one GPU UUID; one CTA; window = chain + loop control + final consumer/barrier; '
              f'{unroll} dependent steps per loop iteration'))

    # Hand calculation from one real raw sample (stamps are begin/end clock64 of CTA 0).
    raw = sorted(groups['ffma_w1_s1_2048'], key=lambda r: r['record_path'])[0]
    short = sorted(groups['ffma_w1_s1_128'], key=lambda r: r['record_path'])[0]
    begin, end = raw['stamps'][0][2], raw['stamps'][0][3]
    hand = [
        f"`{raw['record_path']}`：stamps 的 clock64 起止为 {begin} → {end}，"
        f"窗口 {end - begin} cycle（= elapsed {raw['elapsed']}）。",
        f"工作量 2 FLOP × 32 lane × 1 链 × 2048 步 = {raw['work_flop']} FLOP，"
        f"{raw['work_flop'] / raw['elapsed']:.3f} FLOP/cycle；"
        f"128 步配置的样本 `{short['record_path']}` 窗口 {short['elapsed']} cycle，"
        f"两点斜率 ({raw['elapsed']} − {short['elapsed']}) / (2048 − 128) = "
        f"{(raw['elapsed'] - short['elapsed']) / 1920:.3f} cycle/步，"
        f"与三长度拟合 {fits['ffma']['cycles_per_step']:.3f} 一致。"]

    fmt = lambda v, d=2: f'{v:.{d}f}'
    lines = [
        '# R01 依赖与结果可用：实测（循环展开版）', '',
        f"GPU {env['gpu_uuid']}，{env['hostname']}，作业 {env['job']}，CUDA 12.9，sm_90a。"
        f"每次循环迭代含 {unroll} 个依赖步，循环计数/比较/回跳每 {unroll} 步一次。"
        f"CPU 逐值重算 {len(records)} 个进程（含短检查与控制）、{checked} 个输出通过；"
        f"SASS 检查见 `sass_check.json`。", '',
        '## 三长度拟合 T(N)=a+bN', '',
        'N 为每条链的依赖步数（128/512/2048），T 为单 CTA clock64 窗口均值。', '']
    lines += table(['路径', 'b cycle/步', 'a cycle', '残差 cycle', '进程数', '状态'], [
        [k, fmt(f['cycles_per_step'], 3), fmt(f['intercept_cycles'], 1),
         ', '.join(fmt(x, 1) for x in f['residual_cycles']), '/'.join(map(str, f['counts'])),
         f['status']] for k, f in fits.items()])
    lines += ['', '## R01-B 独立流（4 warp，2048 步）', '']
    lines += table(['路径', '1流 cycle', '4流 cycle', '4流−1流 cycle', '差值标准误', '共同进程',
                    '工作率提升'], [
        [p['op'], fmt(p['s1_mean_cycle'], 1), fmt(p['s4_mean_cycle'], 1),
         fmt(p['mean_delta_cycle'], 1), fmt(p['standard_error_delta_cycle'], 1),
         p['matched_processes'], f"{p['work_rate_gain']:.3f}×"] for p in pairs])
    lines += ['', '## 全部配置', '']
    lines += table(['配置', '进程', '均值 cycle', 'CV', '窗口/N cycle', '工作率', '预热收敛'], [
        [r['case'], r['count'], fmt(r['mean'], 1), f"{r['cv']:.2%}", fmt(r['cycles_per_step'], 2),
         f"{r['work_per_cycle']:.4g} {r['unit']}", r['warmup_all_converged']] for r in rows])
    if reference:
        lines += ['', '## 循环残余（U=32 对照）', '',
                  '同一源码以 `-DR01_UNROLL=32` 编译，FFMA/add 三长度各 3 进程（`unroll32_reference/`）。'
                  '若每迭代循环成本为 o，则 b(U)=b∞+o/U，故 o=64·(b32−b64)。', '']
        lines += table(['路径', 'b, U=32', 'b, U=64', 'o cycle/迭代', 'b∞ cycle/步'], [
            [x['op'], fmt(x['slope_u32'], 3), fmt(x['slope_u64'], 3),
             fmt(x['loop_cycles_per_iteration'], 1), fmt(x['slope_without_loop'], 3)]
            for x in reference])
    lines += ['', '## 手算核对', ''] + hand
    lines += [
        '', '## 边界', '',
        '- 窗口：CTA barrier 后读 clock64 起，到最终 volatile shared 消费者写入并再过 barrier 止；'
        '含依赖链、循环控制与消费者。整卡与多 CTA 不在本组。',
        '- 每步内容（SASS）：FFMA 1 条；add 为 x_{n+1}=x_n+x_{n-1}，ptxas 生成 IADD3/IMAD.IADD 混合；'
        'shared 为 LEA+LDS；global 为 IMAD.WIDE+LDG.STRONG.GPU（.cg）；ldmatrix 为 IADD3/IMAD.IADD+LDSM；'
        'WGMMA 为 HGMMA+WARPGROUP.DEPBAR（wait0），每迭代 1 次 WARPGROUP.ARRIVE。',
        '- global 小/大工作集为 L2/4、4×L2，每个窗口前线性 .cg 读一遍；不证明命中层级。',
        '- WGMMA FLOP=2×64×N×16×步数，不乘线程数。工作量与独立流请求字节只计数据，不计地址转换。',
        f'- 采样：每进程 8–30 次预热（末 5 次 CV≤2%）后 1 个正式窗口；3 进程起，CV>1% 或独立流差'
        f'≤max(64 cycle, 3σ) 补到 10。控制窗口列在 rules.json，未从结果中扣除。']
    if plotted:
        lines += ['', '![每步窗口](plots/dependency_windows.svg)']
    (path / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'CPU checked {len(records)} processes / {checked} values; report {path / "report.md"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    analyze(parser.parse_args().input.resolve())
