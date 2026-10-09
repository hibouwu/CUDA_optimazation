#!/usr/bin/env python3
"""CPU-only R15 fragment mapping, source lifetime, output/background and pair replay."""
import argparse
import array
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction

MODES = ('reg_smem', 'tma_only', 'full_output', 'background', 'serial', 'concurrent')


def value(row, col):
    return (((row % 16) * 7 + (col % 128) * 11) % 31 - 15) / 32


def coordinate(lane, register, group):
    row = (lane // 32) * 16 + (lane % 32) // 4 + ((register // 2) % 2) * 8
    col = (lane % 4) * 2 + register % 2 + (register // 4) * 8
    return row + group * 64, col


def packed(row, col, n):
    return (row % 32) * 128 + col % 128


def lifetime(chunks, buffers):
    pending = []
    events = []
    for chunk in range(chunks):
        if len(pending) == buffers:
            old = pending.pop(0)
            events.append(('read_done', old))
            events.append(('released', old))
        events.append(('prepare', chunk))
        events.append(('issued', chunk))
        pending.append(chunk)
    for old in pending:
        events.append(('read_done', old))
    events.append(('full_done_all', chunks - 1))
    for old in pending:
        events.append(('released', old))
    return events


def cpu_check():
    proof = []
    for n in (128, 256):
        chunks = 128 * n // 4096
        seen = set()
        fragment_slots = [set() for _ in range(chunks)]
        for group in (0, 1):
            for lane in range(128):
                for register in range(n // 2):
                    row, col = coordinate(lane, register, group)
                    assert 0 <= row < 128 and 0 <= col < n
                    assert (row, col) not in seen
                    seen.add((row, col))
                    chunk = (col // 128) * 4 + row // 32
                    index = packed(row, col, n)
                    assert index not in fragment_slots[chunk]
                    fragment_slots[chunk].add(index)
                    # v3: same mapping with one base and64 immediate offsets.
                    chunk_m, chunk_n = chunk % 4, chunk // 4
                    assert group == chunk_m // 2 and lane // 64 == chunk_m % 2
                    local_register = register % 64
                    assert register // 64 == chunk_n
                    row_base = ((lane // 32) % 2) * 16 + (lane % 32) // 4
                    col_base = (lane % 4) * 2
                    immediate_words = ((local_register // 2) % 2) * 8 * 128
                    immediate_words += (local_register // 4) * 8 + local_register % 2
                    assert row_base * 128 + col_base + immediate_words == index
                    # Immutable TMA-only boxes match every full-output chunk.
                    assert value(row, col) == value(row % 32, col % 128)
        assert len(seen) == 128 * n
        assert all(indices == set(range(4096)) for indices in fragment_slots)
        for buffers in (1, 2):
            events = lifetime(chunks, buffers)
            issued = {chunk: i for i, (event, chunk) in enumerate(events) if event == 'issued'}
            read = {chunk: i for i, (event, chunk) in enumerate(events) if event == 'read_done'}
            release = {chunk: i for i, (event, chunk) in enumerate(events) if event == 'released'}
            prepare = {chunk: i for i, (event, chunk) in enumerate(events) if event == 'prepare'}
            for chunk in range(chunks):
                assert prepare[chunk] < issued[chunk] < read[chunk] <= release[chunk]
                if chunk + buffers < chunks:
                    assert release[chunk] < prepare[chunk + buffers]
            proof.append(dict(n=n, fragments=len(seen), chunks=chunks, buffers=buffers,
                              bytes_per_chunk=16384, boxes_per_chunk=1, tma_box_shape=[128,32],
                              tma_box_bytes=16384,
                              background_mma=16 * chunks, lifetime_events=events))
    assert len(proof) == 4
    return dict(status='CPU mapping/lifetime passed', conditions=proof,
                note='No GPU, tensor-map encoding, compilation or timing qualification claimed')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_raw(folder, name, digest):
    with gzip.open(folder / (name + '.gz'), 'rb') as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('raw identity mismatch ' + str(folder / name))
    return raw


def replay(record, folder):
    n = 128 if record['kib'] == 64 else 256
    ldd = n + 32
    if record['ldd'] != ldd or record['padding_errors']:
        raise ValueError('output stride or padding mismatch')
    raw = read_raw(folder, 'output.f32', record['raw_hashes']['output.f32'])
    output = array.array('f'); output.frombytes(raw)
    words = array.array('I'); words.frombytes(raw)
    if len(output) != 128 * ldd or len(output) != record['output_elements']:
        raise ValueError('full output length')
    stores = record['mode'] in (1, 2, 4, 5)
    for row in range(128):
        for col in range(ldd):
            index = row * ldd + col
            if stores and col < n:
                if not math.isfinite(output[index]) or output[index] != value(row, col):
                    raise ValueError('fragment or TMA output mismatch')
            elif words[index] != 0x7fc12345:
                raise ValueError('inactive logical output or padding modified')
    background = array.array('f')
    background.frombytes(read_raw(folder, 'background.f32', record['raw_hashes']['background.f32']))
    expected = record['repeats'] * record['chunks'] if record['mode'] >= 3 else 0
    if len(background) != 8192 or any(x != expected for x in background):
        raise ValueError('complete BG mismatch')
    smem = array.array('f')
    smem.frombytes(read_raw(folder, 'smem.f32', record['raw_hashes']['smem.f32']))
    if len(smem) != 8192:
        raise ValueError('SMEM witness length')
    for q, x in enumerate(smem):
        local = q % 4096
        if x != value(local // 128, local % 128):
            raise ValueError('packed SMEM witness mismatch')
    trace = array.array('Q')
    trace.frombytes(read_raw(folder, 'trace.u64', record['raw_hashes']['trace.u64']))
    if len(trace) != record['trace_records'] * 6:
        raise ValueError('trace length')
    if not record['trace']:
        if any(trace):
            raise ValueError('plain unexpectedly traced')
    else:
        selected = record.get('trace_selected_record')
        for q in range(record['trace_records']):
            prepare, issue, read, release, full, bg = trace[q * 6:q * 6 + 6]
            if selected is not None and q != selected:
                if any((prepare, issue, read, release, full, bg)):
                    raise ValueError('unselected trace record was written')
                continue
            if selected is not None and selected != record['repeats']*record['chunks']-1:
                raise ValueError('trace selector mismatch')
            pair_only = record.get('trace_profile') == 'read_full_pair'
            if pair_only:
                if record['mode'] != 1 or not (0 < read <= full):
                    raise ValueError('read/full pair order')
                if prepare or issue or release or bg:
                    raise ValueError('unobserved event fabricated in read/full-only trace')
            elif stores and not (0 < prepare <= issue <= read <= release and issue <= full):
                raise ValueError('source read/release/full ordering')
            if record['mode'] >= 3 and bg == 0:
                raise ValueError('selected background completion missing')
            if record['mode'] == 4 and not full <= bg:
                raise ValueError('serial background precedes full output')
    return dict(output_values=len(output), background_values=len(background),
                smem_values=len(smem), max_error=0)


def audit_sass(path):
    """Conservative machine-code evidence; refuses unknown serialization and spills.

    One BG collective16 site batch must appear in a runtime chunk/repeat loop.
    This is a representative admission gate, not a complete instruction simulator.
    Review the archived body and branch bounds before assigning physical service rules.
    """
    text = Path(path).read_text()
    functions = []
    for part in text.split('Function : '):
        match = re.match(r'_Z12output_probeILi(128|256)ELi([12])ELi([0-5])ELb([01])', part)
        if not match:
            continue
        n, buffers, mode, traced = map(int, match.groups())
        ops = []
        for line in part.splitlines():
            found = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*/\*', line)
            if found:
                ops.append((int(found[1], 16), found[2]))
        loops = []
        for pc, op in ops:
            found = re.search(r'\bBRA\s+0x([0-9a-f]+)', op)
            if found and int(found[1], 16) < pc:
                loops.append((int(found[1], 16), pc))
        mma = [(pc, op) for pc, op in ops if 'HGMMA.64x128x16.F32' in op]
        if mode >= 3:
            if len(mma) != 16 or any(not any(a <= pc <= b for a, b in loops) for pc, _ in mma):
                raise ValueError('BG dynamic16 batch not established; inspect CFG')
        elif mma:
            raise ValueError('baseline has unexpected BG work')
        if any(' STL' in op or ' LDL' in op for pc, op in ops):
            raise ValueError('local spilling is not an R15 service parameter')
        functions.append(dict(n=n, buffers=buffers, mode=mode, traced=bool(traced),
             background_sites=len(mma), backwards_edges=[(hex(a), hex(b)) for a, b in loops],
             tma_store_sites=sum('UTMASTG' in op for pc, op in ops),
             tma_command_groups=sum('UTMACMDFLUSH' in op for pc, op in ops),
             depbar_sites=[dict(pc=hex(pc), op=op) for pc, op in ops if 'DEPBAR' in op],
             note='Need independent source-wait/full-wait and guarded clock CFG review'))
    if len(functions) != 48:
        raise ValueError(f'expected 48 compiled template variants, got {len(functions)}')
    return functions


def analyze(folder, destination):
    folder, destination = Path(folder), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in json.loads((folder / 'source_hashes.json').read_text()).items():
        if sha(folder / 'source' / name) != digest:
            raise ValueError('source identity changed')
    if sha(folder / 'build/r15') != json.loads((folder / 'build/binary_hashes.json').read_text())['r15']:
        raise ValueError('binary identity changed')
    machine = audit_sass(folder / 'build/r15.sass')
    log = (folder / 'build/compile.log').read_text()
    serialization = any(code in log for code in ('C7510', 'C7514', 'C7520'))
    rows, groups, pairs = [], {}, {}
    sample_files = sorted((folder / 'samples').glob('*/*/result.json'))
    for file in sample_files:
        r = json.loads(file.read_text())
        proof = replay(r, file.parent)
        if json.loads(file.with_name('process.json').read_text())['returncode']:
            raise ValueError('nonzero process result')
        key = (r['configuration']['id'], r['role'], r['trace'])
        groups.setdefault(key, []).append(r)
        ordinal = file.parent.name.rsplit('-', 1)[0]
        pairs.setdefault((r['configuration']['id'], ordinal), {})[r['trace']] = r
        rows.append(dict(case_id=key[0], role=key[1], trace=r['trace'],
                         elapsed_cycles=r['elapsed_cycles'], **proof))
    summaries = []
    for (name, role, traced), records in groups.items():
        values = [r['elapsed_cycles'] for r in records]
        r = records[0]
        stores = r['mode'] in (1, 2, 4, 5)
        bg = r['mode'] >= 3
        summaries.append(dict(case_id=name, role=role, traced=traced, processes=len(records),
            median=statistics.median(values), minimum=min(values), maximum=max(values),
            cv=statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None,
            all_warmup_converged=all(r['warmup_converged'] for r in records),
            registers_per_thread=r['registers_per_thread'],
            background_flop=262144 * 16 * r['chunks'] * r['repeats'] if bg else 0,
            tma_bytes=r['kib'] * 1024 * r['repeats'] if stores else 0,
            register_smem_bytes=r['kib'] * 1024 * r['repeats'] if r['mode'] in (0, 2, 4, 5) else 0))
    perturbations = []
    for (name, ordinal), variants in pairs.items():
        if set(variants) != {False, True}:
            raise ValueError('plain/trace pair incomplete')
        plain, trace = variants[False]['elapsed_cycles'], variants[True]['elapsed_cycles']
        perturbations.append(dict(case_id=name, ordinal=ordinal,
                                 relative=(trace - plain) / plain))
    max_perturbation = max((abs(p['relative']) for p in perturbations), default=math.inf)
    representative = [s for s in summaries if s['role'] == 'representative' and not s['traced']]
    admit = (len(representative) == 2 and all(s['processes'] >= 3 for s in representative)
             and max_perturbation <= .05 and not serialization
             and all(s['all_warmup_converged'] for s in representative))
    recheck_ok = ((folder/'protocol-trace-recheck.json').exists()
                  and len({p['case_id'] for p in perturbations})==2
                  and all(s['role']=='tracecheck' and s['processes']>=3 for s in summaries)
                  and max_perturbation<=.05 and not serialization
                  and all(s['all_warmup_converged'] for s in summaries))
    qualified = dict(admit_matrix=admit, trace_recheck_passed=recheck_ok,
                     max_pair_trace_perturbation=max_perturbation,
                     compiler_serialization=serialization,
                     reason='SASS/CPU/order gates + trace perturbation<=5%; independent CFG review remains')
    (destination / 'qualification.json').write_text(json.dumps(qualified, indent=2) + '\n')
    summary = dict(status='conditional observation; GPU qualification pending' if serialization
                   else 'CPU/SASS preliminary replay passed', cases=summaries,
                   machine=machine, trace_pairs=perturbations, qualification=qualified)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if summaries:
        with (destination / 'cases.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader(); writer.writerows(summaries)
    text = ['# R15 输出服务', '', '数值、padding、完整背景输出和源生命周期已CPU重算。', '',
            '时间是相同384线程/72KiB动态SMEM预留的单CTA完成窗口；各模式实际寄存器需另核对。', '',
            '源read等待不算写完成；full_done只在非.read的bulk wait0后记录。双缓冲最多两组输出在途。', '',
            'trace扰动超过5%不得进入正式矩阵；先缩小标记范围并保留原失败。', '',
            '| 配置 | role | trace | 进程 | 中位cycle | CV |', '|---|---|---|---:|---:|---:|']
    for s in summaries:
        text.append(f"| {s['case_id']} | {s['role']} | {s['traced']} | {s['processes']} | "
                    f"{s['median']} | {s['cv']} |")
    (destination / 'report.md').write_text('\n'.join(text) + '\n')
    (destination / 'analyzer_identity.json').write_text(json.dumps(dict(
        script_sha256=sha(__file__), python=sys.version), indent=2) + '\n')
    return summary


def overlap_windows(windows):
    """Time-average active output windows, including self; one call at a time."""
    changes = defaultdict(int)
    for start, end in windows:
        changes[start] += 1
        changes[end] -= 1
    times = sorted(changes)
    counts, areas = [], [0.0]
    count = 0
    for i, time in enumerate(times):
        if i:
            areas.append(areas[-1] + count * (time - times[i - 1]))
        count += changes[time]
        counts.append(count)
    def integral(time):
        i = bisect_right(times, time) - 1
        return areas[i] + counts[i] * (time - times[i])
    return [(integral(end) - integral(start)) / (end - start)
            for start, end in windows]


def analyze_v08_output(root, destination):
    """Posthoc output overlap estimates and directly observed critical CTA endpoints."""
    import v08_model
    med = statistics.median
    rows = json.loads((root / 'cases.json').read_text())
    setups = {r['case']: r['setup'] for r in json.loads((root / 'static_setup.json').read_text())}
    cases, processes, tails = [], [], []
    for row in rows:
        if row['config'] not in ('cfg_a', 'cfg_c'):
            continue
        records = [json.loads(p.read_text()) for p in sorted((root / 'samples' / row['id']).glob('*.json'))]
        records = [r for r in records if not r['returncode']]
        plain = [r['elapsed_us'] for r in records if r['variant'] == 'plain']
        stamped = [r['elapsed_us'] for r in records if r['variant'] == 'stamped']
        perturbation = med(stamped) / med(plain) - 1
        qualified = (min(len(plain), len(stamped)) >= 10 and abs(perturbation) <= .05
                     and max(statistics.pstdev(v) / statistics.mean(v) for v in (plain, stamped)) <= .05)
        selected = []
        for record in records:
            if record['variant'] not in ('stamped', 'ends'):
                continue
            observed = v08_model.observe(root, record, row, setups[row['id']])
            ctas = [c for c in observed['ctas'] if c['tiles']]
            origin = min(c['entry_ns'] for c in observed['ctas'])
            last_ns = max(c['end_ns'] for c in ctas)
            longest_cycles = max(c['end_c'] - c['entry_c'] for c in ctas)
            longest_ties = [c for c in ctas if c['end_c'] - c['entry_c'] == longest_cycles]
            longest = max(longest_ties, key=lambda c: c['end_ns'])
            last = max(ctas, key=lambda c: c['end_ns'])
            # globaltimer ties are preserved; choosing a different tied CTA is not a miss.
            tails.append(dict(case=row['id'], trial=record['trial'], variant=record['variant'],
                ctas=len(ctas), unique_sms=len({c['sm'] for c in ctas}),
                entry_spread_ns=max(c['entry_ns'] for c in observed['ctas']) - origin,
                final_tied_ctas=[c['cta'] for c in ctas if c['end_ns'] == last_ns],
                longest_tied_ctas=[c['cta'] for c in longest_ties],
                final_cta=last['cta'], final_sm=last['sm'], longest_cta=longest['cta'],
                final_entry_offset_ns=last['entry_ns'] - origin,
                longest_exit_lag_ns=last_ns - longest['end_ns'],
                envelope_ns=last_ns - origin,
                longest_cycles=longest['end_c'] - longest['entry_c'],
                final_cycles=last['end_c'] - last['entry_c'],
                cta_endpoints=[dict(cta=c['cta'], sm=c['sm'], entry_ns=c['entry_ns'] - origin,
                    end_ns=c['end_ns'] - origin, duration_cycles=c['end_c'] - c['entry_c']) for c in ctas]))
            if record['variant'] == 'ends':
                continue
            with gzip.open(root / record['raw'], 'rt') as stream:
                words = next(e['trace'] for line in stream for e in [json.loads(line)] if e['event'] == 'call')
            tm = 256 if row['config'] == 'cfg_c' else 128
            events = []
            rates = [(c['end_c'] - c['entry_c']) / (c['end_ns'] - c['entry_ns']) for c in ctas]
            common_rate = med(rates)
            for c, rate in zip(ctas, rates):
                for j, (tile, (mi, ni)) in enumerate(zip(c['tiles'], c['work'])):
                    offset = c['cta'] * (16 + 2 * 64 * 6) + 16 + (64 + j) * 6
                    issuer = words[offset:offset + 4]  # cooperative thread256, role2
                    valid_bytes = max(0, min(tm, row['m'] - mi * tm)) * max(0, min(128, row['n'] - ni * 128)) * 4
                    if not valid_bytes:
                        continue  # OOB windows issue no valid logical output bytes.
                    phase = 'single' if len(c['tiles']) == 1 else ('first' if j == 0 else 'last' if j == len(c['tiles']) - 1 else 'middle')
                    events.append(dict(phase=phase, cycles=tile[3] - tile[2], issuer_cycles=issuer[3] - issuer[2],
                        start=c['entry_ns'] - origin + (issuer[2] - c['entry_c']) / rate,
                        end=c['entry_ns'] - origin + (issuer[3] - c['entry_c']) / rate,
                        merged_start=c['entry_ns'] - origin + (tile[2] - c['entry_c']) / rate,
                        merged_end=c['entry_ns'] - origin + (tile[3] - c['entry_c']) / rate,
                        entry_start=c['entry_ns'] - origin + (issuer[2] - c['entry_c']) / common_rate,
                        entry_end=c['entry_ns'] - origin + (issuer[3] - c['entry_c']) / common_rate,
                        final_start=c['end_ns'] - origin - (c['end_c'] - issuer[2]) / common_rate,
                        final_end=c['end_ns'] - origin - (c['end_c'] - issuer[3]) / common_rate))
            # Each alternative uses this stamped call's own endpoints and rates, never an ends call.
            overlaps = {}
            for name, start, end in [('affine', 'start', 'end'), ('entry_anchor', 'entry_start', 'entry_end'), ('final_anchor', 'final_start', 'final_end'), ('merged_affine', 'merged_start', 'merged_end')]:
                overlaps[name] = overlap_windows([(e[start], e[end]) for e in events])
            for phase in ('single', 'first', 'middle', 'last'):
                indices = [i for i, e in enumerate(events) if e['phase'] == phase]
                if not indices:
                    continue
                item = dict(case=row['id'], trial=record['trial'], phase=phase, windows=len(indices),
                    E_cycles=med(events[i]['cycles'] for i in indices),
                    issuer_E_cycles=med(events[i]['issuer_cycles'] for i in indices),
                    **{name: med(values[i] for i in indices) for name, values in overlaps.items()})
                selected.append(item)
                processes.append(item)
        feat = v08_model.features(row, setups[row['id']]['grid'])
        phases = {}
        for phase in sorted({r['phase'] for r in selected}):
            values = [r for r in selected if r['phase'] == phase]
            phases[phase] = {k: med(r[k] for r in values) for k in ('windows', 'E_cycles', 'issuer_E_cycles', 'affine', 'entry_anchor', 'final_anchor', 'merged_affine')}
            phases[phase]['E_process_minmax'] = [min(r['E_cycles'] for r in values), max(r['E_cycles'] for r in values)]
        cases.append(dict(**row, T=feat['T'], static_max_tile_ctas=round(feat['q'] * 132),
                          perturbation=perturbation, trace_qualified=qualified, phases=phases))
    # A conditional diagnostic of max(floor, bytes * concurrency / service), not a frozen prediction.
    candidates = [r for r in cases if r['config'] == 'cfg_c' and r['T'] == 1
                  and r['swizzle'] == 1 and r['m'] % 256 == 0 and r['n'] % 128 == 0]
    train = [r for r in candidates if r['set'] == 'calib' and r['trace_qualified']]
    fits = {}
    for response, key in ((response, key) for response in ('E_cycles', 'issuer_E_cycles')
                          for key in ('static_max_tile_ctas', 'affine', 'entry_anchor', 'final_anchor')):
        floor = med(r['phases']['single'][response] for r in candidates if r['id'] in ('cfg_c_g1', 'cfg_c_g3'))
        def x(r):
            return r[key] if key == 'static_max_tile_ctas' else r['phases']['single'][key]
        def loss(slope):
            return sum((max(floor, slope * x(r)) - r['phases']['single'][response]) ** 2 for r in train)
        lo, hi = 0.0, max(r['phases']['single'][response] / x(r) for r in train) * 2
        for _ in range(80):
            a, b = (2 * lo + hi) / 3, (lo + 2 * hi) / 3
            if loss(a) < loss(b): hi = b
            else: lo = a
        slope = (lo + hi) / 2
        fits[response + '/' + key] = dict(floor_cycles=floor, cycles_per_concurrent_cta=slope,
            effective_logical_bytes_per_cycle=131072 / slope, fitted_cases=[r['id'] for r in train],
            comparisons=[dict(case=r['id'], set=r['set'], trace_qualified=r['trace_qualified'],
                observed=r['phases']['single'][response], estimated=max(floor, slope * x(r)),
                relative=max(floor, slope * x(r)) / r['phases']['single'][response] - 1) for r in candidates])
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in [('output.json', dict(cases=cases, processes=processes, candidates=fits,
                scope='Posthoc. Output globaltimer absent. Affine clock mapping and common-rate entry/final anchors are assumptions, not direct concurrency measurements; active windows are not physical write occupancy. Floor is the observed g1/g3 plateau, not a measured isolated CTA.',
                script_sha256=sha(__file__))), ('critical.json', tails)]:
        (destination / name).write_text(json.dumps(value, indent=2) + '\n')
    print(f'{len(cases)} cases, {len(tails)} stamped/ends calls -> {destination}')


def analyze_v08_single_model(root, destination):
    """Two static-q candidates on qualified cfg_c single-tile calibration cases only."""
    import v08_model as model
    import v08_fit as fit
    med = statistics.median
    summary = json.loads((root / 'derived/summary.json').read_text())
    frozen = json.loads((root / 'frozen/v08-predictions.json').read_text())
    if sha(root / 'derived/summary.json') != frozen['summary_sha256']:
        raise ValueError('calibration summary differs from the frozen source')
    rows = {r['id']: r for r in json.loads((root / 'cases.json').read_text())}
    setups = {r['case']: r['setup'] for r in json.loads((root / 'static_setup.json').read_text())}
    points, excluded, failed = [], [], []
    for case in summary.values():
        if case['config'] != 'cfg_c' or case['set'] != 'calib' or case['T'] != 1:
            continue
        if abs(case['perturbation']) > .05 or max(case['plain_cv'], case['stamped_cv']) > .05:
            excluded.append(dict(case=case['id'], perturbation=case['perturbation'],
                                 reason='existing interval qualification'))
            continue
        row = rows[case['id']]
        processes = []
        for path in sorted((root / 'samples' / row['id']).glob('stamped-*.json')):
            record = json.loads(path.read_text())
            if record['returncode']:
                failed.append(str(path.relative_to(root)))
                continue
            observed = model.observe(root, record, row, setups[row['id']])
            ctas = observed['ctas']
            if any(len(c['tiles']) != 1 for c in ctas):
                raise ValueError('single-tile calibration case changed')
            cycles = [c['tiles'][0][3] - c['tiles'][0][2] for c in ctas]
            rates = [(c['end_c'] - c['entry_c']) / (c['end_ns'] - c['entry_ns']) for c in ctas]
            # Same CTA/call endpoint ratios; this is NOT a directly measured output ns window.
            normalized = [e / f for e, f in zip(cycles, rates)]
            processes.append(dict(trial=record['trial'], cycle=med(cycles),
                normalized_ns=med(normalized), measured_cta_cycles_per_ns=med(rates)))
        cycle = med(p['cycle'] for p in processes)
        if cycle != case['intervals']['E0']:
            raise ValueError('merged single-tile E0 replay differs from original summary')
        points.append(dict(id=row['id'], m=row['m'], n=row['n'], k=row['k'], q=case['q'],
            geometry=f"{row['m']}x{row['n']}", cycle=cycle,
            normalized_ns=med(p['normalized_ns'] for p in processes), processes=processes))

    def parameters(train, target, form):
        if form == 'constant':
            return med(p[target] for p in train), 0.0
        return fit.lsq([p['q'] for p in train], [p[target] for p in train])

    def errors(records):
        values = [r['relative'] for r in records]
        return dict(n=len(values), median_absolute=med(abs(e) for e in values),
                    maximum_absolute=max(abs(e) for e in values), rms=math.sqrt(statistics.mean(e * e for e in values)))

    fits, validation = [], []
    for target in ('cycle', 'normalized_ns'):
        for form in ('constant', 'q_linear'):
            intercept, slope = parameters(points, target, form)
            fits.append(dict(target=target, form=form, intercept=intercept, slope=slope,
                comparisons=[dict(case=p['id'], observed=p[target], fitted=intercept + slope * p['q'],
                                  relative=(intercept + slope * p['q']) / p[target] - 1) for p in points]))
            for group_by in ('k', 'geometry'):
                predictions = []
                for group in sorted({p[group_by] for p in points}):
                    train = [p for p in points if p[group_by] != group]
                    intercept, slope = parameters(train, target, form)
                    for p in points:
                        if p[group_by] != group:
                            continue
                        value = intercept + slope * p['q']
                        predictions.append(dict(case=p['id'], withheld_group=group,
                            train_cases=[t['id'] for t in train], intercept=intercept, slope=slope,
                            train_q_range=[min(t['q'] for t in train), max(t['q'] for t in train)],
                            train_k_range=[min(t['k'] for t in train), max(t['k'] for t in train)],
                            observed=p[target], predicted=value, relative=value / p[target] - 1))
                validation.append(dict(target=target, form=form, group_by=group_by,
                                       metrics=errors(predictions), predictions=predictions))

    # Re-evaluate the existing q candidate without editing the model/fit or any frozen file.
    selection = frozen['calibration']['selection']['cfg_c']
    fit.ROWS = rows
    q_params = fit.fit_params([summary[name] for name in selection['cases']], 'cfg_c',
                             dict(selection['choice'], E='q'))
    wiring = []
    for q in (48 / 132, 70 / 132, 1.0):
        p = model.case_params({'cfg_c': q_params}, 'cfg_c', dict(q=q, fp=30.0))
        wiring.append(dict(q=q, E0=p['E0'], Elast=p['Elast'],
            single_last_epilogue=model.cta_cycles(p, 'cooperative', 1, 64, True)[1]['last_epilogue'],
            multi_last_epilogue=model.cta_cycles(p, 'cooperative', 2, 64, True)[1]['last_epilogue']))
    old_e0 = frozen['calibration']['params']['cfg_c']['E0']
    result = dict(input=str(root), gpu=frozen['gpu'], points=points, excluded=excluded,
        failed_processes=failed, full_data_fits=fits, grouped_validation=validation,
        frozen_E0_reference=dict(cycle=old_e0, scope='Already used these calibration cases; not heldout scoring',
            comparisons=[dict(case=p['id'], relative=old_e0 / p['cycle'] - 1) for p in points]),
        q_wiring_check=wiring, analyzer_sha256=sha(__file__),
        model_sha256=sha(Path(model.__file__)), fit_sha256=sha(Path(fit.__file__)),
        summary_sha256=sha(root / 'derived/summary.json'),
        scope='Posthoc grouped calibration diagnostics, equal weight per case. Merged max(done)-max(permit). '
              'normalized_ns uses measured same-CTA full-window cycle/ns, including withheld targets; '
              'not direct output timestamps or an autonomous timing prediction. No observed overlap input, '
              'V08 heldout samples or job738100 samples enter these fits.')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'single-tile-model.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(points)} qualified single-tile cases, grouped by K and geometry -> {destination}')


def role_windows(root, row, variant):
    """Two cooperative consumer leaders; store entry/return is not TMA issue/complete."""
    from analyze_r18 import replay
    processes = []
    for path in sorted((root / 'samples' / row['id']).glob(variant + '-*.json')):
        record = json.loads(path.read_text())
        if record['returncode']:
            continue
        observed = replay(root, record, row)
        with gzip.open(root / record['raw'], 'rt') as stream:
            events = {e['event']: e for line in stream if line.strip() for e in [json.loads(line)]}
        setup, words = events['setup'], events['call']['trace']
        if setup.get('trace_tile_words', 6) != 6:
            raise ValueError('this role comparison uses the archived six-word profiles')
        grouped = defaultdict(lambda: defaultdict(list))
        finals = defaultdict(list)
        for base in range(0, len(words), 784):
            h = words[base:base + 16]
            if h[6] != h[9] or not h[6]:
                raise ValueError('expected cooperative consumers with the same nonempty work')
            finals['other_minus_issuer_final_cycles'].append(h[4] - h[7])
            finals['other_minus_issuer_final_ns'].append(h[5] - h[8])
            for j in range(h[6]):
                a, b = words[base + 16 + j * 6:base + 22 + j * 6], words[base + 400 + j * 6:base + 406 + j * 6]
                m, p, d = max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
                metrics = dict(w=p-m, other_permit_wait=a[2]-a[1], issuer_permit_wait=b[2]-b[1],
                    issuer_start_vs_last_main=b[2]-m, main_end_skew=a[1]-b[1],
                    arrival_gap=p-b[2], issuer=b[3]-b[2], post_permit=b[3]-p,
                    done_join=d-b[3], other_done_minus_issuer=a[3]-b[3], E=d-p, after_main=d-m)
                assert metrics['issuer'] == metrics['arrival_gap'] + metrics['post_permit']
                assert metrics['E'] == metrics['post_permit'] + metrics['done_join']
                assert metrics['after_main'] == metrics['w'] + metrics['E']
                phase = 'first' if j == 0 else 'last' if j == h[6]-1 else 'middle'
                for key, value in metrics.items():
                    grouped[phase][key].append(value)
                if variant == 'global':
                    if setup['trace_version'] != 'r15-first-output-ns' or h[6] != 1:
                        raise ValueError('single-tile direct issuer profile required')
                    grouped[phase]['issuer_ns'].append(h[14]-h[13])
                    grouped[phase]['issuer_cycles_per_ns'].append(metrics['issuer']/(h[14]-h[13]))
        processes.append(dict(trial=record['trial'], raw=record['raw'],
            work_sha256=hashlib.sha256(json.dumps([c['work'] for c in observed['ctas']]).encode()).hexdigest(),
            phases={phase:dict(count=len(values['E']),
                median={k:statistics.median(v) for k,v in values.items()},
                ranges={k:[min(v),max(v)] for k,v in values.items()}) for phase,values in grouped.items()},
            final_median={k:statistics.median(v) for k,v in finals.items()},
            final_ranges={k:[min(v),max(v)] for k,v in finals.items()},
            other_final_later=sum(v>0 for v in finals['other_minus_issuer_final_ns']), ctas=len(finals['other_minus_issuer_final_ns'])))
    return processes


def fit_first_bytes(points):
    """All minimizers of J=max(a,b*x), a,b>=0, with exact active-region enumeration."""
    xy = [(Fraction(str(p['x'])), Fraction(str(p['post_permit']))) for p in points]
    levels = sorted({x for x,y in xy})
    sets = []
    a = sum(y for x,y in xy)/len(xy)
    sets.append([(a, Fraction(0)), (a, a/max(levels))])
    b = sum(x*y for x,y in xy)/sum(x*x for x,y in xy)
    sets.append([(Fraction(0), b), (b*min(levels), b)])
    for low, high in zip(levels, levels[1:]):
        floor = [(x,y) for x,y in xy if x<=low]
        rate = [(x,y) for x,y in xy if x>=high]
        a = sum(y for x,y in floor)/len(floor)
        b = sum(x*y for x,y in rate)/sum(x*x for x,y in rate)
        if b*low <= a <= b*high:
            sets.append([(a,b)])
        for boundary in (low, high):
            gx = [(boundary,y) for x,y in floor] + rate
            b = sum(g*y for g,y in gx)/sum(g*g for g,y in gx)
            sets.append([(b*boundary,b)])
    def loss(q):
        return sum((max(q[0],q[1]*x)-y)**2 for x,y in xy)
    best = min(loss(group[0]) for group in sets)
    vertices = sorted({q for group in sets if loss(group[0])==best for q in group})
    return dict(loss=float(best), unique=len(vertices)==1,
                vertices=[dict(floor_cycles=float(a), ns_per_nominal_cta=float(b),
                               exact_floor=str(a), exact_rate=str(b)) for a,b in vertices])


def analyze_role_output(root, direct_root, destination, assumed_f):
    """One first-cohort byte candidate, with no target overlap/frequency features."""
    import v08_model as model
    med = statistics.median
    rows = json.loads((root/'cases.json').read_text())
    summary = json.loads((root/'derived/summary.json').read_text())
    setups = {s['case']:s['setup'] for s in json.loads((root/'static_setup.json').read_text())}
    cases, points = [], []
    for row in rows:
        if row['config']!='cfg_c' or row['set']!='calib':
            continue
        proc = role_windows(root,row,'stamped')
        phases = {phase:{k:med(p['phases'][phase]['median'][k] for p in proc if phase in p['phases'])
                         for k in next(p['phases'][phase]['median'] for p in proc if phase in p['phases'])}
                  for phase in sorted({phase for p in proc for phase in p['phases']})}
        work = model.scheduled_work('cfg_c',row['m'],row['n'],setups[row['id']]['grid'],row['swizzle'])
        work_hash = hashlib.sha256(json.dumps(work).encode()).hexdigest()
        if any(p['work_sha256'] != work_hash for p in proc) or phases['first']['E'] != summary[row['id']]['intervals']['E0']:
            raise ValueError('role replay differs from static work or the original merged E0')
        def full(coord):
            mi,ni=coord
            return mi*256+256<=row['m'] and ni*128+128<=row['n']
        unpadded = all(full(w) for c in work for w in c)
        first_bytes = sum(max(0,min(256,row['m']-c[0][0]*256))*max(0,min(128,row['n']-c[0][1]*128))*4 for c in work if c)
        old = summary[row['id']]
        qualified = abs(old['perturbation'])<=.05 and max(old['plain_cv'],old['stamped_cv'])<=.05
        included = qualified and unpadded
        cases.append(dict(case=row['id'],phases=phases,processes=proc,trace_qualified=qualified,
            unpadded=unpadded,included=included,first_valid_bytes=first_bytes,
            q_max_tiles=old['q'],T=old['T'],assumed_output_cycles_per_ns=assumed_f))
        if included:
            points.append(dict(id=row['id'],geometry=f"{row['m']}x{row['n']}",k=row['k'],
                k_band='short_le4096' if row['k']<=4096 else 'long_ge8192',
                nominal_first_ctas=first_bytes/131072,x=first_bytes/131072*assumed_f,
                **phases['first']))
    fitted = fit_first_bytes(points)
    validation = []
    for grouping in ('geometry','k','k_band'):
        predictions = {'constant':[], 'first_bytes_max':[]}
        for group in sorted({p[grouping] for p in points}):
            train = [p for p in points if p[grouping]!=group]
            fit = fit_first_bytes(train)
            w, join, constant = (med(p[k] for p in train) for k in ('w','done_join','E'))
            for p in points:
                if p[grouping]!=group:
                    continue
                estimates = [max(Fraction(q['exact_floor']),Fraction(q['exact_rate'])*Fraction(str(p['x'])))+Fraction(str(join)) for q in fit['vertices']]
                for name, interval in [('constant',[constant,constant]),('first_bytes_max',[float(min(estimates)),float(max(estimates))])]:
                    predictions[name].append(dict(case=p['id'],withheld_group=group,train_cases=[t['id'] for t in train],
                        interval_cycles=interval,prediction_unique=True if name=='constant' else min(estimates)==max(estimates),
                        observed_E=p['E'],observed_after_main=p['after_main'],
                        worst_E_relative=max(abs(v/p['E']-1) for v in interval),
                        worst_after_main_relative=max(abs((v+w)/p['after_main']-1) for v in interval),
                        parameter_unique=fit['unique'] if name=='first_bytes_max' else True))
        for name, pred in predictions.items():
            validation.append(dict(grouping=grouping,model=name,predictions=pred,
                unique_predictions=sum(p['prediction_unique'] for p in pred),n=len(pred),
                median_E=med(p['worst_E_relative'] for p in pred),max_E=max(p['worst_E_relative'] for p in pred),
                median_after_main=med(p['worst_after_main_relative'] for p in pred),
                max_after_main=max(p['worst_after_main_relative'] for p in pred)))
    direct = []
    for row in json.loads((direct_root/'cases.json').read_text()):
        for variant in ('stamped','global'):
            proc = role_windows(direct_root,row,variant)
            keys=proc[0]['phases']['first']['median']
            direct.append(dict(case=row['id'],variant=variant,processes=proc,
                median={k:med(p['phases']['first']['median'][k] for p in proc) for k in keys},
                final_median={k:med(p['final_median'][k] for p in proc) for k in proc[0]['final_median']}))
    result = dict(v08_gpu=json.loads((root/'environment.json').read_text())['gpu'],
        direct_gpu=json.loads((direct_root/'environment.json').read_text())['gpu'],cases=cases,points=points,
        candidate=dict(form='E0 = max(J_floor, assumed_f * tau * first_valid_bytes/131072) + done_join',
            assumed_f=assumed_f,fit=fitted,join_cycles=med(p['done_join'] for p in points),
            w_cycles=med(p['w'] for p in points)),
        constant_cycles=med(p['E'] for p in points),validation=validation,direct_constraints=direct,
        scope='Posthoc cfg_c first-output development only. Nominal first-cohort bytes are static work, not observed concurrency. '
              'One fixed assumed frequency for every target, not a frequency model. No target measured overlap/frequency enters prediction. '
              'Direct job738100 is a separate-card endpoint constraint, never fit data. Nonunique predictions scored by worst endpoint.',
        analyzer_sha256=sha(__file__))
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'role-output.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(cases)} calibration role cases; {len(points)} supported first-output cases -> {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-check', action='store_true')
    parser.add_argument('--v08-output', action='store_true', help='V08 cfg_a/c output overlap and critical-CTA posthoc analysis')
    parser.add_argument('--v08-single-model', action='store_true', help='Grouped calibration check of constant/static-q merged output')
    parser.add_argument('--role-output', action='store_true', help='Cooperative role decomposition and one static first-cohort candidate')
    parser.add_argument('--direct-run', type=Path)
    parser.add_argument('--assumed-output-ghz', type=float, default=1.8)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(), indent=2))
    else:
        if args.input is None or args.output is None:
            parser.error('--input and a NEW --output directory required')
        if args.role_output:
            if args.direct_run is None or args.assumed_output_ghz<=0:
                parser.error('--role-output requires --direct-run and positive --assumed-output-ghz')
            analyze_role_output(args.input,args.direct_run,args.output,args.assumed_output_ghz)
        elif args.v08_single_model:
            analyze_v08_single_model(args.input, args.output)
        elif args.v08_output:
            analyze_v08_output(args.input, args.output)
        else:
            analyze(args.input, args.output)


if __name__ == '__main__':
    main()
