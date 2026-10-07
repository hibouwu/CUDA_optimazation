#!/usr/bin/env python3
"""Validate every R14 tile, pair perturbation, export role timelines and handoff intervals."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

EVENTS = ('work', 'first_mma', 'main_end', 'epi_permit', 'store_return',
          'source_release', 'next_start', 'load_return')
WORDS, TILES, CTA_WORDS = 24, 8, 4 + 3 * 8 * 24
PAIRS = {'main': ('first_mma', 'main_end'),
         'output': ('main_end', 'epi_permit', 'source_release'),
         'supply': ('work', 'first_mma'), 'critical': ('work', 'source_release')}
MASKS = {'main': 0x06, 'output': 0x2c, 'supply': 0x03, 'critical': 0x21}


def cv(values):
    return statistics.pstdev(values) / statistics.mean(values) if len(values) > 1 else None


def decode(setup, raw):
    if setup.get('trace_version') != 6:
        raise ValueError('R14 trace version 6 required')
    if len(raw) != 4 * CTA_WORDS: raise ValueError('trace buffer length mismatch')
    pair = setup.get('trace_pair')
    if pair not in PAIRS or setup.get('trace_mask') != MASKS[pair]:
        raise ValueError('unknown compile-time event pair')
    if setup.get('trace_selected_cta') != (-1 if pair == 'critical' else 0):
        raise ValueError('wrong CTA observation scope')
    expected_clock = 'sm_cycle_and_globaltimer' if pair == 'critical' else 'sm_cycle_only'
    if setup.get('trace_clock') != expected_clock:
        raise ValueError('wrong clock observation scope')
    pingpong = setup['config'] == 'cfg_b'
    if setup.get('source_release_role') != (-1 if pingpong else 2):
        raise ValueError('unexpected TMA issuing role')
    tiles = setup['tiles_per_cta']
    if (setup['m'] != 256 or setup['tile'][0] != 128 or
            setup['n'] != 2 * setup['tile'][1] * tiles or tiles not in (1, 8)):
        raise ValueError('shape does not match the fixed R14 scheduler contract')
    if len(setup['grid']) != 3 or setup['grid'][0]*setup['grid'][1]*setup['grid'][2] != 4:
        raise ValueError('expected four CTAs')
    if pair == 'critical': return decode_critical(setup, raw)
    rows = []
    for cta in range(4):
        base = cta * CTA_WORDS
        if cta != 0:
            if any(raw[base:base + CTA_WORDS]):
                raise ValueError(f'unselected CTA {cta} wrote a trace record')
            continue
        counts = raw[base:base + 3]
        expected = [tiles, (tiles + 1)//2, tiles//2] if pingpong else [tiles]*3
        if counts != expected or raw[base + 3]:
            raise ValueError(f'CTA {cta}: role counts {counts}, expected {expected}, overflow={raw[base+3]}')
        for role, count in enumerate(counts):
            unused = base + 4 + (role * TILES + count) * WORDS
            end = base + 4 + (role + 1) * TILES * WORDS
            if any(raw[unused:end]):
                raise ValueError(f'CTA0 role {role} wrote beyond its tile count')
            for seq in range(count):
                at = base + 4 + (role * TILES + seq) * WORDS
                data = raw[at:at + WORDS]
                row = dict(cta=cta, role=role, seq=seq, trace_pair=pair, m_tile=data[0],
                           n_tile=data[1], l_tile=data[2], sm=data[3])
                for i, event in enumerate(EVENTS):
                    row[event + '_cycle'] = data[4 + i*2] or None
                    row[event + '_ns'] = data[5 + i*2] or None
                required = []
                if pair == 'main' and role: required = ['first_mma', 'main_end']
                if pair == 'output' and role:
                    required = ['main_end', 'epi_permit']
                    if pingpong or role == 2: required += ['source_release']
                if pair == 'supply':
                    required = ['work']
                    if role: required += ['first_mma']
                for event in EVENTS:
                    cycle, ns = row[event + '_cycle'], row[event + '_ns']
                    if ns is not None:
                        raise ValueError('fine-stage ns slot must be zero/unobserved')
                    if event in required:
                        if cycle is None:
                            raise ValueError(f'missing required {event}: {row}')
                    elif cycle is not None or ns is not None:
                        raise ValueError(f'unselected event/role was written: {event}, {row}')
                for clock in ('cycle',):
                    times = [row[e + '_' + clock] for e in required]
                    if times != sorted(times): raise ValueError(f'event order violation: {row}')
                rows.append(row)
    # CUTLASS 3.9.2, M tiles=2, max_swizzle_size=1, grid size=4:
    # CTA0 starts at linear work 0 and advances by 4. For t8 (AlongM),
    # its (M,N) coordinates are (0,2*seq); t1 starts at (0,0) in AlongN.
    expected_order = [(0, 2 * seq) for seq in range(tiles)]
    expected_tiles = set(expected_order)
    producer = [r for r in rows if r['role'] == 0]
    consumer = [r for r in rows if r['role'] != 0]
    def coords(items): return [(r['m_tile'], r['n_tile']) for r in items]
    if any(r['l_tile'] != 0 for r in rows): raise ValueError('unexpected batch coordinate')
    if coords(producer) != expected_order:
        raise ValueError('CTA0 producer order differs from the fixed scheduler assignment')
    for role in (1, 2):
        group = [r for r in consumer if r['role'] == role]
        required_order = expected_order[role-1::2] if pingpong else expected_order
        if coords(group) != required_order:
            raise ValueError(f'CTA0 role {role} does not cover its assigned ordered tiles')
    if set(coords(producer)) != expected_tiles or len(producer) != len(expected_tiles):
        raise ValueError('producer tile coverage mismatch')
    if pingpong:
        if set(coords(consumer)) != expected_tiles or len(consumer) != len(expected_tiles):
            raise ValueError('pingpong consumer tile coverage mismatch')
    else:
        for role in (1, 2):
            group = [r for r in consumer if r['role'] == role]
            if set(coords(group)) != expected_tiles or len(group) != len(expected_tiles):
                raise ValueError('cooperative consumer tile coverage mismatch')
    for cta in (0,):
        group = [r for r in rows if r['cta'] == cta]
        if len({r['sm'] for r in group}) != 1: raise ValueError('CTA migrated SM')
        ownership = defaultdict(set)
        for row in group: ownership[row['role']].add((row['m_tile'], row['n_tile']))
        consumers = ownership[1] | ownership[2]
        if consumers != ownership[0]: raise ValueError('CTA producer/consumer ownership mismatch')
    return rows


def expected_first_tile(setup, cta):
    """CUTLASS 3.9.2 static scheduler, Heuristic raster, max_swizzle_size=1."""
    gx, gy, _ = setup['grid']
    x, y = cta % gx, (cta // gx) % gy
    cm, cn = setup['cluster']
    mt, nt = setup['m']//setup['tile'][0], setup['n']//setup['tile'][1]
    along_n = nt <= mt
    linear = x + y*gx if along_n else x*gy + y
    minor, major = (cm, cn) if along_n else (cn, cm)
    cluster_id, major_offset = divmod(linear//minor, major)
    minor_cluster, major_cluster = divmod(cluster_id, (nt if along_n else mt)//major)
    minor_work = minor_cluster*minor + (x % cm if along_n else y % cn)
    major_work = major_cluster*major + major_offset
    return (minor_work, major_work) if along_n else (major_work, minor_work)


def decode_critical(setup, raw):
    rows = []
    for cta in range(4):
        block = raw[cta*CTA_WORDS:(cta+1)*CTA_WORDS]
        if block[0] != 1 or block[1] != 1 or block[3]:
            raise ValueError('critical requires exactly one start/end marker per CTA')
        data = block[4:4+WORDS]
        if (data[0], data[1]) != expected_first_tile(setup, cta) or data[2] != 0:
            raise ValueError('critical first tile differs from static scheduler')
        if block[2] != data[3]+1: raise ValueError('critical start/end SM mismatch')
        if any(block[4+WORDS:]): raise ValueError('critical wrote a non-endpoint record')
        row = dict(cta=cta, role=0, seq=0, m_tile=data[0], n_tile=data[1],
                   l_tile=data[2], sm=data[3], trace_pair='critical')
        for i,event in enumerate(EVENTS):
            cycle, ns = data[4+2*i:6+2*i]
            if event in PAIRS['critical']:
                if not cycle or not ns: raise ValueError('critical endpoint missing')
            elif cycle or ns: raise ValueError('critical wrote an unselected event')
            row[event+'_cycle'], row[event+'_ns'] = cycle or None, ns or None
        if (row['source_release_cycle'] < row['work_cycle'] or
                row['source_release_ns'] < row['work_ns']):
            raise ValueError('critical endpoints reversed')
        rows.append(row)
    return rows


def intervals(rows):
    """Only endpoints from this process; never join independent observation groups."""
    out = []
    def add(row, name, start, end):
        if start is None or end is None: return
        out.append(dict(cta=row['cta'], role=row['role'], seq=row['seq'],
            m_tile=row['m_tile'], n_tile=row['n_tile'], trace_pair=row['trace_pair'],
            interval=name, start_cycle=start, end_cycle=end, duration_cycles=end-start))
    pair = rows[0]['trace_pair']
    if pair == 'main':
        tiles = defaultdict(list)
        for row in rows:
            if row['role']:
                add(row, 'consumer_main', row['first_mma_cycle'], row['main_end_cycle'])
                tiles[(row['m_tile'],row['n_tile'])].append(row)
        ordered = sorted(tiles.values(), key=lambda group: min(r['first_mma_cycle'] for r in group))
        for group,nxt in zip(ordered,ordered[1:]):
            add(group[0], 'next_tile_first_mma_minus_main_end',
                max(r['main_end_cycle'] for r in group), min(r['first_mma_cycle'] for r in nxt))
    elif pair == 'output':
        for row in rows:
            if row['role']:
                add(row,'epilogue_permission_wait',row['main_end_cycle'],row['epi_permit_cycle'])
                add(row,'epilogue_to_source_release',row['epi_permit_cycle'],row['source_release_cycle'])
    elif pair == 'supply':
        producer = {(r['m_tile'],r['n_tile']):r for r in rows if r['role']==0}
        for row in rows:
            if row['role']:
                add(row,'consumer_work_to_first_mma',row['work_cycle'],row['first_mma_cycle'])
                source=producer[row['m_tile'],row['n_tile']]
                add(row,'producer_work_to_first_mma',source['work_cycle'],row['first_mma_cycle'])
    else:
        for row in rows:
            add(row,'cta_first_work_to_final_source',row['work_cycle'],row['source_release_cycle'])
    return out


def write_csv(path, rows):
    if not rows: return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(output, destination, name):
    destination.mkdir(parents=True, exist_ok=False)
    durations = defaultdict(dict)
    observed_pairs = set()
    build_pair = json.loads((output/'environment.json').read_text())['trace_pair']
    all_rows, stage_rows, failures = [], [], []
    failed_cases = set()
    critical_runs = []
    for path in sorted((output/'samples').glob(f'*/{name}-*/result.json')):
        result = json.loads(path.read_text())
        try:
            if result['status'] != 'measured':
                raise ValueError(result.get('failure_kind', 'process failed'))
            raw_path = path.parent/'stdout.txt.gz'
            if result.get('record_format') == 'r14_compact_v1':
                raw = result['raw']
                if (raw['path'] != raw_path.name or raw['bytes'] != raw_path.stat().st_size or
                        raw['sha256'] != hashlib.sha256(raw_path.read_bytes()).hexdigest()):
                    raise ValueError('compact record raw identity mismatch')
            with gzip.open(raw_path, 'rt') as f:
                events = [json.loads(line) for line in f if line.strip()]
            setup = next(e for e in events if e.get('event') == 'setup')
            if setup.get('trace_pair') != build_pair:
                raise ValueError('sample event pair differs from frozen build')
            observed_pairs.add(setup['trace_pair'])
            call = next(e for e in events if e.get('event') == 'call')
            check = next(e for e in events if e.get('event') == 'check')
            if check['status'] != 'ok': raise ValueError('numeric check failed')
            if check['samples'] != setup['m']*setup['n']:
                raise ValueError('not all output elements checked')
            if len(call['warmup_us']) < 8 or cv(call['warmup_us'][-5:]) > 0.02:
                raise ValueError('warmup not converged')
            case = result['case_id'].split('_', 1)[1]
            kind = 'trace' if setup['traced'] else 'plain'
            key = (case, result['trial'])
            if kind in durations[key]: raise ValueError('duplicate sample/retry; choose one explicitly')
            durations[key][kind] = call['elapsed_us']
            if setup['traced']:
                rows = decode(setup, call['trace'])
                meta = dict(case_id=case, trial=result['trial'], source=str(path.parent))
                all_rows += [dict(**meta, **r) for r in rows]
                stage_rows += [dict(**meta, **r) for r in intervals(rows)]
                if build_pair == 'critical':
                    first = min(r['work_ns'] for r in rows)
                    final = max(r['source_release_ns'] for r in rows)
                    critical_runs.append(dict(**meta, first_work_min_ns=first,
                        final_source_max_ns=final, observed_source_span_ns=final-first,
                        max_local_span_cycles=max(r['source_release_cycle']-r['work_cycle']
                                                  for r in rows),
                        max_local_span_ns=max(r['source_release_ns']-r['work_ns'] for r in rows),
                        latest_source_ctas=json.dumps([r['cta'] for r in rows
                                                     if r['source_release_ns']==final])))
        except (ValueError, KeyError, StopIteration) as exc:
            case = result['case_id'].split('_', 1)[1]
            failed_cases.add(case)
            failures.append(dict(path=str(path), case_id=case, error=str(exc)))
    grouped = defaultdict(list)
    for (case, trial), pair in durations.items():
        if set(pair) != {'plain', 'trace'}:
            failed_cases.add(case)
            failures.append(dict(case=case, trial=trial, error='missing plain/trace pair'))
        else: grouped[case].append(pair)
    summary = []
    for case, pairs in sorted(grouped.items()):
        plain = [p['plain'] for p in pairs]
        trace = [p['trace'] for p in pairs]
        delta = [(p['trace']/p['plain']-1)*100 for p in pairs]
        stable = len(pairs) == 10 and cv(plain) <= 0.05 and cv(trace) <= 0.05
        perturbation_ok = abs(statistics.median(delta)) <= 5
        summary.append(dict(case_id=case, processes=len(pairs),
            plain_us_median=statistics.median(plain), trace_us_median=statistics.median(trace),
            plain_cv=cv(plain), trace_cv=cv(trace),
            paired_perturbation_percent_median=statistics.median(delta),
            stable=stable, perturbation_ok=perturbation_ok,
            trace_usable=stable and perturbation_ok and case not in failed_cases))
    expected = 2 if name == 'pilot' else 18
    passed = len(summary) == expected and not failures and all(r['trace_usable'] for r in summary)
    result = dict(set=name, trace_pair=build_pair,
                  stability_policy='complete kernel: 10 processes, CV<=5%; warmup last5 CV<=2%',
                  status='ok' if passed else 'needs_attention',
                  trace_scope=('all CTA first-work/final-source endpoints' if build_pair=='critical' else
                               'CTA0 all tiles/roles for one observation group; independent runs are not joined'),
                  source_manifest_sha256=hashlib.sha256((output/'source_hashes.json').read_bytes()).hexdigest(),
                  expected_cases=expected, cases=summary, failures=failures,
                  event_semantics='source_release proves SMEM source reusable, not global writeback')
    (destination/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    qualified = {case['case_id'] for case in summary if case['trace_usable']}
    write_csv(destination/'tile_roles.csv', all_rows)
    write_csv(destination/'tile_intervals.csv', [row for row in stage_rows if row['case_id'] in qualified])
    write_csv(destination/'critical_endpoints.csv', [row for row in critical_runs if row['case_id'] in qualified])
    print(json.dumps(dict(status=result['status'], cases=len(summary), failures=len(failures))))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--set', choices=('pilot', 'main'), default='pilot')
    args = parser.parse_args()
    raise SystemExit(0 if analyze(args.output, args.analysis_dir, args.set) else 2)


if __name__ == '__main__': main()
