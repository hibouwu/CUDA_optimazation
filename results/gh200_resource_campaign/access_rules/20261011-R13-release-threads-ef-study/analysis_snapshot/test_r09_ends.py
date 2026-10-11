#!/usr/bin/env python3
"""CPU synthetic-record check of the archived ends parser beyond 64 tiles; no GPU use."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-run', type=Path, required=True)
    args = parser.parse_args()
    # Test the exact shared parser, without writing bytecode into the archive.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.shared_run.resolve() / 'source'))
    import v06_model
    import v08_model
    import r09_run
    import r09_analyze

    rows = r09_run.shared_cases('wide-input')
    assert len(rows) == 9 and {r['input_mode'] for r in rows} == {'dyadic', 'zero', 'random'}
    assert all(r['m'] == r['n'] == 20480 and r['k'] == 1024 for r in rows)
    expected = {'cfg_a': {193: 8, 194: 124}, 'cfg_b': {193: 8, 194: 124}, 'cfg_c': {96: 4, 97: 128}}
    with tempfile.TemporaryDirectory(prefix='r09-ends-cpu-') as temp:
        root = Path(temp)
        for row in [r for r in rows if r['input_mode'] == 'zero']:
            cfg = row['config']
            grid = v06_model.emulate_grid(cfg, row['m'], row['n'])
            work = v08_model.scheduled_work(cfg, row['m'], row['n'], grid, 1)
            counts = list(map(len, work))
            assert dict(Counter(counts)) == expected[cfg]
            setup = dict(row, grid=grid, trace_version='v08-ends', requested_sm_count=0)
            words = [0] * (len(work) * 784)
            for cta, count in enumerate(counts):
                base = cta * 784
                entry_c, entry_ns = 10000 + cta * 1000, 1000000 + cta * 128
                duration_ns = 400000 + count * 1000
                last = 0 if cfg == 'cfg_b' and count % 2 else 1
                role_counts = [(count + 1) // 2, count // 2] if cfg == 'cfg_b' else [count, count]
                words[base:base+4] = [entry_c, entry_ns, cta+1, entry_c+64]
                for role in (0, 1):
                    span = duration_ns - (0 if role == last else 128)
                    words[base+4+3*role:base+7+3*role] = [entry_c+2*span, entry_ns+span, role_counts[role]]
                words[base+11:base+13] = [cta+1, cta+1]
            assert all(x == 0 for cta in range(len(work)) for x in words[cta*784+16:(cta+1)*784])
            folder = root / 'samples' / row['id']
            folder.mkdir(parents=True)

            def save(variant, trial, trace):
                raw = folder / f'{variant}-{trial:02}.txt.gz'
                events = [dict(setup, event='setup'),
                          dict(event='call', trace=trace, elapsed_us=650, warmup_us=[650]*8),
                          dict(event='check', status='ok', nonfinite=0, padding_errors=0,
                               checked_indices=list(range(4096)), checked_values=[0]*4096)]
                with gzip.open(raw, 'wt') as stream:
                    for event in events:
                        stream.write(json.dumps(event) + '\n')
                record = dict(case=row['id'], variant=variant, trial=trial, returncode=0,
                              elapsed_us=650, raw=str(raw.relative_to(root)),
                              raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
                (folder / f'{variant}-{trial:02}.json').write_text(json.dumps(record))
                return record

            for trial in range(10):
                save('plain', trial, [])
                record = save('ends', trial, words)
            observed = v08_model.observe(root, record, row, setup)
            assert [c['tiles'] for c in observed['ctas']] == counts
            summary = r09_analyze.summarize_ends_case(root, row, setup)
            assert summary['T'] == max(counts) and summary['ghz_ends'] == 2
            assert summary['window_ends_gt_600us'] == (cfg != 'cfg_c')
            # Count validation must still run even though the per-tile region is unused.
            bad = words[:]
            bad[6] += 1
            if cfg != 'cfg_b':
                bad[9] += 1
            record = save('ends', 9, bad)
            try:
                v08_model.observe(root, record, row, setup)
            except ValueError as error:
                assert 'tile counts differ from scheduler' in str(error)
            else:
                raise AssertionError('wrong header tile count was accepted')
            save('ends', 9, words)
            (folder / 'stamped-00.json').write_text(json.dumps(dict(variant='stamped')))
            try:
                r09_analyze.summarize_ends_case(root, row, setup)
            except ValueError as error:
                assert 'must not contain stamped' in str(error)
            else:
                raise AssertionError('wide-input accepted stamped')
            print(cfg, grid, dict(Counter(counts)), 'synthetic ends count/summary checks passed')


if __name__ == '__main__':
    main()
