#!/usr/bin/env python3
"""R15's three cfg_c single-tile controls, using the shared R18/V08 harness.

prepare --output NEW_RUN --cutlass-root CUTLASS_3_9_2 is CPU-only.
build/setup/sample reuse run_v08; sample --set ctrl runs plain/stamped/ends/global.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import run_r18
import run_v08
import v06_run as common

ROOT = Path(__file__).resolve().parent
VARIANTS = ['plain', 'stamped', 'ends', 'global']


def prepare(root, cutlass):
    # Reuse source copying and overlay generation; no private probe or overlay.
    run_r18.prepare(root, cutlass, 'r18')
    source = root / 'source'
    for name in ('run_r15_output_ns.py', 'run_v08.py', 'v08_model.py',
                 'analyze_r15.py', 'analyze_r15_output_ns.py'):
        shutil.copy2(ROOT / name, source / name)
    (source / 'configs').mkdir()
    shutil.copy2(ROOT / 'configs/r15-output-ns.json', source / 'configs/r15-output-ns.json')
    rows = json.loads((source / 'configs/r15-output-ns.json').read_text())
    commands = json.loads((root / 'build/commands.json').read_text())
    selected = {name: commands[name] for name in ('cfg_c_plain', 'cfg_c_stamped')}
    for variant, macro in (('ends', '-DV08_ENDS'), ('global', '-DR15_OUTPUT_NS')):
        command = commands['cfg_c_stamped'][:]
        command.insert(1, macro)
        command[-1] = f'build/cfg_c_{variant}'
        selected[f'cfg_c_{variant}'] = command
    common.write_json(root / 'build/commands.json', selected)
    common.write_json(root / 'cases.json', rows)
    common.write_json(root / 'run_config.json', dict(family='r15-output-ns',
        cases_sha256=common.sha(root / 'cases.json'), variants=VARIANTS))
    common.write_json(root / 'source_hashes.json',
        {str(p.relative_to(root)): common.sha(p) for p in source.rglob('*') if p.is_file()})
    print('prepared 3 single-tile controls and 4 same-source cfg_c build commands; no GPU work')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('step', choices=['prepare'])
        parser.add_argument('--output', required=True, type=Path)
        parser.add_argument('--cutlass-root', required=True, type=Path)
        args = parser.parse_args()
        prepare(args.output.resolve(), args.cutlass_root.resolve())
    else:
        # The only dispatch extension: global is a full-coordinate trace, not an ends trace.
        shared_run_one = run_v08.run_one

        def run_one(root, row, variant, trial, attempt=0):
            entry = run_r18.run_one if variant == 'global' else shared_run_one
            return entry(root, row, variant, trial, attempt)

        run_v08.VARIANTS = VARIANTS
        run_v08.run_one = run_one
        run_v08.main()
