"""Apply the 2026-10-28 regrouping of access_rules rounds (TOTAL-MAPPING.tsv); round contents stay unchanged.

    python -B microbench/gh200_resource_campaign/access_rules/reorganize_20261028.py [--execute]

Each moved round keeps its directory name and depth (question/family/round). Relative symlinks that left
their family are rewritten to the new location; emptied family directories have their README archived
under non-experiment/records/reorganization-20261028 and are removed. MOVED-PATHS-20261020.tsv gets the new
current paths and MOVED-PATHS-20261028.tsv records 10-27 path -> current path for flat_layout_view.py.
"""
import argparse
import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ACCESS = REPO / 'results/gh200_resource_campaign/access_rules'
MAPPING = REPO / 'Docs/ModelEvaluation/gemm/planning/reorganization-20261027/TOTAL-MAPPING.tsv'
RECORD = ACCESS / 'non-experiment/records/reorganization-20261028'


def plan():
    moves = {r['old']: r['new'] for r in csv.DictReader(open(MAPPING), delimiter='\t') if r['moves'] == 'True'}
    for old, new in moves.items():
        if not (ACCESS / old).is_dir() or (ACCESS / new).exists():
            raise SystemExit(f'cannot move {old} -> {new}')
    relocate = lambda path: next((moves[o] + path[len(o):] for o in moves if path == o or path.startswith(o + '/')), path)
    links = []
    for old, new in moves.items():
        for link in (ACCESS / old).rglob('*'):
            raw = os.readlink(link) if link.is_symlink() else None
            if raw is None or os.path.isabs(raw):
                continue
            target = os.path.relpath(os.path.normpath(link.parent / raw), ACCESS)
            if target == old or target.startswith(old + '/'):
                continue
            new_link = ACCESS / new / link.relative_to(ACCESS / old)
            new_raw = os.path.relpath(ACCESS / relocate(target), new_link.parent)
            if new_raw != raw:
                links.append(dict(link=str(new_link.relative_to(ACCESS)), old_target=raw, new_target=new_raw))
    families = {'/'.join(o.split('/')[:2]) for o in moves}
    remaining = lambda fam: [p for p in (ACCESS / fam).iterdir() if p.is_dir() and f'{fam}/{p.name}' not in moves]
    emptied = sorted(f for f in families if not remaining(f))
    return moves, links, emptied


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    moves, links, emptied = plan()
    print(f'{len(moves)} moves, {len(links)} symlinks to rewrite, emptied families: {emptied}')
    if not args.execute:
        return
    RECORD.mkdir(parents=True, exist_ok=False)
    archived = []
    for old, new in sorted(moves.items()):
        (ACCESS / new).parent.mkdir(parents=True, exist_ok=True)
        os.rename(ACCESS / old, ACCESS / new)
    for item in links:
        path = ACCESS / item['link']
        os.remove(path)
        os.symlink(item['new_target'], path)
        assert path.exists(), item
    for fam in emptied:
        (RECORD / 'old-family-readmes').mkdir(exist_ok=True)
        for leftover in (ACCESS / fam).iterdir():
            target = RECORD / 'old-family-readmes' / f"{fam.replace('/', '__')}__{leftover.name}"
            os.rename(leftover, target)
            archived.append(str(target.relative_to(ACCESS)))
        (ACCESS / fam).rmdir()
    table = ACCESS / 'MOVED-PATHS-20261020.tsv'
    lines = table.read_text().splitlines()
    updated = 0
    for i, line in enumerate(lines[1:], 1):
        flat, current = line.split('\t')
        if current in moves:
            lines[i] = f'{flat}\t{moves[current]}'
            updated += 1
    table.write_text('\n'.join(lines) + '\n')
    (ACCESS / 'MOVED-PATHS-20261028.tsv').write_text(
        'old\tnew\n' + ''.join(f'{o}\t{n}\n' for o, n in sorted(moves.items())))
    (RECORD / 'changes.json').write_text(json.dumps(dict(
        mapping=str(MAPPING.relative_to(REPO)), moved=len(moves), moved_paths_20261020_rows_updated=updated,
        symlinks_rewritten=links, emptied_families=emptied, archived_family_files=archived), ensure_ascii=False, indent=2) + '\n')
    print(f'moved {len(moves)}, MOVED-PATHS-20261020 rows updated {updated}, symlinks {len(links)}, archived {archived}')


if __name__ == '__main__':
    main()
