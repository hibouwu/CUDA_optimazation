"""Build a 2026-10-20-before (flat) view of access_rules for running historical entries.

Results were regrouped into access_rules/<question>/<experiment>/<run>/ on 2026-10-20; the
packages themselves are unchanged. Old replays still look for access_rules/<run> and count
parents[k] from their own location, so run them from a view that restores the flat layout:

    python -B microbench/gh200_resource_campaign/access_rules/flat_layout_view.py VIEW RUN [RUN ...]
    python -B VIEW/results/gh200_resource_campaign/access_rules/RUN/<entry>.py --repo VIEW ...

Listed runs, plus runs named in their text files (``--depth`` rounds, default 1), become
reflink copies at the old flat paths, so edits inside the view never reach the archive. Every
other repository entry and the question directories are symlinked, so current code keeps
working inside the view. VIEW must be new and on the same btrfs filesystem as the repository.

Rounds were regrouped again on 2026-10-28 (MOVED-PATHS-20261028.tsv). Rounds written between the two
regroupings name their siblings by the 2026-10-27 question/family/round path; give such a RUN as that
path (it contains '/') and the view rebuilds the 10-27 layout of every question: real question and
family directories, rounds symlinked to their current location, the listed rounds as reflink copies.
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CAMPAIGN = Path('results/gh200_resource_campaign')
ACCESS = CAMPAIGN / 'access_rules'
TEXT = {'.json', '.md', '.py', '.sh', '.txt', '.csv', '.tsv'}


def moved_paths():
    rows = (REPO / ACCESS / 'MOVED-PATHS-20261020.tsv').read_text().splitlines()[1:]
    return dict(row.split('\t') for row in rows)


def layout_20261027():
    rows = (REPO / ACCESS / 'MOVED-PATHS-20261028.tsv').read_text().splitlines()[1:]
    return dict(row.split('\t') for row in rows)


def old_layout(view, before, copied):
    """Rebuild the 2026-10-27 question/family/round layout with symlinks to the current rounds."""
    after = {new: old for old, new in before.items()}
    for question in (REPO / ACCESS).iterdir():
        if not question.is_dir():
            continue
        for family in question.iterdir():
            entries = family.iterdir() if family.is_dir() else [family]
            for entry in entries:
                current = str(entry.relative_to(REPO / ACCESS))
                old = after.get(current, current)
                target = view / ACCESS / old
                if old in copied or target.exists() or target.is_symlink():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(entry)


def referenced(run_dir, names, pattern):
    found = set()
    for path in run_dir.rglob('*'):
        if path.is_file() and not path.is_symlink() and path.suffix in TEXT and path.stat().st_size < 20_000_000:
            found.update(pattern.findall(path.read_text(errors='ignore')))
    return found & names


def link_tree(view, keep):
    """Symlink every entry of each directory on the way to access_rules, except the one we descend into."""
    for real, mirror, skip in ((REPO, view, 'results'), (REPO / 'results', view / 'results', 'gh200_resource_campaign'),
                               (REPO / CAMPAIGN, view / CAMPAIGN, 'access_rules'), (REPO / ACCESS, view / ACCESS, None)):
        mirror.mkdir(parents=True, exist_ok=True)
        for entry in real.iterdir():
            if entry.name != skip and entry.name not in keep and not (mirror / entry.name).exists():
                (mirror / entry.name).symlink_to(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('view', type=Path)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--depth', type=int, default=1)
    args = parser.parse_args()
    view = args.view.resolve()
    if view.exists():
        raise SystemExit(f'{view} already exists')
    moved, before = moved_paths(), layout_20261027()
    flat = [run for run in args.runs if '/' not in run]
    layout = [run for run in args.runs if '/' in run]
    unknown = [run for run in flat if run not in moved]
    unknown += [run for run in layout if not (REPO / ACCESS / before.get(run, run)).is_dir()]
    if unknown:
        raise SystemExit(f'not in MOVED-PATHS-20261020.tsv or the 2026-10-27 layout: {unknown}')
    names = set(moved)
    pattern = re.compile('|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True)))
    selected, frontier = set(flat), set(flat)
    for _ in range(args.depth):
        frontier = set().union(*(referenced(REPO / ACCESS / moved[run], names, pattern) for run in frontier)) - selected
        selected |= frontier
    questions = {p.name for p in (REPO / ACCESS).iterdir() if p.is_dir()} if layout else set()
    link_tree(view, keep=selected | questions)
    if layout:
        old_layout(view, before, copied=set(layout))
    copies = {run: moved[run] for run in selected} | {run: before.get(run, run) for run in layout}
    for run, current in sorted(copies.items()):
        (view / ACCESS / run).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cp', '-a', '--reflink=always', str(REPO / ACCESS / current), str(view / ACCESS / run)], check=True)
    # 10-28 重组按新位置重写过少数相对软链接；复制回 10-27 位置的轮次恢复原指向
    changes = REPO / ACCESS / 'non-experiment/records/reorganization-20261028/changes.json'
    rewritten = json.loads(changes.read_text())['symlinks_rewritten'] if changes.exists() else []
    for run in layout:
        current = before.get(run, run)
        for item in rewritten:
            if item['link'].startswith(current + '/'):
                path = view / ACCESS / run / item['link'][len(current) + 1:]
                path.unlink()
                path.symlink_to(item['old_target'])
                if not path.exists():
                    raise SystemExit(f'restored link does not resolve in view: {path}')
    print(f'view {view}: {len(selected)} runs at flat paths, {len(layout)} at 2026-10-27 paths')
    for run in sorted(copies):
        print(' ', run)


if __name__ == '__main__':
    main()
