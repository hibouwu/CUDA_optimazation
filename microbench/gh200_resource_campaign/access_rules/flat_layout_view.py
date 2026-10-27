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
"""
import argparse
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
    moved = moved_paths()
    unknown = [run for run in args.runs if run not in moved]
    if unknown:
        raise SystemExit(f'not in MOVED-PATHS-20261020.tsv: {unknown}')
    names = set(moved)
    pattern = re.compile('|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True)))
    selected, frontier = set(args.runs), set(args.runs)
    for _ in range(args.depth):
        frontier = set().union(*(referenced(REPO / ACCESS / moved[run], names, pattern) for run in frontier)) - selected
        selected |= frontier
    link_tree(view, keep=selected)
    for run in sorted(selected):
        subprocess.run(['cp', '-a', '--reflink=always', str(REPO / ACCESS / moved[run]), str(view / ACCESS / run)], check=True)
    print(f'view {view}: {len(selected)} runs at flat paths')
    for run in sorted(selected):
        print(' ', run)


if __name__ == '__main__':
    main()
