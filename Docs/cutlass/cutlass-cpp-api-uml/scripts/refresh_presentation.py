#!/usr/bin/env python3
"""Refresh only the published atlas presentation, without source extraction.

The original atlas.json remains byte-identical. The overlay report records which
templates replaced the prior rendered UI; it never certifies new source facts.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ('atlas.js', 'atlas.css', 'index.html')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_overlay(root=ROOT):
    report_path = root / 'data/presentation-build-report.json'
    if not report_path.exists():
        return None, []
    report = json.loads(report_path.read_text())
    issues = []
    if report['base_atlas_sha256'] != sha(root / 'data/atlas.json'):
        issues.append({'kind': 'presentation_base_changed'})
    for category in ('inputs', 'outputs'):
        for path, expected in report[category].items():
            target = root / path
            if not target.is_file() or sha(target) != expected:
                issues.append({'kind': 'presentation_' + category + '_changed', 'path': path})
    return report, issues


def main():
    import build_overviews
    import build_interface_docs
    import build_main_path
    before = sha(ROOT / 'data/atlas.json')
    build_overviews.build(ROOT)
    build_interface_docs.build(ROOT)
    build_main_path.build(ROOT)
    assert sha(ROOT / 'data/atlas.json') == before, 'Presentation must not change source graph'
    for name in TEMPLATES:
        destination = ROOT / 'site' / (name if name == 'index.html' else 'assets/' + name)
        shutil.copyfile(ROOT / 'templates/atlas' / name, destination)
    inputs = [ROOT / 'scripts/refresh_presentation.py', ROOT / 'scripts/build_overviews.py',
              ROOT / 'scripts/build_interface_docs.py', *(ROOT / 'data/interface-notes').glob('*.json'),
              ROOT / 'scripts/build_main_path.py', ROOT / 'data/main-path-spec.json',
              *(ROOT / 'verification/scheme-layout' / p for p in ('layout_probe.cu', 'run.py', 'results.json', 'artifacts/run.log', 'artifacts/layout-probe')),
              ROOT / 'data/overview-spec.json', *(ROOT / 'templates/atlas' / n for n in TEMPLATES)]
    graphs = json.loads((ROOT / 'data/overviews.json').read_text())['diagrams']
    outputs = [ROOT / 'data/overviews.json', ROOT / 'site/assets/overview-data.js',
               ROOT / 'data/interface-docs.json', ROOT / 'site/assets/interface-docs.js',
               ROOT / 'data/main-path.json', ROOT / 'site/assets/main-path.js',
               ROOT / 'site/evidence/scheme-layout-results.json', ROOT / 'site/evidence/scheme-layout-probe.cu',
               *(ROOT / 'site' / (n if n == 'index.html' else 'assets/' + n) for n in TEMPLATES),
               *(ROOT / 'site' / graph[key] for graph in graphs for key in ('svg', 'plantuml', 'dot'))]
    report = {'base_atlas_sha256': before, 'scope': 'presentation_only',
              'base_source_graph_unchanged': True, 'global_complete': False,
              'actual_browser_verified': False,
              'updated_template_paths': ['templates/atlas/' + n for n in TEMPLATES],
              'inputs': {str(p.relative_to(ROOT)): sha(p) for p in inputs},
              'outputs': {str(p.relative_to(ROOT)): sha(p) for p in outputs}}
    (ROOT / 'data/presentation-build-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    _, issues = verify_overlay()
    assert not issues, issues
    print(json.dumps({k: v for k, v in report.items() if k not in {'inputs', 'outputs'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
