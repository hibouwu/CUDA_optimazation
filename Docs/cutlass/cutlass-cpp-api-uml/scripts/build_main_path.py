#!/usr/bin/env python3
"""Build the scheme-review reading path from the existing, immutable atlas.

Checks ensure traceability and prevent fixture/route drift. They do not prove
the truth of prose by substring matching, runtime correctness, or full coverage.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
STEP_IDS = ['problem', 'prepare', 'execute', 'complete', 'reuse']
PIN = '8f50b052e1099fb982392a622caab69b97b63128'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(root, spec, atlas, overviews):
    if spec['commit'] != PIN or atlas['commit'] != PIN:
        raise ValueError('Main path has changed its fixed source baseline')
    if [s['id'] for s in spec['steps']] != STEP_IDS:
        raise ValueError('The full five-step execution review must be retained')
    modules = {m['module_id']: m for m in atlas['modules']}
    module = modules[spec['module_id']]
    expected = {'architecture_recipe': 'cutlass::arch::Sm100', 'binary_target': 'sm_110a',
                'tile_mnk': [256, 128, 64], 'cluster_mnk': [2, 2, 1],
                'alpha': 1.0, 'beta': 0.5, 'layouts': 'A/B/C/D RowMajor',
                'elements': {'A': 'cutlass::half_t', 'B': 'cutlass::half_t', 'C': 'float', 'D': 'float', 'accumulator': 'float'}}
    if any(module['configuration'].get(k) != v for k, v in expected.items()):
        raise ValueError('Selected Dense configuration drifted from the narrated path')
    fixture = spec['fixture']
    example = root / 'site' / fixture['example_file']
    if sha(example) != fixture['example_sha256'] or fixture['shape_literal'] not in example.read_text():
        raise ValueError('Narrated example no longer matches saved source')
    if fixture['problem_mnkl'] != [256, 256, 128, 1]:
        raise ValueError('Narrated problem dimensions differ from saved example')
    nodes = {n['id'] for n in atlas['nodes']}
    edges = {e['id'] for e in atlas['edges']}
    views = {v['id'] for v in atlas['views']}
    graphs = {g['id'] for g in overviews['diagrams']}
    protocols = {p['id'] for p in atlas['protocols']}
    evidence = {e['id']: e for e in spec['evidence']}
    if len(evidence) != len(spec['evidence']):
        raise ValueError('Duplicated main-path evidence identity')
    cases = spec.get('review_cases', [])
    if {c['id'] for c in cases} != {'layout-contract', 'completion-contract', 'reuse-contract'} or len(cases) != 3:
        raise ValueError('All three contract review cases must be retained')
    for case in cases:
        if not set(case['steps']) <= set(STEP_IDS) or not case['steps']:
            raise ValueError('Review case lost its execution-step context')
        if not set(case['evidence_ids']) <= set(evidence) or not set(case['edge_refs']) <= edges:
            raise ValueError('Review case has an unrecorded source or edge')
        for field in ('title', 'claim_to_test', 'setup', 'observation', 'boundary'):
            if not case.get(field):
                raise ValueError('Missing case distinction: ' + case['id'] + '/' + field)
        if len(case['decisions']) < 2 or any(not d.get(k) for d in case['decisions'] for k in ('when', 'judgment', 'next_evidence')):
            raise ValueError('Review case must distinguish scope and correctness with conditions')
    for step in spec['steps']:
        for field in ('title', 'summary', 'input', 'responsibility', 'handoff', 'conditions', 'questions', 'api_refs', 'edge_refs', 'evidence_ids', 'boundary'):
            if not step.get(field):
                raise ValueError('Missing review responsibility: ' + step['id'] + '/' + field)
        if not set(step['api_refs']) <= nodes or not set(step['edge_refs']) <= edges:
            raise ValueError('Unrecorded API or edge used as main-path evidence')
        if not set(step['evidence_ids']) <= set(evidence):
            raise ValueError('Unknown source evidence')
        for route in step['detail_routes']:
            for key, available in [('api', nodes), ('view', views), ('overview', graphs), ('protocol', protocols)]:
                if key in route and route[key] not in available:
                    raise ValueError('Invalid detail route: ' + str(route))
            if 'mode' in route and route['mode'] not in {'contracts', 'compile', 'issues'}:
                raise ValueError('Unsupported main-path detail mode')
    source_files = {f['path']: f for f in atlas['files']}
    for item in evidence.values():
        path = root / item['artifact'] if item.get('artifact') else root / 'snapshot' / item['path']
        if item.get('path'):
            if sha(path) != source_files[item['path']]['sha256']:
                raise ValueError('Fixed source contents changed: ' + item['path'])
        elif path.resolve() != example.resolve():
            raise ValueError('Unreviewed auxiliary file in path evidence')
        lines = path.read_text().splitlines()
        first, last = item['start_line'], item['end_line']
        if not 1 <= first <= last <= len(lines) or item['needle'] not in '\n'.join(lines[first - 1:last]):
            raise ValueError('Evidence range does not contain the cited source: ' + item['id'])
        item['source_sha256'] = sha(path)
        if item.get('path'):
            item['source_url'] = 'source/' + item['path'] + '.html#L' + str(first)
        else:
            item['path'] = '已保存示例 / dense_baseline.cu'
        target, anchor = item['source_url'].split('#')
        if ('id="' + anchor + '"') not in (root / 'site' / target).read_text():
            raise ValueError('Missing offline source anchor: ' + item['id'])
    return spec


def load_host_probe(root, atlas):
    directory = root / 'verification/scheme-layout'
    result = json.loads((directory / 'results.json').read_text())
    if (result.get('status') != 'host_layout_pass' or result.get('scope') != 'host_layout_coordinate_mapping_only'
            or result.get('commit') != atlas['commit'] or result.get('compile_exit_code') != 0
            or result.get('execution_exit_code') != 0):
        raise ValueError('Host layout execution is missing or failed')
    for field in ('gpu_kernel_launched', 'gemm_runtime_verified', 'performance_verified'):
        if result.get(field) is not False:
            raise ValueError('Host probe must not claim GPU GEMM validation')
    if result['source_sha256'] != sha(directory / 'layout_probe.cu') or result['runner_sha256'] != sha(directory / 'run.py'):
        raise ValueError('Host layout result refers to different probe sources')
    if result['binary_sha256'] != sha(directory / 'artifacts/layout-probe'):
        raise ValueError('Host layout result binary changed')
    files = {f['path']: f for f in atlas['files']}
    for path, expected in result['fixed_header_sha256'].items():
        if files[path]['sha256'] != expected or sha(root / 'snapshot' / path) != expected:
            raise ValueError('Host layout reference header changed')
    expected = [(0, 128, 256, 128, 257), (8, 136, 264, 136, 265)]
    actual = [tuple(c[k] for k in ('padding', 'lda', 'ldb', 'a_offset_1_0', 'b_offset_1_1')) for c in result['cases']]
    if actual != expected or result['coordinates_checked'] != 131072 or result['ignored_padding_mismatches'] <= 0 or result['wrong_B_coordinate_mismatches'] <= 0:
        raise ValueError('Host layout observations do not match the narrated counterexample')
    if (directory / 'artifacts/run.log').read_text() != result['output']:
        raise ValueError('Host probe record does not match execution log')
    result['source_url'] = 'evidence/scheme-layout-results.json'
    result['download_url'] = 'evidence/scheme-layout-probe.cu'
    shutil.copyfile(directory / 'results.json', root / 'site' / result['source_url'])
    shutil.copyfile(directory / 'layout_probe.cu', root / 'site' / result['download_url'])
    return result


def build(root=ROOT):
    source = root / 'data/main-path-spec.json'
    atlas_file = root / 'data/atlas.json'
    overview_file = root / 'data/overviews.json'
    probe_directory = root / 'verification/scheme-layout'
    tracked = [source, atlas_file, overview_file, root / 'scripts/build_main_path.py',
               *(probe_directory / p for p in ('layout_probe.cu', 'run.py', 'results.json', 'artifacts/run.log', 'artifacts/layout-probe'))]
    before = {str(p.relative_to(root)): sha(p) for p in tracked}
    spec = copy.deepcopy(json.loads(source.read_text()))
    atlas = json.loads(atlas_file.read_text())
    report = validate(root, spec, atlas, json.loads(overview_file.read_text()))
    report['host_layout_probe'] = load_host_probe(root, atlas)
    report.update({'base_atlas_sha256': before['data/atlas.json'], 'input_sha256': before,
                   'scope': 'dense_scheme_review_path_not_full_library',
                   'configuration': next(m['configuration'] for m in atlas['modules'] if m['module_id'] == spec['module_id']),
                   'runtime_verified': False, 'actual_browser_verified': False,
                   'independent_use_review_passed': False, 'global_complete': False})
    if any(sha(root / p) != checksum for p, checksum in before.items()):
        raise ValueError('Reading-path inputs changed during build')
    (root / 'data/main-path.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (root / 'site/assets/main-path.js').write_text('window.ATLAS_MAIN_PATH=' + json.dumps(report, ensure_ascii=False, separators=(',', ':')) + ';\n')
    return report


if __name__ == '__main__':
    result = build()
    print(json.dumps({'steps': len(result['steps']), 'evidence': len(result['evidence']), 'runtime_verified': False}))
