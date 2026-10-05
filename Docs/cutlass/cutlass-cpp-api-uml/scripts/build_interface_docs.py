#!/usr/bin/env python3
"""Attach reviewed interface explanations to the immutable published graph.

Declaration metadata determines parameter categories, not spelling such as
constexpr/static. Hand-authored notes explain semantics; absent notes fail the
build for every published callable/type/external interface in scope.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KINDS = {'api', 'type', 'external'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(text):
    return re.sub(r'\s+', '', text)


def source_for(root, node, param=None):
    item = param or node
    path = item.get('path') or node.get('path')
    line = item.get('start_line') or item.get('line') or node.get('line')
    if path and (root / 'snapshot' / path).is_file():
        return {'path': path, 'line': line,
                'source_url': 'source/' + path + '.html#L' + str(line or 1)}
    return {'source_url': node.get('source_url'), 'path': path, 'line': line}


def corrected_parameters(root, node, note):
    override = note.get('parameter_override')
    if not override:
        return node.get('parameters') or []
    if len(node.get('parameters') or []) != override['original_parameter_count']:
        raise ValueError('Stale parameter correction: ' + node['id'])
    source = override['source']
    if source['path'] != node.get('path'):
        raise ValueError('Parameter correction points to another source: ' + node['id'])
    lines = (root / 'snapshot' / source['path']).read_text().splitlines()
    written = '\n'.join(lines[source['start_line'] - 1:source['end_line']])
    for parameter in override['parameters']:
        if compact(parameter['raw']) not in compact(written):
            raise ValueError('Parameter correction missing in source: ' + node['id'])
    return [{**p, 'path': source['path'], 'start_line': source['start_line']}
            for p in override['parameters']]


def callable_category(node, param):
    raw_type = param.get('type') or param.get('raw') or ''
    type_names = {p.get('name') for g in node.get('template_parameters') or []
                  for p in g.get('parameters') or [] if 'type_parameter' in p.get('kind', '')}
    naked_type = re.sub(r'\bconst\b|\bvolatile\b|[&*]|\.\.\.', '', raw_type).strip()
    if re.search(r'\bprefer\s*<|StageCountAutoCarveout\s*<', raw_type):
        return ('type_carried', '编译期信息（以类型标签实参传入）',
                '选择信息已经编码在实参类型或其模板值中；这不是可任意改变的运行时整数。')
    if naked_type in type_names or re.search(r'Shape|Layout|Coord|Tile|Tensor<|CopyPolicy|MMA_Op|MMA_Atom|Traits|Permutations|TiledMma|TiledCopy|Copy_Atom', raw_type):
        return ('mixed', '编译期类型与运行时数据分开看',
                '类型在编译期确定；对象中的指针、坐标或动态尺寸按实际值使用。静态形状/布局可能不携带运行时数值，不能仅因是函数实参便认定其全部动态。')
    label = '函数实参：可运行时，也可常量求值' if re.search(r'\bconstexpr\b', node.get('signature', '')) else '运行时函数实参'
    return ('call_time', label,
            '在本次调用中提供值或对象。若该函数允许常量求值且实参满足条件，可以在编译期求值；constexpr、static 或 const 本身不强制这一点。')


def parameter_docs(root, node, note):
    result = {'templates': [], 'arguments': [], 'macros': []}
    for group_index, group in enumerate(node.get('template_parameters') or []):
        for index, p in enumerate(group.get('parameters') or []):
            key = p.get('name') or '#' + str(index)
            explanation = note.get('template_notes', {}).get(key)
            if not explanation:
                raise ValueError('Missing template explanation: ' + node['id'] + '/' + key)
            start = node.get('signature_range', {}).get('start_line', node.get('line') or 0)
            ownership = '本接口模板' if (p.get('start_line') or 0) >= start else '所属类型或外层模板'
            kind = p.get('kind', '')
            category = 'template_type' if 'type_parameter' in kind else 'template_template' if 'template_template' in kind else 'template_value'
            result['templates'].append({'name': p.get('name'), 'raw': p.get('raw', ''),
                'default': p.get('default'), 'description': explanation,
                'phase': category, 'phase_label': '编译期模板参数', 'ownership': ownership,
                'group_index': group_index, 'index': index, 'source': source_for(root, node, p)})
    if node.get('entity_kind') == 'macro_definition':
        match = re.search(r'#\s*define\s+\w+\(([^)]*)\)', node.get('signature', ''))
        if match:
            for index, name in enumerate(filter(None, (x.strip() for x in match[1].split(',')))):
                description = note.get('parameter_notes', {}).get(name)
                if not description:
                    raise ValueError('Missing macro parameter explanation: ' + node['id'])
                result['macros'].append({'name': name, 'raw': name, 'description': description,
                    'phase': 'preprocessor', 'phase_label': '预处理期文本参数', 'index': index,
                    'source': source_for(root, node)})
        return result
    for index, p in enumerate(corrected_parameters(root, node, note)):
        key = p.get('name') or '#' + str(index)
        explanation = note.get('parameter_notes', {}).get(key)
        if not explanation:
            raise ValueError('Missing function parameter explanation: ' + node['id'] + '/' + key)
        phase, label, caveat = callable_category(node, p)
        result['arguments'].append({'name': p.get('name'), 'raw': p.get('raw', ''),
            'type': p.get('type'), 'default': p.get('default'), 'description': explanation,
            'phase': phase, 'phase_label': label, 'phase_explanation': caveat,
            'index': index, 'source': source_for(root, node, p)})
    return result


def make_doc(root, node, note):
    for key in ('layer', 'summary', 'execution'):
        if not note.get(key):
            raise ValueError('Missing interface prose: ' + node['id'] + '/' + key)
    return {'id': node['id'], 'layer': note['layer'], 'summary': note['summary'],
            'execution': note['execution'], 'parameters': parameter_docs(root, node, note),
            'parameter_correction': note.get('parameter_override'),
            'declaration_available': bool(node.get('signature')),
            'configured_type': node.get('configured_type'),
            'entity_kind': node.get('entity_kind', node['kind']),
            'source': source_for(root, node)}


def build(root=ROOT):
    atlas_file = root / 'data/atlas.json'
    before = sha(atlas_file)
    atlas = json.loads(atlas_file.read_text())
    notes, files = {}, sorted((root / 'data/interface-notes').glob('*.json'))
    for path in files:
        loaded = json.loads(path.read_text())['nodes']
        duplicate = set(loaded) & set(notes)
        if duplicate:
            raise ValueError('Duplicate interface prose: ' + str(sorted(duplicate)))
        notes.update(loaded)
    scope = [n for n in atlas['nodes'] if n['kind'] in KINDS]
    missing = {n['id'] for n in scope} - set(notes)
    extra = set(notes) - {n['id'] for n in scope}
    if missing or extra:
        raise ValueError('Interface explanation coverage differs: ' + str({'missing': sorted(missing), 'extra': sorted(extra)}))
    docs = {n['id']: make_doc(root, n, notes[n['id']]) for n in scope}
    report = {'base_atlas_sha256': before, 'commit': atlas['commit'],
              'scope': 'all_published_api_type_external_nodes_not_full_library',
              'node_count': len(docs), 'nodes': docs,
              'counts': {k: sum(len(d['parameters'][k]) for d in docs.values()) for k in ('templates', 'arguments', 'macros')},
              'input_sha256': {str(p.relative_to(root)): sha(p) for p in [root / 'scripts/build_interface_docs.py', *files]},
              'source_graph_changed': False, 'runtime_verified': False, 'global_complete': False}
    assert before == sha(atlas_file)
    (root / 'data/interface-docs.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (root / 'site/assets/interface-docs.js').write_text('window.ATLAS_INTERFACE_DOCS=' + json.dumps(report, ensure_ascii=False, separators=(',', ':')) + ';\n')
    return report


if __name__ == '__main__':
    result = build()
    print(json.dumps({k: result[k] for k in ('node_count', 'counts', 'scope')}, ensure_ascii=False))
