#!/usr/bin/env python3
"""Emit the reviewed three-header draft to stdout; never writes snapshot/ledger/site."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from reconcile_candidates import top_items
from tree_sitter import Language, Parser
import tree_sitter_cpp

C = 'include/cute/algorithm/clear.hpp'
F = 'include/cute/algorithm/fill.hpp'
A = 'include/cute/algorithm/axpby.hpp'
TI = 'include/cute/tensor_impl.hpp'
P = 'include/cute/algorithm/prefer.hpp'
FN = 'include/cute/algorithm/functional.hpp'
IC = 'include/cute/numeric/integral_constant.hpp'
CX = 'include/cutlass/complex.h'
CF = 'include/cute/config.hpp'
SCOPE = (C, F, A)
PREFIX = 'elementwise.'
RAW = {}


def source(path):
    if path not in RAW:
        RAW[path] = (ROOT / 'snapshot' / path).read_bytes()
    return RAW[path]


def span(path, start, end):
    raw = source(path)
    return {'path': path, 'start_byte': start, 'end_byte': end,
            'start_line': raw.count(b'\n', 0, start) + 1,
            'end_line': raw.count(b'\n', 0, max(start, end - 1)) + 1,
            'raw': raw[start:end].decode()}


def locate(path, text, line, end_line=None, occurrence=None):
    raw = source(path); lines = raw.splitlines(keepends=True)
    lo = sum(map(len, lines[:line-1])); hi = sum(map(len, lines[:end_line or line]))
    pattern = text.encode(); positions = []; pos = lo
    while (pos := raw.find(pattern, pos, hi)) >= 0:
        positions.append(pos); pos += len(pattern)
    if occurrence is None:
        assert len(positions) == 1, (path, line, text, positions)
        occurrence = 0
    assert len(positions) > occurrence, (path, line, text, positions)
    start = positions[occurrence]
    return span(path, start, start + len(pattern))


def evidence(s, **extra):
    return {k: v for k, v in s.items() if k != 'raw'} | {'quote': s['raw']} | extra


def parse_source(path):
    raw = source(path)
    # Byte-preserving parser-only annotation masks. Default arguments are
    # inventoried from physical source independently; no body is removed.
    projected = re.sub(rb'\b(?:CUTE_HOST_DEVICE|CUTE_UNROLL)\b', lambda m: b' ' * len(m[0]), raw)
    projected = re.sub(rb'(?<=p) = \{\}', lambda m: b' ' * len(m[0]), projected)
    tree = Parser(Language(tree_sitter_cpp.language())).parse(projected)
    assert not tree.root_node.has_error, path
    return tree


def descendants(node):
    yield node
    for child in node.named_children:
        yield from descendants(child)


def main():
    scope = json.loads((ROOT / 'data/scope.json').read_text())
    selected_paths = set(SCOPE) | {TI, P, FN, IC, CX}
    ledger = []
    for key, value, item in top_items(ROOT / 'data/declarations.json', arrays=('occurrences',)):
        if item and key == 'occurrences' and value['path'] in selected_paths and '<cutlass_namespace(' not in value['qualified_name']:
            ledger.append(value)
    nodes = []; edges = []; inventories = []; local_to_id = {}; call_refs = {}

    def node(id, kind, name, path, line, role, **kwargs):
        result = {'id': PREFIX + id, 'kind': kind, 'name': name, 'path': path,
                  'line': line, 'role': role, **kwargs}
        nodes.append(result); local_to_id[id] = result
        return result

    def declaration(id, path, line, name, label, role, kind=None):
        choices = [o for o in ledger if o['path'] == path and o['name'] == name
                   and o['signature_range']['start_line'] <= line <= o['signature_range']['end_line']]
        assert len(choices) == 1, (id, len(choices))
        o = choices[0]; s = o['signature_range']
        assert source(path)[s['start_byte']:s['end_byte']].decode().rstrip() == o['raw_signature']
        kind = kind or ('api' if o['kind'] in {'function', 'method', 'operator'} else 'type')
        return node(id, kind, label, path, line, role, qualified_name=o['qualified_name'],
                    entity_id=o['entity_id'], declaration_occurrence_id=o['declaration_occurrence_id'],
                    entity_ref={'space': 'global_source_ledger', 'entity_id': o['entity_id'],
                                'declaration_occurrence_id': o['declaration_occurrence_id']},
                    signature=o['raw_signature'], signature_range=s,
                    template_parameters=o.get('template_parameters', []), parameters=o.get('parameters', []),
                    preprocessor_conditions=o.get('preprocessor_conditions', []))

    def edge(id, owner, target, relation, s, condition='无额外条件', resolution='source_proven', **kwargs):
        e = {'id': PREFIX + 'edge.' + id, 'source': PREFIX + owner, 'target': PREFIX + target,
             'relation': relation, 'source_expression': s['raw'], 'condition': condition,
             'resolution': resolution, 'evidence': [evidence(s)], **kwargs}
        if relation == 'calls':
            e['callsite'] = {k: v for k, v in s.items() if k != 'raw'} | {
                'source_expression': s['raw'], 'raw_source_expression': s['raw'], 'caller': PREFIX + owner}
        edges.append(e)
        return e

    funcs = [
        ('api.clear_rvalue', C, 46, 'clear', 'clear(Tensor&&)', '接收mutable temporary；命名tensor表达式是lvalue。'),
        ('api.clear_lvalue', C, 57, 'clear', 'clear(Tensor&)', '取Tensor::value_type，以T{}调用fill；不是memset。'),
        ('api.fill_rvalue', F, 47, 'fill', 'fill(Tensor&&, value)', '接收mutable temporary，再以命名lvalue转发。'),
        ('api.fill_prefer1', F, 59, 'fill', 'detail::fill(..., prefer<1>)', '返回类型的decltype检验data级fill可形成表达式；函数体才实际调用。'),
        ('api.fill_prefer0', F, 69, 'fill', 'detail::fill(..., prefer<0>)', 'fallback：按int i遍历逻辑size，逐元素赋值。'),
        ('api.fill_lvalue', F, 82, 'fill', 'fill(Tensor&, value)', '构造prefer<1>标签；合法时选择高优先级，否则通过基类转换选择fallback。'),
        ('api.axpby_rvalue', A, 50, 'axpby', 'axpby(..., Tensor&& y, p)', '保持alpha/x/beta/p及命名y，转发到lvalue候选。'),
        ('api.axpby_lvalue', A, 69, 'axpby', 'axpby(..., Tensor& y, p)', '先计算isBetaZero一次，再按size(x)逐坐标、由p(i)控制更新。'),
    ]
    for row in funcs:
        declaration(*row)
    declarations = {
        'type.Tensor': (TI, 136, 'Tensor', 'cute::Tensor<Engine, Layout>', '逻辑布局与Engine存储接口；本模块不把Engine固定为寄存器或任一种地址空间。'),
        'type.value_type': (TI, 139, 'value_type', 'Tensor::value_type', '别名直接依赖Engine::value_type。'),
        'type.prefer': (P, 38, 'prefer', 'cute::prefer<N>', 'prefer<N>公开继承prefer<N-1>。'),
        'type.prefer0': (P, 41, 'prefer<0>', 'cute::prefer<0>', '优先级递归的空终点特化。'),
        'type.constant_fn': (FN, 54, 'constant_fn', 'cute::constant_fn<R>', '默认predicate函数对象模板；R r_为存储值。'),
        'type.true_type': (IC, 57, 'true_type', 'cute::true_type', 'bool_constant<true>的类型别名。'),
        'type.bool_constant': (IC, 55, 'bool_constant', 'cute::bool_constant<b>', '别名C<b>。'),
        'type.C': (IC, 42, 'C', 'cute::C<v>', '类型携带静态value，提供value_type转换。'),
        'type.Int': (IC, 127, 'Int', 'cute::Int<v>', '别名C<v>；此处v=0。'),
        'type.is_complex': (CX, 667, 'is_complex', 'cutlass::is_complex<T>', '主模板value=false；cute通过using引入。'),
        'type.is_complex_complex': (CX, 672, 'is_complex<complex<T>>', 'cutlass::is_complex<complex<T>>', 'cutlass::complex<T>特化value=true；不排除外部合法特化。'),
        'api.tensor_data': (TI, 186, 'data', 'Tensor::data() 非const', '主模板成员返回engine().begin()；本轮不递归展开任意Engine。'),
        'api.tensor_at_mutable': (TI, 236, 'operator()', 'Tensor::operator()(Coord const&) 非const', '本文件中Coord=int；走无underscore分支data()[layout()(coord)]。'),
        'api.tensor_at_const': (TI, 250, 'operator()', 'Tensor::operator()(Coord const&) const', 'x是const Tensor；主模板中int坐标走无underscore分支。'),
        'api.tensor_size': (TI, 552, 'size', 'cute::size(Tensor const&)', '已知Tensor候选；将Is...与tensor.layout()转发给size。'),
        'api.constant_fn_call': (FN, 57, 'operator()', 'constant_fn<R>::operator()', '默认PrdTensor时返回r_；不假定所有自定义p均为此类型。'),
        'api.complex_real': (CX, 296, 'real', 'complex<T>::real() const', '仅当Beta绑定cutlass::complex<T>时的真实成员声明。'),
        'api.complex_imag': (CX, 304, 'imag', 'complex<T>::imag() const', '仅当Beta绑定cutlass::complex<T>时的真实成员声明。'),
    }
    for id, row in declarations.items():
        declaration(id, *row)
    node('binding.prefer1', 'binding', 'cute::prefer<1>', F, 59,
         '这些fill声明/分派实际使用的固定优先级标签实例；不把generic prefer<N>直接画成总是继承prefer<0>。',
         source_entity_kind='template_type_instance', source_expression='prefer<1>',
         instance_of=PREFIX + 'type.prefer', template_arguments={'N': '1'})
    local_to_id['api.fill_prefer1']['notes'] = '返回类型严格保留decltype(fill(tensor.data(), value))；函数体只调用fill而没有return。若自定义ADL fill返回非void，SFINAE成功本身不能证明该重载有定义的返回行为。本模块不替源码补return。'
    namespace = declaration('namespace.cute', C, 37, 'cute', 'namespace cute', '三个header的namespace出现共享同一个全局namespace实体。', kind='namespace')
    namespace['occurrence_refs'] = [{'path': o['path'], 'entity_id': o['entity_id'],
                                    'declaration_occurrence_id': o['declaration_occurrence_id'],
                                    'signature_range': o['signature_range']}
                                   for o in ledger if o['path'] in SCOPE and o['kind'] == 'namespace' and o['name'] == 'cute']
    declaration('namespace.detail', F, 52, 'detail', 'namespace cute::detail', '高低优先级fill实现的声明域。', kind='namespace')

    trees = {p: parse_source(p) for p in SCOPE}
    lambda_ast = next(n for n in descendants(trees[A].root_node) if n.type == 'lambda_expression')
    lam_sig = locate(A, '[&] ()', 75)
    lam = node('api.beta_zero_lambda', 'api', 'axpby局部立即调用lambda', A, 75,
               '捕获beta引用，在逐元素循环开始前执行一次；不是axpby的成员函数。',
               source_entity_kind='lambda', qualified_name='cute::axpby::<lambda@axpby.hpp:75>',
               manual_declaration={'path': A, 'start_line': 75, 'end_line': 84, 'name': '<lambda@75>',
                                   'kind': 'lambda', 'parameters': [], 'raw_signature': lam_sig['raw'],
                                   'signature_range': {k: v for k, v in lam_sig.items() if k != 'raw'},
                                   'definition_range': {k: v for k, v in span(A, lambda_ast.start_byte, lambda_ast.end_byte).items() if k != 'raw'},
                                   'return_type': 'deduced from selected return expression', 'captures': '[&]'})
    alias = locate(C, 'using T = typename Tensor<Engine,Layout>::value_type;', 59)
    node('local.T', 'type', 'clear局部别名T', C, 59, '函数局部依赖别名；全局ledger不收录局部声明，保留独立source身份。',
         qualified_name='cute::clear::<local T@clear.hpp:59>',
         manual_declaration={'path': C, 'start_line': 59, 'end_line': 59, 'name': 'T', 'kind': 'alias',
                             'parameters': [], 'raw_signature': alias['raw'],
                             'signature_range': {k: v for k, v in alias.items() if k != 'raw'}})
    for id, path, line, name, role in [
        ('local.fill_i', F, 72, 'fill循环int i', '独立循环变量；0初始化，每轮++i。'),
        ('local.axpby_i', A, 87, 'axpby循环int i', '与fill的i不是同一个source实体。'),
        ('local.isBetaZero', A, 75, 'isBetaZero', 'auto对象，由立即调用lambda初始化；在循环内反复使用。'),
    ]:
        ast = next(n for n in descendants(trees[path].root_node) if n.type == 'declaration' and n.start_point.row + 1 == line)
        s = span(path, ast.start_byte, ast.end_byte)
        node(id, 'resource', name, path, line, role, entity_kind='local_object', source_declaration=s,
             source_identity=f'{path}@{s["start_byte"]}:{s["end_byte"]}')
    for id, path, line, name, role in [
        ('res.tensor', F, 69, 'tensor逻辑元素', '调用方提供的可写Tensor；底层Engine与Layout未固定。'),
        ('res.value', F, 69, 'value', '按const&传入；fallback每个激活赋值读取此表达式，允许与目标潜在别名。'),
        ('res.x', A, 70, 'x逻辑元素', 'const Tensor视图不等于底层存储全局不可变。'),
        ('res.y', A, 72, 'y逻辑元素', '按p(i)及beta分支进行读取/写入；无独立缓冲分配。'),
        ('res.alpha', A, 69, 'alpha', '模板Alpha const&；运算的具体重载和转换依赖实参。'),
        ('res.beta', A, 71, 'beta', '模板Beta const&；每次调用先进行零值判别。'),
        ('res.p', A, 73, 'p', '模板PrdTensor const&；默认constant_fn<true_type>，但可以自定义。'),
    ]:
        node(id, 'resource', name, path, line, role, entity_kind='local_object', scope='formal_argument_resource_not_new_storage')

    def dependent(id, path, line, expression, parameters, role):
        return node(id, 'binding', expression, path, line, role,
                    source_entity_kind='dependent_expression', source_expression=expression,
                    resolution='symbolic', dependency_parameters=parameters,
                    dependency_status='uninstantiated_template_semantics_not_parser_failure')
    dependent('dep.data_fill', F, 62, 'fill(tensor.data(), value)', ['Engine', 'Layout', 'T'],
              '非限定调用；data返回类型与T控制ADL/候选可行性。记录原表达式，不伪造固定目标函数。')
    dependent('dep.p_call', A, 88, 'p(i)', ['PrdTensor'], '任意predicate调用；只有默认PrdTensor分支可进一步绑定constant_fn::operator()。')
    dependent('dep.beta_real', A, 77, 'beta.real()', ['Beta'], '编译期complex分支内的依赖成员调用。')
    dependent('dep.beta_imag', A, 77, 'beta.imag()', ['Beta'], '编译期complex分支内的依赖成员调用；实际求值还取决于&&。')

    # Each syntax call is listed once by its physical byte identity. One
    # dispatch call may have multiple *conditional targets*, never two sites.
    def call(id, owner, target, path, text, line, end_line=None, condition='无额外条件', resolution='source_proven', occurrence=None, evaluated=True):
        s = locate(path, text, line, end_line, occurrence)
        e = edge(id, owner, target, 'calls' if evaluated else 'type_uses', s, condition, resolution,
                 evaluation='potentially_evaluated' if evaluated else 'unevaluated_decltype',
                 dependency_parameters=local_to_id[target].get('dependency_parameters') or
                 [p['name'] for group in local_to_id[owner].get('template_parameters', []) for p in group['parameters']] or
                 (['Beta'] if owner == 'api.beta_zero_lambda' else []))
        if not evaluated:
            e['expression_range'] = s
        call_refs.setdefault((path, s['start_byte'], s['end_byte']), []).append(e['id'])
        return e
    forward = '命名的&&形参是lvalue；本文件lvalue重载是已知候选；最终依赖调用的ADL/重载决议保留到实例化'
    call('clear_forward', 'api.clear_rvalue', 'api.clear_lvalue', C, 'clear(tensor)', 48, condition=forward, resolution='symbolic')
    call('clear_fill', 'api.clear_lvalue', 'api.fill_lvalue', C, 'fill(tensor, T{})', 61,
         condition='T=Tensor<Engine,Layout>::value_type；已知Tensor& fill候选，保留依赖ADL', resolution='symbolic')
    call('fill_forward', 'api.fill_rvalue', 'api.fill_lvalue', F, 'fill(tensor, value)', 49, condition=forward, resolution='symbolic')
    call('fill_sfinae_expression', 'api.fill_prefer1', 'dep.data_fill', F, 'fill(tensor.data(), value)', 60,
         condition='替换到trailing return的decltype；不求值、不产生一次运行时fill', resolution='symbolic', evaluated=False)
    call('fill_sfinae_data', 'api.fill_prefer1', 'api.tensor_data', F, 'tensor.data()', 60,
         condition='decltype的嵌套不求值表达式；Tensor主模板非const data候选', evaluated=False)
    call('fill_data_call', 'api.fill_prefer1', 'api.tensor_data', F, 'tensor.data()', 62,
         condition='高优先级重载被选中；非const Tensor主模板')
    call('fill_adl_call', 'api.fill_prefer1', 'dep.data_fill', F, 'fill(tensor.data(), value)', 62,
         condition='高优先级重载被选中；实际ADL目标依赖data返回类型和T', resolution='symbolic')
    call('fill_size', 'api.fill_prefer0', 'api.tensor_size', F, 'size(tensor)', 72,
         condition='fallback；每次for条件检查中的Tensor候选，最终ADL保留', resolution='symbolic')
    call('fill_element', 'api.fill_prefer0', 'api.tensor_at_mutable', F, 'tensor(i)', 73,
         condition='fallback && i<size(tensor)；Tensor主模板；Coord=int')
    call('fill_dispatch_prefer1', 'api.fill_lvalue', 'api.fill_prefer1', F, 'detail::fill(tensor, value, prefer<1>{})', 84,
         condition='decltype(fill(tensor.data(), value))替换成功；prefer<1>精确匹配优于基类转换', resolution='symbolic')
    call('fill_dispatch_prefer0', 'api.fill_lvalue', 'api.fill_prefer0', F, 'detail::fill(tensor, value, prefer<1>{})', 84,
         condition='prefer<1>候选因替换失败移除；prefer<1>→prefer<0>基类转换；fallback函数体自身仍须可实例化', resolution='symbolic')
    call('axpby_forward', 'api.axpby_rvalue', 'api.axpby_lvalue', A, 'axpby(alpha, x, beta, y, p)', 56, condition=forward, resolution='symbolic')
    lambda_call = next(n for n in descendants(trees[A].root_node) if n.type == 'call_expression' and n.start_byte == lambda_ast.start_byte)
    call('axpby_beta_once', 'api.axpby_lvalue', 'api.beta_zero_lambda', A,
         source(A)[lambda_call.start_byte:lambda_call.end_byte].decode(), 75, 84, condition='每次进入lvalue axpby先执行一次，位于for循环之前')
    call('beta_real', 'api.beta_zero_lambda', 'dep.beta_real', A, 'beta.real()', 77,
         condition='if constexpr(is_complex<Beta>::value)分支被保留', resolution='symbolic')
    call('beta_imag', 'api.beta_zero_lambda', 'dep.beta_imag', A, 'beta.imag()', 77,
         condition='complex分支；若&&为内建且左侧为false则短路；自定义运算须按实例化结果判断', resolution='symbolic')
    call('axpby_size', 'api.axpby_lvalue', 'api.tensor_size', A, 'size(x)', 87,
         condition='每次for条件检查中的Tensor候选，最终ADL保留', resolution='symbolic')
    call('axpby_predicate', 'api.axpby_lvalue', 'dep.p_call', A, 'p(i)', 88,
         condition='i<size(x)时每个逻辑坐标检验一次；返回值须能用于if条件', resolution='symbolic')
    call('axpby_y_write_ref', 'api.axpby_lvalue', 'api.tensor_at_mutable', A, 'y(i)', 89,
         condition='p(i)条件成立；左侧引用；Tensor主模板，Coord=int', occurrence=0)
    # Explicit byte occurrences preserve the two identical x(i) and y(i)
    # source spellings in the same line as different calls.
    call('axpby_x_beta_zero', 'api.axpby_lvalue', 'api.tensor_at_const', A, 'x(i)', 89,
         condition='p(i) && isBetaZero条件成立；true arm', occurrence=0)
    call('axpby_x_beta_nonzero', 'api.axpby_lvalue', 'api.tensor_at_const', A, 'x(i)', 89,
         condition='p(i) && !isBetaZero条件成立；false arm', occurrence=1)
    call('axpby_y_old', 'api.axpby_lvalue', 'api.tensor_at_mutable', A, 'y(i)', 89,
         condition='p(i) && !isBetaZero条件成立；false arm中的旧值引用', occurrence=1)

    # Necessary source-declared type relations; no synthetic overload identity.
    for id, path, line, *_ in funcs:
        typ = 'Tensor<Engine, Layout>' if path in (C, F) else 'Tensor<XEngine, XLayout>'
        edge(id.replace('api.', '') + '_tensor_type', id, 'type.Tensor', 'type_uses', locate(path, typ, line, line+5),
             '完整模板及引用cv限定见该API全局声明签名', 'symbolic')
        if path == A:
            edge(id.replace('api.', '') + '_y_type', id, 'type.Tensor', 'type_uses', locate(A, 'Tensor<YEngine, YLayout>', line+3),
                 'YEngine/YLayout独立于X；不从源码推断相同shape/地址空间', 'symbolic')
    edge('clear_T_alias', 'local.T', 'type.value_type', 'aliases', alias, 'Tensor主模板的value_type=Engine::value_type', 'symbolic')
    edge('clear_uses_T', 'api.clear_lvalue', 'local.T', 'type_uses', alias)
    edge('prefer1_template', 'binding.prefer1', 'type.prefer', 'instance_of', locate(F, 'prefer<1>', 59),
         '本文件实际标签的模板实参N=1')
    edge('prefer_inheritance', 'binding.prefer1', 'type.prefer0', 'inherits', locate(P, 'struct prefer : prefer<N-1>', 38),
         '源码标签为prefer<1>；将N=1代入直接基类prefer<N-1>，得到prefer<0>')
    for id, line in [('api.fill_prefer1', 59), ('api.fill_prefer0', 69), ('api.fill_lvalue', 84)]:
        text = 'prefer<0>' if line == 69 else 'prefer<1>'
        edge(id.replace('api.', '') + '_prefer_type', id, 'type.prefer0' if line == 69 else 'binding.prefer1', 'type_uses', locate(F, text, line))
    for id, line in [('api.axpby_rvalue', 47), ('api.axpby_lvalue', 66)]:
        edge(id.replace('api.', '') + '_default_predicate', id, 'type.constant_fn', 'type_uses', locate(A, 'constant_fn<true_type>', line),
             'PrdTensor未显式指定时的模板默认类型；函数实参p省略时另外执行{}初始化', 'symbolic')
    edge('default_predicate_result', 'type.constant_fn', 'type.true_type', 'template_binds', locate(A, 'constant_fn<true_type>', 66), '只描述本文件默认R=true_type，不是constant_fn所有实例', 'symbolic')
    edge('true_alias', 'type.true_type', 'type.bool_constant', 'aliases', locate(IC, 'bool_constant<true>', 57))
    edge('bool_alias', 'type.bool_constant', 'type.C', 'aliases', locate(IC, 'using bool_constant = C<b>;', 55), 'b保留为模板参数', 'symbolic')
    edge('Int_alias', 'type.Int', 'type.C', 'aliases', locate(IC, 'using Int = C<v>;', 127), 'v保留为模板参数；本模块使用Int<0>', 'symbolic')
    edge('beta_trait', 'api.beta_zero_lambda', 'type.is_complex', 'type_uses', locate(A, 'is_complex<Beta>::value', 76),
         'cute/numeric/complex.hpp通过using cutlass::is_complex引入；选取Beta的实际特化', 'symbolic',
         extra_evidence=[evidence(locate('include/cute/numeric/complex.hpp', 'using cutlass::is_complex;', 41))])
    edge('beta_trait_specialization', 'type.is_complex_complex', 'type.is_complex', 'specializes', locate(CX, 'struct is_complex<complex<T>>', 672), 'Beta=cutlass::complex<T>', 'symbolic')
    for target, dep, line, expr in [('api.complex_real', 'dep.beta_real', 296, 'T const &real() const'), ('api.complex_imag', 'dep.beta_imag', 304, 'T const &imag() const')]:
        edge(dep.replace('dep.', '') + '_complex_binding', dep, target, 'instance_of', locate(CX, expr, line), '仅当Beta=cutlass::complex<T>且该主模板成员适用', 'symbolic')
    edge('predicate_default_binding', 'dep.p_call', 'api.constant_fn_call', 'instance_of', locate(A, 'constant_fn<true_type>', 66),
         'PrdTensor使用默认类型constant_fn<true_type>；T...从int&形参转发推导', 'symbolic')

    def owner_for(path, start):
        if path == A and lambda_ast.start_byte <= start < lambda_ast.end_byte:
            return 'api.beta_zero_lambda'
        candidates = []
        for id, p, *_ in funcs:
            if p != path:
                continue
            sig = local_to_id[id]['signature_range']
            if sig['start_byte'] <= start:
                candidates.append((sig['start_byte'], id))
        return max(candidates)[1] if candidates else 'namespace.cute'

    # Inventory *every* selected syntax item, not only relationships. Operators
    # are evaluations, not invented C++ calls: builtins versus overloads depend
    # on actual scalar/reference types.
    categories = {'function_definition': 'function', 'namespace_definition': 'namespace',
                  'alias_declaration': 'local_declaration', 'declaration': 'local_declaration',
                  'lambda_expression': 'lambda', 'call_expression': 'call_expression',
                  'compound_literal_expression': 'value_initialization',
                  'binary_expression': 'operator_expression', 'assignment_expression': 'operator_expression',
                  'conditional_expression': 'operator_expression', 'update_expression': 'operator_expression',
                  'for_statement': 'control_flow', 'if_statement': 'control_flow',
                  'return_statement': 'return_statement', 'preproc_include': 'include', 'preproc_call': 'pragma'}
    counts = {}
    for path in SCOPE:
        for ast in descendants(trees[path].root_node):
            if ast.type not in categories:
                continue
            s = span(path, ast.start_byte, ast.end_byte); cat = categories[ast.type]
            # Preprocessor node includes trailing newline; the exact byte span
            # intentionally retains it, while line range ends on directive line.
            key = (path, ast.start_byte, ast.end_byte)
            uid = f'{Path(path).stem}.{cat}.{ast.start_byte}_{ast.end_byte}'
            refs = []; explanation = ''
            if cat == 'call_expression':
                assert key in call_refs, ('unclassified call', s)
                refs = call_refs[key]
                explanation = 'decltype内不求值' if path == F and s['start_line'] == 60 else '单一物理调用点；模板分派可对应互斥条件的多个候选目标'
            elif cat == 'function':
                refs = [PREFIX + owner_for(path, ast.start_byte)]
                explanation = '完整模板/形参/返回类型/引用限定复用全局声明；函数体由后续逐项义务覆盖'
            elif cat == 'namespace':
                refs = [PREFIX + ('namespace.detail' if path == F and s['start_line'] == 52 else 'namespace.cute')]
                explanation = 'namespace出现保留独立occurrence，全局实体按identity复用'
            elif cat == 'local_declaration':
                lid = 'local.T' if path == C else 'local.fill_i' if path == F else 'local.isBetaZero' if s['start_line'] == 75 else 'local.axpby_i'
                refs = [PREFIX + lid]; explanation = '函数局部source身份，不冒充全局ledger实体'
            elif cat == 'lambda':
                refs = [lam['id']]; explanation = '完整捕获/签名/定义范围，立即调用另有独立callsite'
            elif cat in {'operator_expression', 'control_flow', 'value_initialization'}:
                oid = 'expr.' + uid
                dependent_params = ['Engine', 'Layout', 'T'] if path != A else ['Alpha', 'Beta', 'XEngine', 'XLayout', 'YEngine', 'YLayout', 'PrdTensor']
                role = {'operator_expression': '完整运算表达式；保留内建/用户重载/代理引用的模板依赖，不把它虚构成已决API调用。',
                        'control_flow': '已知for/if源码控制容器，不是API或待解析调用目标。内部条件/子表达式的模板依赖另列；结构本身不依赖实例化才可识别。',
                        'value_initialization': '源码{}构造/值初始化；不自动等价为所有类型的算术零或字节零。'}[cat]
                proven_builtin = ast.type == 'update_expression' and s['raw'] == '++i'
                proven_fixed_init = cat == 'value_initialization' and s['raw'] in {'prefer<1>{}', 'Int<0>{}'}
                structural = cat == 'control_flow'
                resolution = 'source_proven' if proven_builtin or proven_fixed_init or structural else 'symbolic'
                if proven_builtin:
                    role = 'i在本for初始化中声明为int；++i是已知内建自增，不是依赖重载或额外call API。'
                    dependent_params = []
                elif proven_fixed_init:
                    role = '显式固定类型的{}值初始化：类型实参为源码常量1或0；不是未实例化Engine/Beta依赖，也不虚构构造函数调用节点。'
                    dependent_params = []
                label = f'{ast.type} 控制容器 @{Path(path).name}:{s["start_line"]}' if structural else s['raw']
                node(oid, 'binding', label, path, s['start_line'], role,
                     source_entity_kind=ast.type, source_expression=s['raw'], source_range=s,
                     dependency_parameters=dependent_params, resolution=resolution,
                     semantic_class='source_control_container' if structural else 'builtin_int_update' if proven_builtin else 'fixed_type_value_initialization' if proven_fixed_init else 'template_dependent_expression')
                ee = edge(uid, owner_for(path, ast.start_byte), oid, 'evaluates', s,
                          '源码控制结构已知；内部依赖条件/子表达式单独保留，不把控制容器当未解析API' if structural else
                          '源码固定int变量的内建自增' if proven_builtin else
                          '源码固定类型prefer<1>或Int<0>的值初始化' if proven_fixed_init else
                          '按所属源码控制结构求值；依赖的运算结果与重载保留到实例化', resolution)
                refs = [PREFIX + oid, ee['id']]; explanation = role
                if cat == 'value_initialization':
                    type_target = 'local.T' if path == C else 'binding.prefer1' if path == F else 'type.Int'
                    edge(uid + '_type', oid, type_target, 'type_uses', s, '保留真实初始化表达式', resolution)
            elif cat == 'return_statement':
                refs = [PREFIX + owner_for(path, ast.start_byte)]; explanation = 'return表达式原文；嵌套调用/运算在独立义务中，不另伪造返回API'
            elif cat == 'include':
                explanation = '完整include指令；依赖header不加入本模块全文件覆盖分母'
            else:
                explanation = '#pragma once；header包含约定，不作为运行时操作'
            obligation = {'id': PREFIX + 'obligation.' + uid, 'category': cat,
                          'syntax_kind': ast.type, 'source_range': s, 'status': 'source_reviewed',
                          'graph_refs': refs, 'explanation': explanation}
            if cat == 'include':
                obligation['target_path'] = 'include/' + re.search(r'<([^>]+)>', s['raw']).group(1)
                obligation['relation'] = 'file_dependency'
                assert (ROOT / 'snapshot' / obligation['target_path']).is_file()
            inventories.append(obligation)
            counts[cat] = counts.get(cat, 0) + 1

    # Defaults are separately inventoried because the independent parser mask
    # only repairs its inability to parse the omitted type in `p = {}`.
    for owner, line in [('api.axpby_rvalue', 54), ('api.axpby_lvalue', 73)]:
        s = locate(A, '{}', line); oid = f'expr.p_default_{line}'
        dependent(oid, A, line, '{}', ['PrdTensor'], '函数实参p省略时，以{}初始化PrdTensor并绑定const&；不假定总是默认predicate类型。')
        e = edge(f'p_default_{line}', owner, oid, 'evaluates', s, '仅当调用者省略函数实参p；PrdTensor本身也可显式指定', 'symbolic')
        inventories.append({'id': PREFIX + f'obligation.axpby.default_argument.{s["start_byte"]}',
                            'category': 'default_argument', 'syntax_kind': 'initializer_list', 'source_range': s,
                            'status': 'source_reviewed', 'graph_refs': [PREFIX + oid, e['id']],
                            'explanation': '原始{}默认值精确保留；parser-only mask不删除覆盖义务'})
        counts['default_argument'] = counts.get('default_argument', 0) + 1
    # Macro uses are a separate source obligation; the host/device and unroll
    # spelling never disappears from API signatures or coverage data.
    for path in SCOPE:
        for match in re.finditer(rb'\b(CUTE_HOST_DEVICE|CUTE_UNROLL|CUTE_GCC_UNREACHABLE)\b', source(path)):
            # Ignore the explanatory include comment in clear.hpp:33.
            line = source(path).count(b'\n', 0, match.start()) + 1
            if path == C and line == 33:
                continue
            s = span(path, match.start(), match.end())
            behavior = {'CUTE_HOST_DEVICE': 'config.hpp33–41：CUDA编译为host/device forceinline，否则inline。',
                        'CUTE_UNROLL': 'config.hpp49–59：编译环境决定pragma或空展开；不是并行调度/向量化保证。',
                        'CUTE_GCC_UNREACHABLE': 'config.hpp93–99：未预定义且__GNUC__时为__builtin_unreachable()，否则为空；位于两个return分支之后。'}[s['raw']]
            macro_refs = [PREFIX + owner_for(path, match.start())]
            expansion_variants = []
            macro_kind = 'host_device_attribute' if s['raw'] == 'CUTE_HOST_DEVICE' else 'compiler_unroll_pragma'
            if s['raw'] == 'CUTE_GCC_UNREACHABLE':
                macro_kind = 'conditional_builtin_or_empty_expansion'
                use = node('macro.unreachable_use', 'binding', 'CUTE_GCC_UNREACHABLE 使用点', path, line,
                           '两个return分支之后的词法宏出现；不宣称有可执行路径到达此位置。',
                           source_entity_kind='macro_use', source_expression=s['raw'], source_range=s)
                builtin = node('external.gnu_unreachable', 'external', '__builtin_unreachable()', None, None,
                               '编译器builtin边界；不是本snapshot的C++函数声明。此处只记录条件宏展开，不记录实际运行调用。')
                builtin.pop('path'); builtin.pop('line')
                node('macro.unreachable_empty', 'binding', 'CUTE_GCC_UNREACHABLE 空展开', CF, 97,
                     'config默认非GNU分支不产生表达式。', source_entity_kind='empty_macro_expansion', source_expression='')
                node('macro.unreachable_override', 'binding', '外部预定义 CUTE_GCC_UNREACHABLE', path, line,
                     'config.hpp以ifndef保护；若调用环境已定义该宏，则本文件不能决定展开文本。',
                     source_entity_kind='external_macro_override', source_expression='CUTE_GCC_UNREACHABLE',
                     dependency_parameters=['preprocessor environment: CUTE_GCC_UNREACHABLE'], resolution='symbolic')
                usage = edge('unreachable_macro_use', 'api.beta_zero_lambda', 'macro.unreachable_use', 'macro_uses', s,
                             '词法位于复数/非复数return之后；不等同实际执行')
                variants = [
                    ('gnu', 'external.gnu_unreachable', locate(CF, '__builtin_unreachable()', 95),
                     '!defined(CUTE_GCC_UNREACHABLE) && defined(__GNUC__)', 'source_proven'),
                    ('empty', 'macro.unreachable_empty', locate(CF, '#    define CUTE_GCC_UNREACHABLE', 97),
                     '!defined(CUTE_GCC_UNREACHABLE) && !defined(__GNUC__)', 'source_proven'),
                    ('override', 'macro.unreachable_override', locate(CF, '#if ! defined(CUTE_GCC_UNREACHABLE)', 93),
                     'defined(CUTE_GCC_UNREACHABLE) before config.hpp', 'symbolic'),
                ]
                macro_refs += [use['id'], usage['id']]
                for tag, target, defn, condition, resolution in variants:
                    ee = edge('unreachable_expansion_' + tag, 'macro.unreachable_use', target, 'expands_to', defn,
                              condition, resolution, evaluation='macro_expansion_only_not_runtime_call')
                    ee['evidence'] += [evidence(s), {'path': CF, 'start_line': 93, 'end_line': 99}]
                    macro_refs += [ee['id']]
                    expansion_variants.append({'condition': condition, 'target': PREFIX + target,
                                               'expanded_spelling': '__builtin_unreachable()' if tag == 'gnu' else '' if tag == 'empty' else None,
                                               'resolution': resolution, 'edge_id': ee['id']})
            inventories.append({'id': PREFIX + f'obligation.{Path(path).stem}.macro.{match.start()}',
                                'category': 'macro_use', 'syntax_kind': 'macro_identifier', 'source_range': s,
                                'status': 'source_reviewed', 'graph_refs': macro_refs, 'macro_kind': macro_kind,
                                'expansion_variants': expansion_variants,
                                'explanation': behavior, 'evidence': [{'path': CF, 'start_line': 33 if s['raw'] == 'CUTE_HOST_DEVICE' else 49 if s['raw'] == 'CUTE_UNROLL' else 85,
                                                                    'end_line': 41 if s['raw'] == 'CUTE_HOST_DEVICE' else 59 if s['raw'] == 'CUTE_UNROLL' else 99}]})
            counts['macro_use'] = counts.get('macro_use', 0) + 1

    # Resource effects are attached to the caller/expression with guards; no
    # API is claimed to perform a barrier or async completion.
    effects = [
        ('fill_reads_value', 'api.fill_prefer0', 'res.value', 'reads', F, 73, 'value', 'fallback，循环体执行；const&可能与目标存储别名'),
        ('fill_writes_tensor', 'api.fill_prefer0', 'res.tensor', 'writes', F, 73, 'tensor(i) = value', 'fallback；赋值的具体语义依赖reference与T'),
        ('axpby_reads_p', 'api.axpby_lvalue', 'res.p', 'reads', A, 88, 'p(i)', '每个被循环遍历的i；predicate可自带副作用'),
        ('axpby_reads_x', 'api.axpby_lvalue', 'res.x', 'reads', A, 89, 'isBetaZero ? alpha * x(i) : alpha * x(i) + beta * y(i)', 'p(i)成立；仅所选条件分支的x(i)求值；不是两次无条件读取'),
        ('axpby_reads_old_y', 'api.axpby_lvalue', 'res.y', 'reads', A, 89, 'beta * y(i)', 'p(i)成立且isBetaZero条件为false；仅源码明确的旧y右侧子表达式'),
        ('axpby_writes_y', 'api.axpby_lvalue', 'res.y', 'writes', A, 89, 'y(i) = (isBetaZero ? alpha * x(i) : alpha * x(i) + beta * y(i))', 'p(i)成立；false时没有这条赋值；代理引用内部副作用仍依赖实例'),
    ]
    for id, owner, target, rel, path, line, text, condition in effects:
        edge(id, owner, target, rel, locate(path, text, line), condition, 'symbolic')
    for id, text, condition, occurrence in [
        ('axpby_alpha_zero', 'alpha', 'p(i)成立且isBetaZero成立；true arm', 0),
        ('axpby_alpha_nonzero', 'alpha', 'p(i)成立且isBetaZero不成立；false arm', 1),
    ]:
        edge(id, 'api.axpby_lvalue', 'res.alpha', 'reads', locate(A, text, 89, occurrence=occurrence), condition, 'symbolic')
    edge('axpby_beta_nonzero', 'api.axpby_lvalue', 'res.beta', 'reads', locate(A, 'beta', 89), 'p(i)成立且isBetaZero不成立；false arm', 'symbolic')
    edge('beta_real_resource', 'api.beta_zero_lambda', 'res.beta', 'reads', locate(A, 'beta.real()', 77), 'complex编译期分支；真实member行为依赖Beta', 'symbolic')
    edge('beta_imag_resource', 'api.beta_zero_lambda', 'res.beta', 'reads', locate(A, 'beta.imag()', 77), 'complex分支且该右操作数实际求值；&&语义保留模板依赖', 'symbolic')
    edge('beta_scalar_resource', 'api.beta_zero_lambda', 'res.beta', 'reads', locate(A, 'beta == Int<0>{}', 80), '!is_complex<Beta>::value编译期分支', 'symbolic')
    edge('beta_result_initialize', 'api.axpby_lvalue', 'local.isBetaZero', 'writes', span(A, 2851, 3076), 'auto对象初始化；lambda先求值一次', 'symbolic')
    edge('beta_result_read', 'api.axpby_lvalue', 'local.isBetaZero', 'reads', locate(A, 'isBetaZero', 89), 'p(i)成立后用作条件表达式的条件', 'symbolic')
    for owner, target, path, line in [('api.fill_prefer0', 'local.fill_i', F, 72), ('api.axpby_lvalue', 'local.axpby_i', A, 87)]:
        edge(target + '_initialize', owner, target, 'writes', locate(path, 'int i = 0;', line), '每次进入此for语句时初始化独立int变量')
        edge(target + '_increment', owner, target, 'writes', locate(path, '++i', line), '本轮循环体执行后；int内建自增，迭代域需可表示')
    edge('beta_before_loop', 'api.beta_zero_lambda', 'api.axpby_lvalue', 'precedes', span(A, lambda_call.start_byte, lambda_call.end_byte),
         'caller=axpby_lvalue；只表示75–84的初始化先于87–91循环，不是lambda内部等待整个axpby',
         caller=PREFIX + 'api.axpby_lvalue', successor_range=locate(A, 'for (int i = 0; i < size(x); ++i)', 87))

    node_ids = {n['id'] for n in nodes}; edge_ids = {e['id'] for e in edges}
    assert len(node_ids) == len(nodes) and len(edge_ids) == len(edges)
    for e in edges:
        assert e['source'] in node_ids and e['target'] in node_ids, e['id']
        if 'extra_evidence' in e:
            e['evidence'] += e.pop('extra_evidence')
    def view(id, title, select):
        chosen = [e for e in edges if select(e)]
        return {'id': PREFIX + 'view.' + id, 'title': title, 'kind': 'relationships',
                'node_ids': list(dict.fromkeys(x for e in chosen for x in (e['source'], e['target']))),
                'edge_ids': [e['id'] for e in chosen]}
    views = [view('clear', 'clear：两个重载、T{}与fill入口', lambda e: any(x in e['source'] for x in ('clear', 'local.T'))),
             view('fill_dispatch', 'fill：转发、SFINAE不求值与两个优先级实现', lambda e: e['source'].startswith(PREFIX + 'api.fill') or 'prefer' in e['id']),
             view('axpby', 'axpby：完整分支与不同元素调用点', lambda e: e['source'].startswith(PREFIX + 'api.axpby') or e['source'] == PREFIX + 'api.beta_zero_lambda'),
             view('resources', '资源与源程序顺序：没有隐含同步或独立缓冲', lambda e: e['relation'] in ('reads', 'writes', 'precedes')),
             view('dependencies', '模板依赖：条件实例、默认predicate与必要类型', lambda e: e['relation'] in ('instance_of', 'aliases', 'specializes', 'template_binds') or e['source'].startswith(PREFIX + 'type.'))]
    views.append(view('preprocessor', '预处理：不可达宏的条件展开，不是运行calls', lambda e: e['relation'] in ('macro_uses', 'expands_to')))
    contracts = [
        {'id': PREFIX + 'contract.fill', 'resource': PREFIX + 'res.tensor',
         'participants': [PREFIX + x for x in ('api.clear_lvalue', 'api.fill_lvalue', 'api.fill_prefer1', 'api.fill_prefer0')],
         'events': [PREFIX + 'edge.' + x for x in ('clear_fill', 'fill_dispatch_prefer1', 'fill_dispatch_prefer0', 'fill_writes_tensor')],
         'preconditions': ['元素类型T{}必须可形成，才可实例化clear；value-initialization不保证任意用户类型的数值零/字节零。',
                           'fallback的每个有效i需要tensor(i)=value合法；int循环必须能表示实际迭代域。',
                           '优先路径的ADL fill必须能按实际iterator/value调用；不能把decltype可形成误当函数体、运行效果已经验证。',
                           '高优先重载的返回型是decltype(...)而body未return；自定义非void data-fill的返回行为需另核，不能仅凭表达式可形成保证。',
                           '存储生命周期、布局可访问性、并发访问同步由调用方保证；本header不分配、不发射barrier。'],
         'release_condition': '本函数源码只顺序调用选中实现并返回。fallback语句完成不额外代表设备全局同步；自定义ADL fill的异步/副作用契约须由其实际实现给出。',
         'evidence': [evidence(locate(C, 'fill(tensor, T{});', 61)), evidence(span(F, 2593, 2680))]},
        {'id': PREFIX + 'contract.axpby', 'resource': PREFIX + 'res.y',
         'participants': [PREFIX + x for x in ('api.axpby_lvalue', 'api.beta_zero_lambda', 'dep.p_call', 'res.x', 'res.y')],
         'events': [PREFIX + 'edge.' + x for x in ('axpby_beta_once', 'axpby_predicate', 'axpby_x_beta_zero', 'axpby_x_beta_nonzero', 'axpby_y_old', 'axpby_writes_y')],
         'preconditions': ['alpha/beta/element/reference运算及predicate条件转换必须可实例化。',
                           '对p(i)成立的i，x/y和predicate访问必须有效；本源码没有shape相等断言或长度检查。',
                           'x/y/value可潜在别名，本源码没有restrict或先快照输入；不能承诺任意别名下数学向量语义不变。',
                           '循环int i的迭代域须可表示；调用方负责正确存储生命周期和并发同步。'],
         'release_condition': '循环结束后函数返回；只承诺源码分支内的求值关系，不承诺某种机器指令、内存空间或外部异步完成。',
         'sequence': ['75–84先计算isBetaZero一次。', '87按逻辑size(x)检查循环；88每个i调用p(i)。',
                      'p(i)为false时不进入89赋值。为true时条件表达式只选一个arm。',
                      '零beta arm源码不求值旧y右侧子表达式；非零arm求值beta*y(i)。不外推用户运算/代理引用内部行为。',
                      '源循环++i后进入下一次条件；CUTE_UNROLL仅编译提示。不额外规定同一算术表达式的独立操作数求值顺序。'],
         'evidence': [evidence(span(A, 2851, 3228))]},
    ]
    issues = [
        {'id': PREFIX + 'issue.template_closure', 'kind': 'template_dependent_boundary', 'status': 'symbolic_not_parse_failure',
         'description': '没有选择Engine/Layout/Alpha/Beta/PrdTensor：ADL目标、scalar/reference运算、用户特化与存储行为保持原表达式和依赖参数。全文件源码义务覆盖不等于所有实例语义闭合。'},
        {'id': PREFIX + 'issue.dependency_bodies', 'kind': 'bounded_dependency_perimeter', 'status': 'explicitly_outside_full_file_denominator',
         'description': 'dependency目标已连到真实声明；tensor_impl/prefer/functional/integral_constant/complex/config只核本模块所需证据，不宣称这些依赖header全文件已覆盖。'},
        {'id': PREFIX + 'issue.validation', 'kind': 'verification_boundary', 'status': 'not_claimed',
         'description': '本草稿是完整源范围与关系审查；独立checker验证位置/调用枚举/覆盖义务/ledger身份。不包含编译实例化、GPU执行、数值正确性或性能实测。'},
    ]
    # Parameter declarations stay individually addressable, including both
    # unnamed prefer parameters and the two defaulted predicate parameters.
    parameter_obligations = []
    for id, *_ in funcs:
        n = local_to_id[id]
        for group in n['template_parameters']:
            for p in group['parameters']:
                parameter_obligations.append({'owner': n['id'], 'category': 'template_parameter', **p})
        for p in n['parameters']:
            parameter_obligations.append({'owner': n['id'], 'category': 'function_parameter', **p})
    return {'schema_version': 1, 'module_id': 'cute_elementwise', 'part': 'complete_three_headers',
            'snapshot_commit': scope['commit'], 'status': 'draft_for_independent_review',
            'scope': {'kind': 'complete_physical_headers_not_configuration_slice', 'paths': list(SCOPE),
                      'files': [{'path': p, 'sha256': hashlib.sha256(source(p)).hexdigest(), 'bytes': len(source(p)),
                                 'line_count': len(source(p).splitlines())} for p in SCOPE],
                      'dependency_headers_are_full_scope': False, 'all_template_instances_resolved': False},
            'nodes': nodes, 'edges': edges, 'views': views, 'contracts': contracts, 'issues': issues,
            'coverage': {'obligations': inventories, 'parameter_obligations': parameter_obligations,
                         'counts': counts, 'unclassified_source_obligations': [],
                         'semantic_completion': 'all_physical_constructs_reviewed_template_dependent_semantics_open',
                         'non_executable_regions': '三个文件1–30版权/许可证全文保留在snapshot；注释/空行无API或运行调用，pragma/include/namespace另有独立义务。',
                         'parser_only_masks': ['CUTE_HOST_DEVICE/CUTE_UNROLL按字节等长空白，仅用于独立语法枚举。',
                                               'axpby两个p = {}默认initializer按字节等长空白；原文和默认值义务另行逐个核对，任何函数body不被移除。']},
            'runtime_execution_claimed': False, 'global_completion_claimed': False}


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
