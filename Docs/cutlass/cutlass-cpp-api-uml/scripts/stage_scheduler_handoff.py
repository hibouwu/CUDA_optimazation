#!/usr/bin/env python3
"""Stage source-selected Scheduler interfaces and exact Kernel callsites.

This contribution is a handoff slice, not a claim that the entire Scheduler
file or every helper call is extracted. The regular atlas builder resolves the
explicit byte selectors against the existing declaration ledger.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHED = 'include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp'
KERNEL = 'include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp'
Q = 'cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100::'

DEFINITIONS = [
    ('work_info', 'WorkTileInfo', 76, 76, 'alias', '当前调度器沿用底层调度器的工作描述类型；它不是 K 分段描述。'),
    ('initial', 'initial_work_tile_info', 366, 369, 'method', '用本CTA的初始blockIdx经swizzle/raster转换得到首个任务，初始valid=true；不是逐元素边界谓词。'),
    ('coordinate', 'work_tile_to_cta_coord', 373, 375, 'method', '把M/N/L任务索引组成CTA坐标，K位置使用占位符；不能把该占位符当作运行时K起点。'),
    ('fetch', 'fetch_next_work', 454, 460, 'method', '等待当前CLC响应槽，读取并转换下一任务，再释放响应槽；返回的true通知调用方推进CLC消费者状态。'),
    ('fetch_compat', 'fetch_next_work', 660, 662, 'method', '单参数兼容重载原样返回任务与true，不读取CLC；不能与三参数重载合并。'),
    ('k_iterator', 'get_k_tile_iterator', 479, 482, 'method', '按问题K形状与CTA Tile K构造完整K遍历；仅高秩K模式使用编译期的迭代次序分支。'),
    ('k_count', 'get_work_k_tile_count', 499, 502, 'method', '此调度器每个输出任务覆盖完整K归约，计数为size(ceil_div(K,TileK))，并不按WorkTileInfo切分K。'),
    ('k_start', 'get_work_k_tile_start', 509, 511, 'method', '兼容接口固定返回K tile 0，当前Kernel主线未调用它。'),
    ('epilogue', 'compute_epilogue', 524, 526, 'method', '当前使用的单参数重载始终允许该任务做Epilogue；它自身不检查任务是否有效。'),
    ('epilogue_compat', 'compute_epilogue', 518, 520, 'method', 'Epilogue输入搬运分支使用的双参数重载同样返回true；与后处理使用的单参数重载分别保留。'),
    ('requires_fixup', 'requires_fixup', 532, 534, 'method', '此完整K任务调度不需要跨split归约，返回false；不能推广为Stream-K的规则。'),
    ('fixup_state', 'fixup', 545, 562, 'method', '当前Kernel调用的六参数模板重载原样返回accumulator消费者状态，不执行归约、拷贝或额外等待。'),
    ('continue', 'continue_current_work', 569, 571, 'method', '完整输出任务处理后不继续沿用该工作描述，返回false；当前主线不是借一个描述处理多个K分段。'),
]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def span(path, first, last, declaration=False):
    lines = (ROOT / 'snapshot' / path).read_text().splitlines(keepends=True)
    text = ''.join(lines[first - 1:last])
    start = sum(len(s.encode()) for s in lines[:first - 1])
    if declaration:
        whitespace = len(text) - len(text.lstrip())
        start += len(text[:whitespace].encode()); text = text.lstrip()
        if '{' in text:
            text = text.rsplit('{', 1)[0]
        # The canonical signature range ends at the opening body brace and
        # includes its preceding space; raw_signature is separately rstripped.
        if '{' not in ''.join(lines[first - 1:last]):
            text = text.rstrip()
    return {'path': path, 'start_line': first, 'end_line': last,
            'start_byte': start, 'end_byte': start + len(text.encode()),
            'source_text': text.rstrip() if declaration else text}


def call_at(line, expression_start, path=KERNEL):
    raw = (ROOT / 'snapshot' / path).read_text()
    lines = raw.splitlines(keepends=True)
    offset = sum(len(s) for s in lines[:line - 1])
    start = raw.index(expression_start, offset)
    if raw[:start].count('\n') + 1 != line:
        raise ValueError('Callsite moved from reviewed line')
    begin = raw.index('(', start)
    depth = 1; end = begin + 1
    while depth:
        if raw[end] == '(': depth += 1
        elif raw[end] == ')': depth -= 1
        end += 1
    return {'path': path, 'start_line': line, 'end_line': raw[:end].count('\n') + 1,
            'start_byte': len(raw[:start].encode()), 'end_byte': len(raw[:end].encode()),
            'source_expression': raw[start:end]}


def build():
    nodes = []
    for identifier, name, first, last, kind, notes in DEFINITIONS:
        declaration = span(SCHED, first, last, True)
        nodes.append({'id': 'scheduler.' + identifier, 'kind': 'type' if kind == 'alias' else 'api',
                      'name': name + ('(WorkTileInfo)' if identifier == 'fetch_compat' else '(WorkTileInfo, Params)' if identifier == 'epilogue_compat' else ''),
                      'qualified_name': Q + name, 'path': SCHED, 'line': last,
                      'role': 'selected_scheduler_handoff', 'notes': notes,
                      'source_selector': {'path': SCHED, 'kind': kind, 'qualified_name': Q + name,
                                          'signature_range': {k:v for k,v in declaration.items() if k != 'source_text'},
                                          'signature_sha256': hashlib.sha256(declaration['source_text'].encode()).hexdigest()}})
    edges = []
    calls = [
        (607, 'initial', 'scheduler.initial_work_tile_info', '共同准备首任务，使用当前blockIdx；selected Scheduler为Sm100动态persistent'),
        (608, 'coordinate', 'scheduler.work_tile_to_cta_coord', '共同准备首任务的CTA坐标'),
        (626, 'k_iterator', 'scheduler.get_k_tile_iterator', 'is_participant.main_load分支，处理当前任务'),
        (627, 'k_count', 'TileScheduler::get_work_k_tile_count', 'is_participant.main_load分支，计算当前任务完整K计数'),
        (664, 'fetch', 'scheduler.fetch_next_work', 'is_participant.main_load；当前输入搬运后取得下一任务'),
        (670, 'coordinate', 'scheduler.work_tile_to_cta_coord', 'main_load中已赋值work_tile_info=next_work_tile_info后更新坐标'),
        (701, 'fetch', 'scheduler.fetch_next_work', 'is_participant.sched && IsSchedDynamicPersistent；读取新CLC响应'),
        (738, 'k_count', 'TileScheduler::get_work_k_tile_count', 'is_participant.mma；next_work取得前为当前任务计算完整K计数'),
        (741, 'fetch', 'scheduler.fetch_next_work', 'is_participant.mma；预先取得next，当前任务尚未结束'),
        (774, 'coordinate', 'scheduler.work_tile_to_cta_coord', 'mma分支当前计算完成、赋值next后更新坐标'),
        (816, 'epilogue_compat', 'TileScheduler::compute_epilogue', 'is_participant.epi_load；双参数重载决定当前任务是否需要Epilogue输入处理'),
        (819, 'fetch', 'scheduler.fetch_next_work', 'is_participant.epi_load；随后提前赋值work_tile_info=next，但仍保留旧坐标和旧compute_epilogue判定处理当前输入'),
        (854, 'coordinate', 'scheduler.work_tile_to_cta_coord', 'epi_load的work_tile_info已提前设为next；旧坐标的本轮输入处理完后，此处才更新下一轮坐标'),
        (877, 'fetch', 'scheduler.fetch_next_work', 'is_participant.epilogue；取得next不跳过current的store'),
        (910, 'epilogue', 'scheduler.compute_epilogue', 'is_participant.epilogue；单参数重载返回true，任务有效性由外层循环控制'),
        (932, 'coordinate', 'scheduler.work_tile_to_cta_coord', 'epilogue当前store完成并采用next后更新坐标'),
    ]
    for line, target, expression, condition in calls:
        point = call_at(line, expression)
        if line >= 626:
            condition += '；位于该角色do-while的当前迭代，尾部用更新后的work_tile_info.is_valid()决定继续。'
        edges.append({'id': 'scheduler.call.' + target + '.' + str(line),
                      'source': 'host.kernel.operator', 'target': 'scheduler.' + target,
                      'relation': 'calls', 'resolution': 'configured', 'condition': condition,
                      'source_expression': point['source_expression'], 'callsite': point,
                      'evidence': [span(KERNEL, point['start_line'], point['end_line'])]})
    edges.append({'id':'scheduler.type.work_info', 'source':'host.scheduler', 'target':'scheduler.work_info',
                  'relation':'type_uses', 'resolution':'configured', 'source_expression':'using WorkTileInfo = typename UnderlyingTileScheduler::WorkTileInfo;',
                  'condition':'Sm100 Scheduler使用底层WorkTileInfo类型；别名不是运行时调用', 'evidence':[span(SCHED,60,76)]})
    subject_sets = [('entry', '工作描述、坐标与完整K', {'initial','coordinate','k_iterator','k_count'}),
                    ('next', '当前任务与下一任务：各参与角色独立取响应', {'fetch','coordinate'}),
                    ('output', '当前任务的Epilogue与无跨split归约', {'fixup_state','epilogue','epilogue_compat'})]
    views=[]
    for key,title,targets in subject_sets:
        selected=[e for e in edges if e['target'] in {'scheduler.'+s for s in targets}]
        if key == 'entry':
            selected.append(next(e for e in edges if e['id'] == 'scheduler.type.work_info'))
        if key == 'output':
            # Reuse the original callsite identity instead of duplicating the
            # same physical call under the Kernel's other graph-role alias.
            selected.append({'id':'contract.edge.kernel_fixup','source':'contract.api.kernel','target':'scheduler.fixup_state'})
        views.append({'id':'scheduler.handoff.'+key,'title':title,
                      'node_ids':list(dict.fromkeys(n for e in selected for n in (e['source'],e['target']))),
                      'edge_ids':[e['id']for e in selected]})
    part={'module_id':'dense_fp16_2sm_sm110a','part':'scheduler_handoff',
          'nodes':nodes,'edges':edges,'views':views,
          'issues':[{'id':'scheduler.handoff.remaining',
                     'description':'此贡献只核对已列接口声明与Kernel直接交接；完整swizzle/raster、CLC发布/完成/无效响应和各方法内部调用仍未全量提取，不表示Scheduler文件或调度唯一性已验收。'}],
          'contracts':[{'id':'scheduler.handoff.full_k','kind':'source_contract','title':'选中Scheduler的任务单位与前后任务交接',
                        'participants':['host.kernel.operator','host.scheduler','scheduler.fetch','scheduler.k_count','scheduler.fixup_state','scheduler.epilogue'],
                        'claims':['选中Scheduler的每个输出任务覆盖完整K空间，K起点为0；基线K=128、CTA TileK=64，对应2个K tile。',
                                  'fetch_next_work三参数重载返回新任务和true；调用方按increment_pipe推进CLC状态，不把它当作累加器状态。',
                                  'MMA/Epilogue预取next之后仍完成current，再赋值current=next并检查有效性；next无效不等于立即跳过current。',
                                  'Epilogue输入搬运分支不同：它提前把work_tile_info设为next，但仍用旧cta_coord_mnkl和旧compute_epilogue判定处理本轮，最后才更新坐标。',
                                  '本选择的fixup不归约也不等待，仅原样返回accumulator消费者状态；compute_epilogue本身不是任务或矩阵边界检查。'],
                        'evidence':[span(SCHED,454,474),span(SCHED,499,535),span(SCHED,545,572),span(KERNEL,737,775),span(KERNEL,876,934)]}]}
    dump(ROOT/'data/modules/dense_fp16/scheduler_handoff.json',part)
    print(json.dumps({'nodes':len(nodes),'edges':len(edges),'views':len(views)}))


if __name__=='__main__':build()
