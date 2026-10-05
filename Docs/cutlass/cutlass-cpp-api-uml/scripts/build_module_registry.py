#!/usr/bin/env python3
"""Single-owner file registry for the fixed scope; never an API coverage claim."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '8f50b052e1099fb982392a622caab69b97b63128'
EXPECTED_LIBRARY_COUNTS = {'cutlass': 712, 'cute': 112}

def selectors(*, exact=(), prefix=(), children=()):
    return {'exact': list(exact), 'prefix': list(prefix), 'immediate_children': list(children)}

SYNC = (
    'cutlass/barrier.h', 'cutlass/semaphore.h', 'cutlass/arch/barrier.h',
    'cutlass/arch/grid_dependency_control.h', 'cutlass/detail/cluster.hpp',
    'cute/arch/cluster_sm90.hpp', 'cute/arch/cluster_sm100.hpp',
    'cute/arch/tmem_allocator_sm100.hpp',
)
RUNTIME = tuple('cutlass/'+name for name in (
    'device_kernel.h', 'kernel_launch.h', 'cluster_launch.hpp', 'cuda_host_adapter.hpp',
    'kernel_hardware_info.h', 'kernel_hardware_info.hpp', 'workspace.h',
))
FOUNDATION_DETAIL = tuple('cutlass/detail/'+name for name in (
    'helper_macros.hpp', 'dependent_false.hpp', 'layout.hpp',
))
MMA_ALGORITHM = ('cute/algorithm/gemm.hpp', 'cute/algorithm/cooperative_gemm.hpp')
COPY_ALGORITHM = ('cute/algorithm/copy.hpp', 'cute/algorithm/cooperative_copy.hpp', 'cute/algorithm/prefetch.hpp')

def rule(number, module_id, title, role, include, exclude=None):
    return {'rule_id': f'R{number:02d}', 'module_id': module_id, 'title': title,
            'role': role, 'include': include, 'exclude': exclude or selectors()}

# Selectors are relative to include/. Independent matches must be unique; ordering
# does not hide overlap. Exact exceptions are subtracted from broader selectors.
RULES = [
    rule(1, 'sync_resource', 'Pipeline、Barrier 与资源生命周期',
         '生产/消费stage、barrier、grid/cluster依赖和TMEM分配；底层提交与完成事件分开。',
         selectors(exact=SYNC, prefix=('cutlass/pipeline/',))),
    rule(2, 'runtime_bridge', 'Host 参数与 CUDA 启动桥接',
         '通用device_kernel、kernel/cluster launch、Host adapter、hardware info与workspace；不专属于GEMM。',
         selectors(exact=RUNTIME)),
    rule(3, 'arch_primitives', '架构原语与启用条件',
         'CUTLASS架构操作、配置、日志与CuTe通用架构设施；路径中的SM名称不是API代际或运行目标证明。',
         selectors(prefix=('cutlass/arch/',), exact=('cute/arch/config.hpp','cute/arch/util.hpp','cute/arch/simd_sm100.hpp')),
         selectors(exact=SYNC)),
    rule(4, 'collective_support', '共享 Collective 类型与布局支持',
         '跨Mainloop、卷积、MoE、block-scaled等使用的detail声明，仍是范围内源码义务。',
         selectors(prefix=('cutlass/detail/',)), selectors(exact=FOUNDATION_DETAIL+SYNC)),
    rule(5, 'gemm_interface', 'GEMM 问题与调度类型契约',
         'gemm根目录的枚举、问题形状与dispatch policy；与实际Kernel实现通过绑定连接。',
         selectors(children=('cutlass/gemm',))),
    rule(6, 'gemm_device', 'GEMM Device 接口',
         'Host生命周期、各adapter及所有重载/特化；同一文件内2.x/3.x仍分别审查。',
         selectors(prefix=('cutlass/gemm/device/',))),
    rule(7, 'gemm_kernel', 'GEMM Kernel 与 Tile Scheduler',
         'Kernel组合、线程角色和tile工作分配；目录内卷积等共享重载仍保留。',
         selectors(prefix=('cutlass/gemm/kernel/',))),
    rule(8, 'gemm_collective', 'GEMM Collective 与 Builder',
         'Mainloop、builder入口和所有.inl偏特化；Auto类型选择不表示运行调用。',
         selectors(prefix=('cutlass/gemm/collective/',))),
    rule(9, 'gemm_tiles', 'GEMM Thread、Warp 与 Threadblock 组件',
         '组件粒度归档，不将threadblock目录或.h扩展名直接解释为legacy代际。',
         selectors(prefix=('cutlass/gemm/thread/','cutlass/gemm/warp/','cutlass/gemm/threadblock/'))),
    rule(10, 'epilogue_collective', 'Epilogue Collective 与 Builder',
         '输出Collective、存储/流水线和builder选择；通过FusionCallbacks接下层操作。',
         selectors(prefix=('cutlass/epilogue/collective/',))),
    rule(11, 'epilogue_fusion', 'Epilogue Fusion 与 EVT',
         '融合操作、callback与树结构；不把单一LinearCombination推广为整个融合模块。',
         selectors(prefix=('cutlass/epilogue/fusion/',))),
    rule(12, 'epilogue_tiles', 'Epilogue Thread、Warp 与 Threadblock 组件',
         'thread输出操作、warp/threadblock访存与visitor组件；保留所有年代/条件实现。',
         selectors(children=('cutlass/epilogue',), prefix=('cutlass/epilogue/thread/','cutlass/epilogue/warp/','cutlass/epilogue/threadblock/'))),
    rule(13, 'cute_mma', 'CuTe MMA 分派、Atom、Traits 与指令包装',
         'gemm/cooperative_gemm、MMA Atom/Traits和MMA架构包装，逐重载和配置连接。',
         selectors(exact=MMA_ALGORITHM, prefix=('cute/atom/mma','cute/arch/mma'))),
    rule(14, 'cute_copy', 'CuTe Copy、TMA 与 TMEM 搬运',
         'copy/cooperative_copy/prefetch、Copy Atom/Traits和架构copy包装；with绑定与copy执行分开。',
         selectors(exact=COPY_ALGORITHM, prefix=('cute/atom/copy','cute/arch/copy'))),
    rule(15, 'cute_core', 'CuTe Tensor、Layout 与基础设施',
         'Tensor/Layout/Pointer、容器、数值、通用算法、partitioner和print/debug接口。',
         selectors(children=('cute',), exact=('cute/atom/partitioner.hpp',), prefix=('cute/algorithm/','cute/container/','cute/numeric/','cute/util/')),
         selectors(exact=MMA_ALGORITHM+COPY_ALGORITHM)),
    rule(16, 'convolution', 'Convolution',
         '卷积问题、Device/Kernel/Collective及tile组件；与GEMM共享的端点保持原文件归属。',
         selectors(prefix=('cutlass/conv/',))),
    rule(17, 'transform', 'Transform',
         '独立transform adapter/kernel/collective以及访问器/重排组件，不归并为GEMM helper。',
         selectors(prefix=('cutlass/transform/',))),
    rule(18, 'reduction', 'Reduction',
         '独立规约、Split-K workspace后处理及其Device/Kernel/thread接口。',
         selectors(prefix=('cutlass/reduction/',))),
    rule(19, 'distributed', 'Experimental Distributed',
         '分布式GEMM包装、CUDA Graph与跨设备barrier；experimental仅为目录标签。',
         selectors(prefix=('cutlass/experimental/distributed/',))),
    rule(20, 'cutlass_foundation', 'CUTLASS 数值、容器、布局与兼容基础',
         '顶层数值/容器/坐标/视图、layout、platform和基础宏；冷门及兼容分支不排除。',
         selectors(children=('cutlass',), exact=FOUNDATION_DETAIL, prefix=('cutlass/layout/','cutlass/platform/','cutlass/thread/')),
         selectors(exact=RUNTIME+SYNC)),
]

def selector_matches(path, selector):
    return (path in selector['exact'] or
            any(path.startswith(p) for p in selector['prefix']) or
            any(path.startswith(p+'/') and '/' not in path[len(p)+1:]
                for p in selector['immediate_children']))

def classify(path, rules=RULES):
    p=PurePosixPath(path)
    if not path.startswith('include/') or p.is_absolute() or '..' in p.parts or str(p)!=path:
        raise ValueError(f'Invalid source path: {path}')
    relative=path[len('include/'):]
    matches=[r for r in rules if selector_matches(relative,r['include']) and not selector_matches(relative,r['exclude'])]
    if len(matches)!=1:
        raise ValueError(f'Expected one owner for {path}; matched {[r["rule_id"] for r in matches]}')
    return matches[0]

def build_registry(scope, *, scope_sha256, generator_sha256, strict_snapshot=True):
    files=scope['files']; paths=[f['path'] for f in files]
    if len(paths)!=len(set(paths)):raise ValueError('Duplicate path in source manifest')
    if scope.get('file_count',len(files))!=len(files):raise ValueError('Manifest file_count does not match records')
    libraries=Counter(PurePosixPath(p).parts[1] for p in paths)
    if strict_snapshot and (scope['commit']!=COMMIT or dict(libraries)!=EXPECTED_LIBRARY_COUNTS):
        raise ValueError('Registry must retain the fixed 824-file / 712+112 source scope')
    records=[]; counts=Counter(); extensions=Counter(); inl_modules=Counter()
    for entry in files:
        path=entry['path']; chosen=classify(path); counts[chosen['module_id']]+=1
        extension=PurePosixPath(path).suffix;extensions[extension]+=1
        if extension=='.inl':inl_modules[chosen['module_id']]+=1
        records.append({'path':path,'primary_module_id':chosen['module_id'],'rule_id':chosen['rule_id'],
            'source_sha256':entry['sha256'],'git_blob':entry.get('git_blob'),'git_mode':entry.get('git_mode'),
            'navigation':{'library':PurePosixPath(path).parts[1],
                'directory_domain':entry.get('navigation_domain'),
                'relative_directory':str(PurePosixPath(path).parent), 'extension':extension,
                'architecture_name_hints':entry.get('architecture_name_hints',[]),
                'api_generation':entry.get('api_generation','pending_entity_review'),
                'version_reference':'registry.library_version_macros',
                'classification_basis':'source_manifest_and_path_rule_not_api_generation'},
            'coverage':{'file_ownership':'assigned','declarations':'not_asserted','relationships':'not_asserted','protocols':'not_asserted'}})
    modules=[{'module_id':r['module_id'],'title':r['title'],'role':r['role'],'rule_ids':[r['rule_id']],
        'primary_file_count':counts[r['module_id']],
        'status':'inventory_assigned_not_api_coverage',
        'layer':'source_ownership_module'} for r in RULES]
    assert len(modules)==len({m['module_id'] for m in modules})
    return {'schema_version':1,'registry_kind':'single_primary_file_ownership',
        'commit':scope['commit'],'scope_path':'data/scope.json','scope_sha256':scope_sha256,
        'library_version_macros':scope.get('library_version_macros'),
        'api_reading_mainline':scope.get('api_reading_mainline'),
        'policy':{'path_base':'snapshot/; selectors omit include/',
            'matching':'exactly one independent rule match; unmatched and overlap are errors',
            'file_inventory_complete':True,'api_coverage_claimed':False,'relationship_coverage_claimed':False,
            'configuration_cases_are_not_file_owners':True,
            'directory_version_architecture_are_navigation_only':True,
            'cross_module_edges':'owned by source occurrence/callsite file; target keeps its canonical identity and primary file owner',
            'macro_provenance':'macro definition file and generated occurrence invocation file keep distinct provenance and ownership',
            'source':'Only fixed scope.json records; no glob, extension exclusion, reachability filter or ignore-aware discovery'},
        'counts':{'files':len(records),'modules':len(modules),'files_by_library':dict(sorted(libraries.items())),
            'files_by_module':dict(sorted(counts.items())),'files_by_extension':dict(sorted(extensions.items())),
            'inl_by_module':dict(sorted(inl_modules.items())),'unassigned_files':0,'multiply_assigned_files':0},
        'modules':modules,'rules':RULES,'files':sorted(records,key=lambda x:x['path']),
        'generator':{'path':'scripts/build_module_registry.py','sha256':generator_sha256}}

def sha(data):return hashlib.sha256(data).hexdigest()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',type=Path,default=ROOT/'data/scope.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/module-registry.json')
    args=parser.parse_args(argv)
    source=args.scope.read_bytes();code=Path(__file__).read_bytes()
    data=build_registry(json.loads(source),scope_sha256=sha(source),generator_sha256=sha(code))
    if args.scope.read_bytes()!=source or Path(__file__).read_bytes()!=code:
        raise RuntimeError('Inputs changed during generation; output not published')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=args.output.parent,prefix='.module-registry-',suffix='.tmp',delete=False) as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2);stream.write('\n');temp=stream.name
    os.replace(temp,args.output)
    print(json.dumps({'output':str(args.output),'counts':data['counts'],'api_coverage_claimed':False},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
