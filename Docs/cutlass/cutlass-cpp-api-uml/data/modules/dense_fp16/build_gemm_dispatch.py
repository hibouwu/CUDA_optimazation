#!/usr/bin/env python3
"""Bounded, compiled Dense gemm overload chain; no edited CUTLASS headers."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parents[2]
RUN = HERE/'probe/dispatch-run-ufh3cn5e'
GEMM = 'include/cute/algorithm/gemm.hpp'
ATOM = 'include/cute/atom/mma_atom.hpp'
TRAITS = 'include/cute/atom/mma_traits_sm100.hpp'
COLLECTIVE = 'include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp'
KERNEL = 'include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp'
LOG = str((RUN/'host_type_dump.stdout.txt').relative_to(ATLAS))
DWARF = str((RUN/'readelf_debug_info.stdout.txt').relative_to(ATLAS))
PROBE = str((RUN/'input.cu').relative_to(ATLAS))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def local(path): return ATLAS/'snapshot'/path if path.startswith('include/') else ATLAS/path
def evidence(path, start, end=None):
    end = start if end is None else end
    lines = local(path).read_text().splitlines()
    assert 1 <= start <= end <= len(lines), (path,start,end)
    return {'path':path,'start_line':start,'end_line':end,'quote':'\n'.join(lines[start-1:end])}
def atomic_json(path, data):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.'+path.stem+'-', delete=False) as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2);stream.write('\n'); temp=stream.name
    os.replace(temp,path)

def debug_index():
    """Resolve DWARF inline origins, not merely a grep for unrelated line markers."""
    ptx = (RUN/'probe.ptx').read_text()
    files = {int(a): b for a,b in re.findall(r'\.file\s+(\d+)\s+"([^"]+)"', ptx)}
    records={}; stack={}; current=None
    for number,line in enumerate((RUN/'readelf_debug_info.stdout.txt').read_text().splitlines(),1):
        header=re.match(r'\s*<(\d+)><([0-9a-f]+)>: Abbrev Number: (\d+)(?: \((\w+)\))?',line)
        if header:
            depth,offset,abbrev,tag=header.groups();depth=int(depth)
            for key in list(stack):
                if key>=depth: del stack[key]
            current=None
            if int(abbrev):
                current={'offset':offset,'tag':tag,'depth':depth,'parent':stack.get(depth-1),'start_line':number,'end_line':number,'attrs':{}}
                records[offset]=current;stack[depth]=offset
        elif current:
            attribute=re.match(r'\s*<[0-9a-f]+>\s+(DW_AT_\w+)\s*:\s*(.*)',line)
            if attribute: current['attrs'][attribute[1]]=attribute[2];current['end_line']=number
    def origin(record):
        while 'DW_AT_abstract_origin' in record['attrs']:
            record=records[re.search(r'<0x([0-9a-f]+)>',record['attrs']['DW_AT_abstract_origin'])[1]]
        return record
    def endpoint(record):
        record=origin(record);a=record['attrs']
        if 'DW_AT_decl_file' not in a or 'DW_AT_decl_line' not in a:return None
        path=files[int(a['DW_AT_decl_file'])]
        if '/snapshot/' in path:path=path.split('/snapshot/',1)[1]
        else:
            try:path=str(Path(path).relative_to(ATLAS))
            except ValueError:pass
        linkage=a.get('DW_AT_MIPS_linkage_name')
        return {'path':path,'line':int(a['DW_AT_decl_line']),'name':a.get('DW_AT_name'),
                'linkage_name':linkage,'demangled':subprocess.check_output(['c++filt',linkage],text=True).strip() if linkage else None,
                'die_offset':record['offset'],'evidence':evidence(DWARF,record['start_line'],record['end_line'])}
    expected={(GEMM,83,GEMM,275,88),(GEMM,275,GEMM,142,298),(GEMM,142,GEMM,190,148),
              (GEMM,190,ATOM,94,197),(ATOM,94,TRAITS,2072,104)}
    result=[]
    for record in records.values():
        if record['tag']!='DW_TAG_inlined_subroutine':continue
        target=endpoint(record)
        parent=records.get(record['parent'])
        while parent and parent['tag'] not in ('DW_TAG_inlined_subroutine','DW_TAG_subprogram'):
            parent=records.get(parent['parent'])
        source=endpoint(parent) if parent else None
        if not source or not target:continue
        call_line=int(record['attrs'].get('DW_AT_call_line','0'))
        key=(source['path'],source['line'],target['path'],target['line'],call_line)
        if key in expected:
            result.append({'source':source,'target':target,'callsite':{'path':source['path'],'line':call_line},
                'inline_die_offset':record['offset'],'inline_evidence':evidence(DWARF,record['start_line'],record['end_line'])})
    assert len(result)==5 and {(x['source']['path'],x['source']['line'],x['target']['path'],x['target']['line'],x['callsite']['line']) for x in result}==expected
    return {'schema_version':1,'kind':'NVCC/PTXAS DWARF inline origin chain','source':DWARF,
        'source_sha256':sha(RUN/'readelf_debug_info.stdout.txt'),'file_number_map_source':'probe.ptx .file directives',
        'entries':result,'note':'Compiler-generated inline call provenance. This is not proof of runtime launches, numerical correctness, or performance.'}

def main():
    metadata=json.loads((RUN/'metadata.json').read_text())
    assert metadata['status']=='static_dispatch_probe_pass' and metadata['all_snapshot_file_hashes_verified']==824
    for name,record in metadata['artifacts'].items():assert sha(RUN/name)==record['sha256'],name
    for name,digest in metadata['inputs'].items():assert sha(ATLAS/name)==digest,name
    lines=(RUN/'host_type_dump.stdout.txt').read_text().splitlines()
    values=dict(x.split('=',1) for x in lines)
    def log(*keys):
        return [evidence(LOG,next(i for i,line in enumerate(lines,1) if line.startswith(key+'='))) for key in keys]
    assert values['ASlice.rank']==values['BSlice.rank']=='2' and values['Accumulator.rank']=='3'
    assert values['Dispatch4.A_V_bytes']==values['Dispatch4.B_V_bytes']=='8'
    assert values['Dispatch4.M']==values['Dispatch4.N']=='1' and values['Mainloop.K_blocks']=='4'
    assert values['DVector.is_reference']=='0' and values['Accumulator.is_tmem']==values['Accumulator.is_rmem']=='1'
    dwarf=debug_index();atomic_json(HERE/'gemm_dispatch_debug_index.json',dwarf)
    dw_by_call={x['callsite']['line']:x for x in dwarf['entries']}
    prior_types=dict(x.split('=',1) for x in (HERE/'probe/run-334xibhm/host_type_dump.stdout.txt').read_text().splitlines())
    mma_op=prior_types['Mainloop.MMA_Op']
    assert all('SM100_MMA_F16BF16_2x1SM_SS' in x['target']['demangled'] for x in dwarf['entries'])
    nodes=[];edges=[]
    def node(key,kind,name,path,line,role,qualified_name=None,**extra):
        item={'id':'dispatch.'+key,'kind':kind,'name':name,'path':path,'line':line,'role':role,**extra}
        if qualified_name:item['qualified_name']=qualified_name
        evidence(path,line);nodes.append(item)
    def edge(key,source,target,relation,expression,condition,proof,**extra):
        edges.append({'id':'dispatch.edge.'+key,'source':source,'target':target,'relation':relation,
            'source_expression':expression,'condition':condition,'resolution':'configured','evidence':proof,**extra})
    def dw_proof(call):
        item=dw_by_call[call]
        return [item['inline_evidence'],item['target']['evidence']]

    node('api.rank4','api','cute::gemm：五参 dispatch [4]',GEMM,275,
         '唯一的D/C rank3、A/B rank2且四个Engine满足is_rmem的五参重载。','cute::gemm',
         signature=evidence(GEMM,264,279)['quote'])
    node('api.five_rvalue','api','cute::gemm：五参 D&& 转发',GEMM,142,
         '接受D(_,m,ns)产生的临时Tensor；命名D变成lvalue后再次调用gemm。','cute::gemm',
         signature=evidence(GEMM,135,146)['quote'])
    for key,name,line,role in [
        ('ASlice','A：tCrA(_,_,k_block,read_stage)',703,'rank2、shape(1,1)；V=1个64位DescriptorIterator值，不是1个FP16输入值。'),
        ('BSlice','B：tCrB(_,_,k_block,read_stage)',704,'rank2、shape(1,1)；V=1个64位DescriptorIterator值。'),
        ('Accumulator','D/C：accumulators',664,'rank3、shape((128,128),1,1)；Engine是TMEM视图，虽同时满足广义is_rmem谓词。'),
    ]:
        node('binding.'+key,'binding',name,COLLECTIVE,line,role,values[key],tensor_type=values[key],shape=values[key+'.shape'])
    for key,name,role in [
        ('AVector','A(_,m)：rank1描述符切片','shape(1)，rank1的A正式参数；DescriptorIterator值为64位。'),
        ('BVector','B(_,ns)：rank1描述符切片','shape(1)，rank1的B正式参数。'),
        ('DVector','D(_,m,ns) / C(_,m,ns)：rank1 TMEM视图','shape((128,128))只有一个顶层mode，故rank1；D临时值进入D&&，命名D再绑定rank1 D&；C类型相同但绑定const&。'),
    ]:
        node('binding.'+key,'binding',name,GEMM,298,role,values[key],tensor_type=values[key],shape=values[key+'.shape'])
    rank4_condition='DLayout::rank=3; ALayout::rank=2; BLayout::rank=2; CLayout::rank=3; is_rmem<TD/TA/TB/TC>::value=true。D/C仍是TMEM。'
    edge('four_to_rank4','contract.api.gemm','dispatch.api.rank4','calls','gemm(mma, C, A, B, C)',rank4_condition,
         [evidence(GEMM,88),evidence(GEMM,269,279)]+log('ASlice.rank','BSlice.rank','Accumulator.rank','Accumulator.is_rmem','Accumulator.is_tmem')+dw_proof(88),
         callsite={'path':GEMM,'line':88},selection_kind='template_overload_resolution',
         template_bindings={'MMA':mma_op,
                            'Tensor<TA,ALayout>':'dispatch.binding.ASlice','Tensor<TB,BLayout>':'dispatch.binding.BSlice','Tensor<TD,DLayout>':'dispatch.binding.Accumulator','Tensor<TC,CLayout>':'dispatch.binding.Accumulator'})
    edge('rank4_to_rvalue','dispatch.api.rank4','dispatch.api.five_rvalue','calls','gemm(mma, D(_,m,ns), A(_,m), B(_,ns), C(_,m,ns))',
         'if constexpr: A V字节=1×8=8且B V字节=1×8=8；源码#if 1选row-major serpentine；for m<M、n<N。本例M=N=1，故m=n=ns=0。',
         [evidence(GEMM,298),evidence(GEMM,288,297),evidence('include/cute/tensor_impl.hpp',233,254)]+log('Dispatch4.A_V_bytes','Dispatch4.B_V_bytes','Dispatch4.M','Dispatch4.N','DVector.is_reference')+dw_proof(298),
         callsite={'path':GEMM,'line':298},selection_kind='if_constexpr_and_preprocessor_branch',
         compile_time_selection={'expression':'decltype(size<0>(A))::value * sizeof(typename TA::value_type) == 8 && decltype(size<0>(B))::value * sizeof(typename TB::value_type) == 8','value':True,'preprocessor':{'path':GEMM,'line':291,'expression':'1','value':True}},
         runtime_condition='for (int m=0;m<M;++m) and for (int n=0;n<N;++n); M=N=1 in this recipe')
    edge('rvalue_to_rank1','dispatch.api.five_rvalue','contract.api.gemm_rank1','calls','gemm(mma, D, A, B, C)',
         'D是命名左值；D/A/B/C的layout rank均1，四个Engine均满足is_rmem。D/C实际仍是TMEM，不经UniversalFMA。',
         [evidence(GEMM,148),evidence(GEMM,178,197)]+log('AVector.rank','BVector.rank','DVector.rank','CVector.rank','DVector.is_tmem')+dw_proof(148),
         callsite={'path':GEMM,'line':148},selection_kind='template_overload_resolution')
    for key,formal in [('ASlice','TA, ALayout'),('BSlice','TB, BLayout'),('Accumulator','TD, DLayout; TC, CLayout')]:
        source_proof=[evidence(COLLECTIVE,554,563),evidence(COLLECTIVE,700,705)] if key!='Accumulator' else [evidence(KERNEL,612),evidence(COLLECTIVE,453,477),evidence(KERNEL,762,766)]
        edge('bind_rank4_'+key,'dispatch.binding.'+key,'dispatch.api.rank4','template_binds',formal+' from '+values[key],rank4_condition,
             source_proof+log(key,key+'.shape',key+'.rank',key+'.value_bytes',key+'.is_rmem',key+'.is_tmem'))
    for key,formal in [('AVector','TA, ALayout'),('BVector','TB, BLayout'),('DVector','TD, DLayout; TC, CLayout')]:
        edge('bind_rank1_'+key,'dispatch.binding.'+key,'contract.api.gemm_rank1','template_binds',formal+' from '+values[key],
             'rank1重载中的实际Tensor绑定；D和C的engine/layout相同，引用类别不同。',
             [evidence(GEMM,298),evidence(GEMM,148),evidence(GEMM,178,194)]+log(key,key+'.shape',key+'.rank'))
        edge('bind_atom_'+key,'dispatch.binding.'+key,'contract.api.mma_atom_call','template_binds',formal+' from '+values[key],
             'gemm rank1直接mma.call(D,A,B,C)，未更换Tensor类型；外层MMA_Atom实现属于本例F16 2×1SM Traits。',
             [evidence(GEMM,197),evidence(ATOM,94,104)]+log(key)+dw_proof(197)+dw_proof(104))
    for before,after,expr in [('ASlice','AVector','A(_,m)'),('BSlice','BVector','B(_,ns)'),('Accumulator','DVector','D(_,m,ns) / C(_,m,ns)')]:
        edge('slice_'+before,'dispatch.binding.'+before,'dispatch.binding.'+after,'type_uses','decltype('+expr+')',
             '切片保留V顶层mode、移除M/N modes；编译期类型关系，不是额外运行调用边。',
             [evidence(GEMM,298),evidence(PROBE,18,35)]+log(before+'.shape',after+'.shape'))
    chain_nodes=['contract.api.mma','contract.api.gemm','dispatch.api.rank4','dispatch.api.five_rvalue','contract.api.gemm_rank1','contract.api.mma_atom_call']
    chain_edges=['contract.edge.mma_gemm','dispatch.edge.four_to_rank4','dispatch.edge.rank4_to_rvalue','dispatch.edge.rvalue_to_rank1','contract.edge.gemm_atom']
    views=[{'id':'dispatch.view.calls','title':'Dense：Mainloop到MMA_Atom的真实重载调用链','node_ids':chain_nodes,'edge_ids':chain_edges},
           {'id':'dispatch.view.tensor_bindings','title':'Dense：描述符与TMEM切片如何决定rank分派','node_ids':[x['id'] for x in nodes if x['kind']=='binding']+['dispatch.api.rank4','contract.api.gemm_rank1','contract.api.mma_atom_call'],
            'edge_ids':[x['id'] for x in edges if x['relation']!='calls']}]
    result={'schema_version':1,'module_id':'dense_fp16','part':'gemm_dispatch','snapshot_commit':metadata['snapshot_commit'],
        'scope':'Fixed Dense FP16 recipe. Exact overload identities, branch predicates, Tensor bindings and compiler inline origins. No GPU run.',
        'nodes':nodes,'edges':edges,'views':views,
        'issues':[{'id':'dispatch.issue.runtime_not_tested','status':'not_in_scope','message':'静态探针不验证合法启动、TMEM分配/地址、同步完成、数值正确性或性能；probe kernel未运行。'},
                  {'id':'dispatch.issue.other_configurations','status':'not_in_scope','message':'未枚举其他Tile/Cluster/数据类型，也不把本例M=N=1的row分支当作普遍遍历策略或优化收益。'}],
        'resolves_external_issues':[{'id':'types.issue.tensor_callsite_bindings','scope':'Only this fixed Dense mma callsite; actual rank1 Tensor bindings and Traits2072 DWARF origin are now provided.',
            'evidence_path':'data/modules/dense_fp16/gemm_dispatch_debug_index.json'}],
        'compile_evidence':[{'id':'dispatch.compile.snapshot_sm110a','status':metadata['status'],'path':str((RUN/'metadata.json').relative_to(ATLAS)),
            'sha256':sha(RUN/'metadata.json'),'target':'sm_110a','recipe_arch_tag':'cutlass::arch::Sm100','gpu_executed':False,
            'probe_source':PROBE,'type_log':LOG,'ptx_path':str((RUN/'probe.ptx').relative_to(ATLAS)),
            'dwarf_index':'data/modules/dense_fp16/gemm_dispatch_debug_index.json',
            'all_commands_returned_zero':all(x['returncode']==0 for x in metadata['commands']),
            'limitations':'One synthetic kernel reproduces source Tensor construction/slices and compiles original gemm bodies. Full real-kernel emission is separately evidenced by immutable run-334xibhm. Debug register/stack properties are not performance measurements.'}],
        'provenance':{'generator_sha256':sha(__file__),'snapshot_scope_sha256':metadata['snapshot_scope_sha256'],
            'external_node_ids':[x for x in chain_nodes if x.startswith('contract.')],
            'external_edge_ids':['contract.edge.mma_gemm','contract.edge.gemm_atom'],
            'source_paths_are_snapshot_include_relative':True}}
    ids={x['id'] for x in nodes};assert len(ids)==len(nodes)
    allowed=ids|set(result['provenance']['external_node_ids'])
    assert all(e['source'] in allowed and e['target'] in allowed for e in edges)
    atomic_json(HERE/'gemm_dispatch.json',result)
    print(json.dumps({'nodes':len(nodes),'edges':len(edges),'views':len(views),'dwarf_verified_edges':len(dwarf['entries']),'output':str(HERE/'gemm_dispatch.json')},ensure_ascii=False))

if __name__=='__main__':main()
