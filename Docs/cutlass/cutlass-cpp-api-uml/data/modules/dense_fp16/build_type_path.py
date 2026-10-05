#!/usr/bin/env python3
"""Build four bounded Dense type views from pinned source and real NVCC evidence."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parents[2]
RUN = HERE/'probe/run-334xibhm'
PROBE = 'data/modules/dense_fp16/probe/dense_type_probe.cu'
TYPE_LOG = str((RUN/'host_type_dump.stdout.txt').relative_to(ATLAS))
BUILDER = 'include/cutlass/gemm/collective/builders/sm100_umma_builder.inl'
DECL = 'include/cutlass/gemm/collective/collective_builder_decl.hpp'
COMMON = 'include/cutlass/gemm/collective/builders/sm100_common.inl'
COMMON1 = 'include/cutlass/gemm/collective/builders/sm1xx_common.inl'
COLLECTIVE = 'include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp'
DISPATCH = 'include/cutlass/gemm/dispatch_policy.hpp'
ATOM = 'include/cute/atom/mma_atom.hpp'
TRAITS = 'include/cute/atom/mma_traits_sm100.hpp'
WRAPPER = 'include/cute/arch/mma_sm100_umma.hpp'
EPI = 'include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def local(path): return ATLAS/'snapshot'/path if path.startswith('include/') else ATLAS/path
def evidence(path, start, end=None):
    end = start if end is None else end
    lines = local(path).read_text().splitlines()
    assert 1 <= start <= end <= len(lines), (path,start,end)
    return {'path':path,'start_line':start,'end_line':end,'quote':'\n'.join(lines[start-1:end])}

def main():
    metadata = json.loads((RUN/'metadata.json').read_text())
    assert metadata['status'] == 'static_probe_pass'
    assert metadata['snapshot_commit'] == '8f50b052e1099fb982392a622caab69b97b63128'
    assert metadata['all_snapshot_file_hashes_verified'] == 824
    for name, record in metadata['artifacts'].items(): assert sha(RUN/name) == record['sha256'],name
    values = dict(line.split('=',1)for line in (RUN/'host_type_dump.stdout.txt').read_text().splitlines())
    old_log = ATLAS.parent/'02_cutlass_and_gemm/exemples/verification/results/dense.types.txt'
    old_values = dict(line.split('=',1)for line in old_log.read_text().splitlines())
    shared_type_keys = sorted(set(values)&set(old_values))
    old_type_comparison = {key:values[key]==old_values[key]for key in shared_type_keys}
    assert all(old_type_comparison.values()), old_type_comparison
    assert values['Mainloop.PipelineStages']=='8' and values['Builder.is_2sm']=='1'
    assert 'SM100_MMA_F16BF16_2x1SM_SS' in values['Mainloop.MMA_Op']
    nodes=[]; edges=[]
    def node(key, kind, name, path, line, role, qualified_name=None, notes=None):
        assert path.startswith('include/')
        item={'id':'types.'+key,'kind':kind,'name':name,'path':path,'line':line,'role':role}
        if qualified_name:item['qualified_name']=qualified_name
        if notes:item['notes']=notes
        evidence(path,line)
        nodes.append(item)
    def edge(key, source, target, relation, expression, condition, proof, resolution='configured'):
        edges.append({'id':'types.edge.'+key,'source':'types.'+source,'target':'types.'+target,
                      'relation':relation,'source_expression':expression,'condition':condition,
                      'resolution':resolution,'evidence':proof})

    node('builder_entry','api','CollectiveBuilder 主模板入口',DECL,93,'接收ArchTag、OpClass、元素/布局/对齐、Tile、Cluster、Stage与Schedule；默认模板不是本例实现。','cutlass::gemm::collective::CollectiveBuilder')
    node('builder_sm100','api','SM100 Dense UMMA Builder 偏特化',BUILDER,169,'固定源码中实际命中的Dense TensorOp偏特化。','cutlass::gemm::collective::CollectiveBuilder')
    node('builder_bound','type','本例 CollectiveBuilder 实例',BUILDER,169,'单个已绑定Builder类型：FP16 A/B RowMajor、FP32累加、Tile 256×128×64、Cluster 2×2×1、Auto。',values['Mainloop.Builder'],'ArchTag=Sm100是C++配方；新静态二进制目标是sm_110a。')
    node('arch_recipe','api','Sm100', 'include/cutlass/arch/arch.h',100,'本例C++ ArchTag；不等于把实际NVCC目标改写成SM100。','cutlass::arch::Sm100')
    node('schedule_auto','api','KernelScheduleAuto',DECL,73,'编译期选择标记，不是运行时调度调用。','cutlass::gemm::collective::KernelScheduleAuto')
    node('epilogue_bound','type','本例 CollectiveEpilogue 实例',EPI,84,'FP32 C/D线性组合Epilogue；SharedStorage实测33792字节。',values['Epilogue.FullType'])
    node('stage_bound','binding','StageCountAutoCarveout<33792>',DECL,50,'Epilogue先确定存储预算，再给Mainloop留下可用SMEM。',values['Mainloop.StagePolicy'])
    node('stage_compute','api','sm100_compute_stage_count_or_override（AutoCarveout）',BUILDER,84,'本例编译期阶段数计算；保留实际公式而非运行时调优步骤。','cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override')
    node('dispatch_bound','type','MainloopSm100TmaUmmaWarpSpecialized<8,2,4,…>',DISPATCH,1029,'本例Stages=8、SchedulerPipelineStageCount=2、AccumulatorPipelineStageCount=4。',values['Mainloop.DispatchPolicy'])
    node('collective_api','api','CollectiveMma：SM100 TMA UMMA 偏特化',COLLECTIVE,81,'DispatchPolicy命中的Collective实现，独立于Builder入口。','cutlass::gemm::collective::CollectiveMma')
    node('collective_bound','type','本例 CollectiveMma 实例',COLLECTIVE,81,'带实际TiledMma、SMEM布局、TMA copy atom的完整Mainloop类型。',values['Mainloop.FullType'])
    node('major_a','api','tag_to_umma_major_A<RowMajor>',COMMON1,95,'A的RowMajor映射到UMMA::Major::K。','cutlass::gemm::collective::detail::tag_to_umma_major_A')
    node('major_b','api','tag_to_umma_major_B<RowMajor>',COMMON1,117,'B的RowMajor映射到UMMA::Major::MN。','cutlass::gemm::collective::detail::tag_to_umma_major_B')
    node('auto_selector','api','sm100_make_trivial_tiled_mma',COMMON,444,'依固定Schedule与静态Cluster/Tile进行编译期2SM/1SM选择。','cutlass::gemm::collective::detail::sm100_make_trivial_tiled_mma')
    node('two_sm_selector','api','sm100_make_2sm_trivial_tiled_mma',COMMON,372,'本例half_t、M=256、N=128落到F16/BF16 2×1SM SS操作。','cutlass::gemm::collective::detail::sm100_make_2sm_trivial_tiled_mma')
    node('factory_op','api','make_tiled_mma(MMA_Op const&, …)',ATOM,548,'以操作类型构造MMA_Atom并转交Atom重载，属于类型构造路径。','cute::make_tiled_mma')
    node('factory_atom','api','make_tiled_mma(MMA_Atom<MMA_Op> const&, …)',ATOM,531,'构造TiledMMA的Atom、线程布局与置换模板参数。','cute::make_tiled_mma')
    node('tiled_bound','type','本例 TiledMMA 实例',ATOM,211,'TiledMMA::Atom为已选MMA_Atom；AtomThrID来自Traits，不把Cluster总CTA数当Atom复制数。',values['Mainloop.TiledMma'])
    node('atom_bound','type','MMA_Atom<本例MMA_Op>',ATOM,45,'操作包装入口，继承基于MMA_Traits的Atom实现。',values['Mainloop.MMA_Atom'])
    node('atom_impl','api','MMA_Atom<MMA_Traits<MMAOperation,…>>',ATOM,49,'通过Traits取得值类型、片段类型、形状与线程—值布局。','cute::MMA_Atom')
    node('traits_bound','type','本例 MMA_Traits 实例',TRAITS,2034,'Shape_MNK实测为256×128×16，ThrID为Layout<2>；Collective K=64不是单条MMA的K。',values['Mainloop.MMA_Traits'])
    node('wrapper','api','SM100_MMA_F16BF16_2x1SM_SS',WRAPPER,552,'实际选中的指令操作包装模板；本例a/b=half_t,c=float,M=256,N=128,Major=K/MN。','cute::SM100_MMA_F16BF16_2x1SM_SS','名称保留SM100；实际sm_110a设备分支由原cutlass/arch/config.h和cute/arch/config.hpp启用，并非按类名前缀推测。')
    node('atom_call','api','MMA_Atom::call(D,A,B,C)',ATOM,94,'4-tensor call重载，以实际Traits参与mma_unpack的ADL选择。','cute::MMA_Atom::call')
    node('mma_unpack','api','mma_unpack（F16/BF16 2×1SM SS Traits）',TRAITS,2072,'这个特定friend重载将寄存器描述符及TMEM地址交给实际wrapper；不是把所有mma_unpack重载合并。','cute::mma_unpack')
    node('fma','api','SM100_MMA_F16BF16_2x1SM_SS::fma',WRAPPER,563,'静态fma操作数为desc_a、desc_b、tmem_c、scaleC、idescE；内部elect_one_sync后发射tcgen05.mma。','cute::SM100_MMA_F16BF16_2x1SM_SS::fma')

    edge('instantiate_builder','builder_bound','builder_entry','template_binds','CollectiveBuilder<Sm100,OpClassTensorOp,half_t,RowMajor,8,half_t,RowMajor,8,float,Shape<256,128,64>,Shape<2,2,1>,StageCountAutoCarveout<33792>,KernelScheduleAuto>','固定Dense recipe；旧baseline hash与新probe均核验。',[evidence(PROBE,18,48),evidence(TYPE_LOG,1)])
    edge('arch_recipe','builder_bound','arch_recipe','type_uses','ArchTag = cutlass::arch::Sm100','C++配方轴；NVCC设备目标另见compile_evidence。',[evidence(PROBE,18),evidence('include/cutlass/arch/arch.h',100,108)])
    edge('auto_tag','builder_bound','schedule_auto','type_uses','KernelScheduleType = KernelScheduleAuto','编译期参数，不表示运行时调用。',[evidence(PROBE,44,48),evidence(DECL,70,73)])
    edge('sm100_specializes_entry','builder_sm100','builder_entry','specializes','CollectiveBuilder<ArchTag,arch::OpClassTensorOp,...,BuilderScheduleTag,enable_if_t<...>>','ArchTag=Sm100，标量非complex输入，Dense/Auto schedule且对齐谓词满足。',[evidence(BUILDER,169,198)],'direct')
    edge('selected_specialization','builder_bound','builder_sm100','template_binds','ArchTag=Sm100; ElementA=ElementB=half_t; BuilderScheduleTag=KernelScheduleAuto; AlignmentA=AlignmentB=8','新NVCC实际类型及TMA/UMMA Collective产物确认该选中结果。',[evidence(BUILDER,184,223),evidence(TYPE_LOG,1,4)])
    edge('epilogue_carveout','epilogue_bound','stage_bound','template_binds','StageCountAutoCarveout<static_cast<int>(sizeof(CollectiveEpilogue::SharedStorage))>','sizeof(SharedStorage)=33792；编译期存储依赖。',[evidence(PROBE,41,43),evidence(TYPE_LOG,19),evidence(EPI,207,226)])
    edge('stage_overload','stage_bound','stage_compute','template_binds','sm100_compute_stage_count_or_override(..., StageCountAutoCarveout<carveout_bytes>)','carveout_bytes=33792；该函数是编译期类型构造辅助。',[evidence(BUILDER,75,99),evidence(TYPE_LOG,2)])
    edge('computed_stages','stage_compute','dispatch_bound','template_binds','PipelineStages = (CapacityBytes - carveout_bytes) / stage_bytes; DispatchPolicy<PipelineStages,2,4,ClusterShape,ArchTag>','本例实际Stages=8，不将这个值推广为所有Auto配置。',[evidence(BUILDER,90,98),evidence(BUILDER,295,325),evidence(TYPE_LOG,16,18)])
    edge('builder_collective','builder_sm100','collective_bound','template_binds','using CollectiveOp = CollectiveMma<DispatchPolicy,TileShape_MNK,ElementA,TagToStrideA_t<...>,ElementB,TagToStrideB_t<...>,TiledMma,...>','本例完整Collective类型由Host-only打印确认。',[evidence(BUILDER,329,344),evidence(TYPE_LOG,4)])
    edge('dispatch_selects_collective','dispatch_bound','collective_api','template_binds','CollectiveMma<MainloopSm100TmaUmmaWarpSpecialized<Stages,SchedulerPipelineStageCount,AccumulatorPipelineStageCount,ClusterShape,ArchTag_>,...>','实际策略为<8,2,4,Cluster<2,2,1>,Sm100>。',[evidence(COLLECTIVE,81,114),evidence(TYPE_LOG,3)])
    edge('collective_instance','collective_bound','collective_api','template_binds',values['Mainloop.FullType'],'固定probe对应这个源偏特化；完整参数不靠短名推测。',[evidence(TYPE_LOG,4),evidence(COLLECTIVE,81,106)])
    edge('collective_tiled','collective_bound','tiled_bound','type_uses','using TiledMma = TiledMma_;','TiledMma_为实际Builder传入类型。',[evidence(COLLECTIVE,104,107),evidence(TYPE_LOG,5)])
    edge('major_a_binding','builder_bound','major_a','template_binds','UmmaMajorA = tag_to_umma_major_A<GmemLayoutATag>(); GmemLayoutATag=RowMajor','结果K=0；由源映射和probe static_assert共同确认。',[evidence(BUILDER,202,203),evidence(COMMON1,93,107),evidence(PROBE,60),evidence(TYPE_LOG,14)])
    edge('major_b_binding','builder_bound','major_b','template_binds','UmmaMajorB = tag_to_umma_major_B<GmemLayoutBTag>(); GmemLayoutBTag=RowMajor','结果MN=1；B与A的映射不同。',[evidence(BUILDER,202,203),evidence(COMMON1,115,128),evidence(PROBE,61),evidence(TYPE_LOG,15)])
    edge('builder_tiled_selector','builder_bound','auto_selector','template_binds','using TiledMma = decltype(sm100_make_trivial_tiled_mma<ElementAMma,ElementBMma,float,Tile,Cluster,Major::K,Major::MN,KernelScheduleAuto>());','half_t经输入元素映射保持half_t；Tile=256×128×64，Cluster=2×2×1。',[evidence(BUILDER,205,222),evidence(COMMON1,65,89),evidence(TYPE_LOG,5)])
    edge('auto_chooses_two_sm','auto_selector','two_sm_selector','template_binds','ClusterShape静态 && size<0>(ClusterShape{}) % 2 == 0 && size<0>(TileShape{}) % 128 == 0','本例2%2=0且256%128=0；仅为编译期选择，不是运行调用。',[evidence(COMMON,444,469),evidence(PROBE,31,32),evidence(TYPE_LOG,13)])
    edge('fp16_two_sm_op','two_sm_selector','wrapper','template_binds','SM100_MMA_F16BF16_2x1SM_SS<ElementAMma,ElementBMma,float,M,N,UmmaMajorA,UmmaMajorB,ANeg,BNeg>','ElementAMma=ElementBMma=half_t，M=256，N=128，Major K/MN，ScaleIn One/One。',[evidence(COMMON,372,391),evidence(PROBE,55,58),evidence(TYPE_LOG,7)])
    edge('op_to_factory','two_sm_selector','factory_op','template_binds','return make_tiled_mma(SM100_MMA_F16BF16_2x1SM_SS<...>{});','选择MMA_Op重载，作为返回类型推导的一部分。',[evidence(COMMON,385,390),evidence(ATOM,543,554)])
    edge('factory_wraps_atom','factory_op','atom_bound','template_binds','MMA_Atom<MMA_Op>{}','MMA_Op为已选F16/BF16 2×1SM SS操作；static_assert核对完全相同类型。',[evidence(ATOM,548,553),evidence(PROBE,58,59),evidence(TYPE_LOG,6)])
    edge('factory_forward','factory_op','factory_atom','template_binds','make_tiled_mma(MMA_Atom<MMA_Op>{}, thr_layout, permutations)','这是类型工厂重载的选择，不用calls表示运行行为。',[evidence(ATOM,531,553)])
    edge('factory_tiled_type','factory_atom','tiled_bound','template_binds','TiledMMA<MMA_Atom<MMA_Op>,decltype(thr_layout_mnk),decltype(permutation_mnk)>','默认Atom layout与permutation的实际类型见新Host打印。',[evidence(ATOM,531,540),evidence(TYPE_LOG,5)])
    edge('tiled_atom','tiled_bound','atom_bound','type_uses','struct TiledMMA : MMA_Atom; using Atom=MMA_Atom;','此处Atom不是按Cluster总CTA数推测；使用TiledMma::Atom的实际类型。',[evidence(ATOM,208,218),evidence(TYPE_LOG,5,6)])
    edge('atom_traits_impl','atom_bound','atom_impl','type_uses','MMA_Atom<MMAOperation> : MMA_Atom<MMA_Traits<MMAOperation>>','已选操作包装进入Traits实现。',[evidence(ATOM,44,53),evidence(TYPE_LOG,6,8)])
    edge('atom_traits_type','atom_impl','traits_bound','template_binds','using Traits = MMA_Traits<MMAOperation,Args...>;','MMAOperation=本例ExpectedOp；完整Traits类型由NVCC编译的Host程序打印。',[evidence(ATOM,49,61),evidence(TYPE_LOG,8)])
    edge('traits_uses_wrapper','traits_bound','wrapper','type_uses','MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS<a_type,b_type,c_type,M,N,a_major,b_major,a_neg,b_neg>>','ThrID=Layout<2>；Atom K=256/16=16，不能与Collective K=64混同。',[evidence(TRAITS,2030,2058),evidence(TYPE_LOG,7,10)])
    edge('traits_unpack_binding','traits_bound','mma_unpack','template_binds','mma_unpack(MMA_Traits const& traits, Tensor<D>&, Tensor<A> const&, Tensor<B> const&, Tensor<C> const&)','外层Traits已绑定；本页未枚举每个调用点的Tensor布局模板参数。',[evidence(TRAITS,2065,2086),evidence(TYPE_LOG,8)])
    edge('atom_calls_unpack','atom_call','mma_unpack','calls','mma_unpack(static_cast<Traits const&>(*this), D, A, B, C)','此边表示实际C++调用表达式；通过本例Traits确定这个friend重载，而非所有同名重载。',[evidence(ATOM,94,104),evidence(TRAITS,2072,2076),evidence(TYPE_LOG,8)])
    edge('unpack_calls_fma','mma_unpack','fma','calls','SM100_MMA_F16BF16_2x1SM_SS<a_type,b_type,c_type,M,N,a_major,b_major,a_neg,b_neg>::fma(desc_a,desc_b,tmem_c,uint32_t(traits.accumulate_),idesc)','fma内部另以elect_one_sync控制指令发射；未执行GPU。',[evidence(TRAITS,2083,2092),evidence(WRAPPER,563,582)])

    views=[]
    def view(key,title,node_keys,edge_keys):
        views.append({'id':'types.view.'+key,'title':title,'node_ids':['types.'+x for x in node_keys],
                      'edge_ids':['types.edge.'+x for x in edge_keys]})
    view('entry','Dense配置与SM100 Builder选择',
         ['builder_bound','builder_entry','builder_sm100','arch_recipe','schedule_auto'],
         ['instantiate_builder','arch_recipe','auto_tag','sm100_specializes_entry','selected_specialization'])
    view('collective','存储预算、DispatchPolicy与CollectiveMma',
         ['epilogue_bound','stage_bound','stage_compute','dispatch_bound','builder_sm100','collective_bound','collective_api','tiled_bound'],
         ['epilogue_carveout','stage_overload','computed_stages','builder_collective','dispatch_selects_collective','collective_instance','collective_tiled'])
    view('mma_construction','Auto选择2-SM操作与TiledMMA／Atom构造',
         ['builder_bound','major_a','major_b','auto_selector','two_sm_selector','wrapper','factory_op','factory_atom','tiled_bound','atom_bound'],
         ['major_a_binding','major_b_binding','builder_tiled_selector','auto_chooses_two_sm','fp16_two_sm_op','op_to_factory','factory_wraps_atom','factory_forward','factory_tiled_type','tiled_atom'])
    view('wrapper','已选Traits到指令包装：静态调用证据',
         ['atom_bound','atom_impl','traits_bound','wrapper','atom_call','mma_unpack','fma'],
         ['atom_traits_impl','atom_traits_type','traits_uses_wrapper','traits_unpack_binding','atom_calls_unpack','unpack_calls_fma'])

    ptx_path=str((RUN/'probe.ptx').relative_to(ATLAS));sass_path=str((RUN/'nvdisasm.stdout.txt').relative_to(ATLAS))
    ptx_lines=local(ptx_path).read_text().splitlines();sass_lines=local(sass_path).read_text().splitlines()
    ptx_sites=[i+1 for i,line in enumerate(ptx_lines)if 'tcgen05.mma.cta_group::2.kind::f16' in line]
    sass_sites=[i+1 for i,line in enumerate(sass_lines)if 'UTCHMMA.2CTA' in line]
    ptx_text=local(ptx_path).read_text()
    entries=list(re.finditer(r'^\s*(?:(?:\.visible|\.weak)\s+)?\.entry\s+(\S+)\(',ptx_text,re.M))
    assert len(entries)==1
    body_start=ptx_text.index('{',entries[0].end());depth=0;body_end=None
    for offset in range(body_start,len(ptx_text)):
        depth+=(ptx_text[offset]=='{')-(ptx_text[offset]=='}')
        if depth==0:body_end=offset+1;break
    assert body_end is not None
    mma_offsets=[m.start()for m in re.finditer(r'tcgen05\.mma\.cta_group::2\.kind::f16\b',ptx_text)]
    assert len(mma_offsets)==4 and all(body_start<x<body_end for x in mma_offsets)
    compile_evidence=[{'id':'types.compile.snapshot_sm110a','status':'static_probe_pass','path':str((RUN/'metadata.json').relative_to(ATLAS)),
        'snapshot_commit':metadata['snapshot_commit'],'nvcc_version':metadata['nvcc_version'],'target':'sm_110a','recipe_arch_tag':'Sm100',
        'commands':metadata['commands'],'artifacts':metadata['artifacts'],'actual_types':values,
        'device_entry_count':1,'device_entry_symbol':entries[0][1],
        'entry_body_byte_range':[body_start,body_end],
        'instruction_count_interpretation':'Four static instruction sites inside the one emitted entry; this is not a count of runtime GEMM instructions.',
        'old_type_log_comparison':old_type_comparison,
        'ptx_mma_sites':[evidence(ptx_path,x)for x in ptx_sites],
        'sass_mma_sites':[evidence(sass_path,x)for x in sass_sites],
        'architecture_evidence':[evidence('include/cutlass/arch/config.h',131,147),evidence('include/cute/arch/config.hpp',100,113),evidence(PROBE,67,73)],
        'host_type_execution':'Only the PROBE_HOST_TYPES main was run; it prints typeid/sizeof and invokes no launch anchor or CUDA API.',
        'runtime_gpu_launched':False,'bounds':'Static type selection, device emission, PTXAS and SASS only. Launch legality, synchronization, numeric correctness and performance were not tested.',
        'host_compatibility':{'flags':['-Xcompiler -U_GNU_SOURCE','-D_DEFAULT_SOURCE=1','-D_POSIX_C_SOURCE=200809L','--expt-relaxed-constexpr'],
            'probe_local_shim':'data/modules/dense_fp16/probe/host_compat.hpp','shim_sha256':metadata['host_compatibility_shim_sha256'],
            'meaning':'Only optional libstdc++ GNU pthread clockwait/clocklock overload paths are disabled for this Host header compatibility check; snapshot/system headers are unchanged.'}}]
    for failed,reason in [('run-fqjcmaf1','Host pthread clockwait/clocklock declarations were hidden after the GNU_SOURCE workaround.'),
                          ('run-topnb8e3','Device PTX/PTXAS/SASS succeeded, but the Host-type branch incorrectly applied the probe device-feature guard. Metadata also used an overly narrow visible-entry regex; this run remains an incomplete probe.')]:
        prior=HERE/'probe'/failed/'metadata.json';record=json.loads(prior.read_text())
        archived_input=prior.parent/'input.cu'
        assert sha(archived_input)==record['probe_source_sha256']
        compile_evidence.append({'id':'types.compile.'+failed,'status':'incomplete_or_failed_probe_preserved','path':str(prior.relative_to(ATLAS)),
            'commands':record['commands'],'reason':reason,'runtime_gpu_launched':False,
            'archived_source_path':str(archived_input.relative_to(ATLAS)),'archived_source_sha256':sha(archived_input)})
    node_ids={n['id']for n in nodes};edge_by_id={e['id']:e for e in edges}
    assert len(node_ids)==len(nodes) and len(edge_by_id)==len(edges)
    for e in edges:assert e['source']in node_ids and e['target']in node_ids and e['evidence']
    for v in views:
        assert set(v['node_ids'])<=node_ids
        for key in v['edge_ids']:
            assert key in edge_by_id
            assert {edge_by_id[key]['source'],edge_by_id[key]['target']}<=set(v['node_ids'])
    result={'schema_version':1,'module_id':'dense_fp16','part':'type_path','snapshot_commit':metadata['snapshot_commit'],
        'scope':'One fixed Dense FP16 recipe. Type selection is separate from runtime calls and from the sm_110a binary target.',
        'nodes':nodes,'edges':edges,'views':views,
        'issues':[{'id':'types.issue.tensor_callsite_bindings','status':'not_expanded_in_this_part',
            'scope':'types.edge.atom_calls_unpack',
            'message':'该边依据已选Traits定位具体mma_unpack重载；未逐调用点列出TD/TA/TB/TC及其Tensor Layout模板实参。不能把本类型分图当作完整callsite实参对账。',
            'does_not_invalidate':'已核验的Builder、Collective、TiledMMA、Atom、MMA_Op类型选择与静态设备指令发射。'}],
        'compile_evidence':compile_evidence,
        'provenance':{'baseline_verification':metadata['prior_evidence'],'builder_script_sha256':sha(__file__),
            'baseline_sha256':metadata['baseline']['sha256'],'source_paths_are_snapshot_include_relative':True,
            'api_node_policy':'Every API node identifies one declaration/overload or one specialization using its original path and line; complete signatures are joined by the atlas root.'}}
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=HERE,prefix='.type-path-',suffix='.tmp',delete=False) as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n');temporary=stream.name
    os.replace(temporary,HERE/'type_path.json')
    print(json.dumps({'nodes':len(nodes),'edges':len(edges),'views':len(views),'compile_status':compile_evidence[0]['status'],'output':str(HERE/'type_path.json')},ensure_ascii=False))

if __name__=='__main__':main()
