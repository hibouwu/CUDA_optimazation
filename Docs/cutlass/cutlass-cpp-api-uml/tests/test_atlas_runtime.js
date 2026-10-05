// Pure JavaScript unit tests with a minimal DOM double. No browser is launched,
// no file URL is navigated, and this is NOT viewport/interaction acceptance.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const root=path.resolve(__dirname,'..'),data=JSON.parse(fs.readFileSync(path.join(root,'data/atlas.json'),'utf8'));
const overviewData=JSON.parse(fs.readFileSync(path.join(root,'data/overviews.json'),'utf8'));
const interfaceData=JSON.parse(fs.readFileSync(path.join(root,'data/interface-docs.json'),'utf8'));
const mainPathData=JSON.parse(fs.readFileSync(path.join(root,'data/main-path.json'),'utf8'));
const listeners={},elements=new Map();
function element(id){
  if(!elements.has(id))elements.set(id,{id,innerHTML:'',textContent:'',value:'',hidden:false,style:{},dataset:{},clientWidth:940,events:{},
    addEventListener(name,fn){this.events[name]=fn;}});
  const value=elements.get(id);
  if(id==='diagram-object'){
    const text=elements.get('content')?.innerHTML||'';
    value.dataset.width=/data-width="([0-9.]+)"/.exec(text)?.[1]||'900';
    value.dataset.height=/data-height="([0-9.]+)"/.exec(text)?.[1]||'600';
  }
  return value;
}
const document={getElementById:element,querySelectorAll(){return[];},addEventListener(name,fn){listeners['doc:'+name]=fn;}};
const location={search:'',hash:''};
const window={ATLAS_DATA:data,ATLAS_OVERVIEWS:overviewData,ATLAS_INTERFACE_DOCS:interfaceData,ATLAS_MAIN_PATH:mainPathData,addEventListener(name,fn){listeners[name]=fn;},scrollTo(){}};
const context={document,window,location,URLSearchParams,URL,Map,Set,console};
vm.runInNewContext(fs.readFileSync(path.join(root,'templates/atlas/atlas.js'),'utf8'),context);
let passed=0;
function test(name,fn){fn();passed++;console.log('PASS '+name);}
function route(params){location.hash='#?'+new URLSearchParams(params);listeners.hashchange();return element('content').innerHTML;}
test('initial page starts with the complete five-step scheme path, not the internal Kernel graph',()=>{
  assert(element('scope').textContent.includes('824'));
  const text=element('content').innerHTML;
  assert(text.includes('同一次 GEMM 的五步审查'));assert(text.includes('D = A × B + 0.5 × C'));
  assert(!text.includes('id="diagram-object"'));assert(!text.includes('signature-label'));
  assert.equal(element('reference-navigation').open,false);
});
test('all main-path stages keep the same fixture and essential boundaries visible',()=>{
  for(const step of mainPathData.steps){
    const text=route({mode:'mainline',step:step.id});
    assert(text.includes('M=256 · N=256 · K=128 · L=1'));
    assert(text.includes(step.handoff));assert(text.includes(step.boundary));
    assert(text.includes('必须成立的条件'));assert(text.includes('现场怎样核对'));
    assert.equal((text.match(/class="path-step /g)||[]).length,5);
    assert(text.includes('aria-current="step"'));
    assert(text.includes('<details class="path-evidence">'));
    for(const id of step.api_refs)assert(text.includes('api='+id),id);
  }
});
test('invalid main-path step cannot hide the failure/reuse stage',()=>{
  const text=route({mode:'mainline',step:'not-a-step'});assert(text.includes('没有这个主线步骤'));
  assert(text.includes('复用、失败与验证证据'));
});
test('details retain a return path to scheme review and the full API reference',()=>{
  const text=route({api:'host.initialize'});assert(text.includes('完整源码签名'));
  assert.equal(element('back').textContent,'返回方案验证主线');
  assert.equal(element('reference-navigation').open,true);
  element('back').onclick();assert(location.hash.includes('mode=mainline'));
  assert(element('content').innerHTML.includes('同一次 GEMM 的五步审查'));
});
test('main path does not claim GPU execution or require all scenario branches',()=>{
  const text=route({mode:'mainline',step:'complete'});
  assert(text.includes('没有运行 GPU Kernel'));assert(text.includes('只在同事实际承诺支持'));
  assert(text.includes('不能单独证明 D 的全局写入完成'));
  assert(text.includes('主线')&&text.includes('源码'));
});
test('contract review examples are attached to their relevant steps without replacing the main path',()=>{
  for(const step of mainPathData.steps){
    const text=route({mode:'mainline',step:step.id});
    const expected=mainPathData.review_cases.filter(c=>c.steps.includes(step.id));
    assert.equal((text.match(/data-review-case=/g)||[]).length,expected.length);
    for(const c of expected)assert(text.includes(`data-review-case="${c.id}"`));
    assert.equal((text.match(/class="path-step /g)||[]).length,5);
    assert(text.includes('不是同事已经给出的实现'));
  }
});
test('review examples allow restricted and synchronous designs instead of automatically rejecting differences',()=>{
  const layout=route({mode:'mainline',step:'problem'});
  assert(layout.includes('这是支持范围取舍'));assert(layout.includes('接口缺少区分'));
  const completion=route({mode:'mainline',step:'complete'});
  assert(completion.includes('可以是一种同步设计'));assert(completion.includes('缺少结果交付条件'));
  const reuse=route({mode:'mainline',step:'reuse'});
  assert(reuse.includes('重复调用本身不构成错误'));assert(reuse.includes('非空指针也不证明容量'));
});
test('Host layout execution is presented separately from GPU validation',()=>{
  const text=route({mode:'mainline',step:'problem'});
  assert(text.includes('HOST_LAYOUT_PASS'));assert(text.includes('未运行 GEMM'));
  assert(text.includes('evidence/scheme-layout-results.json'));
  assert(text.includes('evidence/scheme-layout-probe.cu'));
  assert(text.includes('128变为136'));assert(text.includes('257变为265'));
});
test('Scheduler handoff views default to whole process diagrams and retain all callers',()=>{
  for(const view of ['scheduler.handoff.entry','scheduler.handoff.next','scheduler.handoff.output']){
    const text=route({view});assert(text.includes('value="overview" selected'));
    assert(text.includes('overview-'+view));
  }
  const next=route({view:'scheduler.handoff.next'});
  for(const line of [664,701,741,819,877])assert(next.includes('scheduler.call.fetch.'+line));
});
test('Scheduler overloads show independent parameters and resolved fixup target',()=>{
  const three=route({api:'scheduler.fetch'});assert(three.includes('scheduler_pipeline'));assert(three.includes('scheduler_pipe_consumer_state'));
  const one=route({api:'scheduler.fetch_compat'});assert(one.includes('不读取CLC')||one.includes('不等待响应'));
  const detail=route({edge:'contract.edge.kernel_fixup'});assert(detail.includes('api=scheduler.fixup_state'));
  assert(detail.includes('指定配置下确定'));assert(detail.includes(':898</a>'));
});
test('main path names the selected full-K handoff without claiming all scheduling correct',()=>{
  const text=route({mode:'mainline',step:'execute'});
  assert(text.includes('CTA 级 Tile 坐标'));assert(text.includes('2个K块'));
  assert(text.includes('取得无效的下一任务不等于可以跳过当前任务'));
  assert(text.includes('scheduler.handoff.entry'));assert(text.includes('全工作覆盖性仍未验证'));
});
test('Kernel overview precedes source details and includes independent members and aliases',()=>{
  const text=route({api:'host.kernel'});
  assert(text.indexOf('overview-card')<text.indexOf('类型完整声明'));
  assert(text.includes('host.kernel.lower'));assert(text.includes('host.kernel.can_implement'));
  assert(text.includes('contract.type.acc_pipeline'));assert(!text.includes('panel-select'));
  const deviceCalls=data.edges.filter(e=>e.source==='contract.api.kernel'&&e.relation==='calls');
  assert(deviceCalls.length>0);deviceCalls.forEach(e=>assert(text.includes(e.id),e.id));
  assert(route({api:'contract.type.kernel'}).includes('overview-host.kernel'));
});
test('all Host topic views select complete overview unless auxiliary panel requested',()=>{
  for(const view of ['host.lifecycle','host.preparation','host.launch']){
    const text=route({view});assert(text.includes('value="overview" selected'));
    assert(text.includes(overviewData.view_overviews[view]));
    const detail=route({view,panel:'0'});assert(!detail.includes('value="overview" selected'));
    assert(detail.includes('局部关系分组'));
  }
});
test('overview selector supports semantic process diagrams and safe fallback',()=>{
  const id=overviewData.type_process_diagrams['host.kernel'].find(id=>id.includes('lowering'));
  const text=route({api:'host.kernel',overview:id});assert(text.includes(`value="${id}" selected`));
  const device=overviewData.type_process_diagrams['host.kernel'].find(id=>id.includes('device'));
  assert(route({api:'host.kernel',overview:device}).includes(`value="${device}" selected`));
  assert(route({api:'host.kernel',overview:'untrusted-unknown'}).includes('value="overview-host.kernel" selected'));
});
test('member API preserves full signature and route back to type overview',()=>{
  const text=route({api:'host.kernel.lower'});assert(text.includes('返回 GemmUniversal 接口总览'));
  assert(text.includes('完整源码签名'));assert(text.includes('Arguments const&amp; args'));
  assert(route({api:'contract.api.kernel'}).includes('返回 GemmUniversal 接口总览'));
});
test('edge detail offers every containing overview and keeps original evidence',()=>{
  const text=route({edge:'host.adapter.init_lower.328'});
  assert(text.includes('返回 Kernel 接口总览'));assert(text.includes('process-host.kernel.lowering'));
  assert(text.includes('view=host.preparation'));assert(text.includes('实际调用位置'));assert(text.includes(':328</a>'));
});
test('Host launch has semantic subgraphs and selector navigates without an auxiliary panel',()=>{
  const ids=overviewData.view_process_diagrams['host.launch'];assert.equal(ids.length,3);
  for(const id of ids){const text=route({view:'host.launch',overview:id});assert(text.includes(`value="${id}" selected`));}
  element('panel-select').value=ids[0];element('panel-select').onchange();
  assert(location.hash.includes('overview='+ids[0]));assert(!location.hash.includes('panel='));
  element('panel-select').value='overview';element('panel-select').onchange();
  assert(!location.hash.includes('overview='));assert(!location.hash.includes('panel='));
});
test('every published interface exposes its own level and explanation',()=>{
  for(const node of data.nodes.filter(n=>['api','type','external'].includes(n.kind))){
    const text=route({api:node.id});assert(text.includes('所属层级与作用'),node.id);
    assert(text.includes('参数含义与确定阶段'),node.id);
    assert(text.includes(interfaceData.nodes[node.id].layer.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')),node.id);
  }
});
test('Builder compile parameters and callable runtime arguments are displayed separately',()=>{
  const builder=route({api:'types.builder_entry'});assert(builder.includes('本接口模板参数 · 编译期'));
  assert(builder.includes('AlignmentA'));assert(builder.includes('KernelScheduleType'));
  const call=route({api:'host.mainloop.lower'});assert(call.includes('所属类型或外层模板'));
  assert(call.includes('函数参数 · 逐项区分'));assert(call.includes('编译期类型与运行时数据分开看'));
  assert(call.includes('ProblemShape'));assert(call.includes('constexpr 仅允许'));
});
test('macro text and parser correction are not presented as ordinary extra arguments',()=>{
  const macro=route({api:'alignment.macro.cuda'});assert(macro.includes('宏参数 · 预处理期'));
  assert(macro.includes('预处理期文本参数'));assert(!macro.includes('函数参数 · 逐项区分'));
  const kernel=route({api:'host.device_kernel'});assert(kernel.includes('参数展示已按声明原文校正'));
  const section=kernel.split('函数参数 · 逐项区分')[1].split('</ol>')[0];
  assert.equal((section.match(/class="parameter-heading"/g)||[]).length,1);
});
test('absent external declaration is not claimed to have zero parameters',()=>{
  const text=route({api:'host.cuda.launch_ex'});assert(text.includes('不能将未记录的形参解释为接口没有参数'));
  assert(!text.includes('此声明没有显式函数形参'));
});
test('API source signature and cross-part aliases render',()=>{
  const text=route({api:'host.initialize'});assert(text.includes('完整源码签名'));assert(text.includes('cuda_adapter'));assert(text.includes('相关关系'));
  assert(text.includes('https://github.com/NVIDIA/cutlass/blob/'+data.commit+'/include/cutlass/gemm/device/gemm_universal_adapter.h#L'));
  assert(text.includes('固定提交原文（联网）'));
});
test('alignment attribute renders its owner and both definitions without a network lookup',()=>{
  const node=data.nodes.find(n=>n.id==='host.initialize'),old=node.alignment_specifiers;
  const p='include/cute/container/alignment.hpp';
  node.alignment_specifiers=[{path:p,start_line:60,semantic_spelling:'CUTE_ALIGNAS(Alignment)',alignment_expression:'Alignment',owner_hint:{kind:'field'},conditional_expansions:[
    {conditions:[{expression:'defined(__CUDACC__)'}],expanded_spelling:'__align__(Alignment)',definition:{path:p,start_line:52}},
    {conditions:[{expression:'!defined(__CUDACC__)'}],expanded_spelling:'alignas(Alignment)',definition:{path:p,start_line:54}}]}];
  try{const text=route({api:node.id});assert(text.includes('当前字段声明'));assert(text.includes('__align__(Alignment)'));assert(text.includes('alignas(Alignment)'));assert(text.includes('alignment.hpp.html#L52'));assert(text.includes('alignment.hpp.html#L54'));assert(text.includes('未代入模板值'));}
  finally{node.alignment_specifiers=old;}
});
test('shared namespace keeps separate declaration source locations visible',()=>{
  const node=data.nodes.find(n=>n.id==='host.initialize'),old=node.source_declaration_occurrences;
  node.source_declaration_occurrences=[{path:'include/cute/container/alignment.hpp',start_line:38,source_url:'source/include/cute/container/alignment.hpp.html#L38',signature:'namespace cute',declaration_occurrence_id:'occ_A'},
    {path:'include/cute/container/array_aligned.hpp',start_line:36,source_url:'source/include/cute/container/array_aligned.hpp.html#L36',signature:'namespace cute',declaration_occurrence_id:'occ_B'}];
  try{const text=route({api:node.id});assert(text.includes('同一实体的不同声明位置'));assert(text.includes('alignment.hpp.html#L38'));assert(text.includes('array_aligned.hpp.html#L36'));assert(text.includes('occ_A'));assert(text.includes('occ_B'));}
  finally{node.source_declaration_occurrences=old;}
});
test('multiple source conditions are alternatives, not a selected declaration masquerading as global availability',()=>{
  const node=data.nodes.find(n=>n.id==='host.initialize'),old=node.declaration_availability;
  node.declaration_availability={operator:'any_of',occurrence_condition_groups:[[{expression:'!defined(__CUDACC_RTC__)'}],[{expression:'defined(CUTE_STL_NAMESPACE_IS_CUDA_STD)'}]]};
  try{const text=route({api:node.id});assert(text.includes('满足下列任意一组条件'));assert(text.includes('!defined(__CUDACC_RTC__)'));assert(text.includes('defined(CUTE_STL_NAMESPACE_IS_CUDA_STD)'));}
  finally{node.declaration_availability=old;}
});
test('callsite differs from context evidence start',()=>{
  const text=route({edge:'host.adapter.init_lower.328'});assert(text.includes('实际调用位置'));assert(text.includes(':328</a>'));
});
test('search includes API and full fixed source path matches',()=>{
  element('search').value='gemm_universal_adapter.h';element('search').events.input();
  assert(element('search-results').innerHTML.includes('include/cutlass/gemm/device/gemm_universal_adapter.h'));
  assert(element('search-results').innerHTML.includes('file='));
});
test('all 824 fixed files remain in offline navigation',()=>{
  const text=route({mode:'files'});assert.equal((text.match(/class="list-item"/g)||[]).length,824);
});
test('resources and static compilation remain separate claims',()=>{
  assert(route({mode:'contracts'}).includes('允许复用 / 释放'));
  const text=route({mode:'compile'});assert(text.includes('未运行GPU'));assert(text.includes('sm_110a'));
});
test('single-owner modules retain all 40 inl files',()=>{
  assert.equal(data.source_ownership.modules.length,20);
  assert.equal(data.files.filter(f=>f.path.endsWith('.inl')).length,40);
  const text=route({owner:'gemm_collective'});assert(text.includes('84个文件'));assert(text.includes('sm100_umma_builder.inl'));
});
test('file entry separates recorded APIs from full file denominator',()=>{
  const text=route({file:'include/cutlass/gemm/device/gemm_universal_adapter.h'});
  assert(text.includes('host.initialize'));assert(text.includes('不是本文件API分母'));assert(text.includes('源码'));
});
test('architecture search includes the actual configuration',()=>{
  element('search').value='sm_110a';element('search').events.input();
  assert(element('search-results').innerHTML.includes('host.initialize'));
});
test('closed configured issues expose closure rather than silently hiding history',()=>{
  assert(route({mode:'issues'}).includes('指定配置已关闭'));
  assert(element('content').innerHTML.includes('关闭依据与限制'));
});
test('full-file work package exposes source obligations rather than only graph counts',()=>{
  const text=route({coverage:'cute_elementwise_full_headers'});
  assert(text.includes('逐源码义务'));assert(text.includes('axpby.hpp:83</a>'));
  assert(text.includes('模板与函数参数'));assert(text.includes('不是API调用'));
});
test('conditional candidate and non-evaluated expression retain their bindings',()=>{
  const text=route({edge:'elementwise.edge.fill_adl_call'});
  assert(text.includes('dependency_parameters'));assert(text.includes('Engine'));
  assert(route({api:'elementwise.dep.data_fill'}).includes('dependency_status'));
});
test('contract relation references are navigable and do not imply unconditional order',()=>{
  const text=route({mode:'contracts'});
  assert(text.includes('edge=elementwise.edge.fill_dispatch_prefer1'));
  assert(text.includes('edge=elementwise.edge.fill_dispatch_prefer0'));
  assert(!text.includes('事件 / API顺序'));
});
test('type representation contract needs no fictitious resource lifecycle',()=>{
  const text=route({mode:'contracts'});
  assert(text.includes('alignment.contract.types'));
  assert(text.includes('类型约束不构成事件序列'));
  assert(text.includes('返回与访问边界'));
});
test('array full header exposes physical declarations, conditional bindings and plural parameter owners',()=>{
  const text=route({coverage:'cute_array_full_header'});
  assert(text.includes('物理声明 87 · 条件绑定 92'));
  assert(text.includes('array.api.fn_422'));assert(text.includes('array.type.cuda_std.tuple_size'));
  assert(!text.includes('undefined ·'));
});
test('array source contracts and implicit calls remain distinguishable',()=>{
  const text=route({mode:'contracts'});assert(text.includes('逐项源码契约'));assert(text.includes('Free clear calls a.fill(T(0))'));
  const edge=data.edges.find(e=>e.relation==='implicit_calls'&&e.source.startsWith('array.'));
  const detail=route({edge:edge.id});assert(detail.includes('源码起点，不是字面函数调用'));assert(detail.includes('__range.begin()')||detail.includes('__range.end()'));
});
test('TMEM protocol event keeps API, predicate and thread domain visible',()=>{
  const text=route({protocol:'dense.protocol_draft.tmem_lifetime',event:'tmem.cta0.arrive793'});
  assert(text.includes('对应API完整声明'));assert(text.includes('predicate_value'));
  assert(text.includes('false'));assert(text.includes('instance_domain'));
  assert(text.includes('793</a>'));assert(text.includes('关联分图'));
});
test('protocol navigation exposes quantified order and logical pair validity separately',()=>{
  const p=data.protocols.find(p=>p.id==='dense.protocol_draft.tmem_lifetime');
  const view=p.views.find(v=>v.kind==='partial_order');
  const text=route({protocol:p.id,protocol_view:view.id});
  assert(text.includes('本图的明示约束'));assert(text.includes('source_domain_ref'));
  assert(text.includes('配对参与的必要条件'));assert(text.includes('paired_deallocation_participation_validated'));
  assert(text.includes('不是配对物理释放时刻'));
});
test('logical state view shows derivation and never treats it as a physical timeline',()=>{
  const p=data.protocols[0],view=p.views.find(v=>v.kind==='state');
  const text=route({protocol:p.id,protocol_view:view.id});
  assert(text.includes('历史事实的状态推导'));assert(text.includes('adds_temporal_order'));
  assert(text.includes('不产生新的时间边'));
});
console.log(JSON.stringify({unit_tests_passed:passed,actual_browser_used:false,viewport_acceptance_claimed:false}));
