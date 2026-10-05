#!/usr/bin/env python3
"""Stage the selected CLC response publication/consumption/reuse handoff."""
import hashlib
from stage_scheduler_handoff import ROOT,SCHED,KERNEL,span,call_at,dump

PIPE='include/cutlass/pipeline/sm100_pipeline.hpp'
BARRIER='include/cutlass/arch/barrier.h'
P='cutlass::PipelineCLCFetchAsync::'
S='cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100::'
DECLS=[
 ('type',PIPE,934,935,'class_specifier','cutlass::PipelineCLCFetchAsync','CLC响应流水线，full按transaction bytes就绪，empty按消费者到达允许复用。'),
 ('advance',SCHED,438,440,'method',S+'advance_to_next_work','生产者先取得可复用槽并登记期待字节，再由elected lane发出CLC请求。'),
 ('issue',SCHED,392,394,'method',S+'issue_clc_query','发起异步CLC取消查询，响应多播到集群共享内存；不是响应完成通知。'),
 ('decode',SCHED,409,411,'method',S+'work_tile_info_from_clc_response','在响应可读后解码取消结果，valid决定是否有下一任务，不决定是否应释放响应槽。'),
 ('barrier_address',PIPE,1072,1073,'method',P+'producer_get_barrier','返回当前full屏障的共享内存地址，不等待也不完成事务。'),
 ('acquire',PIPE,1026,1027,'method',P+'producer_acquire','PipelineState重载将index、phase和token分别传给底层重载。'),
 ('acquire_stage',PIPE,1094,1095,'method',P+'producer_acquire','等待empty后，按lane目标为各CTA登记full到达与期待的响应字节。'),
 ('wait',PIPE,1060,1061,'method',P+'consumer_wait','PipelineState重载按当前轮次等待本CTA响应full屏障。'),
 ('wait_stage',PIPE,1123,1124,'method',P+'consumer_wait','WaitAgain token时等待对应stage/phase的full；不能把请求已发出当作WaitDone。'),
 ('release',PIPE,1067,1068,'method',P+'consumer_release','读取响应后按index转发释放通知，不代表所有消费者已经释放。'),
 ('release_stage',PIPE,1131,1132,'method',P+'consumer_release','向producer_blockid对应的empty屏障发送到达通知。'),
 ('tail',PIPE,1040,1041,'method',P+'producer_tail','生产者退出前检查全部环形槽的empty轮次；不重置期待字节，也不完成整个GEMM输出。'),
 ('commit',PIPE,1032,1033,'method',P+'producer_commit','手工完成transaction字节的入口；当前硬件CLC查询路径没有调用它。'),
 ('commit_stage',PIPE,1106,1107,'method',P+'producer_commit','手工complete_transaction，当前函数体不使用phase参数。'),
 ('expect_remote',BARRIER,558,559,'method','cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx','向指定CTA的屏障登记到达及期待字节，区别于单参数本地重载。'),
 ('complete_remote',BARRIER,577,578,'method','cutlass::arch::ClusterTransactionBarrier::complete_transaction','软件手工完成指定CTA的transaction字节，不能等同于发起真实CLC请求。'),
]


def build():
 nodes=[];edges=[]
 for key,path,first,last,kind,qname,notes in DECLS:
  d=span(path,first,last,True)
  nodes.append({'id':'clc.'+key,'kind':'type' if kind=='class_specifier' else 'api',
    'name':qname.split('::')[-1], 'qualified_name':qname,'path':path,'line':last,
    'role':'selected_clc_handoff','notes':notes,
    'source_selector':{'path':path,'kind':kind,'qualified_name':qname,
     'signature_range':{k:v for k,v in d.items() if k!='source_text'},
     'signature_sha256':hashlib.sha256(d['source_text'].encode()).hexdigest()}})
 for key,name,path,line,notes in [
  ('response','CLCResponse[stage]',KERNEL,200,'各CTA本地共享内存中的16字节响应槽；相同index仍需区分轮次。'),
  ('full','CLC full[stage]（各CTA本地）',PIPE,928,'full结合到达计数和期待transaction字节，表示对应轮次响应就绪。'),
  ('empty','CLC empty[stage]（生产者CTA）',PIPE,929,'生产者等待的empty槽，需收到约定消费者到达，单个release不足以代表全部读完。'),
 ]:
  nodes.append({'id':'clc.res.'+key,'kind':'binding','name':name,'path':path,'line':line,'role':'runtime_resource_binding','notes':notes})
 nodes.append({'id':'clc.hw.query','kind':'hardware_event','name':'CLC异步查询与响应写入','path':SCHED,'line':400,
               'role':'asynchronous_hardware_operation','notes':'启用CLC时发起b128响应的集群多播；complete_tx是硬件效果，API返回不表示响应已经写完。'})
 calls=[
  ('host.kernel.operator','clc.advance',KERNEL,697,'scheduler.advance_to_next_work','is_participant.sched && IsSchedDynamicPersistent && requires_clc_query；仅集群首CTA的Scheduler角色'),
  ('host.kernel.operator','clc.tail',KERNEL,721,'clc_pipeline.producer_tail','Scheduler循环结束后，生产者等待各响应槽的消费者释放'),
  ('clc.advance','clc.barrier_address',SCHED,441,'clc_pipeline.producer_get_barrier','取得当前full屏障地址；不是等待'),
  ('clc.advance','clc.acquire',SCHED,443,'clc_pipeline.producer_acquire','发起请求前等待可复用响应槽并登记期待字节'),
  ('clc.advance','clc.issue',SCHED,446,'issue_clc_query','cute::elect_one_sync()成立的lane发起；其他lane仍参与此前的barrier登记'),
  ('scheduler.fetch','clc.wait',SCHED,462,'scheduler_pipeline.consumer_wait','已选PipelineCLCFetchAsync；读取该stage响应之前'),
  ('scheduler.fetch','clc.decode',SCHED,464,'work_tile_info_from_clc_response','consumer_wait后读取响应；有效与无效响应都走此调用'),
  ('scheduler.fetch','clc.release',SCHED,465,'scheduler_pipeline.consumer_release','响应已复制到局部WorkTileInfo；解码无效也执行，不等下一任务有效才释放'),
  ('clc.acquire','clc.acquire_stage',PIPE,1028,'producer_acquire','以state.index()/state.phase()和token转发；两个重载不合并'),
  ('clc.wait','clc.wait_stage',PIPE,1062,'consumer_wait','以state.index()/state.phase()和token转发'),
  ('clc.release','clc.release_stage',PIPE,1069,'consumer_release','仅传stage index；目标CTA来自Params'),
  ('clc.acquire_stage','contract.api.cluster_wait_member',PIPE,1100,'empty_barrier_ptr_[stage].wait','barrier_token==WaitAgain；等待生产者CTA的empty对应轮次'),
  ('clc.acquire_stage','clc.expect_remote',PIPE,1103,'full_barrier_ptr_[stage].arrive_and_expect_tx','lane_idx_ < cluster_size_时作用于对应CTA；每个目标登记transaction_bytes'),
  ('clc.wait_stage','contract.api.cluster_wait_member',PIPE,1127,'full_barrier_ptr_[stage].wait','barrier_token==WaitAgain；FullBarrier继承ClusterBarrier的wait'),
  ('clc.release_stage','contract.api.dealloc_arrive',PIPE,1134,'empty_barrier_ptr_[stage].arrive','目标是params_.producer_blockid；复用已有ClusterBarrier::arrive声明，不是TMEM释放事件'),
  ('clc.tail','contract.api.cluster_wait_member',PIPE,1046,'empty_barrier_ptr_[state.index()].wait','逐槽test_wait未通过时等待；不重新登记transaction字节'),
  ('clc.commit','clc.commit_stage',PIPE,1034,'producer_commit','手工完成入口；未声称当前Scheduler.advance调用它'),
  ('clc.commit_stage','clc.complete_remote',PIPE,1109,'full_barrier_ptr_[stage].complete_transaction','lane_idx_ < cluster_size_；当前分支cluster z=1，手工完成与硬件complete_tx不能混用解释'),
 ]
 for source,target,path,line,expression,condition in calls:
  point=call_at(line,expression,path)
  edges.append({'id':'clc.call.'+source.replace('.','_')+'.'+str(line),'source':source,'target':target,
   'relation':'calls','resolution':'configured','condition':condition,'source_expression':point['source_expression'],
   'callsite':point,'evidence':[span(path,point['start_line'],point['end_line'])]})
 def effect(key,source,target,relation,condition,path,first,last,expression):
  edges.append({'id':'clc.effect.'+key,'source':source,'target':target,'relation':relation,'resolution':'source_proven',
   'condition':condition,'source_expression':expression,'evidence':[span(path,first,last)]})
 effect('issue','clc.issue','clc.hw.query','issues','defined(CUTLASS_ARCH_CLC_ENABLED)；否则走NOT_IMPLEMENTED，不是备用有效响应',SCHED,392,407,'clusterlaunchcontrol.try_cancel.async...b128')
 effect('write_response','clc.hw.query','clc.res.response','writes','异步硬件响应写入，不表示issue函数返回时已经完成',SCHED,396,403,'multicast::cluster::all.b128')
 effect('notify_full','clc.hw.query','clc.res.full','signals','响应写入的complete_tx硬件效果；消费仍须满足full的相应phase/字节条件',SCHED,400,403,'mbarrier::complete_tx::bytes')
 effect('wait_full','clc.wait_stage','clc.res.full','waits_for','WaitAgain路径等待对应轮次响应；已有正确WaitDone token的路径可不再次等待',PIPE,1123,1128,'full_barrier_ptr_[stage].wait(phase)')
 effect('read_response','clc.decode','clc.res.response','reads','在fetch中已经完成consumer_wait后解码；不是请求发出后立即读',SCHED,415,435,'ld.shared.b128; query_cancel.is_canceled')
 effect('release_empty','clc.release_stage','clc.res.empty','signals','一次消费者到达，不能单独推出约定的全部消费者均已结束',PIPE,1131,1135,'empty_barrier_ptr_[stage].arrive(params_.producer_blockid)')
 effect('wait_empty','clc.acquire_stage','clc.res.empty','waits_for','生产者等待本轮empty；复用需要正确token/index/phase与到达计数',PIPE,1094,1103,'empty_barrier_ptr_[stage].wait(phase)')
 groups=[('publish','CLC响应发布：先取得空槽，再发起请求',{'clc.advance','clc.barrier_address','clc.acquire','clc.acquire_stage','clc.issue','clc.hw.query'}),
         ('consume','CLC响应消费与复用：有效和无效响应都要释放',{'scheduler.fetch','clc.wait','clc.wait_stage','clc.decode','clc.release','clc.release_stage','clc.tail'})]
 views=[]
 for key,title,subjects in groups:
  selected=[e for e in edges if e['source'] in subjects or (e['source']=='host.kernel.operator' and e['target'] in subjects)]
  views.append({'id':'clc.handoff.'+key,'title':title,'node_ids':list(dict.fromkeys(n for e in selected for n in (e['source'],e['target']))),'edge_ids':[e['id'] for e in selected]})
 # The manual completion API remains separately inspectable, not inserted into
 # the normal hardware response timeline.
 manual=[e for e in edges if e['source'] in {'clc.commit','clc.commit_stage'}]
 views.append({'id':'clc.handoff.manual','title':'CLC手工完成接口：不是当前查询路径的必经调用',
  'node_ids':list(dict.fromkeys(n for e in manual for n in (e['source'],e['target']))),'edge_ids':[e['id'] for e in manual]})
 contract={'id':'clc.contract.response_lifetime','kind':'source_contract','title':'CLC响应槽的发布、消费和复用条件',
  'participants':['host.kernel.operator','clc.type','clc.advance','scheduler.fetch','clc.tail'],
  'claims':[
   'Kernel仅让集群首CTA的Scheduler角色发起查询；Params.role的ProducerConsumer标记不等于其他CTA的Scheduler线程也实际参与。',
   '每个响应是16字节。producer_arv_count=1；前cluster_size个lane按目标CTA分别登记full到达与transaction_bytes，发出CLC指令则由elected lane完成。',
   'consumer_arv_count按实际线程计：32 + cluster_size*(32+128+32)，若需要Epilogue输入再加cluster_size*32。本例cluster_size=4时分别为800或928，不是4个CTA或若干warp的计数。',
   'fetch先等full，再读响应，再release empty；取消失败/无下一任务的响应也释放，valid决定任务循环是否继续而不是槽是否需要释放。',
   '单个消费者release不等于槽已可覆盖；生产者再次使用前必须观测相应empty条件。state.index定位槽，phase区分轮次，CLC状态与累加器状态独立。',
   'producer_tail等待各槽empty以防过早退出，不重置transaction bytes；正常CLC硬件complete_tx路径没有额外调用producer_commit。'],
  'evidence':[span(KERNEL,449,517),span(KERNEL,590,598),span(KERNEL,680,723),span(PIPE,982,1000),span(PIPE,1026,1135),span(SCHED,392,474),span('include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp',123,128)]}
 dump(ROOT/'data/modules/dense_fp16/clc_handoff.json',{'module_id':'dense_fp16_2sm_sm110a','part':'clc_handoff','nodes':nodes,'edges':edges,'views':views,'contracts':[contract],
  'issues':[{'id':'clc.handoff.remaining','description':'已列关键发布/消费/释放调用与参数公式。初始化屏障辅助函数、throttle全协议、PTX运行时正确性、全部CLC配置及完整swizzle仍未全量核对；未宣称全库或GPU验收。'}]})
 print({'nodes':len(nodes),'edges':len(edges),'views':[(v['id'],len(v['edge_ids'])) for v in views]})

if __name__=='__main__':build()
