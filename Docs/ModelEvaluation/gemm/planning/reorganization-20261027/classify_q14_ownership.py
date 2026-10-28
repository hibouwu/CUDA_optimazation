"""生成 Q14 问题归属审阅清单；只写本目录，不移动或修改结果。"""
from pathlib import Path
import collections, csv, hashlib, json, os, re

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'results/gh200_resource_campaign/access_rules'
Q14 = BASE / 'Q14-composition'
ROUNDS = {p.name: p for fam in Q14.iterdir() if fam.is_dir()
          for p in fam.iterdir() if p.is_dir()}
assert len(ROUNDS) == 113
ROWS = {}

def assign(owner, topic, names, obj, variable, endpoint, reason, refs='', kind='实验/模型检验', unresolved=''):
    for name in names.split():
        matches = [p for n,p in ROUNDS.items() if n == name or n.endswith('-Q14-'+name)]
        assert len(matches) == 1, (name, matches)
        p = matches[0]
        assert p.name not in ROWS, p.name
        ROWS[p.name] = dict(old_path=str(p.relative_to(BASE)), round=p.name,
            previous_family=p.parent.name, proposed_question=owner, review_topic=topic,
            object=obj, variable=variable, endpoint=endpoint, reason=reason,
            cross_references=refs, record_kind=kind, unresolved=unresolved,
            action='只列归属；不生成可执行迁移路径')

# gemm-connection：按主问题归属，不按是否运行完整 GEMM、目录名或结果是否成功归属。
assign('Q02','请求共享、路由与应答', '''
source-line-pool-001 domain-line-pool-001 bounded-sector-merge-001
line-subscription-screen-001 address-routed-input-001 request-reply-network-001
read-sharing-network-001 bounded-read-fanout-001 source-only-calibration-001
''','输入请求的共享服务与逐目的应答','共享粒度/范围、路由、订阅上限或该服务价格',
       '请求释放/早晚窗口及 G3 L/plain 的响应；各轮覆盖不同',
       '被检验的是同一输入路径的服务组织；GEMM 是约束该候选的负载，不是另一个归属理由。', 'Q03;Q14')
assign('Q02','等值源重用干预', '''
equal-value-source-001 equal-value-source-002 equal-value-source-003 equal-value-source-004 source-fold-replay-001
''','跨 CTA 输入请求的源重用关系','保持计算值和请求形状，折叠同行 A 源地址；含换卡、换序与模型回放',
       'CTA0 两段 L 的行距响应、plain 和观察/桥接资格',
       '主问题是请求源重用如何改变输入供给。源覆盖也可能影响缓存，故交叉引用 Q03，不把归属当成物理归因。', 'Q03;Q14')
assign('Q02','请求准备、形成与返回', '''
producer-preparation-transfer-001 formation-before-cache-001 formation-return-001 address-matched-return-001
''','producer 请求准备/形成/返回','准备价格、形成与缓存先后、返回反馈；同地址评分修正',
       '请求 release/早晚窗口，或接回 GEMM 的 L/plain 响应',
       '主要替换或纠正请求侧规则，尚未在这一轮共同检验独立计算/输出规则。修复和回归跟随原请求实验。', 'Q03;Q04;Q14')
assign('Q02','发布到请求完成的观测与核对', '''
full-entry-mask-001 full-entry-mask-002 producer-publication-001 producer-publication-002
producer-publication-003 publication-time-alignment-001 published-request-age-001
full-boundary-screen-001 query-return-connection-001 operand-completion-001
''','请求发布后、消费者查询时是否已完成','观察位图/发布标记/时戳或 A/B 请求分组；时间线原点修正',
       '发布计数、ready/unready、发布后年龄区间或 A/B 完成集合',
       '查询 full 是读取请求完成状态；这些轮次没有独立测得 full 已完成之后的 wait 返回耗时，主归 Q02。', 'Q04;Q14')
assign('Q02','输入读取和目的投递的容量约束', '''
copy-work-bound-001 gpc-delivery-bound-001 independent-service-bounds-001
''','输入请求的读取/逐副本投递服务','固定服务拓扑、工作量和候选价格的必要约束',
       '服务时间上/下界、G3 L/plain 或纯 TMA 完整窗口',
       'copy 指向输入数据的逐 CTA 副本投递，不是 GEMM 输出写回；界的推导仍属于输入服务。', 'Q03;Q14', '模型筛查/必要界')
assign('Q03','缓存与源地址状态', '''
banked-source-cache-001 port-ordered-cache-001 public-fold-correction-001
''','源缓存的分区、组索引和查询状态','组相联结构、端口受理后查询、分区折叠转写修正',
       '缓存/源状态及 G3 L/plain 响应；修正轮的语义一致性',
       '改变的是 L2/HBM 源状态或地址映射规则；候选公开配置用于 GH200 的资格仍是工作假设。', 'Q02;Q14')
assign('Q03','逐 CTA 私有源工作集', '''private-source-001 private-source-transfer-001''',
       '源数据工作集及复用状态','相同值和形状的逐 CTA 独立副本，及对应地址/暖机接线',
       '原源/副本的两段 L、plain 和对应预测',
       '预注册目标是大源工作集响应，主归 Q03；不据此声称缓存容量已被单独识别，Q02 保留请求共享关联。', 'Q02;Q14')
assign('Q03','源读取工作量计数', 'input-counter-localization-001',
       '真实 GEMM 的 L2 读取工作量','base/A16 行距对照',
       'NCU L2 sector 比值和计数采集资格',
       '直接目标是源读取量；计数采集失败仍保留在此问题，不能用成功 plain 运行替代计数结果。', 'Q02', '计数实验（失败历史保留）')
assign('Q04','full 等待策略', '''full-wait-policy-001 runtime-full-wait-001 full-wait-plain-002 stride-wait-policy-001''',
       'consumer full 等待的控制路径','blocking/polling；部分轮次加 stage 或行距对照',
       '等待策略的 plain 响应及可用的 L 响应/资格',
       '被操纵的是 full 等待策略，而非请求源或计算服务；runtime 只是同二进制控制的实现方式。', 'Q02;Q14')
assign('Q04','活动槽与复用', 'stage-scope-screen-001',
       '固定物理 S6 下的槽复用','已有活动4/6槽记录的只读对照',
       'FIRST_MMA→MAIN_END 两段 cycle 响应',
       '问题是活动槽/复用是否足以产生响应；不是重新研究物理 SMEM 容量或全系统模型。', 'Q07;Q14', '已有数据分析')
assign('Q05','计算完成规则与正常 WGMMA 等待', '''
compute-completion-transfer-001 gemm-wait-001 gemm-wait-prediction-001
''','WGMMA 计算完成和正常 wait 连接','完成滞后 d0/d95 或正常 wait1/wait0',
       'S6/S4 的 L、plain 及完成规则的预测差',
       '主判别针对计算完成/等待规则，其余服务固定；在 GEMM 中检验单机制的迁移不自动变成 Q14。', 'Q04;Q14')
assign('Q06','计算尾部与跨输出交接', '''compute-handoff-contract-001 runtime-tail-001 wait0-boundary-001''',
       '跨输出计算状态或末尾 wait0 到后续结束边界','跨输出状态传递、循环外尾部观察的实现',
       '逐节点一致性，或末尾 wait0 时戳与结束/跨度资格',
       '对象是数学 WG 交接和尾部阶段；状态契约与失败观察均作为该机制的支撑轮次保留。', 'Q05;Q10', '机制支撑/观察实验')
assign('Q07','TMA 与 WGMMA 的共同 SMEM 服务', '''
joint-smem-bound-001 local-joint-service-001 gather-joint-service-001
''','SMEM 的输入写入与计算操作数读取','独立/共同资源、B 复用、逻辑字节/源片段计费',
       '原生 D 必要下界，或 GEMM 的 L/plain/full 未完成响应',
       '虽然涉及两条路径，但 Q07 已专门研究 SMEM 多路共享；不重复归 Q14。下界和模型回放的端点差别写入主题族 README。', 'Q02;Q05;Q14')
assign('Q10','计时与桥接重复性', 'bridge-null-001',
       '同二进制对照的可重复性','相邻重复、交换顺序',
       'plain、L cycle/ns 的配对偏差',
       '用于判断测量/桥接是否可靠，没有操纵输入或计算的硬件服务。', '', '测量资格对照')
assign('Q14','同行活动的整体效应', 'active-peer-001',
       '其他 CTA 活动对目标 CTA 的影响','132/1 活动 CTA，前缀与目标工作列表受控',
       'CTA0 两段 L 行距响应与 plain',
       '关闭同行同时去掉输入、计算和输出，测的是同行多路径活动的聚合影响，默认归 Q14；尚未定位到单一路径。',
       'Q02;Q03;Q07', '实机聚合交互定位', '科学结论仍未定位到单一路径；目录主归属已确定为 Q14')
assign('Q02','源组与逻辑工作组的空间跟随', '''
completion-spatial-screen-001 raster-contrast-prediction-001 raster-age-discrimination-001
''','请求年龄的空间分布','分组分析与 raster 改变源坐标/逻辑工作组对应',
       '中心化年龄、空间投影与控制重复性',
       '测量对象是发布后的请求年龄，默认主归 Q02；raster 是改变源坐标与逻辑工作组对应的手段，三轮保持同一主题。',
       'Q01;Q03', '定位诊断/预测/实机尝试', '源组或逻辑组尚未区分，后续也可能归 Q03；当前不阻塞 Q02 的目录归属')
assign('Q14','输出推进与输入/计算的交互', '''
epilogue-defer-001 shared-output-service-001 output-write-gemm-001 peer-output-001
''','输出工作对主循环输入/计算推进的影响','延后输出依赖、输入输出共享占用、屏蔽同行输出',
       '另一阶段/CTA 的 early、post、L 响应及同版 plain 回归',
       '不是测输出写回自身的延迟，而是检验输出改变另一条执行路径的时间；具体共同资源尚未唯一定位。', 'Q02;Q06;Q07')
assign('Q14','计算量与输入等待/槽复用的交互', '''
runtime-intervention-001 stride-compute-discrimination-001 direct-repeat-path-001
static-stage-repeat-001 static-repeat-dispatch-001 program-budget-scope-001
''','计算推进能否隐藏输入行距代价或槽复用等待','单遍/三遍×行距或活动槽；实现桥接及程序预算诊断',
       '行距/活动槽响应随计算量的变化、L 与 plain',
       '判别对象是一个因素的代价怎样随另一因素改变，属于跨计算、供给和复用的交互。program-budget 是此对照的诊断支撑，不另算机制实验。', 'Q02;Q04;Q05')
assign('Q14','完成、继续、释放规则的联合迁移与关键路径', '''
local-bundle-transfer-001 local-bundle-regression-001 local-bundle-G6-001 critical-path-current-001
''','多项局部规则接回同一程序后的推进与瓶颈','计算完成+消费者继续+empty 完成的联合替换；扩展条件和同版关键路径分析',
       '多条件 L/plain、stage 响应及关键 CTA/依赖路径',
       '直接检验多个已有局部规则共同决定的时间，不是寻找某一个局部服务价格；G6 补测留在同一联合验证。', 'Q04;Q05;Q15')
assign('非实验','跨模块程序接入', 'program-intervention-001',
       '共同程序是否准确表达已测源折叠及输出 copy 开关','修正程序事实和服务接线',
       '请求身份、输出计数、默认回归与已有模型响应',
       '本轮核心交付是跨模块程序接线；作为源/输出实验引用的 integration 记录，避免把纯实现接入单立为组合机制实验。',
       'Q02;Q06;Q14', 'non-experiment/integration 候选')

# COMPUTE 是有比较关系的递进系列：整组保留 Q14，各级机制由其他 Q 交叉引用。
C='20261017-COMPUTE-'
assign('Q14','逐步组合系列：A 驻留计算基线', ' '.join(C+x for x in ['01-preparation-001','01-execution-001','01-execution-002','01-fit-001','01-heldout-001']),
       '操作数驻留时的计算体','Ktile 长度4/13/32，plain/trace',
       '完整驻留计算窗口 cycle、长度外推与观察资格',
       'A 提供后续 B/C/D/E 的驻留计算基线；为保留比较链，整组留 Q14。Q05 引用 A 的计算规律，不另存一份。', 'Q05', '逐步组合系列：准备/执行/拟合/评分链')
assign('Q14','逐步组合系列：B/C 同步与复用', ' '.join(C+x for x in ['02-handoff-preparation-001','02-execution-001','02-prediction-001','02-score-001']),
       '驻留操作数下的输入同步和槽复用','B/C 首次复用后延迟，S4/S6',
       'C−B 窗口和跨 stage 响应、同步观察资格',
       'B/C 是 A 到真实 TMA 链之间的同步/复用对照，保留系列完整性；Q04 引用该级，不拆开准备、预测、执行和评分。', 'Q04;Q05', '逐步组合系列：准备/预测/执行/评分链')
assign('Q14','逐步组合系列：D 真实 TMA 短链', ' '.join(C+x for x in ['03-tma-preparation-001','03-execution-001','03-score-001']),
       '真实供给、计算和槽复用组成的 D 短链','相对于 B/C 接入全部 TMA；Ktile 长度和 stage',
       'D 完整窗口、D−B 及 S4−S6 响应',
       '主窗口包含全部 TMA、计算和释放，D−B 明确只是完整程序响应，不能当作 TMA 独立服务价；因此保留组合短链。', 'Q02;Q04;Q05', '准备/执行/评分链')
assign('Q14','逐步组合系列：E pingpong 与输出', ' '.join(C+x for x in ['04-pingpong-preparation-001','04-execution-001','04-score-001']),
       'WG0→WG1→WG0 交接和输出排空','真实单 CTA 三输出 pingpong，plain/trace',
       '角色调用边界、输出/交接见证、完整窗口与观察资格',
       'E 补入真实双数学 WG 和输出生命周期，整组留 Q14 以保留与前级比较；Q06 引用该级。程序组织/资源也有变化，差值不能拆成裸交接价格。', 'Q02;Q04;Q06', '逐步组合系列：准备/执行/评分链')

# cfg-b-composition 中真正的输入服务、组合验证和非实验汇总分别归位。
B='20261015-cfg-b-'
assign('Q02','输入信用、服务和源组织', ' '.join(B+x+'-001' for x in [
    'async-source-interface','dual-granule-service','program-source-adapter','resident-arbitration',
    'return-credit-candidate','service-price-check','sharing-spatial-audit','size-lanes-service',
    'source-scope-discrimination','stateful-probe-replay']),
       '输入请求的源组织、信用、仲裁和逐目的完成','源映射、服务粒度/价格/信用；相关地址和回调接口',
       '纯 TMA 请求窗口、CTA 释放跨度、语义约束；部分轮次带 G3 回归',
       '逐轮内容主要约束输入服务；不少候选在纯探针门槛已停止，根本没有进入 GEMM。接口记录跟随 Q02 实验。', 'Q03;Q14', '输入服务实验/相关实现支撑')
assign('Q01','启动前缀', '20261016-cfg-b-prefix-001',
       'entry→producer_first_work','每 CTA 固定 P0 前缀接入',
       '启动前缀及原条件时间响应',
       '唯一新增项是启动段；完整时间回归只验证其影响，不能据 cfg-b 名称归组合。', 'Q10;Q14', '单机制经验接入/回归')
assign('Q14','计算活跃对 producer 发出的影响', B+'local-producer-candidate-001',
       '本地计算活跃与输入发出的耦合','只在本 CTA 计算活跃时给 producer 指令增加延迟',
       'S6/S4 的 L、stage 响应、CTA 空间跨度和 plain',
       '规则显式依赖计算状态来改变发出，研究跨计算与输入路径的交互；尚未归因到某一共同硬件资源。', 'Q02;Q05;Q07')
assign('Q14','完整服务组合的阶段回归与同版基线', '''
20261015-cfg-b-combination-regression-001 20261015-cfg-b-full-matrix-001
20261017-cfg-b-system-001 20261020-Q14-current-system-001
''','程序、源状态、计算、释放和输出组成的统一时间线',
       '同一整套规则接入条件矩阵；完整事件依赖/默认版本核对',
       '阶段端点、请求级依赖、CTA 跨度及 plain 对照',
       '评估的是多模块在同一程序中的联合解释和可执行基线。含交付/回归支撑，不声称每个目录都是新机制实验；纯完整时间验收由 Q15 引用。', 'Q01;Q02;Q03;Q04;Q05;Q06;Q09;Q10;Q15', '组合回归/同版基线')
assign('Q15','冻结观测与评分验收', '20261011-V09-gate-audit',
       'V09 冻结预测和验收指标','聚合误差、关键 CTA 指标与观察扰动的审计',
       '分项误差、完整时间与验收覆盖',
       '对象是模型验证/评分合同，不是新增机制；建议回 Q15/V09，与请求的 Q01–Q13 回归分开列出。', 'Q10;Q14', '已有验证数据审计')
assign('非实验','多问题模块交付汇总', '20261017-module-campaign-001',
       '九方向材料的汇总与交付状态','没有单一实验操纵变量',
       '模块证据覆盖、接口和剩余缺口',
       '这是跨问题汇总包；各来源实验独立归属，本包建议 non-experiment/integration。', 'Q01–Q14', 'non-experiment/integration 候选')
assign('混合批次','历史建模窗口（包内实验另列）', '20261016-cfg-b-window-001',
       '一个执行窗口内的多个不同候选','27 个独立登记的候选及配套筛查',
       '不同轮次各自的窗口/阶段/完整时间端点',
       '不能把整个包称为一个 Q14 实验，也不能强归一个局部问题。保留包边界，建议 non-experiment/campaigns；子轮逻辑归属见附表。',
       'Q02;Q03;Q04;Q05;Q14', 'non-experiment/campaigns 候选')

# 程序图不是性能实验；具体请求和源分配查询可归相应机制的证据记录。
assign('非实验','跨机制程序图与端点绑定', '''
20261015-cfg-a-program-001 20261015-cfg-a-stage4-program-001
20261015-cfg-a-boundary-program-001 20261015-cfg-a-long-trace-program-001
20261015-cfg-c-program-001 20261015-cfg-c1-program-001
''','共同程序的工作列表、请求、同步和输出依赖','配置/边界/长工作列表的程序支持范围',
       '字节和请求守恒、节点/角色端点一致性、默认图回归',
       '导出和核对程序事实，没有检验硬件时间交互；建议 non-experiment/integration，相关 Q 页面引用同一证据。', 'Q01;Q02;Q04;Q06;Q12;Q14', 'non-experiment/integration 候选')
assign('Q02','真实请求描述符', '20261015-cfg-descriptor-query-001',
       '实际 launch 参数中的 tensor-map','八候选、两种形状的 setup 查询',
       'rank/shape/stride/box/swizzle 与实际描述符绑定',
       '为 TMA 请求定义提供事实证据；不是性能测量，也不是组合交互实验。', 'non-experiment/integration（程序图）;Q01', '请求事实查询/准备')
assign('Q03','源分配和调用前缓存准备', '20261015-g3-allocation-query-001',
       'A/B/D/workspace/scratch 地址与调用前访问','S6/S4、plain/dual 的 host 分配记录',
       '指针偏移、scratch 大小及计时前清零事实',
       '用于源地址和缓存初态建模。只证明访问事实，没有证明清零在 L2 的实际分配行为。', 'Q10;Q14', '源状态事实查询')

assert set(ROWS)==set(ROUNDS), {'missing':sorted(set(ROUNDS)-set(ROWS)), 'extra':sorted(set(ROWS)-set(ROUNDS))}

# 混合包只给逻辑分类，不提出拆包或搬动子目录。
WINDOW = Q14/'cfg-b-composition/20261016-cfg-b-window-001'
CHILD_OWNERS = {
 1:('Q02','共享域返回服务'), 2:('Q14','输入/输出共用接入槽'),
 3:('Q02','输入行信用'),4:('Q02','请求覆盖计费'),
 5:('Q14','计算活跃排斥 producer 发出'),6:('Q14','提交前计算活动约束发出'),
 7:('Q02','逐 sector 订阅'),8:('Q02','请求 tag 覆盖计费'),
 9:('Q03','缓存访问时点'),10:('Q02','输入源服务仲裁'),
 11:('Q03','HBM miss 容量'),12:('Q03','HBM 发出准入'),
 13:('Q03','缓存副本 sector 有效性'),14:('Q03','dirty owner 写回'),
 15:('Q14','缓存慢路向请求源槽反压'),16:('Q05','计算提交与完成连接'),
 17:('Q02','按 home 路由请求服务'),18:('Q03','调用前清零带来的缓存初态'),
 19:('Q02','输入 atom 接入与 home 分部汇合'),20:('Q02','共享目的投递'),
 21:('Q02','GPC 输入信用'),22:('Q03','缓存远端副本返回路径'),
 23:('Q02','目的分块轮转'),24:('Q04','wait1/release 软件窗口'),
 25:('Q02','请求完成计量'),26:('Q02','请求本地准备与接入'),
 27:('Q02','目的并发及参数校准')}

def write_tsv(name, fields, rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(rows)

def evidence(p):
    names=['README.md','plan.json','PLAN.md','EXECUTION.md','RESULTS.md']
    found=[p/n for n in names if (p/n).is_file()]
    if not found:
        found=[p/n for n in ['fit.json','dev-freeze.json','score.json','report.json'] if (p/n).is_file()]
    assert found, p
    return found

def main():
    snapshots={}; rows=[]
    for name,row in sorted(ROWS.items()):
        p=ROUNDS[name];fs=evidence(p)
        row['evidence']=';'.join(str(f.relative_to(ROOT)) for f in fs)
        row['evidence_titles']=';'.join(f.read_text(errors='replace').splitlines()[0][:180] for f in fs if f.suffix=='.md')
        row['decision_status']='建议主归属，待用户审阅'
        rows.append(row)
        for f in fs:
            b=f.read_bytes(); snapshots[str(f.relative_to(ROOT))]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode(errors='replace')}
    fields=['old_path','round','previous_family','proposed_question','review_topic','record_kind','object','variable','endpoint','reason','cross_references','unresolved','evidence','evidence_titles','decision_status','action']
    write_tsv('Q14-question-ownership.tsv',fields,rows)
    children=[]
    for p in sorted(WINDOW.glob('r[0-9][0-9]-*')):
        i=int(p.name[1:3]);owner,topic=CHILD_OWNERS[i];fs=evidence(p)
        children.append(dict(old_path=str(p.relative_to(BASE)),proposed_question=owner,topic=topic,
                             evidence=';'.join(str(f.relative_to(ROOT)) for f in fs),action='仅逻辑归属；保留混合包，不拆文件'))
        for f in fs:
            b=f.read_bytes();snapshots[str(f.relative_to(ROOT))]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode(errors='replace')}
    assert len(children)==27
    write_tsv('Q14-mixed-window-ownership.tsv',['old_path','proposed_question','topic','evidence','action'],children)
    with (OUT/'Q14-ownership-evidence.jsonl').open('w') as f:
        for path,v in sorted(snapshots.items()): f.write(json.dumps({'path':path,**v},ensure_ascii=False)+'\n')
    counts=collections.Counter(r['proposed_question'] for r in rows)
    family_counts={fam:dict(collections.Counter(r['proposed_question'] for r in rows if r['previous_family']==fam)) for fam in sorted({r['previous_family'] for r in rows})}
    q02_existing=sum(p.is_dir() for fam in (BASE/'Q02-tma-request').iterdir() if fam.is_dir() for p in fam.iterdir())
    summary=dict(directory_count=len(rows),question_counts=dict(sorted(counts.items())),by_old_family=family_counts,
                 mixed_children_count=len(children),mixed_children_counts=dict(collections.Counter(r['proposed_question'] for r in children)),
                 q02_existing=q02_existing,q02_incoming=counts['Q02'],q02_projected_before_other_outgoing=q02_existing+counts['Q02'],
                 source_documents=len(snapshots),original_results_modified=False,original_results_moved=False)
    (OUT/'Q14-ownership-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    render(rows,children,summary)
    print(json.dumps(summary,ensure_ascii=False))

def render(rows,children,summary):
    lines=['# Q14 问题归属调整清单','',
      '本次核对 Q14 下 4 个现有族、113 个轮次目录。目录数包含准备、拟合、评分、实现支撑和汇总，并非 113 个独立实验。只提出主问题归属，不移动目录、不修改实验内容、不修历史链接。', '',
      '当前采用的组织口径：族按主题划分，族内用表格逐轮列出对象、变量、端点、条件与结论；不把每个细微差别单独变成一个族。本轮先校正第一层问题归属，下表的主题仍待与目标 Q 的现有族一起整合。此前仅在 Q14 内提出的 29 族草案已被本清单取代，不能继续当作迁移表。','',
      '## 分类边界','',
      '- 主归属看被检验的服务、事件区间或交互假设。真实 GEMM、CPU 模型、plain 完整时间都可以是验证载体，不能单独决定归 Q14。',
      '- Q02 覆盖请求发出到目的投递/full 事务完成；Q04 覆盖 full 完成后的等待返回、槽释放与再取得。看到 full 位图，不等于在测同步返回耗时。',
      '- Q03 覆盖源地址、缓存/工作集和 HBM 状态；请求合并、订阅、准入及逐目的应答优先 Q02。主归属不代表物理机制已经被证实。',
      '- Q07 已覆盖 TMA/WGMMA/输出对 SMEM 的多路共享。已有专门问题能容纳的交互归该问题，不重复放进 Q14。',
      '- Q14 保留计算量与输入等待、输出与其他路径、跨模块反馈、完整短链与多模块联合规则的阶段验证。相关准备、评分和必要对照跟随主实验，并标明记录性质。',
      '- 纯跨模块实现/汇总放非实验支撑；对能根据测量对象确定主问题的定位轮次，给默认归属并保留科学结论的不确定性，不因物理路径尚未定位而长期挂起。','',
      '## 两项归属说明','',
      '**COMPUTE-01–04 整组保留 Q14，共 15 个目录，作为“逐步组合”系列。** A 是驻留计算基线，B/C 加入同步、复用与延迟对照，D 接入真实 TMA，E 使用真实 pingpong 和输出。Q05 引用 A，Q04 引用 B/C，Q02 引用 D 的请求事实，Q06 引用 E；原件只留一份，各级对象、变量、端点和可比较范围写进同一族 README。',
      '这是一条递进比较链，不是严格可相减的组件定价梯子。A→B 还改变线程/资源和控制组织；D 窗口包含全部 TMA，而 B 首批槽预填；E 也改变角色和工作组织。尤其 D−B 只能报告同条件下完整程序响应，不能直接称为 TMA 独立耗时。','',
      '**full-entry-mask 两轮继续主归 Q02，交叉引用 Q04。** 本实验固定 barrier 协议，用 consumer 进入等待前的 full 状态检验“输入请求是否已完成”，没有以相位计数、代次正确性或完成后返回耗时为操纵对象。Q04 负责 barrier 协议、phase/代次、返回和释放机制，不按“凡使用 mbarrier 都归 Q04”分类。若后续主张变成相位正确性、状态可见性机制或 wait 返回路径，则对应新问题主归 Q04。','',
      '## 数量','', '| 建议去向 | 轮次目录数 |','|---|---:|']
    for owner,n in sorted(summary['question_counts'].items()):lines.append(f'| {owner} | {n} |')
    lines += ['', '逐轮对象、变量、端点、理由和来源见 [113 行 TSV](Q14-question-ownership.tsv)。以下按问题列出所有目录；无重复归档，其他问题只交叉引用。', '',
      '## 保留在 Q14 的内容','',
      '| 主题 | 目录数 | 必须保留在组合问题中的原因 |','|---|---:|---|']
    kept=collections.defaultdict(list)
    for r in rows:
        if r['proposed_question']=='Q14':kept[r['review_topic']].append(r)
    for topic,rs in kept.items():lines.append(f"| {topic} | {len(rs)} | {rs[0]['reason']} |")
    lines += ['', '## 逐问题清单','']
    for owner in sorted(summary['question_counts']):
        rs=[r for r in rows if r['proposed_question']==owner]
        lines += [f"### {owner}（{len(rs)} 个目录）",'']
        topics=collections.defaultdict(list)
        for r in rs:topics[r['review_topic']].append(r)
        for topic,items in topics.items():
            first=items[0]
            lines += [f'**{topic}**','',f"主对象：{first['object']}。",f"操纵/比较：{first['variable']}。",f"端点：{first['endpoint']}。",'',f"归属理由：{first['reason']}", '']
            if first['unresolved']:lines += [f"尚未区分的科学结论：{first['unresolved']}。",'']
            if first['cross_references']:lines += [f"交叉引用：{first['cross_references']}。",'']
            for r in items:
                fs=r['evidence'].split(';');target=next((x for x in fs if x.endswith('.md')),fs[0])
                lines.append(f"- [{r['round']}]({Path(os.path.relpath(ROOT/target, OUT)).as_posix()})")
            lines.append('')
    lines += ['## 混合窗口内 27 个子轮：只调整逻辑归属','',
      '`20261016-cfg-b-window-001` 建议作为混合历史批次保留包边界。下表不增加 113 个顶层目录的统计，也不授权拆包。包内 baseline、各种 screen、执行记录仍作为配套材料；不是另建独立实验。','',
      '| 包内子轮 | 主归属 | 检验对象 |','|---|---|---|']
    for r in children:
        target=next((x for x in r['evidence'].split(';') if x.endswith('RESULTS.md')),r['evidence'].split(';')[0])
        lines.append(f"| [{Path(r['old_path']).name}]({Path(os.path.relpath(ROOT/target, OUT)).as_posix()}) | {r['proposed_question']} | {r['topic']} |")
    lines += ['', '## 审阅状态','',
      '本清单已有全部 113 个目录的建议主归属；没有悬空的待定目录，物理机制未定和失败结论原样保留。Q08/Q09/Q11/Q12/Q13 没有发现需要从本次 Q14 目录独立归回的主实验；程序图的相关内容通过引用覆盖。',
      '族按研究主题组织，主题内部的不同对象、操纵和端点在 README 逐轮表中明确区分；换卡/修复/补测仍跟随原实验。当前尚未建立这些目标族。','',
      '## 联合实施计划（尚未执行）','',
      f"1. **Q02 分族与 Q14 外移合并审阅、一次搬到最终路径。** 当前 Q02 有 {summary['q02_existing']} 个轮次目录，本修订从 Q14 归入 {summary['q02_incoming']} 个，暂计 {summary['q02_projected_before_other_outgoing']} 个。旧版 213+42=255 尚未计入本次归 Q02 的 3 个 raster 轮次。这个数不包含混合包里只增加引用的 14 个子轮，也未扣除未来 Q02 向 Q03 的外移。先合并盘点原有族与新增主题，明确最终族名/归属、冲突和三层深度，再批准一份联合映射表；不先移进临时族后再搬一次。",
      '2. **每个主题族 README 提供逐轮表。** 列日期/卡、轮次、对象或事件区间、操纵变量、读出端点、条件/资格、结论和原始记录链接。失败、修复与补测保留；不同窗口或资格不能共用一句泛化结论。COMPUTE 单独保留 A→B/C→D→E 的对照关系和不可直接相减的边界。',
      '3. **同步重分配 Docs 结论。** 逐条审阅 Q14/gemm-connection.md、COMPUTE-01-resident-chain.md 及相关组合页，建立“旧条目→目标问题页/主题页→原始轮次”的对应表。归 Q02–Q07/Q10 的结论和必要证据链写入目标页，保留原条件、失败/不确定性、数值和来源；不会只改 results 链接而把正文留在错误问题下。',
      '4. **Q14 文档保留组合主线与跨问题入口。** gemm-connection.md 收敛为组合交互及导读，单机制结论用链接指向目标问题页；原有外链或锚点按需留简短指引。COMPUTE 整组的正文和系列对照保留，Q04/Q05/Q06 等问题页添加对应级别的交叉引用。混合窗口保持原包，Docs 只接收逐子轮的逻辑索引。',
      '5. **同次更新两侧索引与核验。** results 根/问题/族 README、Docs 的问题页和 experiments/README.md 跟随同一映射更新；仓库内 Markdown 使用相对路径。核对迁移前后结论条目覆盖、链接目标、冻结身份和目录深度，再按已确认的运行方式检查评分入口。冻结内容不直接重写。',
      '本轮只修订规划文件。上述 Q02 主题族设计、Docs 正文分流、备份与实际迁移尚未执行；前轮大树校验清单不重复生成，原始数据与实验目录保持原位置。','']
    (OUT/'Q14-question-ownership.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
