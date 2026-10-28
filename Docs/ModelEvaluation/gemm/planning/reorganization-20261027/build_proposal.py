"""生成审阅文件；候选路径不用于执行移动。数字索引由固定轮次排序校验。"""
from pathlib import Path
import csv, hashlib, json, os

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'results/gh200_resource_campaign/access_rules'
Q14=BASE/'Q14-composition/gemm-connection'
ROUNDS=sorted(p for p in Q14.iterdir() if p.is_dir())
assert len(ROUNDS)==71, '轮次数变化，必须重新核对'
assert hashlib.sha256('\n'.join(p.name for p in ROUNDS).encode()).hexdigest() == 'e236c0919531cfd8e7132deaa2ba18adcea6055c4c6d6f2e2c14db248bbccb1b', '轮次身份或排序变化，禁止复用索引归类'

# 主题仅用于 README 导航，不产生新的目录层。
GROUPS=[
 ('计算与程序','compute-completion-lag',[0],'组合中的计算完成','完成滞后参数','G3 S6/S4 的 L 与 plain 预测','CPU 局部完成规则迁移；与 wait0 实机干预分开'),
 ('计算与程序','wgmma-wait-policy',[3,4],'GEMM 正常计算等待','正常 wait1/wait0','S6/S4 的 L、plain 与对应预测','预测和实测围绕同一等待干预'),
 ('等待与完成','full-wait-policy',[2,7,14,70],'GEMM consumer full 等待','blocking/polling，含行距对照','plain 响应；具备 dual 的轮次补充 L','14 只有 plain；若要求所有辅助端点也完全相同，应继续拆分'),
 ('等待与完成','wait0-tail-observer',[11,13],'末尾 wait0 返回至结束','循环外时戳观察实现','边界时戳与观察资格','包含失败实现，不能按 runtime 字样另分'),
 ('计算与程序','mainloop-active-repeat',[6,8,12],'固定物理 S6 的主循环','活动槽额度及已测重复计算','FIRST_MMA→MAIN_END、阶段响应','6/12 为已有数据诊断，8 为程序接入；是否并入原 G3 族待核对'),
 ('计算与程序','mainloop-compute-repeat',[19,21,22,52],'GEMM 行距代价的隐藏','单遍/三遍及其实现修复','CTA 两段 L 的行距增量、plain 桥接','恢复执行路径和静态 dispatch 属于同一干预的实现演进'),
 ('计算与程序','gemm-active-peers',[23],'同行 CTA 对目标 CTA 的影响','132/1 活动 CTA','CTA0 两段 L 与 plain','不能并入仅修改源地址或输出 copy 的族'),
 ('输出路径','epilogue-defer-order',[1],'输出与下一数学 WG 交接','E0 permit 延后依赖','C16−C1、M−C16、E0、plain','计划和 EXECUTION 代替缺失的 README；保留独立'),
 ('输出路径','output-shared-service',[5,9],'输出后台工作与输入共享','输出源读取后的共享占用规则','早段/后段响应及 plain 回归','5 是 9 的同版完整时间回归，标为关联回归而非新增物理测量'),
 ('输出路径','gemm-peer-output',[40],'同行 SMEM→global 输出 TMA','同行输出 copy 开关','CTA0 行距周期增量与 plain','输出保留/屏蔽与 epilogue 顺序不是同一种操纵'),
 ('源共享','gemm-source-sharing',[31,32,33,34,44,50,64,65],'真实 GEMM 的跨 CTA 源重用','等值地址折叠/逐 CTA 等值副本','CTA0 两段 L、plain 及对应预测','001–004 换卡、换序与修复保持同族；44/50 是该干预的程序接入'),
 ('源共享','input-cache-organization',[25,63],'输入服务中的源缓存状态','分区组相联/端口受理后查询','G3 两段 L、plain、full 状态','缓存组织候选；不与实机等值源改址直接混合'),
 ('源共享','input-line-merge',[26,30,39,51],'输入源行发出与订阅','32/128B 合并粒度、域范围、合并截止','G3 两段 L、plain、请求工作计数','共同主端点为 G3 行距响应；每轮具体合并规则必须保留'),
 ('读取与请求服务','input-read-reply-network',[24,49,54,66,68,69],'输入读取与逐接收者应答','分区路由、在途共享、订阅上限与服务价格','G3 L/plain，部分轮次另读纯 TMA release 窗口','较宽候选：69 是参数标定，纯 TMA 与 G3 双端点需审阅是否再拆'),
 ('读取与请求服务','tma-formation-return',[53,55,56],'请求形成与 producer 返回','形成服务位置、返回反馈及地址绑定修复','纯 TMA release/早晚窗口；55 另做 GEMM 迁移','53 为同址修正，保留同一次模型谱系；不能归成实机发布时戳实验'),
 ('等待与完成','full-entry-state',[35,36],'consumer 进入等待前的 full 状态','位图观察实现/overlay 修复','各 Ktile ready/unready 位图及扰动资格','实现1未采到机制状态，失败轮次仍保留'),
 ('等待与完成','operand-completion-set',[62],'g43 分操作数完成集合','A/B 请求分组的观察','A-only/B-only/A+B 未完成集合','端点区别于整组 full 位图，不强并单轮'),
 ('发布与时序','producer-publication-progress',[41,42,43,45],'g43 查询处 producer 发布进度','发布标记的全局/CTA 本地实现','B1 后发布计数、full 未完成状态','45 是对应计数和时间线的 CPU 核对'),
 ('发布与时序','request-query-age',[46,57,67],'B1 后发布至 full 查询/完成边界','软件计时原点及返回规则对齐','年龄区间、query−FIRST、MAIN−query、full 状态','57 修正上界原点；不得沿用已撤销的 issue 年龄推断'),
 ('发布与时序','request-age-raster',[28,47,48],'请求年龄的空间分布','源组/逻辑组与 raster 对应','中心化年龄预测/空间投影','28 为干预前诊断，48 为预注册可分性，47 为实机尝试'),
 ('局部规则迁移','gemm-local-completion-release',[15,16,17,18],'局部完成/继续/释放规则接回组合','固定联合规则替换旧 reuse 项','G3及开发条件的 L/plain、关键 CTA','G6 补测不另分族；18 为同版时间线分析'),
 ('局部规则迁移','producer-preparation-cost',[20],'producer 逐轮请求准备','两档已测准备价格替换软件 gap','G3/A16 的 L、plain、CTA 结束跨度','准备成本与消费者完成/释放的对象不同，保留独立'),
 ('局部规则迁移','input-compute-service-bound',[38],'TMA 与 WGMMA 共同服务必要容量','全逻辑引用/相邻 B 复用的假设','原生 D 窗口与服务下界','下界推导与 GEMM 时间预测不同，保留独立候选'),
 ('局部规则迁移','gemm-input-compute-sharing',[58,61],'GEMM 输入与计算共同服务','逻辑字节/源片段引用计费','G3 两段 L、plain、后续 full 未完成数','同一共同资源模型的需求计数变化'),
 ('读取与请求服务','gpc-copy-service',[29,37],'逐 CTA 副本的 GPC 投递','固定价格的 FIFO/工作守恒约束','完整时间预测/必要下界、L 响应','29 是 37 的必要下界诊断；若只收测量轮次则作为关联分析保留'),
 ('读取与请求服务','tma-read-service-bound',[59],'独立 TMA 读取/应答容量','按实际地址与资源归属统计工作','每片段/每16KiB服务时间上界','与 copy 下界对象不同；不能归入 input-counter'),
 ('读取与请求服务','gemm-input-sector-counter',[60],'真实 GEMM 输入路径工作量','base/A16 行距','NCU L2 sector 比值及采集资格','实际计数未取得；plain 有效不等于 counter 实验成功'),
 ('重复性与契约','gemm-repeatability-control',[27],'同二进制桥接的可重复性','相邻重复/顺序','plain、L cycle/ns 配对偏差','Q10 跨问题候选；本表暂保留 Q14 等待确认'),
 ('重复性与契约','compute-handoff-contract',[10],'跨输出计算模块状态','上一输出 service_end 状态传递','CPU 逐节点一致性和默认回归','软件契约检查，Q06 或关联实现记录候选；暂保留 Q14'),
]

def tsv(name, fields, rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(rows)

def main():
    used=[];rows=[]; evidence=[]; group_rows=[]
    for theme,family,indices,obj,var,end,reason in GROUPS:
        group_rows.append(dict(theme=theme,family=family,rounds=len(indices),object=obj,variable=var,endpoint=end,review_note=reason))
        for i in indices:
            used.append(i);p=ROUNDS[i];old=p.relative_to(BASE);new=Path('Q14-composition')/family/p.name
            docs=[p/n for n in ('README.md','plan.json','EXECUTION.md') if (p/n).exists()]
            snippets=[]
            for d in docs:
                content=d.read_text(errors='replace')
                evidence.append(dict(round=p.name,path=str(d.relative_to(ROOT)),sha256=hashlib.sha256(d.read_bytes()).hexdigest(),text=content))
                snippets.append(str(d.relative_to(ROOT)))
            rows.append(dict(old=str(old),proposed_new=str(new),status='proposal_only',theme=theme,
                             object=obj,variable=var,endpoint=end,reason=reason,evidence=';'.join(snippets),
                             depth_before=len(old.parts),depth_after=len(new.parts),
                             target_exists=(BASE/new).exists(),round_name_changed=False))
    assert sorted(used)==list(range(71)), '遗漏或重复轮次'
    assert len({r['proposed_new'] for r in rows})==71
    assert all(r['depth_before']==r['depth_after']==3 and not r['target_exists'] for r in rows)
    fields=['old','proposed_new','status','theme','object','variable','endpoint','reason','evidence','depth_before','depth_after','target_exists','round_name_changed']
    tsv('Q14-move-proposal.tsv',fields,sorted(rows,key=lambda r:r['old']))
    tsv('Q14-family-proposal.tsv',['theme','family','rounds','object','variable','endpoint','review_note'],group_rows)
    with (OUT/'Q14-source-evidence.jsonl').open('w') as f:
        for e in evidence:f.write(json.dumps(e,ensure_ascii=False)+'\n')
    inventory=[]
    for q in sorted(BASE.iterdir()):
        if not q.is_dir() or q.is_symlink():continue
        for fam in sorted(q.iterdir()):
            if not fam.is_dir() or fam.is_symlink():continue
            for p in sorted(fam.iterdir()):
                if not p.is_dir():continue
                inventory.append(dict(path=str(p.relative_to(BASE)),question=q.name,family=fam.name,round=p.name,
                                      symlink=p.is_symlink(),readme=(p/'README.md').exists(),
                                      review_scope='Q14_pilot' if fam==Q14 else 'not_classified_in_this_pilot'))
    tsv('all-rounds-inventory.tsv',['path','question','family','round','symlink','readme','review_scope'],inventory)
    lines=['# Q14 实验族候选清单','',
           '本表只提出路径，不执行迁移。原 71 个轮次逐一登记；族名和端点仍供审阅，不是已确认的分类。',
           f'按对象、变量、端点先提出 {len(GROUPS)} 个候选族；8 个主题只作页面导航。相关模型回放、必要界和准备记录已明确标注，不伪装成新增实机测量。部分较宽候选的边界在下文单列，仍需审阅。',
           '同一轮次名保持原样，避免无证据地补卡名。等值源 001–004、失败构建和换卡历史仍在同族。','']
    for theme,family,indices,obj,var,end,reason in GROUPS:
        lines += [f'## {family}（{len(indices)} 轮）','',f'对象：{obj}。变量：{var}。端点：{end}。', '',f'归属依据／待审边界：{reason}。','']
        for i in indices:
            p=ROUNDS[i];d=p/'README.md'
            if not d.exists():d=p/'EXECUTION.md'
            lines.append(f'- [{p.name}]({Path(os.path.relpath(d, OUT)).as_posix()})')
        lines.append('')
    (OUT/'Q14-proposal.md').write_text('\n'.join(lines))
    print(json.dumps(dict(q14_rounds=len(rows),candidate_families=len(GROUPS),inventory_rounds=len(inventory),depth_checks='71/71',moves=0),ensure_ascii=False))

if __name__=='__main__':main()
