# R15：输出服务与完成边界

当前输出候选按 **single/multi、real/OOB、首/中/末 tile** 分类，以直接 merged ns 接入 [V09 r2](RULES.md#v09-model)。V09整体未通过，末次输出分项误差中位/最大为 **3.73%/35.60%**；参数与范围不能称为通用输出带宽。V09 partial 的主要漏算在主循环；输出残差在 P1 还部分抵消了它，不能靠增大输出费用补总时间。
固定CTA0本地处理后，只延后其它CTA输出仍使stage4早段10/10缩短，说明局部服务依赖全卡执行组织；完整调用没有因此变快，见[背景配对](#stage-output-background)。长K输出周期下降主要伴随局部cycle/ns变化；OOB仍有输出准备/同步成本；单tile规则不能直接用于多tile首轮。旧标量STS探针不提供CUTLASS epilogue常数。完整旧推导与准备记录见 [cee978a固定正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R15-output-service.md)。

<a id="cutlass-epilogue"></a>

## 要预测哪个输出窗口？

当前cfg_b的实际epilogue tile是`64×32`，完整`128×128` float输出分8个8KiB子tile；源码选择`StagesD=2`、`DelayTmaStore=true`、`ReuseSmemC=false`。准备当前子tile时延迟提交前一个，逐次fence/sync、TMA commit和源buffer acquire，最后再排空。组合模型必须保留这条双缓冲链；总E0不是一笔64KiB传输，除以8也不能得到独立子tile服务价。静态配置见[本批setup](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R13-output-background/static_setup.json)。

cooperative在两consumer的`mma_tail → MAIN_END → PERMIT → store() → DONE`之后，最终还执行`store_tail → final`。role2/thread256为TMA issuer，`store()`内部包括寄存器/SMEM准备、fence、同步、TMA提交与等待；PERMIT不是首条TMA发出时刻。

同CTA、同tile的两角色事件记为m1/m2、p1/p2、d1/d2：

```text
M=max(m1,m2), P=max(p1,p2)
W=P−M                    merged许可间隔
A=P−p2                   issuer早到部分
I=d2−p2                  issuer整个store()窗口
J=d2−P                   较晚permit到issuer返回
R=max(d1,d2)−d2           另一consumer的DONE补差
I=A+J; E=J+R; max(DONE)−M=W+J+R
```

当前递推保留W，再收费E；若改收整个I，会重复计入A。E_middle/Elast已是merged窗口，不能再加R；单tile只收E0，不再收Elast。角色差值可能含重叠准备和等待，不命名为纯barrier或纯TMA成本。

最终`store_tail()`在本CUTLASS版本使用`.read0`，只保证源SMEM被读完、可复用，不证明全局写完成。完整CUDA event才覆盖最终写完成。不同CTA/进程分别取的中位数不保持上述加法恒等式；精确分解先在同一记录内计算。

<a id="next-mainloop-overlap"></a>

## 前一输出是否拖慢下一主循环？

**改变输出顺序会改变早段，但不能解释全部缺口。** [739717同卡对照](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R13-epilogue-order)保持cfg_b 2816×1536×4096、stage4/6、A/B和分配容量不变，只让CTA0前WG在释放全部输入slot后，等待后WG的mma返回再做epilogue。正常/延后使用同一二进制，预热flag均为0、全部控制的预热均为8次。40个progress进程的工作与SM一致；20个延后进程均确认E0开始晚于同调用M1返回，早段重叠为零。

| 同调用窗口或同trial配对 | stage4 | stage6 |
|---|---:|---:|
| 正常→延后，早段C1→C16中位cycle | 11130.5→10481.5 | 7947.5→7890 |
| 早段配对差中位[范围] | −772 [−1091,−146]，10/10缩短 | −124 [−731,319]，5/10缩短 |
| 后段C17→C63配对差中位 | +304.5 | +1 |
| 整个L配对差中位 | −140.5 | +109 |
| E0自身窗口中位 | 6429.5→2524 | 8098→2679.5 |
| M1返回→MAIN配对差中位 | +94 | +93 |
| plain完整时间配对变化中位 | −0.39% | +1.43% |

早段收益没有完整转化成L或整调用收益。延后的E0仍与M1的tail重叠约376/377 cycle，不能把尾段增长直接称为硬件drain变慢。E0自身明显缩短，说明固定几何的输出窗口也依赖执行重叠或所处时段；延后同时改变了它遇到的全卡输出背景，不能唯一归因本SM端口竞争。

两种顺序各用自身两stage后段检查[R13固定复用子类](R13-async-retirement.md#stage-reuse)：stage4早段上界分别8888.58/8970.58，仍少算约2242/1511 cycle。该界沿用计算受限c和默认继承状态等前提，不是任意模型的下界。不能给early补一个772或1511的专属常数，也不能继续单独修E0来宣称整个缺口已经解释。

[逐进程分析](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-epilogue-order.md)保留原端点、重叠、VA、预热与失败来源。100个有效进程无失败；四次同分配全部4325376个D元素逐位相同；四binary均168寄存器、16条静态HGMMA、无spill。两个先行plain参照相对739686为+0.52%/+1.04%。progress相对plain的整调用配对中位扰动为1.08%～2.13%，不代表局部扰动同样小。sender记录在自身epilogue结束后写出，正常组仍可能落在receiver主循环内；不同进程VA也有变化，stage4的四个同VA配对仍全部早段缩短，但相同VA不证明相同物理映射。

旧[V08F D-only对照](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08F-job737322-v1/reanalysis/followup-v1/followup.json)保留为弱干预：旧099卡、stage6、2304×3072×1024、ldd3072→3080，后续L仅+0.423%、完整时间+0.226%；它没有当前早段端点，也未固定输入VA，不能替代本次顺序对照。E0末端沿用store_tail边界，不是全局D可见完成。


<a id="stage-output-context"></a>

### 整卡首输出重叠集中在第二主循环的前段

复用739095的stage4/6各10个dual进程，所有132 CTA均有两tile的直接globaltimer端点。下表以每次调用的CTA0第二L为窗口，统计其它131 CTA首输出PERMIT→DONE；数值为进程中位，不与739717的progress提交点拼接。

| stage | CTA0第二L，µs | L入口时其它E0活跃数 | 其它E0至少一个活跃的覆盖比例 | 整段平均并发数 | FIRST1−本CTA E0开始，ns |
|---|---:|---:|---:|---:|---:|
| 4 | 22.592 | 107.5 | 18.95% | 17.99 | +208 |
| 6 | 19.440 | 13.5 | 27.35% | 31.17 | −352 |

20次都曾同时有其它131个E0活跃，但这些记录窗口在L后半均已结束。stage6累计重叠更多、L却更短，不能直接按重叠面积给stage4加价；两个stage进入该背景的相位和预取余量不同。stage4的FIRST1有8/10次严格晚于本CTA E0开始、1/10同一记录时刻；stage6十次均早于。端点差落在32ns格点，同值不证明严格同时。[逐进程上下文及端点](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-output-context.md)。

源码的输入复用依赖也给出不同预取机会：第二输出首输入group64依赖前输出group(64−S)的槽，由commit(65−S)之后的wait1释放；到前输出末次commit63，stage4/6分别相隔2/4个group。这个结构关系不能直接换成2c/4c的就绪时间，全部线程release、重新发出请求和服务完成仍未定值。

<a id="stage-output-background"></a>

### 固定本CTA处理后，其它CTA的输出背景仍会改变早段

[739740同卡配对](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R13-output-background)保持上述几何和stage不变。两组均让CTA0首输出等待其第二WG mma返回；flag1只有CTA0延后，flag2对全部132 CTA执行相同规则。同stage两处理用同一binary，全部预热均为flag0。新增直接globaltimer同时记录每CTA的E0及CTA0的C1→C16，避免把旧dual与progress拼成一个窗口。

| 同调用中位或同trial配对差 | stage4 | stage6 |
|---|---:|---:|
| 仅CTA0→全部延后，early cycle | 10511→9373.5 | 7873.5→7684 |
| early配对差cycle [范围] | −1008.5 [−1991,−399]，10/10缩短 | −189.5 [−478,0]，7/10缩短、3/10相同 |
| early配对差ns [范围] | −544 [−1312,−192] | −128 [−256,0] |
| 其它E0在early中的平均并发数 | 73.54→0 | 117.50→0 |
| 后段C17→C63配对差cycle | +438.5 | −5 |
| 整个L配对差cycle | −599.5 | −231 |
| FIRST→C1、M→MAIN配对差cycle | 0、0 | +6、+1 |
| plain完整时间配对差µs | +0.272 | +0.560 |

flag1每次只有CTA0满足延后顺序；flag2全部132 CTA满足，20个progress进程的其它E0均完全移出CTA0早段。目标CTA0的SM始终124、工作序列不变。stage4四个同VA配对也全部缩短；相同VA仍不保证相同物理映射。cycle与ns同时缩短，不能仅用时间换算解释。stage4后段变长、完整plain未改善，局部收益不能直接加到总时间上。

暖机次数相同的stage4九对仍9/9缩短，中位−930 cycle/−512 ns；同VA四对为−869 cycle/−480 ns。每调用先用early cycle除以对应直接ns，再做处理配对，比率中位仅−0.31%，不足以将early变化归为提频。两个读时钟指令仍有先后，不能把该比率当作精确硬件频率。

原[R13共同复用子类](R13-async-retirement.md#stage-reuse)用每种处理自身的两stage后段约束early：flag1中位上界9007.95、实测10511，缺1503.05 cycle；flag2上界9194.08、实测9373.5，缺179.42。逐进程分别代入自身late时，flag2差值范围−225～709，8正2负；中央179不是严格排除，也没有证明全部分项已被同一组参数解释。它只把下一步离线检验范围收紧到无E0背景的有限slot基线及其背景迁移。

延后的输出并没有消失，而是进入了后段。使用同调用C16和M的直接ns，不插值C17/C63，得到：

| C16→M窗口，进程中位 | stage4 flag1→2 | stage6 flag1→2 |
|---|---:|---:|
| 其它E0参与CTA数 | 0→60 | 8.5→60 |
| 其它E0平均并发数 | 0→1.94 | 0.058→1.30 |
| 完全没有其它E0的进程数 | 10/10→0/10 | 2/10→0/10 |
| 配对周期变化 | +268 [76,1614] | −18.5 [−325,468] |

C16→M包含相邻间隔、后46间隔及最终返回，不等于单独C17→C63。stage4在这个完整窗口10/10变长；它与早段10/10缩短同时发生。不能把flag2的两stage后段都当作无输出背景，用其重新拟合后再将179cycle称为固定基线的迁移成绩。重叠转移解释了需要检查的执行上下文，仍不证明所有变化都是物理输出竞争。

[固定校准窗口检验](R13-async-retirement.md#stage-background-baseline)改用stage6 flag2 early及stage4 flag1 late，这两个实际窗口均无已记录E0覆盖。条件c/T固定后，stage4 flag2 early距包络上限365.26cycle，late距上限293.15cycle。它约束后续共享服务候选，没有形成完整L预测。

[共享服务事件候选](R13-async-retirement.md#stage-shared-service)已保留上述8子tile双缓冲链并自行生成E0。联合拟合可以接近12个窗口摘要，却漏掉C16→M的真实全卡相位，且本CTA输出干预效应太小；没有据此给输出加入一个新共享价格。模型最后store源读完也没有替代全局D完成。

**支持的范围是全卡组织参与局部服务。** 仅依赖CTA0固定几何、本地处理和stage的独立服务规则不充分。flag2也会改变其它CTA的输入请求进度，新增等待影响其调度，因而尚不能唯一定位输出物理资源，更不能按E0重叠面积加价。E0截止仍是store-tail源复用边界。

[逐进程数据与对照](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/B-stage-background.md)保留100个有效进程，无失败；八次完整D比较均4325376元素逐位相同。四binary均168寄存器、16条静态HGMMA、无spill。progress/plain整调用配对中位扰动为0.60%～2.15%，不能约束局部扰动。两个先行plain参照相对739717为−2.13%/−0.34%，跨批漂移不算本批处理效果。少数进程暖机次数不同，全样本保留并单列共同暖机配对。

## 周期变短是否等于输出服务变快？

<a id="v08-输出窗口离线复核2026-10-09"></a>

旧V08以逐tile clock64和CTA首末globaltimer作仿射换算，cfg_c的低/高输出档与重叠变化相容；长K结果却对入口/退出锚点敏感。该分析提出直接输出端点补测，不能把其Nbar当作独立预测输入。[原离线分析](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/B-20261009-output-issuer-v2/output.json)。

### job738100：直接issuer窗口推翻长K的入口换算

a057/GPU-43269fbc，同卡cfg_c三条件，120个进程；只给首tile issuer的PERMIT/DONE增加globaltimer。g3为48个完整单tile CTA，c2/c6为132个；c2→c6固定M/N与输出字节、K4096→16384。

| 条件 | 直接I_ns | issuer cycle | 直接平均重叠 | 仿射/入口锚点/退出锚点重叠 | 输出启动跨度ns |
|---|---:|---:|---:|---:|---:|
| g3 | 2496 | 4720 | 42.55 | 42.97 / 45.33 / 42.78 | 1040 |
| c2 | 3264 | 6259 | 114.16 | 116.61 / 121.68 / 116.31 | 1264 |
| c6 | 3144 | 4892 | 87.80 | 92.08 / 123.98 / 91.96 | 3776 |

长K实际重叠下降23.09%，入口锚点估计反而上升；c6逐调用入口估计误差中位+40.51%，否定“全CTA共同速率从入口换算即可准确定位输出”。仿射/退出锚点保留趋势，但平均重叠误差仍约5%。重叠数统计的是处于issuer `store()`窗口的CTA，不是物理store队列。

| 条件 | 输出局部cycle/ns | CTA全程cycle/ns | 同CTA局部/全程比值 |
|---|---:|---:|---:|
| g3 | 1.89183 | 1.81630 | 1.04219 |
| c2 | 1.91343 | 1.64988 | 1.15871 |
| c6 | 1.59063 | 1.46762 | 1.08346 |

比值先在每CTA算，再取中位。c2→c6的输出cycle下降21.84%，直接ns只下降3.68%；局部有效率同时下降16.87%。周期下降不能全归为并发减少而服务变快。g3→c2虽见更多重叠与更长ns窗口，但几何、总字节、足迹和前序供给同时变化，尚非纯并发干预。

global/plain完整时间变化≤1.03%，四变体CV≤1.141%；内部issuer周期仍比stamped高约0.77%–2.05%，整调用扰动小不证明局部无扰动。Nbar依赖已测窗口，不作为测前自变量；旧V08另一卡的时间线也不能由本卡补写。[直接窗口与逐CTA比值](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R15-output-ns-job738100/reanalysis/B-20261009-r15-rates-and-perturbation/output-ns.json)。

### job738397：双角色直接ns把换算差与服务差分开

同卡c2/c6使用现成dual协议，同时记录两角色MAIN_END/PERMIT/DONE的cycle/ns；80个进程通过。wide匹配scratch，dual完整扰动中位为1.075%/0.397%，J_cycle相对stamped仅变0.279%/0.041%。

| 条件 | W，cycle/ns | A，cycle/ns | I，cycle/ns | J，cycle/ns | R，cycle/ns | 最终尾段，cycle/ns |
|---|---:|---:|---:|---:|---:|---:|
| c2 K4096 | 114 / 64 | 554.75 / 320 | 6224.50 / 3248 | 5755.25 / 3008 | 6 / 0 | 656.25 / 320 |
| c6 K16384 | 113 / 88 | 602.75 / 384 | 4847.00 / 3136 | 4254.50 / 2752 | 6 / 0 | 528.00 / 320 |

全部2640个dual tile在cycle/ns下分别满足角色恒等式。dual的R_cycle恒6，stamped恒4；R_ns多数为0、少量32 ns，说明它不能当通用join指令成本。最终`.read0`边界保持原义。

按同trial和CTA逻辑工作配对，c6/c2的J_cycle、J_ns、局部cycle/ns比值中位为 **0.79775/0.96883/0.82698**。同一批配对的平均log精确分解中，局部换算项占周期差88.9%；这是观测分解，非锁频因果实验。每个trial的J_ns差仍为负（−64至−192 ns），所以也不能写成纳秒完全不变。

条件中位J_ns为3008→2752（−8.51%），配对比值中位却约−3.1%，来自分布/聚合差，不可混用。旧job738100仅测I，本次I_ns为3248/3136，与旧值差−0.49%/−0.25%，复现issuer窗口，但不能反推旧J。固定几何开发初值可由同一批逐CTA均值给出W=81.830、J=2845.515、R=3.600、Etail=329.661 ns；两个相同B0条件不能同时识别floor和字节系数。[双角色分析报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R15-dual-roles-job738397/reanalysis/B-20261009-dual-roles-final/report.md)、[逐CTA重放](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R15-dual-roles-job738397/reanalysis/B-20261009-dual-roles-final/dual-roles.json)。

## 单tile与多tile首轮能用同一个字节规则吗？

旧V08的cooperative单tile首先走E0，q形式只改Elast，实际上没有改变单tile输出。cfg_c旧E0=5784.25 cycle对少量CTA平台高估约43%。静态“持有最大tile数的CTA比例”也不是首轮输出人数：g3与g4的q同为48/132，首轮有效CTA却分别48/132，E0约4042.5/5888.25 cycle。

因此当前用软件调度的首轮有效**字节总量B0**，并按整个case的T_max=1或≥2分组；静态参与者仍不等于同时写出者。同case内只有一个tile的CTA不能自行改走single规则。

### job738496：哪些参数有校准支持？

同卡新校准读取23个cfg_a/c条件，另复用job738397的两个cfg_c单tile条件。先逐CTA得到直接merged ns，再按条件等权汇总；当前参数以 [r2 frozen/calibration.json](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/frozen/calibration.json) 为准，下面数值解释候选来源。

single使用 `J=max(floor,beta×B0/MiB)`、`E0=J+R`：

| 配置 | K4096两个B0，MiB | floor_ns | beta，ns/MiB | R_ns |
|---|---:|---:|---:|---:|
| cfg_a | 3 / 8.25 | 1051.933 | 139.960 | 3.061 |
| cfg_c | 6 / 16.5 | 2159.133 | 174.947 | 3.691 |

两点在既定max形式下分别激活floor/字节分支，给出唯一解；只有两个规模，尚未验证中间分支或外推。cfg_c高点属于旧job738397，未伪称同期；未拟合的K16384 E0高估2.93%。两配置beta不同，不能强迫为共同物理输出率。

把single规则用于普通multi首轮，cfg_a低估 **12.04%–33.44%**、cfg_c低估0.62%–7.38%。因此multi只保留当前B0的有效E0。普通multi各只有B0=8.25/16.5 MiB，等价参数集合可由floor-only或byte-only表达，不能分别冻结其floor/beta。

| 进入r2的cooperative候选，ns | cfg_a | cfg_c | 来源限制 |
|---|---:|---:|---|
| E0_multi | 1518.727 | 3019.018 | 普通校准仅一个B0，最大残差15.39%/3.80% |
| E_middle_real | 1443.403 | 2867.051 | c无普通T≥3，来自padding上下文real子集 |
| Elast_real | 1166.683 | 2346.673 | 普通完整输出，最大残差9.98%/5.90% |
| E_middle_oob | 1143.504 | 1958.081 | 保留准备/同步，不按零字节清零 |
| Elast_oob | 902.883 | 1835.828 | padding中的全OOB末tile |
| Etail_single | 322.800 | 290.773 | 两规模等权均值，最大残差20.69%/17.85% |
| Etail_multi_real | 304 | 272 | r2使用普通条件中位口径 |
| Etail_oob | 128 | 160 | r2使用OOB中位口径 |

归档还给multi Etail均值307.495/291.224 ns；它与r2中位参数不同，不能混成同一个观测。cfg_c middle代理保留其补齐上下文限制。multi首轮E0仅收费J+R，后续E/Elast不再加R；single只收费E0。

当前partial按有效输出面积u，在real/OOB E与最终tail之间线性插值；它是迁移假设，尚未通过。cfg_b保留原pingpong输出重叠模型，不能直接套a/c的cooperative参数。[校准分析报告](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/B-20261009-composition-output-final/report.md)、[参数集合与残差](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/B-20261009-composition-output-final/composition-output.json)。

## 越界tile没有有效输出，为什么仍有成本？

八个padding条件首轮B0与普通multi相同；最后132个tile中116个OOB、16个real。混合Elast为cfg_a约900–956、cfg_c约1852–1933 ns，real子集仍约1094–1201/2277–2346 ns。OOB路径保留输出准备与同步；有效字节为零不足以设E=0。

padding Etail均值约133–144/165–171 ns，也小于普通real输出尾段，不能互相替代。普通与single中另一consumer final不晚于issuer；padding中1100个CTA晚32 ns，80次调用里两次改变全调用final包络32 ns。跟随role2 final的递推必须保留此端点范围。

这些现象允许按real/OOB分组开发，却不证明面积线性插值或所有上下文通用。V09 partial/sw8的真正最后完成CTA在10/10次不属于冻结并列集合；完整时间误差−5.75%仍不能证明输出/调度定位准确。[V09分项与关键CTA](V09-component-validation.md#失败定位与下一步)。

## 哪些旧候选已被排除？

| 旧候选/解释 | 淘汰依据 | 原分析 |
|---|---|---|
| 用观测Nbar直接预测输出 | Nbar包含待解释窗口本身，长K还依赖周期换算 | [issuer/重叠诊断](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/B-20261009-output-issuer-v2/output.json) |
| single常数或q线性式 | 按几何留出最大误差约31%，低q平台无法识别满卡斜率 | [single-tile-model](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/B-20261009-single-tile-static-q/single-tile-model.json) |
| 固定1.8GHz的首轮字节式 | 短/长K分组最大E0误差56.57%，部分参数非唯一 | [role-output](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/B-20261009-role-output-final/role-output.json) |
| 只加一个join常数解释E0 | R仅为观察协议中的4/6cycle，主要变化仍在J | [双角色复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R15-dual-roles-job738397/reanalysis/B-20261009-dual-roles-final/review.json) |

这些负结果说明为什么当前保留single/multi与直接ns分组；没有再平行维护多份可启用规则。准备包、接线推导和旧命令仍可从[固定旧正文](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/R15-output-service.md)查回。

## 标量STS探针回答了什么？

job735876，GPU-54896349，与当前参考卡不同。固定384线程、72KiB动态SMEM，两个输出warpgroup用标量`STS`整理64/128KiB，按16KiB块TMA写出；第三组执行独立WGMMA背景，比较单/双缓冲、串行/并发。它没有复现CUTLASS的STSM、向量化、fusion与动态配额，不能解释原小tile约1500cycle固定段或直接供给输出常数。

| 输出KiB / buffer | full_output，cycle/序列 | serial | concurrent | 并发相对串行缩短 |
|---|---:|---:|---:|---:|
| 64 / 1 | 7144.500 | 11669.125 | 10167.969 | 12.864% |
| 64 / 2 | 5897.875 | 11687.000 | 8838.969 | 24.369% |
| 128 / 1 | 14288.438 | 23402.750 | 20036.469 | 14.384% |
| 128 / 2 | 11303.406 | 23423.875 | 17061.188 | 27.163% |

每进程32序列，时间含发布、同步、等待和末尾消费，未扣控制成本。完整窗口终点使用非`.read` wait0，可与源读完/release事件区分；窗口包含控制，不是裸TMA延迟或物理带宽。

v4共18个plain条件有效，16个六事件trace满足5%扰动门槛；两个TMA-only trace扰动约9%–10%，定量事件判定失败。v5只补这两点末块read/full，两点各3对、扰动0.57%–1.90%通过；未观察的prepare/issued/released仍缺失，不能据此追认完整六事件链。read→full均48cycle也包含打点/等待/控制。

v4核对4,866,048个保存FP32位置，v5核对540,672个，数值与padding通过。程序只保存elapsed标量，未保存原始begin/end，独立复核能重算统计但不能再次相减原始时钟。v1/v2 spill与v3逐块trace超标保留为一条历史失败；不进入现行参数。[v4独立报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R15-job735876-v4/final-analysis-A/report.md)、[v5复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R15-job735876-v5/independent-review-A/report.md)。

## 作业与复核索引

| 实验问题 / 作业 | 一句结果 | 证据入口 |
|---|---|---|
| STS/TMA与背景能否重叠，735876 v4/v5 | 并发缩短完整窗口，TMA-only只补证两个末端事件 | [v4](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R15-job735876-v4/final-analysis-A/report.md)、[v5](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R15-job735876-v5/independent-review-A/report.md) |
| V08高低输出档是否由重叠解释 | 现象相容，换算敏感且观测Nbar不能做预测输入 | [output.json](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/B-20261009-output-issuer-v2/output.json) |
| 长K输出直接窗口，738100 | cycle降约22%，ns仅降约3.7%，入口重叠估计失败 | [直接测量](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R15-output-ns-job738100/reanalysis/B-20261009-r15-rates-and-perturbation/output-ns.json) |
| 双角色共同端点，738397 | 精确角色计账成立，局部换算主导周期差 | [report.md](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R15-dual-roles-job738397/reanalysis/B-20261009-dual-roles-final/report.md) |
| single/multi与real/OOB，738496 | 分组候选进入r2，partial与c-middle仍有限制 | [report.md](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/B-20261009-composition-output-final/report.md) |
| 完整新留出，739011 | 输出最大分项误差35.60%，整体未通过 | [V09](V09-component-validation.md) |

当前离线复核入口是 `analyze_r15_output_ns.py` 与 `analyze_r15.py` 的 `--dual-roles`/`--composition-output`；各归档报告保存完整命令与来源。分析写新目录，旧source/build/samples/frozen保持只读；无需为了复核重复GPU采样。

[输出消融](V01-validation.md#v09-ablation)显示分类对总中位误差的收益较小（4.33%→4.11%），但末次输出中位误差6.19%→3.73%，小规模single改善；保留现有分类，不为总误差追加新分支。


<a id="v09-output-failure"></a>

## V09 partial 的输出不是主要漏算项

针对每个 dual 进程真正最后完成的 CTA，直接把其窗口拆成 prefix、各 tile 的 L 总和、E 总和、间隔和最终 tail，再与**同一 CTA**的冻结时间线比较。P1/P2 十进程中位残差如下；每项先在单进程内相减，中位数之间不要求可加。

| 对象 | sum L 残差 ns | sum E 残差 ns | prefix 残差 ns | 最终 tail 残差 ns |
|---|---:|---:|---:|---:|
| cfg_a P1 | −4484 | +445 | −240 | +32 |
| cfg_c P2 | −12712 | −496 | +320 | +16 |

P1 中输出项偏高、主循环偏低，构成局部误差抵消。P2 输出偏低约0.50 μs，远小于 L 漏算的12.71 μs。这里排除的是“调整最后输出或 tail 就能修复这两个反例”的解释；不能据此认定所有输出规则都正确。E 仍是 merged PERMIT→DONE，tail 是观测到的 post-loop 端点，均不命名为物理写完成。

固定输出的 P1 map 补测进一步分离了两项：新 a058 卡上，将输入 map960 扩到1024，plain缩短4.080 μs，dual相对plain扰动仍仅2.41%/2.11%。同角色配对中 first partial 及其完整cluster伙伴的 L各缩短约4.13/4.14 μs，而各类E配对中位变化介于−32与+80 ns，末尾tail也没有同量级改善。最后完成CTA的sum E反而由约2.256变为2.528 μs。收益由输入边界相关的主循环变化主导；这些是新卡同期差值，不是旧V09输出系数的跨卡校准。配对明细见[R18](R18-cluster-boundary.md#v09-boundary-failure)。

逐 CTA 分解保存在 [C-processes](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/failure-decomposition-v1/C-processes.json)；输入路径的干预证据集中在 [R18](R18-cluster-boundary.md#v09-boundary-failure)，身份错位集中在 [R19](R19-critical-cta-tail.md#v09-critical-failure)。无需为上述输出归因另加一组 epilogue 测量。

## cfg_a 的输出缓冲依赖（2026-10-11）

`output_model.cfg_a_rank_dependencies` 现将本批实际输出配置写入无服务价的事件图。cfg_a的epilogue tile为128×32，完整128×128 FP32输出对应四个名义16KiB子tile；`ElementC=void`，源码选择`DelayTmaStore=true`、`ReuseSmemC=false`和`StagesD=2`。这与前文cfg_b的八个8KiB子tile不同；名义字节数不等于partial/OOB的物理流量。

子tile q的寄存器准备可先于q−1的TMA提交；写入共享缓冲前，必须经过q−1提交后的acquire。两缓冲状态跨输出连续，q使用`q % 2`，generation为`q // 2`。`acquire_after(q)`保证除最近提交批次外的旧源读取已完成，所以中间输出DONE允许最后一笔源读取继续进行。下一输出的首笔提交/acquire可提供上一输出最后子tile的源复用保证，其间包含下一主循环，不能另算一笔裸输出服务。

| 事件 | 依赖所保证的边界 |
|---|---|
| `DONE` | store调用返回，最后一笔源读取可以仍在进行 |
| `output_source_read` | 全部已提交子tile的源读取完成；最终issuer `.read0` 返回提供该保证 |
| `D_visible` | 全部D写入全局可见，与源读取完成分开 |
| `cuda_complete` | 输入producer尾部、consumer退出及D写入均完成 |

已有consumer final戳约束最终issuer的源读取完成，但不能给D可见或producer尾部提供有限上界。图没有把CUDA event的elapsed标量当作内部globaltimer绝对端点。此图只保留执行和复用约束，没有改变现行`output_intervals`参数、partial插值或V09输出成绩；尚不能据此报告输出模型通过。源码SHA、逐事件可行区间及跨输出缓冲测试见[独立分析](../../../../../../results/gh200_resource_campaign/access_rules/20261010-R18-cluster-fill-discrimination/reanalysis/C-20261011-rank-dependencies-v5/README.md)。R14跨域页面由主集成统一更新。
