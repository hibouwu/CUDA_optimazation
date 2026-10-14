# GH200：现行候选与开发条件关系

<a id="v09-model"></a>

## 现行候选：V09 job738972-r2（整体未通过）

唯一现行组合候选是 **V09 r2**：同卡 first/later 供给、single/multi 与 real/OOB 输出、输入相关时钟接入同一 CTA 事件递推。它已完成新留出，**不是通过规则**：30 条件中 29 个受支持预测的完整时间绝对误差中位/最大为 **4.11%/35.46%**，另一个保持冻结前拒绝；分项与关键 CTA 仍有显著失配。[完整判定与逐例表](../../experiments/gh200_sm90/access_rules/V09-component-validation.md)。

参数唯一来源为 [r2 frozen/calibration.json](../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/frozen/calibration.json)，冻结身份为 [manifest.json](../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/frozen/manifest.json)。下列公式说明接线与单位，不在正文复制另一套可修改参数表。

### 对象、静态输入与预测入口

FP16 输入、FP32 累加与输出，α=1、β=0，CUTLASS 3.9.2，GPU-43269fbc-449d-3e0f-908a-9c81229546d3。cfg_a 为 128×128×64 cooperative、cluster 2×1、6 stage；cfg_b 同 tile、pingpong、cluster 1×1、6 stage；cfg_c 为 256×128×64 cooperative、cluster 1×2、4 stage。实际 stage/grid/资源由构建查询提供。

[v09_model.predict_components(row, setup, calibration)](../../../../../microbench/gh200_resource_campaign/access_rules/v09_model.py) 只从形状、行距、输入模式、输入 map 边界、swizzle、静态资源和冻结参数构造候选数值。软件调度必须保留补齐/OOB tile；有效输出面积不能替代执行工作量。供给、输出和时钟分别由 [supply_model](../../../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)、[output_model](../../../../../microbench/gh200_resource_campaign/access_rules/output_model.py)、[clock_model](../../../../../microbench/gh200_resource_campaign/access_rules/clock_model.py) 提供，CTA 依赖仍沿用 `v08_model.cta_cycles`。

### first/later 条件供给

对 CTA i 的第 j 个输出 tile，令 p 为 first（j=0）或 later（j>0），K_t=K/64，q_cfg 为 512/512/1024 cycle/Ktile。完整主循环候选为

\[
L_{i,j}(f)=b_{cfg,p}+K_t\max\!\left(q_{cfg}/f,\;\theta_{cfg,p}^{\mathsf T}x_{i,j}\right),
\]

其中 f 用 GHz（cycle/ns），L 与 b 用 ns；x 按配置投影到可辨识的请求组合。原始特征分别记录有效源 KiB、A/B 额外 32 B/128 B 地址覆盖与目标 SMEM 填零 KiB。覆盖量是逻辑请求代理，不是实测 L2/HBM 流量；各配置、first/later 参数不共用。

供给请求按真实软件轮次构造，并用 `min(stage,K_t)/K_t` 在本轮与下一轮请求之间计入有限预填。该权重是候选假设，未从 stage 扫描确认为通用定律。[R13](../../experiments/gh200_sm90/access_rules/R13-async-retirement.md)维护构造、校准与局部负结果，[R10](../../experiments/gh200_sm90/access_rules/R10-layout-cache.md)维护行距对照，[R18](../../experiments/gh200_sm90/access_rules/R18-cluster-boundary.md)维护填零路径。

候选静态接口 `supply_model.boundary_fill_features(row, setup)` 另给出每CTA/tile的自身fill、同操作数multicast mask内max/sum、完整cluster的max/sum/any；输入只来自形状、descriptor与软件调度，不读取trace。它能表达首轮完整伙伴所处的边界类别，但尚未给出可靠的时间价格，未接入r2的X或计时。[四点辨识及条件近似](../../experiments/gh200_sm90/access_rules/R18-cluster-boundary.md#boundary-candidate-features)保留其价值和失败范围。

[R09共同前序对照](../../experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md)进一步表明，频率条件必须声明前序kernel、次数或停止规则、输入与缓存准备操作；目标调用的窗口不能隐式代替历史。已有V09使用目标自身的adaptive暖机，冻结身份与失败判定不变。后续候选尚未实现这项历史依赖，未知前序时不能宣称通用频率或完整时间点预测，也不能由两种已测协议构造通用频率区间。

静态调度接口另接受 `raster_order=heuristic/along_m/along_n`，默认行为与原30条V09预测严格一致；739571的trace已验证显式方向的工作坐标。这个解析扩展没有修复计时：[R10](../../experiments/gh200_sm90/access_rules/R10-layout-cache.md)中两方向的七列供给输入完全相同，实测B32响应却不同。连续调用也显示同CTA纳秒服务窗口随历史变化，不能只更新时钟公式。跨rank full/empty的源码依赖见[R18](../../experiments/gh200_sm90/access_rules/R18-cluster-boundary.md)，这些结构约束尚未形成经过新留出验证的价格或组合模型。 739585进一步固定目标B面板及每组18个使用者，单输出36/126 CTA的行距响应仍不同；下一输出预取不是必要条件。局部规则必须保留更大全局执行组织对当前服务的影响，现有证据尚不提供可迁移的物理资源单价。

r2 在请求特征支持之外，还用等价校准参数集合检查目标窗口是否唯一；正式冻结的支持检查位于 `run_v09.check_prediction_support`，不能仅用数值入口的局部 Jacobian 标志或某个优化器给出的解接受预测。cfg_c G2 的两个首轮 CTA 在目标频率下可转入供给受限，同样校准拟合可给不同价格，因此保持“不支持”。固定 f 下的范围不等于联立时钟后的完整时间区间，也不是置信区间。

### single/multi 与 real/OOB 输出

cooperative 的输出使用 merged 边界：`M=max(MAIN_END)`、`P=max(PERMIT)`、`D=max(DONE)`；`w=P−M` 已单独收费，E=D−P。issuer 的整个 `store()` 窗口从更早的 permit 开始，不能在 w 后整段加入。

令 B0 为每个 CTA 首个 tile 的有效 FP32 输出字节总和，T_max 为整个 case 中最大的每 CTA tile 数：

| 条件 | 当前候选 | 计账与限制 |
|---|---|---|
| T_max=1 | `E0=max(floor_ns, beta_ns_per_MiB × B0/MiB)+R_ns` | 每配置仅两个 K4096 校准规模；只收 E0，不再收 Elast |
| T_max≥2 的首轮 | `E0_multi.effective_ns` | 普通校准只有一个 B0，不能分别辨识 floor 与 beta；case 内仅做一个 tile 的 CTA 仍走 multi 规则 |
| 后续完整输出 | 中间 E_middle、最后 Elast | 已是 merged 窗口，不再加 R；cfg_c middle 来自 padding 中的 real 子集 |
| 后续部分/OOB 输出 | `(1−u)E_oob+uE_real` | u 为该 tile 有效输出面积比；partial 线性插值尚未通过迁移验证，OOB 不设为零 |
| 最终尾段 | single 常数；multi 按最后 tile 的 u 混合 real/OOB 尾段 | V09 multi 使用冻结中位口径，不与归档均值表混用 |

这套覆盖只接入 cfg_a/c；cfg_b 保留 pingpong 的 E 与重叠递推。输出/最终 `.read` 边界只支持源 SMEM 可复用，完整 CUDA event 才覆盖目标写完成。[R15](../../experiments/gh200_sm90/access_rules/R15-output-service.md)保留候选为何这样分组及其负结果。

### 输入相关时钟与时间换算

从软件调度的全部名义 tile 计算每 SM 平均工作：

\[
Q=K_t\sum_iN_iq_{cfg}/132,\qquad
S_{src}=K_t\sum_iN_is_{cfg}/132,
\]

其中 s_cfg 为 24/32/32 KiB/Ktile，按理想 cluster 多播折算；Q/S_src 包括补齐工作。对输入模式 m∈{dyadic,zero,random}：

\[
f_m(W)=a-d_m\frac{Q}{1000W}-e_m\frac{S_{src}}{W}g(W/\tau),\qquad
g(x)=1-\frac{1-e^{-x}}x.
\]

W、τ 用 µs，f/a 用 GHz；Q/S_src 是名义活动代理，不能命名为物理功率项。时钟拟合目标是观测有效频率，训练使用观测窗口；自由预测只使用候选递推得到的 W，不读目标时间或频率。[R09](../../experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md#source-clock-dev-v1)保留训练域和跨 K 失败。

r2 还按静态必零乘积比例 z 混合该输入模式与全零端点的 d/e，不缩减 Q/S_src。这是零新参数的活动假设；一侧面板为零与两输入全零不同，尚无物理功率证据。

事件常数与 L/E 以 ns 进入 `v08_model.cta_cycles`，依赖与递推对应关系见[CUTLASS 程序图](../mechanism/program/cutlass_program_graph.md#recurrence)。`P0/S/h/gm/w/we/Etail` 是条件拟合间隔；pingpong 的 `epilogue_end` 把 E 作为输出工作量，与下一轮主循环重叠时按冻结比例 r 推进，其他时间按 1 推进。这些数值假设留在半经验模型中，不能由程序依赖推出。

给定 f，递推得到每 CTA 完成时间 D_i(f)，令 `W(f)=max_i D_i(f)/1000`，联立 `f=f_m(W(f))`。求根域为 `0<f≤a`；正、连续、随 f 非增的包络是求解器的前提，不能把固定周期模型的唯一根证明自动推广到任意重叠递推。模型忽略 CTA 入口分散，并列最大者全部保留。

最终 `T_plain=F_us+κW`；另报 `T_dual=W+dual_event_extra_ns/1000`。F/κ 来自参照条件的观测 dual 包络与 plain event 映射，没有用供给模型残差拟合。它们包含观察协议差，不是纯 launch 常数；不能再把 W 当 cycle 除一次 f。

### 适用范围与已知失败

- 参数限上述参考卡、编译/观察协议和校准输入分布；clock 的 21 条训练全部 K1024、seed17，窗口 13.600–2546.928 µs。小 grid、stage4、partial、新 random seed 与 K32768 都是 V09 检验的迁移，代数支持不等于经验验证。
- cfg_b K32768 的频率低估 25.00%，总时间高估 35.46%；代入观测频率后仍是测后诊断，不替换冻结分数。cfg_b A+16B 实测慢 28.94%，模型只给 1.40%；stage4 实测慢 10.47%，模型响应为零。
- 首段、预填、later 主循环与尾段仍有较大分项误差；cfg_c partial/sw8 真正最后完成 CTA 在 10/10 次均不属于预测并列集合。总时间中位 4.11% 不能覆盖这些失败。
- 未覆盖跨卡、A+32/64/96B、基址偏移、rotation、K 尾部、cfg_b partial、A16+B 非零组合。时钟不按历史最小频率夹紧；r2 拒绝项不因测后数值接近而追认为受支持。

[测后消融](../../experiments/gh200_sm90/access_rules/V01-validation.md#v09-ablation)暂不支持整体退回V08聚合形式：共享时钟下完整候选4.11%/35.46%，同卡重拟合V08形式5.56%/48.68%。局部收益并不一致，保留现行候选而不增加分支；这不是新的留出成绩。

后续对照给下一版增加了三项结构约束，尚未产生可迁移的新系数：[R13](../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#v09-stage-failure)的stage响应不能仅用固定启动项或恒定每Ktile罚项描述；[R18](../../experiments/gh200_sm90/access_rules/R18-cluster-boundary.md#v09-boundary-failure)的输入边界代价必须区分首轮、后续轮次及同cluster的完整伙伴；[R09](../../experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md#v09-three-layer-diagnostic)的长K条件时钟失配不能仅归结于服务误差，也没有依据设置硬频率下限。新卡局部配对只约束候选形式，不替换旧卡参数或V09成绩。

<a id="source-sharing-dev"></a>

### 开发候选：共享需求与B源组响应

该候选只用于当前cfg_b完整tile、B32对照，尚未接入上面的V09预测。令C为当前软件输出轮的CTA数，p_B为同B面板的其它使用者数，E_B为该B请求多覆盖的128 B granule对应KiB，则

\[
\Delta L=K_t\left[\max(512,aC+bE_Bp_B)-\max(512,aC)\right],\qquad a,b\ge0.
\]

max中的两项单位为cycle/Ktile，ΔL为SM cycle。纯计算入口为 `supply_model.source_sharing_delta_cycles(a,b,C,E_B*p_B,K_t)`；源覆盖和软件轮次仍不等于物理流量或同时发起的请求。仅用739585两个单输出条件约束参数，并以非负启动项所需的 `K_t*a*C≤aligned L` 排除共同基线大错被相减隐藏的解。保留相关分片参数域，不把a、b范围当独立矩形；目前没有物理上优先的一对参数。

迁移评估只对B32响应作开发检查，实测对齐L仍作为锚，不能称完整L或总时间预测。所有迁移组必须共用同一对参数；看过迁移后发现的好顶点仅是存在例，不能追认为事前选中的预测器。完整训练参数域对稳定位置仍有18%～20%的必要误差下界，放宽训练到5%整L后仍超过15%；分组中位数的小误差不能替代逐CTA检验，当前形式不满足局部服务目标。证据见[R10](../../experiments/gh200_sm90/access_rules/R10-layout-cache.md#shared-service-candidate)，完整域见[候选参数与迁移](../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/reanalysis/candidate-development-v1/R10-shared-service-candidate.json)。

硬件层的有限槽、共享源与初态候选见[TMA 机制假设](../mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md)，本节只维护上述半经验响应式。

## 计量与组合约定

- 有效数学工作为 `2MNK FLOP`；补齐 tile、消费者归约、地址、同步与排空按实际窗口计入时间，不能混入同一有效 FLOP 分子。
- 同一 SM 的周期差用 `clock64`；跨 SM 包络用 `globaltimer`；完整 GEMM 用 CUDA event。先在同次调用、同对象内相减/求比，再明确汇总层级，不跨 SM 相减 `clock64`。
- 完整调用不含分配、初始化、驱逐准备或结果检查。不同端点、工作量、输入模式与观察协议的服务率不能直接相加；沿用[模型接口](../common/interfaces.md#5-计时与层间边界)，已含成本不再收费。
- bank/端口、缓存命中、物理 HBM 流量与异步队列深度尚未知；下列已测条件关系只在各自范围内使用，不自动成为 V09 参数。

### 开发候选：有限槽与共享请求

已迁至[TMA 机制假设](../mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md#finite-source)，原实验与开发代码路径保持。

## 已测条件关系

测量与历史条件价格只在原 R/B 页维护；当前硬件解释见[路径目录](../mechanism/hardware_rules/README.md)。这些数据不会因列在此处自动成为 V09 参数。

### 源供给、推进和结果消费必须分开

见[R02](../../experiments/gh200_sm90/access_rules/R02-register-service.md)的测量与历史分析。

### 独立操作流不自动满足 max(A,B)

见[R11](../../experiments/gh200_sm90/access_rules/R11-mixed-issue.md)的测量与历史分析。

### SMEM逻辑区域独立，仍可能有联合服务代价

见[R12](../../experiments/gh200_sm90/access_rules/R12-smem-path-contention.md)的测量与历史分析。

### 输入就绪、消费完成和槽可复用是不同事件

见[R13](../../experiments/gh200_sm90/access_rules/R13-async-retirement.md)的测量与历史分析。

### 容量上限、执行窗口重叠和动态配额分别建模

见[R16](../../experiments/gh200_sm90/access_rules/R16-residency-quota.md)的测量与历史分析。

### cluster边界必须区分零值、越界和补齐工作

见[R18](../../experiments/gh200_sm90/access_rules/R18-cluster-boundary.md)的测量与历史分析。

### 最慢CTA与中位CTA不能互换

见[R19](../../experiments/gh200_sm90/access_rules/R19-critical-cta-tail.md)的测量与历史分析。

### 带宽坐标按方向、工作集和完成语义使用

见[B01](../../experiments/gh200_sm90/access_rules/B01-bandwidth-cache.md)的测量与历史分析。

## 旧组合与复现来源

V07 的事件递推、手算和分项未通过记录见 [V07](../../experiments/gh200_sm90/access_rules/V07-rule-validation.md)；V08 的后续行距、供给和补齐对照分别见 [R10](../../experiments/gh200_sm90/access_rules/R10-layout-cache.md#v08-stride)、[R13](../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#v08-supply)、[R18](../../experiments/gh200_sm90/access_rules/R18-cluster-boundary.md#v08-padding)。它们是现行候选的来源与反例，不是另一份当前参数。

2026-10-08 的一次性离线复算保存于 [delivery-acceptance-v1](../../../../../results/gh200_resource_campaign/access_rules/20261008-delivery-acceptance-v1/)；复算通过不改变分项未通过的判定，也不要求每轮重跑。旧完整规则正文保留在 [cee978a 固定版本](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/RULES.md)。

## m11 的当前资格

两次冻结留出均未通过，现有候选不接入 V09；当前主张、备选和重启条件统一见[m11](../mechanism/hardware_rules/M11-SMEM.md)，逐例冻结评分保留在[R03](../../experiments/gh200_sm90/access_rules/R03-access-demand.md#m11-holdout-v2-result)。

## 版本与参数绑定

| 模型/组件 | 当前入口与参数权威 | 数据用途与资格 |
|---|---|---|
| V09 r2 | 本页公式与顶部 frozen/calibration.json、manifest | 冻结比较基线，整体未通过；不回写 |
| V06/V08 软件递推 | [程序图](../mechanism/program/cutlass_program_graph.md)、`v08_model.cta_cycles` | 软件事件组合沿用；各版时间常数按该版冻结文件读取 |
| R08 波次、R09 时钟及 R10/R13/R18 条件价 | 各原实验的当时模型与冻结文件；V09 采用者以 r2 参数为准 | 历史开发/校准；测后修正不能覆盖正式成绩 |
| 共享 B 源响应 | 本页 `source-sharing-dev` 与所链接参数域 | 开发失败，未接入 V09 |
| C1-01 的 wave_roofline、structural 等 | [C1-01](../../experiments/gh200_sm90/access_rules/C1-01-selection.md)的模型清单与该轮冻结产物 | 半经验比较对象；规格参照另见[计算入口](../spec/spec_baseline.md)，TileSight 为公开模型比较对象 |
| 后续频率版本 | [频率任务](../mechanism/hardware_rules/FREQUENCY-MECHANISM.md) | 尚未新增通过候选；采用后在此登记新版本与参数绑定 |

新增版本必须单列公式入口、参数文件、开发数据、冻结身份与评分链接。原实验保留“当时用何参数得到何结论”，不承担当前参数维护。
