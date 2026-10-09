# GH200：现行候选与已测条件规则

<a id="v09-model"></a>

## 现行候选：V09 job738972-r2（整体未通过）

唯一现行组合候选是 **V09 r2**：同卡 first/later 供给、single/multi 与 real/OOB 输出、输入相关时钟接入同一 CTA 事件递推。它已完成新留出，**不是通过规则**：30 条件中 29 个受支持预测的完整时间绝对误差中位/最大为 **4.11%/35.46%**，另一个保持冻结前拒绝；分项与关键 CTA 仍有显著失配。[完整判定与逐例表](V09-component-validation.md)。

参数唯一来源为 [r2 frozen/calibration.json](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/frozen/calibration.json)，冻结身份为 [manifest.json](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/frozen/manifest.json)。下列公式说明接线与单位，不在正文复制另一套可修改参数表。

### 对象、静态输入与预测入口

FP16 输入、FP32 累加与输出，α=1、β=0，CUTLASS 3.9.2，GPU-43269fbc-449d-3e0f-908a-9c81229546d3。cfg_a 为 128×128×64 cooperative、cluster 2×1、6 stage；cfg_b 同 tile、pingpong、cluster 1×1、6 stage；cfg_c 为 256×128×64 cooperative、cluster 1×2、4 stage。实际 stage/grid/资源由构建查询提供。

[v09_model.predict_components(row, setup, calibration)](../../../../../../microbench/gh200_resource_campaign/access_rules/v09_model.py) 只从形状、行距、输入模式、输入 map 边界、swizzle、静态资源和冻结参数构造候选数值。软件调度必须保留补齐/OOB tile；有效输出面积不能替代执行工作量。供给、输出和时钟分别由 [supply_model](../../../../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)、[output_model](../../../../../../microbench/gh200_resource_campaign/access_rules/output_model.py)、[clock_model](../../../../../../microbench/gh200_resource_campaign/access_rules/clock_model.py) 提供，CTA 依赖仍沿用 `v08_model.cta_cycles`。

### first/later 条件供给

对 CTA i 的第 j 个输出 tile，令 p 为 first（j=0）或 later（j>0），K_t=K/64，q_cfg 为 512/512/1024 cycle/Ktile。完整主循环候选为

\[
L_{i,j}(f)=b_{cfg,p}+K_t\max\!\left(q_{cfg}/f,\;\theta_{cfg,p}^{\mathsf T}x_{i,j}\right),
\]

其中 f 用 GHz（cycle/ns），L 与 b 用 ns；x 按配置投影到可辨识的请求组合。原始特征分别记录有效源 KiB、A/B 额外 32 B/128 B 地址覆盖与目标 SMEM 填零 KiB。覆盖量是逻辑请求代理，不是实测 L2/HBM 流量；各配置、first/later 参数不共用。

供给请求按真实软件轮次构造，并用 `min(stage,K_t)/K_t` 在本轮与下一轮请求之间计入有限预填。该权重是候选假设，未从 stage 扫描确认为通用定律。[R13](R13-async-retirement.md)维护构造、校准与局部负结果，[R10](R10-layout-cache.md)维护行距对照，[R18](R18-cluster-boundary.md)维护填零路径。

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

这套覆盖只接入 cfg_a/c；cfg_b 保留 pingpong 的 E 与重叠递推。输出/最终 `.read` 边界只支持源 SMEM 可复用，完整 CUDA event 才覆盖目标写完成。[R15](R15-output-service.md)保留候选为何这样分组及其负结果。

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

W、τ 用 µs，f/a 用 GHz；Q/S_src 是名义活动代理，不能命名为物理功率项。时钟拟合目标是观测有效频率，训练使用观测窗口；自由预测只使用候选递推得到的 W，不读目标时间或频率。[R09](R09-inkernel-clock-stages.md#source-clock-dev-v1)保留训练域和跨 K 失败。

r2 还按静态必零乘积比例 z 混合该输入模式与全零端点的 d/e，不缩减 Q/S_src。这是零新参数的活动假设；一侧面板为零与两输入全零不同，尚无物理功率证据。

所有事件常数与 L/E 以 ns 进入递推。cooperative：首个 `fm=P0+S`，后续 `fm=前一DONE+h`，`DONE=fm+L+w+E`，最终加 Etail。这里事件 S 是首段等待，与时钟源工作 S_src 不同。

pingpong 使用 `fm_0=P0+S`、`me_j=fm_j+L_j`，随后保留三条依赖：

```text
fm_j = max(me_{j−1}+gm, DONE_{j−2}+h)   第二项从j=2起
ep_j = max(me_j+w, DONE_{j−1}+we)       第二项从j=1起
DONE_j = epilogue_end(ep_j, 下一轮主循环窗口)
```

`epilogue_end` 将E作为输出工作量，与下一轮主循环重叠时按冻结比例r推进，其他时间按1推进；最后取 `max(DONE)+Etail`。不能把全部L/E直接求和。

给定 f，递推得到每 CTA 完成时间 D_i(f)，令 `W(f)=max_i D_i(f)/1000`，联立 `f=f_m(W(f))`。求根域为 `0<f≤a`；正、连续、随 f 非增的包络是求解器的前提，不能把固定周期模型的唯一根证明自动推广到任意重叠递推。模型忽略 CTA 入口分散，并列最大者全部保留。

最终 `T_plain=F_us+κW`；另报 `T_dual=W+dual_event_extra_ns/1000`。F/κ 来自参照条件的观测 dual 包络与 plain event 映射，没有用供给模型残差拟合。它们包含观察协议差，不是纯 launch 常数；不能再把 W 当 cycle 除一次 f。

### 适用范围与已知失败

- 参数限上述参考卡、编译/观察协议和校准输入分布；clock 的 21 条训练全部 K1024、seed17，窗口 13.600–2546.928 µs。小 grid、stage4、partial、新 random seed 与 K32768 都是 V09 检验的迁移，代数支持不等于经验验证。
- cfg_b K32768 的频率低估 25.00%，总时间高估 35.46%；代入观测频率后仍是测后诊断，不替换冻结分数。cfg_b A+16B 实测慢 28.94%，模型只给 1.40%；stage4 实测慢 10.47%，模型响应为零。
- 首段、预填、later 主循环与尾段仍有较大分项误差；cfg_c partial/sw8 真正最后完成 CTA 在 10/10 次均不属于预测并列集合。总时间中位 4.11% 不能覆盖这些失败。
- 未覆盖跨卡、A+32/64/96B、基址偏移、rotation、K 尾部、cfg_b partial、A16+B 非零组合。时钟不按历史最小频率夹紧；r2 拒绝项不因测后数值接近而追认为受支持。

[测后消融](V01-validation.md#v09-ablation)暂不支持整体退回V08聚合形式：共享时钟下完整候选4.11%/35.46%，同卡重拟合V08形式5.56%/48.68%。局部收益并不一致，保留现行候选而不增加分支；这不是新的留出成绩。

## 计量与组合约定

- 有效数学工作为 `2MNK FLOP`；补齐 tile、消费者归约、地址、同步与排空按实际窗口计入时间，不能混入同一有效 FLOP 分子。
- 同一 SM 的周期差用 `clock64`；跨 SM 包络用 `globaltimer`；完整 GEMM 用 CUDA event。先在同次调用、同对象内相减/求比，再明确汇总层级，不跨 SM 相减 `clock64`。
- 完整调用不含分配、初始化、驱逐准备或结果检查。不同端点、工作量、输入模式与观察协议的服务率不能直接相加；沿用[模型接口](../../../model/interfaces.md#5-计时与层间边界)，已含成本不再收费。
- bank/端口、缓存命中、物理 HBM 流量与异步队列深度尚未知；下列已测条件关系只在各自范围内使用，不自动成为 V09 参数。

## 已测条件关系

### 源供给、推进和结果消费必须分开

8192次FFMA/lane、8条累加链、4 warp时，同一kernel、40 reg/thread的复用一对源为9880 cycle，轮换八对为10400 cycle，慢5.26%。旧独立编译帧为56/40 reg/thread，差约22%。固定帧后仍有差异，但数值明显依赖编译帧。

`128×8192×2=2,097,152 FLOP`；除以9880得到212.26 FLOP/SM-cycle/CTA。链数已包含在8192次中，不能再乘8。两个代码位置、plain/trace各10进程复现；短非均匀输入与全部保留结果独立重算通过。

[R02](R02-register-service.md)的SASS说明，同一逻辑累加链可在循环首尾重命名到不同物理寄存器；13个目的编号仍可能是8条链。部分乘数使用uniform寄存器，两个分支的复用与排程也不同，因此本表不是普通RF物理读端口带宽。

结果消费另有直接对照：FADD目的池8→32时，生产者推进同为9791 cycle，消费尾部40→136，增量`96=(32−8)×4 cycle`来自指定的串行消费者。这个条件下可以描述消费者成本，不能称每结果RF写成本4 cycle，也不能直接填入CUTLASS epilogue。

- L0输出：具体源形式、链数和SASS下的操作服务。
- L1输出：保留结果、寄存器资源和消费者依赖。
- 尚未确定：RF bank/端口容量、任意kernel的源池倍率、覆盖过的中间结果的物理写回时刻。

![RF条件服务](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R02-job736989-v8/analysis/plots/r02-service.png)

### 独立操作流不自动满足 max(A,B)

[R11](R11-mixed-issue.md)每路固定128动作/单元、256单元，分别测A-only、B-only、两种串行、同组交错和跨组并发。

| 配对 | 观察 | 可以使用的规则 |
|---|---|---|
| WGMMA＋独立IMAD | 交错2338677 cycle，A-only2337092；差约0.07% | 该帧中整数动作可近似被TC覆盖 |
| FFMA＋IMAD.WIDE | 交错478321，A后B431031 | 交错慢约11%，引用联合服务，不强制取max |
| LDS.128＋CVT动作 | 并发1153894，串行1216219 | 仅部分覆盖；CVT动作含更新和数值摘要 |

WGMMA工作为`32768×(2×64×128×16)=8,589,934,592 FLOP`；除以交错窗口为3672.99 FLOP/cycle/CTA。IMAD的8,388,608整数OP另计，不与FLOP合成一个分子。归约、同步与排空都在分母中。

组合接口需要同时携带组织、共同消费者、阶段位置和空控制。`T_A+T_B`包含两份公共成本，不能把它无条件当串行下界；空单元也不是任意操作图的可加常数。L1按具体操作流查询联合服务，L2决定哪些流确实可同时推进。

### SMEM逻辑区域独立，仍可能有联合服务代价

[R12](R12-smem-path-contention.md)用三个warpgroup分别发TMA、执行WGMMA、发STS；区域互不重叠，所有组织保留相同SMEM容量和两个阶段的消费者。

| 并发配对 | 相对匹配阶段较慢单路的完整窗口 |
|---|---:|
| TMA＋WGMMA | +0.23–0.28% |
| WGMMA＋STS | +9.73–11.34% |
| TMA＋STS | +6.26–7.52% |

例如INTER下，`max(50152,157647)=157647`；并发158094，多447 cycle，比例0.2835%。WGMMA＋STS同布局多17882 cycle，不能在流水模型里忽略。

每单元64条m64n128k16，32单元为536,870,912 FLOP；TMA和STS各自请求2,097,152 B。各路径的字节不能相加后命名物理SMEM带宽。INTER与SW128的具体地址映射分别核对，布局收益也不是统一倍率。

早期消费者的重复归约被CSE消除，制造出“并发比较慢单路还快5%”。修订后SASS保留两条64-FADD归约，数值与消费者工作匹配。旧版本保留，但不进入本规则。

L1提供独立路径/联合窗口，L2在实际角色和缓冲组织下组合；物理共享端口和队列深度仍未知。

### 输入就绪、消费完成和槽可复用是不同事件

```text
producer: 等empty → 发TMA → full就绪 ──────────────────────┐
consumer:                 等full → 发WGMMA → wait完成 → 退役 │
producer:                                                等empty成功 → 下一次补发
```

TMA请求发出不等于输入可读；WGMMA提交不等于输入槽可覆盖；输出源操作数可复用不等于目标输出已可消费。[R13](R13-async-retirement.md)分别保存ready、retire、producer确认可复用与下一次发出请求的观察边界，不能拼接不同run的绝对时间线。

固定4槽、每SM一个CTA、共享小源、真实WGMMA消费时，32 KiB输入的stage4序列率58.046 B/SM-covered-cycle，48 KiB为45.337。这些分母覆盖供给、等待、消费和退役；它们不能替换成统一的55或64 B/cycle“单SM上限”。过度扰动的事件只保留顺序，plain服务仍有效。

L0定义具体搬运/完成原语，L2负责每代缓冲的占用与复用依赖。[R15](R15-output-service.md)的标量STS输出探针与实际CUTLASS的STSM、向量化和寄存器重分配不同，其数值不能直接替代完整epilogue；完整输出和交接使用实际kernel中的[R14](R14-stage-handoff.md)、R18/R19与V07校准。

### 容量上限、执行窗口重叠和动态配额分别建模

[R16](R16-residency-quota.md)的驻留探针固定64×64×64 TC、每CTA32个Ktile、264个CTA，stage=1/2/4，总SMEM预留128/96 KiB，查询得到1/2 CTA/SM上限。

小源stage1的整卡包络48112→34128 ns，容量增至2有29.1%收益；stage4仍有13.9%收益。大源stage4则55024/55312 ns，差0.5%，与波动相当。因此软件stage与另一CTA的覆盖能力不同，更高occupancy也不自动增加吞吐。

小源stage4、容量2的有效矩阵工作`264×32×2×64×64×64=4,429,185,024 FLOP`，除以27792 ns得159.37 TFLOP/s。时间含运输、控制、消费和结果处理，不是整卡TC峰值。记录到的执行窗口重叠不等于完整CTA存活区间。

配额探针384线程，初始`384×168=64,512`个寄存器；producer40、两个consumer各232的稳态同为`128×40+256×232=64,512`。先释放后做0/64/256次依赖IMAD，consumer inc中位数均144.5 cycle；先工作再释放则190/2106.5/7866.5 cycle。把释放前工作保留在事件图中，不能给setmaxnreg附一个通用固定成本。

配额后的192次依赖FADD窗口为787–788 cycle，只作活跃值和后续执行见证。所有源值、FP32归约、整数结果和首末SM都有检查；这不是CUTLASS实际consumer的计算率。

L1使用容量、编译资源和配额合法性；L2使用释放/申请依赖；L3用实际CTA工作与完整包络。

### cluster边界必须区分零值、越界和补齐工作

[R18](R18-cluster-boundary.md)把普通数据、整块越界和有效地址内显式零置于相同物理tile列表、grid、stride和资源下。

cfg_a、K=8192：普通90.080 μs、整块越界108.192、有效零89.776。零值本身不能解释整块越界慢20.1%。首tile匹配增量中位数：越界21668 cycle、伙伴21776、同列其他CTA8091.5；有效零的差异接近波动。

cfg_c同样的边界对照没有cfg_a的正惩罚，不能共享系数。cfg_c切cluster1后接近90 μs，是144个物理tile的两轮变成132个tile的一轮等因素共同造成，不能当成多播成本。

原始窗口先各自相减再配对：一条真实记录为97330−65950=31380 cycle。该单点与按进程/类别汇总的21668不同；应保存分布与统计层级。两个K点得到的窗口增量只构成插值假设，V07负责检验新形状迁移。平均cycle/Ktile不是独立斜率，不能再加一次旧截距。

L2携带首/后tile及边界类别的候选窗口差；L3保持真实物理工作列表。共享B导致扩散仍是推断，没有唯一物理机制证据。

### 最慢CTA与中位CTA不能互换

[R19](R19-critical-cta-tail.md)直接记录cfg_b的实际调度坐标。1408²逻辑工作11×11=121个tile；swizzle8补到256个，分配为`124×2+8×1=256`。近单波长K的47.456→115.520 μs变化先要解释工作量，不能先叫缓存惩罚。

4096²的两种swizzle均为1024个tile，分配`100×8+32×7=1024`；长K swizzle8快5.41%，短K的0.22%差异不超过波动。长K最大/中位CTA时长比在swizzle1约1.050、swizzle8约1.003；近单波swizzle8则约1.164。尾差绑定规模和分布，不是统一百分比。

完整包络为`max(end_ns)−min(entry_ns)`。最长clock64窗口与最晚退出的CTA分别保存；入口分散96–128 ns不能自动解释所有数万周期尾差。L3需要实际调度列表、关键CTA和负载内频率，不能删除慢CTA来满足误差目标。

### 带宽坐标按方向、工作集和完成语义使用

[B01](B01-bandwidth-cache.md)分别列global读/写/独立读写、TMA tensor输入/输出、SMEM独立读/写/联合读写和GEMM缓存准备。有效请求字节、不同地址量、payload与padding分开。

已复用的整卡TMA：1 KiB payload、528 CTA、32槽，不同地址量`528×32×1024=17,301,504 B`；8 KiB为138,412,032 B。设备L2为62,914,560 B。两个坐标同时改变payload，不构成“只改缓存大小”的因果A/B；单CTA旧最大2 MiB不能填“大于L2”一格。

驱逐准备只是先写独立缓冲并完成同步的协议，没有计数器时不称保证冷HBM。global整卡GB/s与单CTA B/clock64-cycle不能直接取最大值或相加；新卡校准不拼接旧卡时间拟合。旧资源索引覆盖计算、矩阵交换、cp.async、同步、DSM/多播和辅助操作，来源和范围保留在覆盖清单。

## 旧组合与复现来源

V07 的事件递推、手算和分项未通过记录见 [V07](V07-rule-validation.md)；V08 的后续行距、供给和补齐对照分别见 [R10](R10-layout-cache.md#v08-stride)、[R13](R13-async-retirement.md#v08-supply)、[R18](R18-cluster-boundary.md#v08-padding)。它们是现行候选的来源与反例，不是另一份当前参数。

2026-10-08 的一次性离线复算保存于 [delivery-acceptance-v1](../../../../../../results/gh200_resource_campaign/access_rules/20261008-delivery-acceptance-v1/)；复算通过不改变分项未通过的判定，也不要求每轮重跑。旧完整规则正文保留在 [cee978a 固定版本](https://github.com/hibouwu/CUDA_optimazation/blob/cee978a47fba73d4507a20a5cc90d37da9910121/Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/RULES.md)。
