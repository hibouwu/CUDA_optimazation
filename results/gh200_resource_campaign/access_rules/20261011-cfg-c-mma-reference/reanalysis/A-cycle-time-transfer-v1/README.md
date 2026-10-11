# R09：同卡独立MMA参考的周期与时间迁移

同卡、同SM的原生工作量近似接近本批GEMM共同计算窗口，参考C/N仍不能直接用作其时间换算。该结论来自独立MMA参考740095和此前740091/740092的完整记录；没有用目标时间拟合参考价格或修正量，没有新的独立留出成绩。

## 固定参考与候选

参考GPU-ef8692f3，六进程均在SM124；64/1024组各三次，交替执行，固定16次同探针前序。N128、FP16 SS、B MN-major、两WG×两累加链，每WG每组八条MMA，wait1/final0。正式uniform输入驻留SMEM，两链及每次q使用同一描述符；不能将工作量等价解释为遍历GEMM的独立K子块或M分区。

| 组数 | 三次cycle | 三次globaltimer差，ns |
|---|---|---|
| 64 | 66339 / 66339 / 66339 | 36704 / 36672 / 36704 |
| 1024 | 1049379 / 1049379 / 1049379 | 578304 / 578272 / 578304 |

长窗口长度在参考采样前指定；本轮将它的C/N用于时间迁移是后验开发诊断，不能称为采样前冻结的时间候选。按三个长参考的中位得到C_R=1049379、N_R=578304。C_R三次相同，故C_R/median(N_R)恰好等于三个C/N的中位，不混用一般情况下不闭合的汇总关系。

一个GEMM tile256×128×4096执行268435456 FLOP，长参考为4294967296 FLOP，工作量比例为1/16：

```text
C_hat = C_R/16 = 65586.1875 cycle
r_R = C_R/N_R = 349793/192768 cycle/ns
N_hat = C_hat/r_R = N_R/16 = 36144 ns
```

参考两长度相对4096条件容量都多803cycle；该量没有拟合、减去或作为通用截距。本候选的缩放保留它在长参考中的份额。公式只用静态工作量和独立参考，在读取目标窗口之前构造。

## 逐窗口分项

GEMM采用同CTA两WG最早clock-FIRST至最晚clock-MAIN的共同计算窗口，ns使用相同WG标签。前一A分析已核对所有8640原窗口的ns值恰好等于ns域计算包络，原生cycle参考与目标仍包含不同软件端点。

```text
e_C = C_hat/C_target - 1
e_f = (C_target/N_target)/r_R - 1
e_N = N_hat/N_target - 1
(1+e_C) × (1+e_f) = 1+e_N
```

e_f固定目标实际native cycle，属于条件换算诊断；e_N是静态参考迁移候选相对记录窗口的误差。C_target及两种原始计时端点都含等待/软件/读取效应，e_C不是已识别的裸指令服务价格误差，e_f也不是独立物理频率误差。各项分组中位不相乘闭合，窗口也不闭合完整CUDA event。

| 批次／范围／输出序号 | 窗口数 | e_C中位，% | e_f中位，% | e_N中位，% | e_N最大绝对，% |
|---|---:|---:|---:|---:|---:|
| on/off，全SM，0 | 2640 | −0.5923 | −4.8941 | −5.4812 | 7.2660 |
| on/off，全SM，1 | 1680 | −0.4248 | −11.9252 | −12.3739 | 14.4318 |
| off/off，全SM，0 | 2640 | −0.5939 | −5.1762 | −5.8750 | 7.5696 |
| off/off，全SM，1 | 1680 | −0.4203 | −12.3041 | −12.7125 | 15.0752 |
| on/off，仅SM124，0 | 20 | −0.6511 | −5.3277 | −6.0316 | 7.1898 |
| on/off，仅SM124，1 | 20 | −0.5366 | −12.7809 | −13.2488 | 14.3020 |
| off/off，仅SM124，0 | 20 | −0.6616 | −5.5983 | −6.2656 | 6.9605 |
| off/off，仅SM124，1 | 20 | −0.5366 | −13.0750 | −13.5145 | 15.0752 |

全部8640个目标窗口逐项有理数闭合。全SM周期最大绝对误差2.9115%；80个SM124窗口单列，避免把单SM参考扩大成全SM价格资格。on/off与off/off分别保留，未作背景相减或删选进程。`transfer.json`还保存全部160个批次／范围／序号／pair／arm分层，未只保留表中合并结果。

## 端点、负载与时序限制

参考包含循环ARRIVE、最终wait0后的checksum/drain和CTA同步。实际formal kernel SASS只读四个计时器：GT起点14d0、CLOCK起点1510、CLOCK终点1fd0、GT终点1fe0。源C++与SASS的名义次序相符；它不是硬件采样包围或固定读成本证明。GEMM FIRST/MAIN的每个site为CLOCK→GT，两者的C/N不能无条件视为同一种时钟测量，也不使用两长度拟合固定ns补偿。

参考是resident repeated SMEM、uniform、单CTA、两链复用descriptor和16次同探针前序；目标为真实TMA、S4、cluster1×2、dyadic、132CTA和16次同GEMM前序。设备和SM身份相同没有消除这些差异。本批限制了这个直接时间迁移候选，未把差异唯一归因于DVFS、功耗、数据模式、读取误差或任何单一机制。

独立复验原globaltimer端点显示：最晚目标记录端点为1791713759969427104，参考最早起点为1791714305942919904，相差545973492800ns。参考在目标之后取得；不是在线预测或新留出。两个原始计时协议的硬件精度、GEMM观察失败及所有旧资格保持不变。

## 身份与复验

绑定165项输入SHA，包含参考冻结原/returned文件、binary、回执、六记录与完整输出文件、原参考数值验证报告、formal source/SASS及两目标批次的原raw和冻结链。沿用主集成已独立核验的196608个正式FP32值，本轮验证其报告及输出文件哈希，没有重复数值参考计算。短坐标检查只保留原程序自检身份；不宣称独立短数组复核或能够重算未导出的16次warmup CV。

`verify.py`不导入分析器或公共模型。重新读取六份参考和40份GEMM raw，重构8640个共同端点，逐项验证有标签的e_C/e_f/e_N，而非只检查乘积。两个负例均保持乘积闭合：互换e_C与e_f仍因标签错误被拒绝；改变ns并重算e_f/e_N仍因原始整数不符被拒绝。这区分了原数据身份、分项含义和代数自洽。

复放：

```sh
r09_transfer_output=$(mktemp -d /tmp/r09-cycle-time-XXXXXX)
cp /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-mma-reference/reanalysis/A-cycle-time-transfer-v1/{analyze,verify}.py "$r09_transfer_output/"
python "$r09_transfer_output/analyze.py" --base /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules --output "$r09_transfer_output"
python "$r09_transfer_output/verify.py"
```

直接复验：

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-mma-reference/reanalysis/A-cycle-time-transfer-v1/verify.py
```

没有修改B的名义源工作／价格、公共模型或runner，没有GPU操作。没有接受新时钟、修正量、独立留出或跨卡支持；完整cfg_a/b/c组合目标仍未完成。
