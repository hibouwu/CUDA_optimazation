# R09：GT_before／clock64／GT_after的条件区间

纯CPU入口r09_gt_brackets.py为拟议的GT-CLOCK-GT观察提供数学合同；当前没有这种新设备记录。它不解析旧CLOCK-GT-CLOCK帧，不使用其中点，不拟合读成本或增加时钟价格。实际设备接口、reader、观察控制与GPU仍由主集成负责。

## 输入与假设

每个端点由GPU UUID、SM编号和三个非零uint64值`gt_before,cycle,gt_after`构成。起止必须同GPU/SM，且原生cycle差C>0；拒绝零值、倒置、类型错误、越界或wrap差。调用者须先绑定正确调用、CTA/WG、源marker、完整协议、源码和二进制；数学入口不认证这些身份。

调用者明确给定单次GT读数的额外坐标误差e，使用整数或Fraction，不能省略、传None或静默设默认。e是条件预算，尚未测量。另独立假设中央clock值的采样时刻被真实GT读取包围，才有：

```text
start_time ∈ [G_start_before − e, G_start_after + e]
end_time   ∈ [G_end_before − e, G_end_after + e]
```

CUDA12.9.1归档的PTX8.8文档将clock64定义为64位cycle counter，globaltimer以ns计并具有目标相关行为。单位说明不替代采样包围或精度证明；本接口保留这两个前提。[NVIDIA PTX clock64](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#special-registers-clock64)、[globaltimer](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#special-registers-globaltimer)。

## 窗口与误差

窗口时间T的条件范围为：

```text
T_low  = max(0, G_end_before − G_start_after − 2e)
T_high = G_end_after − G_start_before + 2e
C/T ∈ [C/T_high, C/T_low]
```

T_high≤0时，在声明前提下没有正时间窗口，入口拒绝。T_low=0时下端为开区间，C/T上界无穷，不能把分母夹到1ns；其它情况上下端闭合。四角枚举给线性差T的极值，正T上的倒数单调，因此上述率范围覆盖整个可行时间区间。未截零时窗口区间宽度为两个原GT括号宽度之和再加4e；2e是窗口每侧的扩张，不能与单读误差e或旧128ns总窗口预算混用。

给定只依赖预测输入的C_hat、r_hat，N_hat=C_hat/r_hat，两项同样使用整数或Fraction。函数保留固定e_C=C_hat/C−1，条件e_f(T)=(C/T)/r_hat−1及自由e_N(T)=N_hat/T−1的区间。同一个T在每一点满足`(1+e_C)(1+e_f)=1+e_N`，对应上下端也闭合。e_C仍是原生窗口误差，不等于裸MMA服务价格误差。三个分项不能独立挑选；跨窗口中位不相乘，多个共享端点的边际可行值也不构成联合可行性证明。

relative_error_status将单窗口条件误差范围分为within（所有值达标）、outside（没有值达标）和undetermined（两种都有）。不得在undetermined中挑有利时间当成通过。这里没有内置5%/10%、64/256ns或物理速率上限；阈值由已冻结实验传入，结果不授观察资格。

纯数学示例：start GT=[100,108]、end GT=[200,212]、C=150、e=2ns，得到T∈[88,116]ns、C/T∈[75/58,75/44]cycle/ns。C_hat=144、r_hat=3/2时N_hat=96ns，自由误差为[−5/29,1/11]；5%条件判定是undetermined。以上数字全部合成，不是GH200实测。

## 复验与范围

九项单元测试覆盖解析公式、误差标签与闭合、大整数原点平移、重叠开区间、零时间预算、保守状态、非法字段/计数域/预算和单位缩放。独立verify.py通过端点矩形四角而非调用窗口公式建立范围，在540个合成盒上核对界及正角点误差，另拒绝错误界和乘积仍闭合的分项互换负例。

```sh
cd /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/A-gt-bracket-contract-v1
python -m unittest test_r09_gt_brackets
python verify.py
```

旧740071是反向CLOCK-GT-CLOCK，未作为本模块样本。新增GT包围只约束读取时刻；硬件包围、GT误差界、observer扰动、off/off基线和新独立留出均未通过，不能据本数学合同解除C已指出的216系列残差失败。B源口径、价格、公共预测、runner、原门槛和全cfg_a/b/c及跨卡目标保持原范围。
