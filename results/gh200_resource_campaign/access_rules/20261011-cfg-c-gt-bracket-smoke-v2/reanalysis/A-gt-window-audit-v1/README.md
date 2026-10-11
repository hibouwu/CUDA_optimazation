# 740160 GT三元组与共同计算窗口

独立按原始trace/shadow索引复算864个GT_before→CLOCK→GT_after三元组、432个WG窗口和216个两WG共同窗口。新binary为`c5476d9b00679859eef64bff34e87532e4936ea73f7f432c476597b8c36a6480`，GPU为`GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114`；固定cfg_c 2304×3072×4096、S4、cluster1×2、dyadic/seed17/sw1、16次同buffer关闭观察的前序。每个warm臂仅一个进程；132CTA中84个两次输出、48个一次输出。

显式`gt_error_ns=0`只表示在记录的GT坐标上做精确算术。中央CLOCK真实采样被两GT包围是前提，GT对连续物理时间的误差界仍未知。零宽GT三元组也不证明零采样延迟或零物理误差。本报告不取得观察、预测或联合可行性资格。

共同窗口在同GPU/SM原生cycle域选`FIRST=argmin_WG(C_FIRST)`、`MAIN=argmax_WG(C_MAIN)`，相等时选WG0；GT沿用所选端点的标签与原始槽位。216个首/末WG选择计数为(0,0):33、(0,1):3、(1,0):179、(1,1):1。没有独立取GT的极值来替代已选CLOCK端点。

对每个窗口，`C=end.cycle−begin.cycle`，`T∈[end.GT_before−begin.GT_after,end.GT_after−begin.GT_before]`，`C/T∈[C/T_high,C/T_low]`。全部648窗口下界为正，432个WG三元组与区间均同主线冻结reader逐值相符。864次读取宽度直方图为0:58、32:729、64:75、96:1、128:1。

| 输出序号 | 共同窗口数 | 原生cycle差中位 | 全样本时间下界最小／上界最大，GT ns | 条件计数率区间的全样本范围，cycle/GT ns | 窗口区间宽度中位／最大，GT ns |
|---|---:|---:|---:|---:|---:|
| 0 | 132 | 66111.5 | 37536 / 38240 | 1.731360 / 1.761402 | 64 / 192 |
| 1 | 84 | 66017 | 41248 / 42272 | 1.561767 / 1.600126 | 64 / 128 |

同CTA首/次输出的84个共同窗口配对、168个同WG配对，条件平均计数率区间全部不相交。共同窗口中`first.rate_low/second.rate_high−1`的最小／中位／最大为9.745247%、10.195148%、10.969350%。因此在本条件前提下，三元组自身的记录宽度不能使两输出共享一个精确常数率；这不唯一识别物理频率，也不自动排除允许误差的时间预测。没有拟合额外读数预算、规则或费用，没有复用旧binary校准。

若已有点读数`N_post=GT_end_after−GT_begin_after`，则`N_post−T∈[−width_begin,+width_end]`，上下误差来自各自端点。文件同时保留这个非对称范围，不能把两个宽度相加后冒称必要误差。

`inputs.json`绑定17项输入SHA；独立复验另外核对14项smoke冻结文件、858项构建source SHA及输入/输出archive的15/30个文件与落地副本相同。全部648窗口和252配对用原始整数与有理数重算，四个错误标签、错误边界、误差符号和错误相容判定负例均拒绝。新临时目录重放的三份分析文件与独立复验输出逐字节相同。主线7,077,888个D元素与抽样数值的精确reference证明仅作绑定复用，A没有独立重算全D；来源report和D文件SHA保留。

后续740161完整控制见[主集成结果](../../../20261011-cfg-c-gt-controls/README.md)：off/on 4912/6372、off/off 4729/6372序列失败，L各432/432失败，观察资格仍未通过。本目录的单进程区间检查不替代这些分布结果，不改变门槛或完整event/新留出/跨卡边界。

```sh
task_run=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-smoke-v2
task_build=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-build
task_audit="$task_run/reanalysis/A-gt-window-audit-v1"
task_replay=$(mktemp -d)
python "$task_audit/r09_gt_window_audit.py" --root "$task_run" --build-root "$task_build" --inputs "$task_audit/inputs.json" --output "$task_replay/analysis"
python "$task_audit/verify.py" --root "$task_run" --build-root "$task_build" --inputs "$task_audit/inputs.json" --analysis "$task_replay/analysis"
```

`analysis/windows.jsonl`保留原始槽位、端点选择、工作坐标、精确区间及点读数误差；`analysis/phase-pairs.jsonl`保留逐CTA/WG的相交判定，`analysis/audit.json`汇总；`verification.json`与`replay-verification.json`记录独立复验和新目录重放，`output-sha256.json`绑定本目录产物。仅CPU离线，无GPU提交或预测评分。
