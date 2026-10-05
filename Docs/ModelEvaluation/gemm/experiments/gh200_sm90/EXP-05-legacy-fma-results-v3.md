# EXP-05 实测：FMA 依赖链、线程数与固定长度

本记录对应 `20261001-resource-suite-v2/legacy_fma/formal-v3-b`。93 个配置各完成 10 个独立进程，共 930 条合并样本；当前统计均达到预定稳定条件。离线后审已通过，完整 C 审查尚未完成，本文与图表暂不授予合格参数导出资格。

原 [FMA 实验约定](EXP-05-legacy-fma.md) 和 [计时修订合同](../../../../../microbench/gh200_resource_campaign/contracts/legacy_fma_timer_v3.json) 保持冻结。首次正式作业 `730335` 按预算保存检查点，随后 `730337` 在同一 run 中补齐剩余进程。前一部署 `formal-v3-a` 因文件可执行权限丢失，在 probe 启动前失败，未产生样本；它不参与本次统计，失败证据保留。

## 实验问题和条件

在给定精度、依赖链数、参与线程和循环长度下，测量计算与结果排空所需时间。单 CTA 用同一 CTA 的 clock64 差，报告 FLOP/clock64 cycle/CTA；全 GPU 用所有 CTA 的 globaltimer 包络，报告 GFLOP/s/GPU。两种单位分别展示，不用固定频率把整卡时间伪装成实测 SM 周期，也不将单 CTA 数值乘以 SM 数。

设备为 GH200，132 SM，CUDA 12.9、`sm_90a`。二进制直接复用通过 B 审查的原件，36 个动态长度点复用原 pilot 解析结果，未重编译、未重新挑选长度。详细环境、资源和输入条件绑定于归档，真实正式启动仍校验 GPU UUID、驱动、工具链和依赖身份。

| 来源组 | 有限配置 | 点数 |
|---|---|---:|
| 旧 initial | 6 个精度形式，32 线程，1/8 条依赖链，固定 128/512/2048 轮 | 36 |
| 旧 sustained | 6 个精度形式，32/128/256 线程，8 条链，单 CTA/全 GPU，使用已冻结的校准长度 | 36 |
| 旧 audit | FP32、FP64、FP16×2，256 线程、8 条链，单 CTA/全 GPU，固定 8192/32768/65536 轮 | 18 |
| 空计时窗口 | 3 个旧控制坐标，只报告 cycles/window | 3 |

六种形式为 FP32、FP64、FP16、FP16×2、BF16、BF16×2。每条链每轮 16 个逻辑 FMA，乘法与加法各计一个 FLOP；打包形式每条指令包含两个有效 lane。相同名称不自动合并旧来源，所有点仍保留完整 mapping ID。

## 一条真实记录的手算

在 case `legacy-561e08351ce3cb7d2b1b4f8da91db0cbd987435dec29b214be807fe3f0c2fcf6` 的 `batch_00/trial_00/attempt_00/raw.jsonl` 中：

- 1 CTA、32 线程、1 条链、每轮 16 个 FMA、128 轮、每个 FMA 1 个有效 lane。
- 逻辑工作量 `2×1×32×1×16×128×1=131,072 FLOP`。
- 同一 CTA 的 clock64 差为 `9,247 cycle`。
- `131,072 / 9,247 = 14.1745430951 FLOP/clock64 cycle/CTA`。

整卡例子在分析目录的 `worked_examples.json`：32 线程、528 CTA、8 条链、每链每轮 16 个 FMA、1,048,576 轮，工作量为 `4,535,485,464,576 FLOP`；globaltimer 包络为 `74,245,920 ns`，因此 `work/ns=61,087.3360392 GFLOP/s/GPU`。这里 `FLOP/ns` 数值等于 `GFLOP/s`，并非 FLOP/cycle。

## 代表结果与完整图表

下面列出 sustained 组的单 CTA、128 线程、8 条链、每轮每链 16 个 FMA。循环长度由各自冻结的校准结果决定，表中明确列出，因此不能把差异只归因于精度。所有行均为 10 个独立进程的中位数。

| 形式 | 实际轮数 | FLOP/clock64 cycle/CTA | 样本 CV |
|---|---:|---:|---:|
| f32 | 1048576 | 232.396784 | 0.000000% |
| f64 | 735928 | 122.268523 | 0.000000% |
| f16 | 1048576 | 230.760184 | 0.000000% |
| f16x2 | 1048576 | 414.784212 | 0.000000% |
| bf16 | 1048576 | 208.713070 | 0.000000% |
| bf16x2 | 1048576 | 414.784212 | 0.000000% |

CV 为零表示本批记录的主周期指标相同，不表示时钟、电源或主机启动时间没有变化。

![FP32 单 CTA 条件比较](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-fma-formal-v3-b-r3/f32-one_cta-1.png)

图中 `t` 是线程数，`c` 是每线程链数，`q` 是每链每轮逻辑指令数，`lanes` 是有效打包 lane 数，`n` 是实际轮数；方括号是 case ID 后八位，完整 ID 与旧来源保存在 `cases.csv`。点显示全部合格进程样本，菱形为中位数，误差条为最小至最大值，不是置信区间。点或误差条重合不意味着只采了一个样本。

全部精度、两个 scope 和所有固定长度见 [图表索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-fma-formal-v3-b-r3/index.md)。空窗口列在 `cases.csv`，不会混入 FLOP 速率图。所有正式样本保存在 `samples.csv`，原始记录路径和哈希逐条绑定。

## 固定长度怎样拆出启动项与增量

对同一个实际编译 kernel、相同参数与范围的固定长度点，描述性拟合 `T(n)=a+b×n`。脚本从冻结源的 case 表取得真实函数符号，并要求该符号在原 SASS 审查记录中；不只按精度名称分组。校准长度点不加入固定长度拟合。

例如上述 FP32、32 线程、1 条链使用同一函数 `lc_f5dd4e591744a76ed846`。128/512/2048 轮的中位数分别为 9247/36511/145567 cycle，本组得到：

`T(n)=159+71×n cycle/CTA`。

这里 159 是本窗口下的拟合常数项，71 是包含该循环控制与计算的每轮增量。它们不是某条 FMA 的裸延迟，也没有证明能外推到其他线程数、依赖链或 kernel。18 组拟合的坐标、斜率、截距、样本内最大相对误差保存在 `fixed_length_fits.json`；没有独立留出验证，不作为完整性能预测器或最优 tile 搜索依据。

## 正确性与实际生成指令

正式结果包含完整输出校验，另有独立的非均匀短输入检查；两者分别保留。FLOP 数按有效标量运算计，SASS 实际发射需求另看编译审查：部分 scalar FP16/BF16 的八链形式，将每轮 128 个逻辑标量操作降低为 64 条双 lane HFMA2。有效 FLOP 守恒，但逻辑指令数不等于机器指令数，不能把每个标量 PTX 都当作一次独立 HFMA2 发射。

本次全部 93 点达到统计稳定条件，其中 90 点是 payload，3 点是计时控制。稳定判定仍需结合数值、SASS 和条件身份使用；当前等待独立 C 审查。NCU 沿用同设备、驱动和权限身份下已记录的权限拒绝，不逐配置反复尝试；本次没有硬件计数器动态发射证明。

本地离线重放通过 `bounded_float_replay_v1`，保留 79 个允许范围内的派生浮点差异，独立核对 1023 条阈值判定。只对白名单派生统计采用至多 2 ULP 比较，raw、FLOP、长度、单位、哈希与状态均严格检查。原 Python 3.9 环境的 preflight 严格审查和二进制导入已验证通过；本地后审不改写严格比较的失败证据。

## 复现

从项目根目录执行。分析输出必须为原 run 外的新目录；依赖 matplotlib，本次版本 3.10.8。先重放完整原始证据，再生成分析：

```bash
python -B microbench/gh200_resource_campaign/audit_replay.py \
  results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_fma/formal-v3-b \
  --policy microbench/gh200_resource_campaign/contracts/offline_replay_v1.json \
  > /tmp/fma-replay.json

python -B microbench/gh200_resource_campaign/report_compute_v2.py \
  results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_fma/formal-v3-b \
  --replay /tmp/fma-replay.json --output /tmp/fma-analysis-new
```

原运行与恢复入口分别保存在 suite 的 `implementation/job-730335/driver.sh`、`implementation/job-730337/driver.sh`。GPU 重现必须处于有效单 GPU Slurm 分配；新实验新建 run，同一次预算检查点使用 `resume`，不能再次执行 `run` 或把新源码写入原快照。数据与图表在本地 `results/gh200_resource_campaign/20261001-resource-suite-v2/`，未自动提交或推送。
