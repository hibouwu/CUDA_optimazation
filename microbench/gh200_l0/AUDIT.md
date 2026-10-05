# GH200 FMA / MMA 测试过程审查

审查日期：2026-09-30。对象为本目录首批单 CTA 测试、持续发射压测与周期换算。原始数据保留；本次新增代表配置复测，不以静态审查代替 GPU 执行。

结论：旧结果可以保留为限定输入与代码条件下的探索性服务率。WGMMA 持续发射接近 32 cycles/条的单 CTA 观察经复测保持；不能将它解释为单指令完成延迟、独立发射端口间隔或所有形状通用的硬件峰值。固定 1.98 GHz 参考周期继续作为归一化指标，硬件周期参数优先采用实测 clock64。

## 依据及落实

| 依据 | 对本次测试的要求 | 审查与调整 |
|---|---|---|
| [EDP 第 1 课](</home/jianyeshi/Note/SiteCHPS/CHPS/M2/EDP/CM/1_edpCpl_zh.md>)：系统、服务、指标、影响参数与实验步骤 | 先说明测量哪个服务及其条件 | 区分依赖成本、持续吞吐、局部 SM 服务和整卡完成；不把 compute-only 当完整 GEMM |
| [EDP 第 2 课 §2.5–2.6](</home/jianyeshi/Note/SiteCHPS/CHPS/M2/EDP/CM/2_edp_zh.md:95>)：负载等级、重复性与置信区间 | 饱和吞吐不能当响应延迟；误差估计须说明样本关系 | 预先固定代表配置，12 轮交错采集，以轮号进行 WGMMA 配对；同一进程重复不宣称机器会话独立 |
| [EDP 第 3 课 §2.5](</home/jianyeshi/Note/SiteCHPS/CHPS/M2/EDP/CM/3_edp_zh.md:78>)：瞬态与稳态 | 预热次数本身不证明稳定 | 预热最多 30 次，至少 8 次后检查最近 5 次 kernel 时长 CV≤1%；失败状态保留 |
| [EDP 第 4–5 课 §4–6](</home/jianyeshi/Note/SiteCHPS/CHPS/M2/EDP/CM/45_edp_zh.md:234>)：测量开销、准确性/精密度、因素与交互作用 | 重复一致不代表准确；计时开销和工作量须检查 | 补空窗口、三种固定循环长度、非均匀输入；只对当前形状和并发配置作结论 |
| [Thor 静态重新标定](../mma_config/Docs/ExperimentReport.md) | 将动态分派移出热循环，保留 completion 对照及 SASS | GH200 已使用编译期配置；新增空窗口及长度敏感性，保留必要同步成本 |
| [Thor campaign](../sm110_gemm_campaign/README.md) | 设备身份、SM 覆盖、源码/二进制、重复与原始输出可追溯 | 新批次保留 job ID、设备日志、实际源码、编译命令、SHA-256、逐 CTA 计时；不覆盖旧批次 |
| [NVIDIA NVBench 采样流程](https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help.md) | 预热、最小样本、噪声判据及有界停止分别定义 | 自定义 runner 借鉴流程；没有声称使用 NVBench 框架或其默认参数直接保证可靠性 |
| [Nsight Compute 可重复性](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#reproducibility) | 区分未插桩运行与 profiler；时钟和 replay 会改变结果 | profiler 独立运行，显式关闭 clock/cache control；计数器权限失败单独记录 |
| [CUDA 计时方法](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/) | 异步执行必须明确完成边界 | WGMMA 最终 wait 0 后排空结果，再停止计时；CUDA event 交叉检查覆盖整个 kernel |

## 发现与处理

| 优先级 | 发现 | 对旧结论的影响 | 本次处理 |
|---|---|---|---|
| 高 | 全程遥测中位频率不能代表每个 workload 的实际周期 | 1.98 GHz 换算本身是合法参考指标，但不能作为实测硬件周期 | 新增 start_cycle/stop_cycle，只在同一 SM 内取覆盖区间；跨 SM 只累加区间长度，不相减异源计数器 |
| 高 | 原持续压测每配置连续采 5 次，且展示已测最优值 | 不足以稳健判定不到 1% 的等待协议差异；选优还引入选择效应 | 新批次预先指定 7 个代表配置，2 个范围、3 个长度、12 轮随机交错，不再重新挑选最优配置 |
| 高 | FMA 的收敛常量结果不能检出部分漏算；MMA 全常量输入掩盖地址映射错误 | 原数值 PASS 仅验证简单数据，不能代表一般正确性或动态计数 | 新增 FP32 精确计数输入，以及 FP16 MMA/WGMMA 行列/K 非均匀输入，9 项短测试全部通过 |
| 中 | 热循环含循环控制，末尾含归约、SMEM 捕获及 barrier | 原周期是代码服务成本，不是裸硬件延迟/启动间隔 | 补 84 次空窗口；所有代表配置空窗口低于最短窗口的 0.012%，长度间吞吐变化低于 0.016%；不把空窗口机械扣除为纯延迟 |
| 中 | 只检查 SASS 指令存在，未检查动态计数 | 静态 presence 不证明运行中准确发射了全部预期指令 | 补独立 NCU 尝试；返回 ERR_NVGPUCTRPERM，动态计数仍未闭合。补非均匀校验提高可信度，但不宣称它替代计数器 |
| 中 | WGMMA ptxas 插入 warpgroup.arrive | 测得值包含生成代码所需的同步 | 归档实际 SASS，核验 wait 0/3/7；保持条件化解释 |
| 中 | 单 CTA 或所有 SM 被访问，不等于一直满占用 | 不能以单 CTA 值乘 SM 数构造实测整卡峰值 | 保留原整卡全局完成指标；新增全 GPU 负载下的每 SM 覆盖周期服务率，两者分开 |

## 复测结果

[复测报告](results/20260930-audit/REPORT.md) 和 [结构化统计](results/20260930-audit/summary.json) 记录 504 次正式测量、84 次空窗口、预热及独立采样状态。

周期分母定义为：对 SM s，将该次 grid 中运行于该 SM 的 CTA 覆盖区间记为 `C_s = max(stop_cycle) - min(start_cycle)`。全 GPU 负载下的平均 SM 服务率为 `sum(work) / sum(C_s)`，包括区间内可能的空隙，单位为 FLOP/SM 覆盖周期。它不等同于整卡全局 makespan 吞吐，也不将单 CTA 视为独占 SM 的理论能力。

| 代表配置 | 单 CTA FLOP/cycle | 全 GPU 负载下 FLOP/SM 覆盖周期 |
|---|---:|---:|
| FP32 FMA，256 线程 | 245.448 | 247.807 |
| FP64 FMA，256 线程 | 124.830 | 127.214 |
| FP16×2 FMA，256 线程 | 250.136 | 254.307 |
| FP16 mma.sync M16N8K16，256 线程 | 2712.990 | 2712.992 |
| FP16 WGMMA M64N64K16，128 线程，wait 0 | 3892.570 | 4095.895 |
| 同上，wait 3 | 4095.937 | 4095.521 |
| 同上，wait 7 | 4095.937 | 4095.521 |

单 CTA 的 wait 3 / wait 0 配对比值为 1.052245，wait 7 / wait 3 为 1.000000。本次数据支持单 warpgroup 放宽逐组等待可提高约 5.22%，继续增加到 wait 7 没有观察到收益。全 GPU 运行的每 SM 周期服务率都接近 4096；不能继续把旧的 wait 3 最佳参考周期点解释为普遍更优。

**预热例外：** wait 3 的完整 kernel 时长未在 30 次上限内满足 CV≤1% 的判据。数据保留且标记失败；观察到周期服务率稳定不等于时间、频率和功耗均达到稳态。

**计数器例外：** 两次 NCU 运行均明确返回 ERR_NVGPUCTRPERM。没有修改集群驱动或权限设置。当前验证包括 CUDA 执行、逐元素结果、SASS、工作量重算和 SM/周期记录，不包括动态硬件指令计数。

**非均匀输入：** [独立验证记录](results/20260930-audit-validation/validation.jsonl) 包含 FP32 精确计数、FP16 MMA 寄存器片段和 WGMMA interleaved SMEM 布局，各测试 1/3/17 轮，合计 9 项、55296 个输出均零误差。此补测不提供非均匀数据的性能/功耗分布，也不覆盖所有类型和布局。

## 建模可用性

- 可用：上述精度、形状、线程组织、并发及等待条件下的经验服务率；单 CTA 长度斜率；已验证的局部到整卡并发趋势。
- 保留为探索性证据：未复测的其他类型/形状、原 1.98 GHz 参考周期整卡结果、初始短链测量。
- 尚不可当作已知硬件参数：纯单指令完成延迟、无控制开销的硬件启动间隔、任意数据/形状的峰值、动态计数器证实的执行量、长期热稳态、完整 GEMM 性能。
- 置信区间仅描述本次 12 轮样本内不确定性。重复值一致可能产生退化 bootstrap 区间，不代表系统误差为零；没有因高 R² 或零宽区间而扩大结论。

## 复现

```bash
python3 microbench/gh200_l0/generate_audit.py
python3 microbench/gh200_l0/generate_validation.py
# audit.cu、audit_cases.json、run_audit_remote.sh 放入新的远程实验目录。
# 在已分配的 GH200 节点登录 shell：
source /path/to/new-run/run_audit_remote.sh /path/to/new-run
# 取回原始结果后：
python3 microbench/gh200_l0/analyze_audit.py microbench/gh200_l0/results/20260930-audit
```

`generate_audit.py` 从原持续压测归档构造新源码，冻结原配置作为对照。验证程序另外编译运行，数值结果和性能数据分开保存。
