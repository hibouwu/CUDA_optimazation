# GH200 微基准的阅读与复现

本轮按 [PLAN.md](PLAN.md) 为性能建模收集组件服务条件。访存基线、FMA、MMA、WGMMA、cp.async 和全局读写联合服务的正式十进程采样、实测说明和独立 C 审查已经完成；统一参数导出与整轮交付尚未完成。其他家族的 preflight 或短正确性记录只证明各自声明的验证范围，不能视作正式性能结果。最新阅读入口和资格记录见 [README.md](README.md)，阶段是否通过以绑定文件哈希的独立审查为准。

## 先找到实验和原始记录

| 要了解的内容 | 入口 |
|---|---|
| 旧三批配置如何保留差异、映射到新实验 | [EXP-03](EXP-03-compute-baseline.md) |
| 访存结果、计时空窗口与不稳定配置 | [EXP-04 实测](EXP-04-memory-baseline-results-v3.md)；[实验约定](EXP-04-memory-baseline-v2.md) |
| FMA、MMA、WGMMA 的实测、固定长度分析和复现 | [FMA 实测](EXP-05-legacy-fma-results-v3.md)、[MMA 实测](EXP-06-legacy-mma-results-v3.md)、[WGMMA 实测](EXP-07-legacy-wgmma-results-v3.md) |
| FP8/INT8 约定及 FP8 数值差异 | [实验约定](EXP-08-low-precision.md)、[四点诊断结果](EXP-08-boundary-result-v1.md)；诊断不授性能资格 |
| 标量、向量、跨步、广播及独立读写 | [EXP-09](EXP-09-shared-memory.md) |
| 异步复制与全局读写联合服务的正式结果 | [cp.async 实测](EXP-12-async-copy-results-v1.md)、[全局读写实测](EXP-13-global-duplex-results-v1.md)；资格与封存记录见README |
| 总控及 preflight 的接受边界 | [INTERFACE.md](INTERFACE.md)、[INTERFACE-PREFLIGHT.md](INTERFACE-PREFLIGHT.md) |

从仓库根目录看，源码在 `microbench/gh200_resource_campaign/`，本轮归档在 `results/gh200_resource_campaign/20261001-resource-suite-v2/`。后者受现有忽略规则管理；拿到源码仓库并不意味着同时拿到了原始记录。

归档中的 `implementation_status.json` 给出推进状态，`reviews/` 保存独立审查。一个运行的 `run_spec.json` 记录身份及依赖，`snapshot/` 冻结代码和约定，`environment/` 保存设备和分配，`build/` 保存编译及 SASS，`preflight/` 或正式采样目录保存进程收据和原始输出。看到二进制或 Slurm 的 `COMPLETED`，仍需查看每个子步骤的收据。

## 用一条真实记录手算

以下使用 `shared_memory/preflight-a` 的 `read_w4_stride1`。这是一条最小设备检查记录，用来演示计量，不能据此确定 GH200 的稳定带宽。

配置是一个 CTA、256 个线程，每线程每轮执行 8 次 4 字节读，共 8192 轮。因此逻辑读请求量为：

\[
Q=1\times256\times8\times4\times8192=67\,108\,864\ \text{B}.
\]

原始字段 `read_payload_bytes=67108864`、`write_payload_bytes=0`。同一 CTA 的 `stop_cycle-start_cycle=770205`，所以这条记录的请求服务率为：

\[
Q/\Delta c=67\,108\,864/770\,205\approx87.1312\ \text{B/clock64 cycle/CTA}.
\]

这里统计的是线程提出的逻辑请求，不是不同地址的总量，也不是物理 bank 事务或 HBM 流量。只有一个 CTA，分母也来自该 CTA；不能乘以 SM 数就宣称整卡带宽。广播配置同样记录线程请求量，另外记录不同地址量，用于解释多个线程访问同一位置的条件。

该行位于运行目录下：

```text
preflight/read_w4_stride1/batch_00/trial_00/attempt_00/raw.jsonl
```

文件 SHA256 为 `5d3a835506de9de27f7a09aa9703a00b732694d04784dc2968f0c7d6c08187dd`。它包含 device 和 trial 两条记录；手算使用 trial 的字段。独立重算检查器位于该运行冻结快照中的 `auditors/shared_memory.py`，不会直接相信探针输出的工作量总数。

## 本地重放这次预检

在源码仓库根目录执行，`RUN` 指向已复制完整的运行目录：

```sh
RUN=results/gh200_resource_campaign/20261001-resource-suite-v2/shared_memory/preflight-a
python -B microbench/gh200_resource_campaign/run_suite.py status "$RUN"
python -B microbench/gh200_resource_campaign/run_suite.py audit "$RUN"
```

总控先检查快照身份，再使用该运行冻结的审查器。离线重算不需要 GPU。成功的预检审查输出是 `preflight_evidence_validated_pending_independent_B_review`，其中 `hardware_qualification` 仍为 `false`。

这次共 15 个配置通过原始记录检查；广播读 `read_w4_broadcast` 的预热未收敛。该状态必须保留，不能选掉这一点，也不能把其余配置的一次进程样本当作十次独立进程统计。正式运行还需要框架 B、实验 B 和后续 C 的独立审查。

如果需要把归档移到另一台机器，复制整个运行目录，保留 `snapshot/`、二进制、原始输出和收据，再把 `RUN` 改为新路径。缺失快照或哈希不匹配时应修复归档来源，不能用当前源码填补后继续宣称同一运行。

## 计时为空不等于操作耗时为零

`memory_baseline/preflight-c2` 的一个零工作窗口出现两次 `globaltimer` 读数相同，而同 CTA 的 `clock64` 相差 25 个周期。现有严格计时检查拒绝该记录，因此这次运行没有通过。

这条失败证据说明现有接受规则需要审查，不能将 0 纳秒替换为 1 纳秒、偷偷加入延时，或把周期值写进纳秒字段。[计时分辨率补充草案](INTERFACE-TIMER-RESOLUTION.md) 正在定义零工作控制的特殊处理；通过独立审查并实施后，需要新运行验证。此前失败的原始记录继续保留。

## 什么结果能用于建模

先确认数值和计量正确，再确认完成事件、资源条件及统计稳定性，最后确认该参数的依赖证据齐全。正式经验参数需要附带精度、指令、线程范围、输入条件、访问布局、缓冲、工作集、计时单位以及证据路径。

一个组件的请求服务率只适用于其测量条件。计算与搬运能否重叠，需要后续受控组合实验；单项峰值不能直接相加或自动组合成整个 kernel 的预测。本轮也不把不稳定、数值失败或仅编译成功的点导出为合格标量参数。
