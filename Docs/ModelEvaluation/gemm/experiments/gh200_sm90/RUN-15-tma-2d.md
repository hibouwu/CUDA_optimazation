# S15：TMA 二维搬运批量测量方案

测量 GMEM↔SMEM 的二维 TMA 完成服务，比较有效 payload、全局行 padding 与合法 SW128。transport 字节数不作为物理 HBM 流量。

## 固定配置与已有结果

原 68 点：两个方向、单 CTA / 全 GPU、每方向 17 种 payload/布局。1/4/8/16/32/64 KiB 的 continuous-none 与 padding-none，加 1/4/8/16/32 KiB 的 continuous-SW128。

第一组 56 点已有 56 个自身 pilot 和 560 个正式样本；作业 734213 完成，普通 ARM 审查器重放通过。完整归档已传回本地，并通过[有限数据 C 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/review/first56-data-C-review.json)；图表、报告和参数发布仍待复核，未授全 68 配置完成。已有正式采样不重做。

剩余 12 点均为全 GPU 的 padding-none：两个方向各六种 payload。原存储组 8/2/1/1 继续作为内部目录，连续在一个作业内运行，不再逐组等待人工批准。

## 一次运行

使用完整、已签的 formal-sampling 源码包，新 namespace 中部署一次。`run_s15_once.sh` 连续调用已有 `run_formal.py` 的四个剩余 cohort；原采样器负责自身 pilot、全值参考、冻结 N、预热、十进程采样及有限复测。普通通过后立即运行下一组。

运行结束后用该包自身 `audit_formal.py` 离线重算，并生成表、图及真实样本算例。独立 C 在 GPU 释放之后核对，不在 GPU 作业里等待人签录。

## 容量条件

旧设备 pilot 的剩余 12 个压缩包合计 158877160 B。一批十进程加自身 pilot 的粗估约 1.75 GB；最多三批粗估约 4.93 GB。它们不是压缩后的可靠上界，不能据此保证未来配额。正式 raw 的峰值、持久归档余量和原预留必须在提交前按实际设备检查。

当前原采样器仍在每点保护真实配额，预算不足会保存检查点而停止。新连续入口不会绕过此保护。要真正一次作业跑完，须先为全部剩余点准备足够持久容量；不能只预算一个 cohort。已有第一组和旧数值证据保留。

连续入口在提交前检查原源码门禁、全部源码身份及组配额，要求扣除原预留后的余量至少 6 GiB。这个阈值是最多三批的运营预算，不是压缩输出的数学上界；运行中仍核每点真实容量。

## 计量与交付

每次有效 payload 为 Q=2WH，计时内逻辑工作量 X=BNQ。单 CTA 为 X/Δclock64，全 GPU 为 X/Δglobaltimer_ns。G2S 循环结束后导出两份 tile 的 2BQ 字节不计入 X；padding 不计入 Q。

结果包括原 68 点终态、分别按范围及方向展示的图、完整 CSV 和条件参数。先完成的 56 点可在独审后发布；合并后保留每个采样组的真实设备身份，不将不同设备条件的进程样本混为一个参数。

当前连续入口为 [run_s15_once.sh](../../../../../microbench/gh200_resource_campaign/runners/run_s15_once.sh)，统计图表入口为 [report_tma_tensor_2d.py](../../../../../microbench/gh200_resource_campaign/report_tma_tensor_2d.py)。命令与执行边界见脚本帮助；CPU 检查不冒充实际新 GPU 执行。

在 ROMEO 已准备好完整源码包和容量后，只需一次提交：

```sh
bash microbench/gh200_resource_campaign/runners/run_s15_once.sh \
  --submit /ABS/DEPLOYMENT/repo remaining
```

该命令提交一个 GPU 作业连续完成剩余四组，并自动提交依赖它成功结束的无 GPU CPU 重算作业。CPU 使用包自身普通审查器，生成 `report-<cohort>/tables/` 下的 CSV、真实样本、中文表及参数候选；原生环境没有 Matplotlib 时先生成表，取回本地后用同一报告脚本生成图。默认不重做第一组 56 点，也不自动授最终 C 或合格参数。
