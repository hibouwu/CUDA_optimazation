# S20：BF16 TMA→WGMMA→输出批量测量方案

固定 128 线程、64×64×64 BF16 tile、FP32 累加，研究 TMA 与 WGMMA 的组合服务。每 K tile 搬运 16 KiB，完成四条 M64N64K16 WGMMA，统一 wait0。

## 固定配置

与 S19 相同五 mode、stage=1/2/4、K=1/2/4/8/16/32，64 留作检查点。正式拟合坐标 90 个，加 15 个 K64 检查点。短数值矩阵继续采用已签约定中的 periodic/tagged 与边界 seed，不因批量执行而缩减。

计算每序列为 `524288×Ktiles FLOP`；运输每序列为 `16384×Ktiles B`。transport 不报告计算吞吐；output 完成时间包含约定输出处理，其他模式停止后校验写不计入主窗口。

## 一次运行

已有 device 实现和 host-v2 候选，先在 CPU 分配完成真实 `sm_90a` 编译、链接、SASS/资源检查并修复发现的错误，不占 GPU 分配开发。随后一次 GPU 作业内连续完成全部短验证、自身校准、预热和正式采样。

每个 stage 的依赖必须闭合：TMA expect_tx/arrive token → acquire 输入就绪 → proxy fence/CTA 发布 → WGMMA fence/commit/wait0 → 消费者离开 → 下一 tile 复用。最后所有请求和使用者排空。脚本不能只凭最终 C 数值正确跳过生命周期检查。

短 trace 保存所有输入 tile、中间 C、最后槽、完成计数、输出和 guard；正式关闭 trace，保留完整末态及完成记录。已签参考、4096 fragment 映射和适用 CPU 检查直接复用，不另写一套模型。

校准只看 K≤32，比较时使用相同重复规则。按 mode/stage 拟合启动项与每 tile 增量，K64 检查误差；同时直接比较 serial、overlap、output 的实测差值。

## 当前需要补的代码与交付

host-v2 源码已通过[当前独立源码审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/combinations/reviews/S20-host-target-compile-source-B-v2.json)，尚无该版本目标编译或 GPU 结果。旧 host README 的待审标签保留为历史版本，当前资格以这份签录为准。需要完成目标编译、实际短证据、关闭 trace 的正式采样及报告；不能把 CPU 全值模型当作 GPU 正确性。

原实现与计量见[原 S20 约定](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/combinations/s20-v1/CONTRACT.md)，[host-v2](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/combinations/s20-v1/host-v2/README.md)为准确候选入口。

最终交付组合曲线、90 个拟合坐标和 15 个检查点终态、拟合有效性与残差、条件参数和一条真实 tile 算例。与 S19 共用报告形式，不新增全流程框架。流程见[批量执行约定](RUN-REMAINING.md)。
