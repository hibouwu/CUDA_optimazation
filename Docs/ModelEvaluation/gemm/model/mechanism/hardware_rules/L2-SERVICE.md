# L2 条件服务：阶段1

更新：2026-10-17。两个先导及开发48进程已完成并合格；六个预置形式因稳定单调反例停止，未拟合参数、未冻结、未采留出。阶段顺序见[模块地图 §6](MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)，条件和执行记录统一见[L2-01](../../../experiments/gh200_sm90/Q03-source-service/L2-01-saturation.md)。

## 1. 现象与缺口

[EXP-13](../../../experiments/gh200_sm90/Q03-source-service/EXP-13-global-rw.md)已测小工作集的`.cg`读、写与混合请求；[R10](../../../experiments/gh200_sm90/Q03-source-service/R10-layout-cache.md)、[B01](../../../experiments/gh200_sm90/Q03-source-service/B01-bandwidth-cache.md)及[TMA共享服务](TMA-SHARED-SERVICE.md)保留复访、共享源和访问次序反例。

旧`global_duplex`固定CTA数，并在每次load后立即累计checksum；数组还按grid舍入。它不能直接提供固定工作量下的独立加载批量和活动规模曲线。第一轮先辨识指定缓存准备下的完整读窗口，写与1:1只做选点。

## 2. 工作假设

在同一生成代码、工作集、输入与完成边界下，读窗口由局部请求供给和共享源服务共同限制。定义：

- `B`：完整窗口的逻辑读取字节。
- `S`：实际参与访存的SM数，按SMID和活动区间核对。
- `J`：每线程的软件独立16B加载／累加流数；生成代码的实际次序另行核对。
- `b(J) = b1 · min(J, Jsat)`：单个活动者条件供给率的经验候选函数，单位B/ns。
- `Bg`：共享条件服务率，单位B/ns；`a`为此访问体的窗口固定项，单位ns。

保留两个候选族：

| 候选 | 读窗口预测 | 可区分的响应 |
|---|---|---|
| Hmax | `T = a + max(B / (S·b(J)), B / Bg)` | 局部限制与共享限制可以覆盖，转折后出现平台 |
| Hsum | `T = c + B / (S·b(J))`，`c = a + B/Bg` | 两段服务在此窗口累加，接近平台前仍有随S下降的增量 |

`Jsat`只取1、4、8，共六个具体形式；Hmax用同一组`a、b1、Bg`，Hsum用同一组`c、b1`解释所有读开发条件，不按case加项。固定B时Hsum的`a`与`B/Bg`不可分别辨识，因此只拟合共同项`c`。拟合前先核对SASS中的加载和消费次序及资源差异；按实际生成代码报告J的组织。这里的`Bg`是该请求窗口的条件率，缓存状态与物理流量证据引用原实验。

两族均预测固定B时增加S或J不会使窗口变长，超过其平台后收益接近零。若有效数据稳定违反此方向，保存反例并结束该形式；不靠修改已测条件或增加逐点补偿保留它。

## 3. 验证与结论

开发数据先核对结果、机器码、活动覆盖、重叠与暖机，再比较六个形式的逐点残差。Hmax同时覆盖局部受限与共享受限分支、且局部分支有至少两点时，再检验`a、b1、Bg`的分别辨识；全局部条件只给`Bg`下界，全平台条件只给`a+B/Bg`的组合。形式及参数冻结后才执行留出；本轮留出检验同工作量和程序组织下活动规模的插值。候选预测无法分开的条件只约束服务范围。写与混合结果单列，不从读曲线推算它们的参数。

[开发实测](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/L2-01-saturation/20261017-L2-01-execution-002/dev/summary.json)包含16条件／48独立进程，全部暖机收敛且性能资格通过；CUDA12.9.41的[九模板SASS审核](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/L2-01-saturation/20261017-L2-01-fit-001/target-review.json)通过。固定S时，J1→4的吞吐改善约2.1倍，J4→8接近平台；固定J时吞吐随S近似线性增加。本组没有单独辨识共享平台或裸L2服务率。

按第2节预置单调规则，S32的J4三个进程窗口为353856–354592 ns，J8为355328–355552 ns，两个区间不重叠；中位数分别354528、355424 ns，J8反而增加约0.253%。[六形式判定](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/L2-01-saturation/20261017-L2-01-fit-001/service.json)均为`refuted_stable_monotonic_reversal`，`selected=null`；判定在参数拟合前结束，没有参数、拟合误差或开发冻结，四个留出条件未采。

本轮排除的是`b(J)=b1·min(J,Jsat)`及其六个具体窗口形式在全部开发点上的共同单调预测，不将这一小幅逆向命名为某个物理资源原因。“局部供给与共享源服务共同限制”仍是待辨识的宽泛解释。本轮按既定停止条件收口，不扩点、不搜索新形式；写与混合选点保留原资格。

## 4. 模型接口

预定产物为`read_window(B, S, J, preparation, build) -> ns`的整窗条件服务；[l2_service.py](../../../../../../microbench/gh200_resource_campaign/access_rules/l2_service.py)实现其数值内核，参数包的`logical_bytes`绑定B，runner绑定构建、设备与缓存准备身份。它覆盖该探针的加载、地址计算、消费和窗口排空。Hsum的共同项只用于本轮固定B；其长度迁移另需观测。本轮没有生成可用参数包；[l2_fit.py](../../../../../../microbench/gh200_resource_campaign/access_rules/l2_fit.py)保存开发证据核对、有限求解、冻结及留出评分入口，本次只执行到反例判定。

整窗扫描先服务局部时间预测与饱和判断；请求级的命中、在途合并、逐请求完成和TMA目的投递继续由[TMA路径](TMA-SHARED-SERVICE.md)辨识。未来接入`input_source_async.issue`时须补齐相应事件规则，不能把整窗率直接作为每请求固定延迟。现有cfg_b源服务与已失败条件保持原版。
