# 早段累计等待的缺失量

主线核对 B 报告的 85 个输入 SHA，并直接读取 20 份 progress、20 份 sparse 的原始 trace 字。stage4−6 的 C16−C1 配对差中位仍为 2941.5 cycle。progress 源码只有 local16/17 等待前标记，没有循环内等待返回标记；sparse 仅一个 group 有该返回。

下一个可辨识问题是：早段额外时间发生在 full 等待内，还是等待之外的循环、发射和释放路径。需要同一 CTA、同一输出、同调用的累计量 `Q_early = Σ(B_i−H_i), i=2…16`，H 在 full try/wait 之前，B 在 wait 成功返回之后。它与同调用 `C16−C1` 满足：

`C16−C1 = Q_early + Σ[(H_i−C_(i−1)) + (C_i−B_i)]`。

不能将旧 sparse 单点等待差乘以 15，不能跨协议借用，也不能把 Q 当裸 TMA、真实 ready 或共享源服务价格。这个量只区分等待内外，完整模型仍需依赖和共享状态；新增观测自身还须通过数值、源码/SASS及局部扰动检查。当前未新增 probe、GPU、参数或观测资格。见[原始核对](verification.json)和[B 证据](../B-20261011-minimal-service-observable-v1/B-minimal-service-observable.json)。
