# R10 联合端点与统计对象：主线复核

B 提交 `1eb0b5ec846cbb9d7588c7962aa9ab58dacd81f0` 已接入为 `ad34fe27`。原 739619 的 40 个独立 dual 进程重放完成，5280 个 CTA 首输出向量、1056 个条件摘要、1320 个响应摘要与 13200 个调度块响应向量全部重现。CTA55 的原报告 400 个标量保持一致。

两侧各 90 个输入 SHA 已核对，86 个共同文件字节相同；四个模块重定位，其中三个字节相同。`analyze_r18.py` 在主线增加了其他 trace 格式及其他分析功能，不能称为同一 SHA。其 `replay` 的 AST 差异只有 sparse trace 分派，实际 `r18-dual-clock-events` 在两版均不进入该分支。完整报告除输入路径/对应模块身份外一致，两份完整 gzip 账本字节相同，详见 [main-receipt.json](main-receipt.json)。

另已重放 [完整解码路径限定审查](../main-paired-decoder-review-v1/B-decoder-review.json)：`validate_warmup` 新增的 prefix 检查在本批输入正常返回，release 校验分支未启用。分别加载两版解码器后，40 个完整 replay 字典、5280 个 CTA 字典和 17280 个输出 tile 记录全部相同；这只证明本批输入的结果一致，不宣称整个解码模块或所有 helper 相同。

复制 B 的独立 `verify.py` 后在主线新输出上运行，不导入模型或分析模块，直接解码原始 trace 字并取排序中间两项。52800 个时长标量、10560 个同调用闭合关系、全部摘要和响应均通过，见 [verification.json](verification.json)。

端点保持原定义：`P0=prod−entry`、`S=FIRST−prod`、`L=MAIN−FIRST`。`S+L` 不包含显式起步等待。

| CTA55 统计 | 数值 / ns |
|---|---:|
| B32 延迟效应：配对 ΔS 的中位数 | 848 |
| B32 延迟效应：配对 ΔL 的中位数 | −512 |
| 上述两个中位数之和 | 336 |
| 逐块先求 Δ(S+L)，再取中位数 | 288 |
| B0：调度块 Δ(S+L) 中位数 | 816 |
| B0：两个条件的 S+L 中位数相减 | 832 |
| 延迟×行距：调度块交互中位数 | −576 |
| 延迟×行距：四条件中位数的线性组合 | −544 |

因此分段中位数、联合窗口中位数、调度块响应中位数、条件中位数之差须各自声明。十个 trial 是四个顺序独立进程组成的调度块，不代表相同隐藏初态；同进程 CTA/分段不是额外进程重复。

本复核只更新候选观测协议的联合数据要求，不改变其选定的两臂中位数差目标，不增加置信区间、价格、GPU 或资格。旧失败保留。

复现入口：仓库公共 `analyze_r10_paired_estimands.py --root results/gh200_resource_campaign/access_rules --output NEW_OUTPUT_DIRECTORY`；本目录 `verify.py` 校验本目录报告和两份账本。
