# A 中位数组合反例：主线复核

A 提交 `4669075053e94b78af656667e8f1ee34e94eeaad` 已接入为 `1ce87c36`。九个交付 SHA、公共模块与交付副本身份均核对；主线新目录重算的报告和独立验证输出与 A 逐字节相同，见 [main-receipt.json](main-receipt.json)。

四个精确三状态总体表明：分量的完整边际相同，仍可能有不同的 `median(C/r)`、`median(X+Y)`、`median(max(X,Y))` 和逐次最后分支。固定 C 列后枚举本三状态表示中的六种 rate 列置换，时间中位数可为 20/3、8、48/5；固定 X 列后的三种 Y 置换给出串行中位数 2 或 10、并行中位数 1 或 9。四项错误组合主张均被拒绝。

这些是合成精确反例，不是 GPU 误差、性能下界或新服务价格。它们要求同执行的端点、换算及联合分量先匹配再统计，不要求先建立完整随机硬件模型。独立完整时间和逐次关键 CTA 验证仍保留。

[report.json](report.json) 保存精确分数，[verification.json](verification.json) 保存有限 CDF 与置换验证。复现使用本目录 `r09_median_composition.py`、`cases.json` 与 `verify.py`；未读取 GPU 样本，资格全部 false。
