A 的零宽 GT 双点反例已在主线完整重放。`30e5a211` 接入为 `ef02d2e5`；8 项交付产物 SHA、88 项输入 SHA 均核对，复制后的分析程序与公共版本逐字节相同。完整 summary、55 个原始见证和独立 verifier 输出三份文件逐字节相等；四种不构成证书的负例检查通过。

740160 与 740161 共 11 个开启进程包含 9504 个三元组，完整枚举 29040 组同 SM、不同 CLOCK 的有序双点，得到 55 个矛盾见证：smoke 9 个、controls 46 个。全部为两个零宽 GT 盒同坐标；54 组 CLOCK 差 8 cycles，1 组差 12 cycles，没有严格反向 GT 盒。

例如 740160 CTA11/SM14、ordinal0 的 FIRST：WG1 CLOCK=1453442399967139，WG0 CLOCK=1453442399967147；两个 GT_before/GT_after 都为 1791725328316007232。若把两个盒当作精确物理采样时刻，它们强制同一时刻；同一单值非递减 CLOCK 函数却不能在这一时刻返回两个值。

该反例拒绝“e=0 精确时刻盒且同 SM CLOCK 是该时间坐标上的单值非递减函数”的合取。它不识别 GT 实际误差、读取乱序、计数器故障或应补多少 e。已有单窗口边际公式仍是条件代数；先前生命周期图的零 lag 可行性也不等于这些时钟读数有共同物理实现。后续不能把这些 e=0 盒作为已经认证的物理时间锚点，必须先有独立成立的采样与误差前提。

原 740161 观察控制失败、全部资格 false 保留，没有新增 GPU、价格、模型参数或重评分。此结论已交给 C，防止后续生命周期适配误用锚点；当前 C 的旧 R10 输出源码工作不依赖该假设。

```sh
python results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-gt-zero-width-witness-v1/r09_gt_order_witness.py \
  --base results/gh200_resource_campaign/access_rules \
  --inputs results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-gt-zero-width-witness-v1/inputs.json \
  --output NEW_OUTPUT_DIRECTORY
python results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-gt-zero-width-witness-v1/verify.py \
  --base results/gh200_resource_campaign/access_rules \
  --inputs results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-gt-zero-width-witness-v1/inputs.json \
  --analysis NEW_OUTPUT_DIRECTORY
```
