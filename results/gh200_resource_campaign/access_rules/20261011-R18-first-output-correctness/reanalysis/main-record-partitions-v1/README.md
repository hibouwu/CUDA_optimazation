C 对 740067 旧 correctness 记录的分段分析已在主线完整重跑。7 项交付 SHA、两侧各 37 项输入 SHA 均重新核对；36 项路径共同，迁移后的分析程序逐字节相同。完整 report 除程序路径外相等，调用身份、完整角色记录、分段 CSV 和有符号差 CSV 四份数据逐字节相等，见 `verification.json`。

两次 on 调用各 12 CTA×2 consumer，合计 48 角色、288 分段、144 有符号分段差。96 项调用内 cycles/ns 闭合及 48 项差值闭合全部成立，16384 个 warm 数值独立核对通过。完整 D 的既有精确证据复用并重核 SHA。

该作业在 a057 单 GPU、4 CPU、节点非排他分配，实际卡为 GPU-43269fbc-449d-3e0f-908a-9c81229546d3；并非后续控制的 ef 卡。map960/map1024 各只有一个 on 调用，前序 mainloop 参数分别沿用各自目标 map，进程地址也不同。现有分段混合 full 等待、MMA 发射/等待、释放和控制路径；不能从精确算术差识别 map 因果效应或独立供给价格。

没有新增 GPU、公共代码修改、服务价格、候选或预测成绩。原控制失败与全部资格 false 保留。

```sh
python results/gh200_resource_campaign/access_rules/20261011-R18-first-output-correctness/reanalysis/main-record-partitions-v1/analyze.py \
  --run results/gh200_resource_campaign/access_rules/20261011-R18-first-output-correctness \
  --output NEW_OUTPUT_DIRECTORY
python results/gh200_resource_campaign/access_rules/20261011-R18-first-output-correctness/reanalysis/main-record-partitions-v1/verify.py
```
