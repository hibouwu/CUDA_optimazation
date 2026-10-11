原始 R10 的边界工作与输入阶段生命周期已在主线重算。B 的 `ff4a7423` 接入为 `8a5413a0`；两份完整报告除输入路径外相等，说明文件逐字节相等，两侧各 18 项输入 SHA 均重新核对。相关 R13 测试共 199 项通过。`verify.py` 另按独立轴区间计数与释放指针循环核对三种行距的 MMA 工作、672 条释放记录、6 个最后槽位退出证明。

每个输出固定 32 个 commit 组、256 次 m64n128k16 atom 调用。最后 K16 保留 8 次调用，其中 6 次不含有效 K 数学量；M40/N56/K16 角落仅 1 次 atom 含有效数学量。每条件总计 129024 次 atom 调用，有效数学量 31200000000 FLOP，名义密集 tile 工作 33822867456 FLOP。计数不构成裸 MMA 服务价格。

普通 wait1 释放前一组，尾部 wait0 释放末组；每组 full 义务为 32768B，empty 完成需要拥有该组的一个 WG 共 128 次到达。S6/Kt32 输出切换后 q0–4 复用前输出 q26–30，q5 才复用其 q31。下一 WG 的 MMA 顺序通知在前一 `mma()` 返回后、`mma_tail()` 前；例如 g95 的 tail 释放限制 g101 的槽位复用，不能额外作为 g96 MMA 顺序的前提。

producer_tail 遍历持续 ring 的最后 6 个槽位：T3 为 g90–95，T4 为 g122–127。较早组由后继同槽 acquire 覆盖；两部分合计覆盖每个输入组一次。这只关闭输入阶段生命周期，不能代替 D 全局可见或 kernel 完成。

现有时间图仍缺零源 B1 的填充完成、128 次 empty 到达完成/acquire 返回、明确 producer_tail 以及对应旧 pingpong 输出边界。原 full-tile guard 与供给时间模块保持原样。已把报告中 `C_boundary_request` 的具体源码接口交给 C；没有新 GPU 运行、服务定价、预测分数或资格。

```sh
python microbench/gh200_resource_campaign/access_rules/analyze_r10_boundary_lifecycle.py \
  --run results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4 \
  --binding results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/B-20261011-boundary-binding-v1 \
  --output NEW_OUTPUT_DIRECTORY
python results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-boundary-lifecycle-v1/verify.py
```
