C 的旧 cfg_b D 生命周期 v3 已在主线完整重放。正式输入为 `../C-20261011-legacy-D-lifecycle-v1/delivery-final-v3.json` 与 `data-final-v3/`，中间版未作为接线依据。10 项交付 SHA、两侧各 15 项输入 SHA 均核对；13 项共同路径及重定位的分析程序/R15 文档字节一致。完整报告除路径外相同，三份完整 ledger 逐字节相同，18 项 CPU 检查通过。

主线预审发现并交回了一条缺失边：各线程自己的 mma_tail 返回必须先于同线程 epilogue order wait 返回。原聚合 D entry 已同时受两者约束，但单独 wait 端口的顺序不完整。C 在 v3 补齐逐线程边与测试，原 v1/v2 中间版保留。

每个条件有 132 CTA、504 输出和 4032 个 64×32 D 子块，其中 full3720、partial134、zero-valid178。有效全局 D 共 31200000B，名义源 33030144B；后者不是物理流量。源码顺序为 epi_n 外层、epi_m 内层，不能直接沿用 B ledger 的坐标排序。全越界子块仍保留准备、同步和 store 协议，物理源读取量与时间均未知。

每输出末尾的 read0 只关闭对应 issuing-thread bulk groups 的源读取义务。非 issuing 线程可以提前返回及到达；下一 WG 的 epilogue 需要全部 128 次到达。同 WG 的 j+2 经过自身前输出的到达，再等待另一 WG 的 MMA 通知。有效 D 全局可见是独立义务，不能由 read0 推出；输入 producer_tail、D 源生命周期、384 线程返回和 CUDA 完成分别保留。

只有必要源码约束，无 GPU、价格、GT 锚点、局部留出或预测资格。与输入图的组合另见 `../main-lifecycle-composition-v1`。

```sh
python results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-legacy-D-lifecycle-v1/analyze.py \
  --run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4 \
  --out NEW_OUTPUT_DIRECTORY
python -m unittest discover \
  -s results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reanalysis/main-legacy-D-lifecycle-v1 \
  -p test_lifecycle.py -v
```
