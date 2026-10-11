# 主线复核 cfg_a 原子源组合

52f18ce 已合入为94992c8；74项cfg_a/pingpong/shared/ledger/release/raster测试通过。主线用当前C output_model.py独立重放22条件、220进程、28200 CTA及12种图模板，结构和实际work检查全部通过。

与B-v2逐项核对：完整source-bindings解压后字节相同；每一图仅新增C已合入的两rank store_tail_read0节点、相应FINAL依赖和source_guarantees元数据。将这项明确升级投影回旧节点后，12份完整图JSON全部精确相同；没有其他边、输入请求、目的投递或工作量变化。差异核对见integration-comparison.json。

每K-group两个rank合计A16×2+B4×4，6源请求/48KiB；10目的投递/64KiB；每目的full为本地producer arrival+A16+两issuer各两个B4，共32KiB。B_issue/B_to_r是汇合节点，不另收传输费。这里没有服务价格、目标时间输入或cfg_c迁移；闲置CTA也未被赋予生命周期时间。
