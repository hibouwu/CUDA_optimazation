# 740071 CTA 头字段拒绝边界复核

A 的只读破坏测试发现原冻结 reader 接受缺失 entry/producer 端点及一致但越界的 SM 字段。当前 reader 补齐 entry/producer cycles/ns 正值、entry ≤ producer ≤ FIRST，以及本固定 132-SM GH200 协议的 SM 编号范围检查；每个失败 CTA 的两个角色均保留错误原因，不丢弃失败角色来形成有效记录。

运行 `python replay.py` 先复核原归档身份及抽样数值，再独立载入当前 reader。plain 仍为 0 条、matched/guarded 各 264 条有效角色记录；两观测臂各 5 个原始记录副本破坏实验全部拒绝（4 个头端点置零、3 个 SM 字段同时置 133）。源单元测试另覆盖两个时钟域的前后顺序反转，9 项测试通过。报告保存当前 parser SHA 和原 raw SHA。

原冻结 reader、raw、manifest 和判定保持原样；这是解析器加严后的补充回放。真实数据全部通过，因此此前条件括号数值结论不变。这里只验证本实验的固定完整 GH200 设备合同，不将 SM 数量当作所有 CUDA 设备 SM 编号的通用上界。硬件包围和观察扰动资格仍未成立，`observation_qualified=false`。
