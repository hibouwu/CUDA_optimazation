# Task A 首轮 cluster 加性项

结论：没有可冻结的边界候选。P2 首轮 A、B 的 cluster 填充面积都为零，因此首轮加性项对所有 CTA 都为零，任意有限非负系数都无法降低 P2 误差。按当前 PLAN 的终点 3 结束，不继续拟价或改为 later 修正。

拟合前声明的形式为：

`extra_L_ns[cta,j=0] = (K/64) * beta_config * cluster_max_fill_fraction`

单位为 ns/Ktile/单位 cluster 最大填充比例；cfg_a 使用 A，cfg_c 使用 B。修正放在冻结 r2 的 max 外，只加到首轮 L，保留 r2 自身填充项。原拟价目标为扣除 r2 自身填充增量与匹配内部完整 CTA 漂移后的 R18 配对残差。该形式已由静态反例淘汰，未读取原始计时样本或拟合价格，`beta` 保持 `null`。

| 条件 | A 首轮/后续填充窗口数 | B 首轮/后续填充窗口数 | 非零首轮权重 CTA 数 |
|---|---:|---:|---:|
| cfg_a P1 | 32 / 12 | 0 / 0 | 32 |
| cfg_c P2 | 0 / 0 | 0 / 256 | 0 |
| G6，cfg_a/b/c 各自 | 0 / 0 | 0 / 0 | 0 |

窗口计数包含静态 cluster 的全部 rank，不用目标 trace 选择 CTA。P2 冻结 r2 的完整时间预测为 188.003031 µs，观测为 199.472003 µs，误差为 −5.749665%；该项无法改变此误差。P1 未定价，不能声称其改善。G6 全零也意味着该项无法修复组合中时钟分支的 G6 失败。

固定比较保留 30 条预列条件，包括 29 条可评分条件、其中 7 条新范围外的数值诊断，以及原有 cfg_c G2 拒绝。冻结 r2 的 29 条完整时间绝对误差中位为 4.114183%，最大为 35.461907%。已失败的 cluster-max 重拟合报告只读保留；新形式的非目标采用检查未执行，也不记为通过。只有 P1 的新形式数值依赖未定价系数，其余可评分条件因修正向量为零而沿用基线数值。

R18 738707 的 cfg_a 仅有 M832×N4096，738865 的 cfg_c 仅有 M1536×N2624，均覆盖 K1024/4096，来源为 romeo-a057 / GPU-43269fbc。对新组件按 M×N 形状留一后，每个配置均没有剩余定价形状，诊断为 `not_estimable`。不从 V09 或另一配置借价；这不是完整模型的留一验证，也不是已拟合系数的不可辨识结论。

交付目录：

`/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/Q15-full-time/long-k-mainline/20261011-V10-taskA/C-boundary-v1`

- `interface-contract.json`：拟合前声明。
- `model-snapshot/`、`model-snapshot.json`：6 个公共模块的只读副本和 SHA。
- `cluster_first_L_extra.py`：纯静态权重与未定价拒绝接口。
- `analyze_r18_v10_cluster_residual.py`：独立 CPU 重放入口。
- `report/static-scope.json`：全部 30 条的逐 CTA 首轮权重与 A/B 填充计数。
- `report/comparison.json`：固定 29+1 表、7 条范围外数值、旧 cluster-max 比较。
- `report/structural-proof.json`：P2 参数无关反证及完整 JSON 相等检查。
- `report/cluster-residual.json`：结构性淘汰、未定价参数、无冻结支持。
- `report/leave-MN-shape.json`：两项新组件留形状诊断。
- `provenance.json`、`delivery-manifest.json`：输入和交付哈希。

重放时输出目录须尚不存在：

```sh
python3 -B /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/Q15-full-time/long-k-mainline/20261011-V10-taskA/C-boundary-v1/analyze_r18_v10_cluster_residual.py \
  --bundle /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/Q15-full-time/long-k-mainline/20261011-V10-taskA/C-boundary-v1 \
  --output /tmp/C-boundary-taskA-replay-new
```

已完成独立目录重放，6 个生成 JSON 逐字节一致；P2 的完整预测、频率和全部 CTA 事件在零修正下完全相同；未定价接口明确拒绝 `null`，P1 静态缩放另以审计系数检查。GPU 作业数为 0。公共模型、runner、PLAN/RULES/EXECUTION、原始与冻结结果均未由 C 修改。旧持续目标仍为 `paused`；本次有界任务以不可行结论结束。
