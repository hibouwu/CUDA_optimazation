本目录只分解 740067 已保存的 correctness 记录，不估计 input map 的因果效应或性能。作业为 a057 上单 GPU、4 CPU、节点非排他分配；实际 GPU 为 `GPU-43269fbc-449d-3e0f-908a-9c81229546d3`，不是后续控制使用的 ef 参考卡。两个 map 各只有一个 on 调用，没有同卡合格观察器控制。此前 observer 失败和全部资格字段保持。

两次 on 调用的 12 CTA×2 consumer 角色全部保留。按 FIRST、commit g0/g16/g32/g63、mma-return、MAIN 七个源码切点手工读取原生槽位，得到 48 个角色记录、288 行六段分解，以及 144 行 `map960记录时长−map1024记录时长` 的有符号算术差。每个角色分别核对 cycles/ns 的六段和等于 L；每个对应角色的六段差也等于 L 差。没有跨调用或跨 SM 相减原始 clock64，没有将负分段差截零。

| 核对项 | 结果 |
|---|---:|
| 原有 verification 中的完整原始角色与六段结果 | 48/48 相等 |
| 调用内分段闭合，cycles/ns 分别检查 | 96/96 |
| 两调用有符号差闭合，cycles/ns 分别检查 | 48/48 |
| 对应角色 SM 改变 | 0/24 |
| 四个 warm 记录独立数值检查 | 16384/16384 |
| 各记录显式零存储检查 | 262144 个值全为零 |

原有两个完整 D 数组各 2703360 个值的独立精确参考证据直接复用，并重新核对数组 SHA。原始数据还包含 `input_zero_storage` 包，分析保留并检查该包；没有将它丢弃后强行套用其它协议的三包布局。所有新计算输出在 `data`，旧归档、verification、控制结果与冻结均未覆写。

`call-identities.json` 保存四个 warm 调用的原始 SHA、GPU、binary、目标和 prefix 地址、完整 16 次前序 event 记录及准备参数。每个调用内部 prefix 与 target 使用相同 A/B 地址，observer 在 prefix 关闭、target 按 flag 设置。源码 `warmup_launch_params.mainloop = launch_params.mainloop` 表明前序 mainloop 参数沿用该调用的目标 input map：map960 与 map1024 各有自己的前序执行，不能称两次调用具有相同历史状态。两进程的分配地址也不同。没有证据把记录差唯一归于目标 map、背景执行、前序状态或采样。

例如 CTA14/role2 的记录 L 差为 8066 cycles、4928ns；其六段 cycle 差依次为 `0, 2153, 2600, 3300, 9, 4`，ns 差为 `0, 1536, 1632, 1760, 0, 0`。这是这两个记录的精确算术分解，没有重复样本或合格观察条件支撑其作为早/晚供给效应、延迟价格或性能估计。全部角色都保留，未按差值大小或最后完成者选点。

| 源码窗口 | 仍混合的对象 |
|---|---|
| commit g0→g16、g16→g32、g32→g63 | 后续 full 等待、GMMA 发射/commit、组等待及 release；不能区分输入供给与跨 rank empty 回压 |
| commit g63→mma-return | 最后一次 wait1、release62、退出及返回路径 |
| mma-return→MAIN | wait0、release63 及端点路径；不能直接命名为裸 GMMA 排空成本 |

FIRST 已在首次 consumer_wait 之后，所以没有第一次 full 的就绪时刻。这些记录也没有四个实际 empty 发送线程的发出前后界、producer acquire/issue、目标 full-ready、producer_tail 或 D_visible 的新包围记录。五切点确实提供了可核验的源码分段，但仍不能识别哪一条等待链造成差异，或据此预测输出退出和最后完成者。独立 SR 读也没有形成 GT_pre→clock→GT_post 证书。

```sh
python3 analyze.py \
  --run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-R18-first-output-correctness \
  --output NEW_C_RECORD_PARTITION_DIRECTORY
```

`report.json` 保存 37 项输入 SHA、分配和源码/历史边界；`records.json.gz` 保存完整 on 端点，`record-partitions.csv` 和 `signed-record-differences.csv` 保存逐角色分解。所有观察、采样包围、物理 ns、map 因果效应、性能估计、服务价格、局部留出、关键 CTA 与完整 GEMM 资格为 false。本轮没有新增 GPU 运行、公共代码改动、价格或候选。
