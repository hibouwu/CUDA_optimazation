# 共同输入

更新：2026-10-15。规格基线、半经验模型、机制模型与实测使用同一份问题描述，各自只读取所需字段。本页定义 GH200 CUTLASS GEMM case 的字段含义、单位、缺省与推导、来源和读取方。默认值和可离线判断的值域与组合约束来自 [r18 probe](../../../../../microbench/gh200_resource_campaign/access_rules/probes/r18.cu) 及表中对应的实验构建，解析由 [case_inputs.py](../../../../../microbench/gh200_resource_campaign/access_rules/case_inputs.py) 实现；依赖实验构建或设备的约束（非零干预需要对应构建、`check_output` 需要特定 mode 等）由 probe 在运行时检查，这里不离线判定。

## 解析规则

1. **只写差异。** case 行只写与缺省不同的字段；缺省或写 0 的推导项按下表推导。检查只要求每个读取方所需的有效输入能完整解析，不要求历史 case 显式写全，也不为满足检查补写不存在的字段。
2. **先核对同一程序，再以 setup 为准。** setup 事件回显的 `config`、`m`、`n`、`k`、行距、存储、填零、输入 map、swizzle、evict、`input_mode`、`seed`、`raster_order`、`b_offset_bytes`、`b_allocation_extra_bytes` 必须与 case 的有效值一致，`requested_sm_count` 对应 case 的 `sm_count`（setup 的 `sm_count` 是设备 SM 数），stage 须等于声明值或配置缺省值。一致后，程序的静态属性（grid、tile、cluster、epilogue tile、stage、线程、SMEM、驻留上限、调度 SM 数、设备身份）以 setup 为准；不一致即报错，不混合两个程序的属性。
3. **实验干预默认关闭。** 第 6 节的字段只在对应实验构建中合法；模型对非零值默认不支持，任务另行声明除外。
4. **注记不作输入。** 第 8 节的历史注记与第 7 节的登记字段只用于识别、分组和采样安排。

## 1. 计算问题

| 字段 | 含义 | 单位 | 缺省／推导与约束 | 读取方 |
|---|---|---|---|---|
| `m`、`n`、`k` | GEMM 形状 | 元素 | 必填，均为正 | 全部 |
| `lda`、`ldb`、`ldd` | A、B、D 的行距 | 元素 | k、n、n；须不小于逻辑列数 | 半经验（供给地址）、实测 |
| `storage_m`、`storage_n` | 物理存储的行数与列数 | 元素 | m、n；须覆盖逻辑矩阵，`ldb`、`ldd` 不小于 `storage_n` | 半经验、实测 |
| `input_map_m`、`input_map_n` | 输入 tensor map 覆盖的行与列 | 元素 | m、n；介于逻辑尺寸与存储尺寸之间 | 半经验（填零与供给）、实测 |
| `zero_m`、`zero_n` | 从该行／列起输入置零 | 元素 | −1 表示不置零 | 半经验（时钟的零乘积比例）、实测 |
| `alloc_lda`、`alloc_ldb` | 分配行距 | 元素 | `lda`、`ldb`；不小于物理行距 | 实测 |
| `b_offset_bytes`、`b_allocation_extra_bytes` | B 基址偏移与额外分配 | 字节 | 0、0；16 B 对齐，偏移不超过额外分配 | 半经验（供给）、实测 |

普通构建的精度、布局与 α/β 固定为：FP16 输入、FP32 累加与输出，α=1、β=0，A 行主序（K-major），B 行主序（N-major）；计算定义见[问题定义](../../problem.md)。

## 2. 程序

| 字段 | 含义 | 缺省／推导与约束 | 读取方 |
|---|---|---|---|
| `config` | probe 配置：cfg_a、cfg_b、cfg_c、cfg_a1、cfg_c1，对应编译宏 V06_CFG=0–4 | 必填 | 半经验、机制、实测 |
| `stages` | 构建中分配的主循环 stage 数 | cfg_a、cfg_a1、cfg_b 为 6，cfg_c、cfg_c1 为 4；显式值对应 `R18_STAGE_COUNT` 构建，须与 setup 一致 | 半经验、机制、实测 |
| `swizzle` | 调度器 max_swizzle_size | 1；只允许 1、8 | 半经验、机制 |
| `raster_order` | 调度的光栅方向 | heuristic；另有 along_m、along_n | 半经验、机制 |
| `sm_count` | 请求的调度 SM 数 | 0 表示全部；模型读取 setup 的 `scheduler_sm_count` | 实测 |
| setup：`grid`、`tile`、`cluster`、`epilogue_tile`、`threads`、`smem`、`smem_epilogue`、`max_active_ctas_per_sm`、`scheduler_sm_count`、`gpu_uuid` | `--mode setup` 的构建查询结果，解析后传给读取方 | 以 setup 为准 | 半经验、机制 |
| setup：`requested_sm_count` | setup 回显的请求 SM 数，用于核对 case 的 `sm_count` | 须与 case 一致 | 实测 |
| setup：`active_stages`、`wgmma_repeats`、`epilogue_alpha` | 同一 S6 二进制的活动槽数、同一 commit 内的 WGMMA 遍数及输出缩放 | [746824](../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-compute-credit-discrimination-003/README.md)回显活动 4／6 槽、1／3 遍及 α=1／FP32(1/3)；普通构建缺省为 `stages`、1、1 | 机制程序图、请求级组合、实测 |
| setup：`fold_a_source`、`store_target_ctas` | 正式调用的 A 源折叠开关与执行 global 输出 TMA 的 CTA 阈值 | 缺省为 0、全部 CTA；[等值源实验](../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-reuse/20261022-Q14-equal-value-source-004/README.md)的 G3 S6 dyadic 分支将 A 的 m17–21 改为 m0–4，输出阈值为1。本地STS与提交／等待保留 | 机制程序图、请求级组合、实测 |
| setup：`private_source`、`allocated_a_elements`、`allocated_b_elements` | 正式调用的逐CTA等值A/B副本开关，以及每份完整分配的元素数 | 开关缺省0；每CTA源偏移为CTA编号×每份元素数×2 B，未回显元素数时按存储行数与分配行距推导；[实际副本实验](../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/gemm-source-workset/20261023-Q14-private-source-001/README.md) | 机制程序图、请求级组合、实测 |
| 构建身份 | 二进制与 SASS 哈希、编译命令、CUDA 与 CUTLASS 版本 | 构建记录：`binary_hashes.json`、`sass_hashes.json`、`commands.json` | 实测 |
| 变体 | plain、wide、stamped、dual 的编译选项与观察端点 | 由任务声明 | 实测 |

## 3. 输入数据

| 字段 | 含义 | 缺省与约束 | 读取方 |
|---|---|---|---|
| `input_mode` | 输入数值的生成方式：dyadic、zero、random | dyadic | 半经验（时钟）、C1 模型、实测 |
| `seed` | 生成种子 | 17；不小于 0 | 实测 |

调用前可知的输入统计描述就是 `input_mode` 与 `seed`；运行前预测不读取实测的数值统计。

## 4. 硬件

| 字段 | 含义 | 来源 | 读取方 |
|---|---|---|---|
| `gpu_uuid` | 设备身份 | setup 事件、`environment.json` | 全部 |
| 规格配置名 | 例如 `gh200_product`、`c1_reference` | [规格配置](../../hardware/gh200_sm90/spec_baseline_profiles.json) | 规格基线 |
| 资源参数 | 计算、存储与搬运资源 | [hardware/gh200_sm90](../../hardware/gh200_sm90/README.md) | 全部 |

## 5. 运行协议

| 字段 | 含义 | 缺省 |
|---|---|---|
| `mode` | 运行模式；`warm` 为测量，`setup` 为构建查询 | warm |
| `warmup_policy` | adaptive：最多 30 次预热，至少 8 次后最后 5 次的 CV ≤2% 即停，否则报错；common-sw1-16：固定 16 次 swizzle 1 预热；fixed-count：按声明次数预热 | adaptive |
| `warmup_fixed_calls` | fixed-count实验构建的预热调用次数，单位为次；当前30–33；[构建与协议](../../../../../results/gh200_resource_campaign/access_rules/Q01-scheduling/work-placement/20261015-d226-fixed-history-discrimination-001/README.md)读取 | fixed-count为30 |
| `history_mode`、`history_source`、`mirror_target_slot` | 调用历史观察（R09 用） | off、original、0 |
| setup：`warmup_fold_a_source`、`warmup_store_ctas` | 源折叠实验前缀调用使用的程序分支，与正式调用共用源状态 | 0、全部 CTA；调用数由组合协议 `calls` 给出，最后一次为正式调用 |
| setup：`warmup_private_source` | 前缀调用使用原源或逐CTA副本 | 缺省与正式`private_source`相同；最后一次为正式调用 |
| `evict` | 每次调用前在计时窗口外运行驱逐 kernel | 0 |
| 计时窗口 | cudaEvent 只包住一次 `Gemm::run`；驱逐、trace 清零与同步在窗口外；预热后测一次 | probe 固定 |
| 进程与重复 | 每个 case 的 plain、dual 进程数与顺序随机化 | 任务声明，例如 V09 各 10 个进程，C1-01见[当前采样协议](../../experiments/gh200_sm90/Q16-config-selection/C1-01-selection.md#恢复前要定的事) |

## 6. 实验干预

`work_swap55_91`、`producer_delay_ns`（0 或 512）、`first_supply_events`、`ready_observer`、`epilogue_defer`（0–2）、`second_prefill`、`a_row_rotation`、`release_warp_observer`、`check_output`、`plain_prefix_module`：默认关闭，非零值只在对应实验构建中合法。

## 7. 任务登记与预测要求

| 字段 | 含义 |
|---|---|
| `id`、`set`、`kind`、`group`、`cond`、`candidate`、`original_experiment`、`original_reference`、`purpose` | case 身份、开发／留出集合、类别、选配置分组、C1 候选名与来源说明 |
| 预测要求 | 由任务冻结：输出理想时间、plain 完整时间、dual 包络、有效频率或事件；C1 候选集合；支持域与拒绝规则 |

## 8. 历史注记

`tile`、`cluster`、`schedule`、`grid_ctas`、`ktiles`、`scheduled_units`、`scheduler_status`、`original_m` 是 V05 等早期 case 预先写入的静态信息，取值以构建查询为准；`label`、`pitch`、`input_path`、`ready_generation_bound`、`sampling_order`、`variants`、`repeats`（每variant的计划进程数）是分组标签与采样安排，属于任务协议。

## 9. 读取方

“记录可解析”与“读取方输入齐备”分开判断：case 字段可由缺省补齐，setup 字段只认构建查询给出的值。没有 setup 的历史记录可以保留、不必补写，但不算 V09 或程序图可运行。

| 读取方 | 需要的有效输入 |
|---|---|
| 规格基线 | `m`、`n`、`k` 与规格配置名 |
| 半经验 V09 | `config`、`m`、`n`、`k`、`lda`、`ldb`、`swizzle`、`raster_order`、`input_mode`、`zero_m`、`zero_n`、`input_map_m`、`input_map_n`、`storage_n`、`b_offset_bytes`、`stages`，以及 setup 的 `grid`、`stages`；卡身份 `gpu_uuid` 优先取 setup，缺失时取同一 run 目录下 `environment.json` 的 GPU UUID，两处都没有则输入不齐备 |
| C1-01 模型 | `m`、`n`、`k`、`input_mode`、`config`、`stages` |
| 机制程序图 | `config`、`m`、`n`、`k`、`swizzle`、`raster_order`、`stages`，以及 setup 的 `grid`、`stages`；同版计算量干预另读 setup 的 `active_stages`、`wgmma_repeats`、`epilogue_alpha` |
| 实测 | 全部字段与身份 |

## 检查

[文档检查](../../../../../microbench/gh200_resource_campaign/access_rules/check_model_docs.py)核对三件事：probe 的每个命令行参数和 `case_inputs.py` 的每个字段都在本页登记；受检清单内的 GEMM case 全部可解析并满足上述约束；各读取方的输入是否齐备单独计数。受检清单为 `results/gh200_resource_campaign/access_rules/` 下的 `*/cases.json`、`*/*/cases.json`、`*/*-cases.json`、`*/*/*-cases.json`；文件损坏、混有非 GEMM 行或 `static_setup.json` 损坏时报错，其他格式的 JSON 按非目标格式排除并计数。

2026-10-15 的本地结果：148 个 GEMM case 文件、14236 行（含 C1 设计清单及其上传包副本），其中 943 行带 setup；另有 96 个非目标格式文件被排除。记录全部可解析；规格基线与 C1 模型的输入在全部 14236 行齐备，V09 的输入在 943 行齐备，程序图在 943 行齐备。
