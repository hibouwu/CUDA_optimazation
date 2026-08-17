# SM110 GEMM 性能参数 runner 对抗式审查

## 1. 结论

现有 compute、18-case component 和 full-GEMM runner 足以复现 2026-08-14
报告中的既有数字，但**不足以证明所有影响经验理想包络的性能参数均已测量**。
历史 `all_common_resources_closed=true` 只检查十个独立 resource ID 是否各有一个
`closure_qualified` 数字；它没有检查独立 read/write 是否可同时达到、TMA rate
对 payload 的敏感性，也没有要求 component 的 L2/HBM residency 由 profiler
counter 证明。

本次增加两个可执行且可独立审计的 campaign：

1. `sm110_tma_payload_campaign` 测量 4、8、16、32、64 KiB 单笔 TMA payload
   在 `hot_l2` 与 `cold_hbm` 两种入口状态下的服务率，每点十次外部 trial，并
   强制 NCU 证明 TMA bytes、L2 bytes 与 hit/miss residency。
2. `sm110_memory_duplex_campaign` 在同一个 kernel 内同时发射 128-bit global
   load 与 128-bit global store，测量精确 read:write 比例下的联合总 B/s；
   cold case 用 L2 read-miss sector 定量证明 DRAM reads，并用 L2 write sector
   证明 write-path issue；hot-L2 强制检查 L2 hit/miss。Thor 不提供直接 external
   write-byte counter，因此 cold case 明确保持 proxy qualification。

这两个 campaign 修补此前最危险的两个“把假设当参数”漏洞。cold duplex 仍不
关闭 physical external write-byte 证据。Thor 数据回传前，
只能声明 `payload_duplex_runner_definition_complete=true`，不能声明新参数已经测量
闭环，更不能声明 `all_performance_parameter_runner_definition_complete=true`。

机器审计当前给出的精确计数是：

- compute runner：12/12 精度、每精度 3 个 full-SM MMA atom，runner 合同完整；
- L2/HBM duplex：21 个由 manifest 和 `N=1024/2048/4096` 推导的唯一 ratio，runner 合同完整；
- serialized TMA payload：5 个 payload × 2 个 residency，共 10 点，runner 合同完整；
- 精确 TMA schedule/precision：2/28 对，只有 `tc5a` 的 FP16/BF16 合同精确；
- 独立 joint-pipeline runner：0；完整 GEMM 只能做反证，不能倒灌成独立容量；
- 完整 GEMM：5/12 精度有可执行且同合同的 candidate/reference；
- 严格 compute upper：12 精度中 7 个有当前 source，缺 TF32、两种 FP6、raw
  E2M1 和 MXFP4；microbenchmark 不能补成物理上界。

因此本次确实补上了两个可以不猜硬件语义而定义的 surface；剩余缺口不能靠增加
一个接受任意参数的壳 runner 伪装闭合。特别是其它 schedule 的 tensor-map
维度、swizzle、A/B/scale request 数、controller warp 和 barrier topology 尚未进入
schedule manifest；在这些字段冻结前，所谓“exact runner”没有可审计的 exact
对象。

## 2. L2 参数在模型中的物理含义

定义 \(C_{L2,r}^{ub}=1024\ \mathrm{B/cycle/GPU}\) 为整 GPU 共享 L2 read 总线
的条件上界；定义 \(C_{L2,w}^{ub}=512\ \mathrm{B/cycle/GPU}\) 为整 GPU 共享
L2 write 总线的条件上界。二者都是 per-GPU 参数，不是 per-SM 参数。在锁定频率
\(f=1.575\ \mathrm{GHz}\) 下，二者分别换算为 1.6128 TB/s 和 806.4 GB/s。

定义 \(C_{TMA,SM}(p,d,s)\) 为单个 SM 的 TMA→SMEM sustained ingress，单位
B/s/SM；其中 \(p\) 是单笔 payload，单位 B/request；\(d\) 是 destination/inflight
合同，单位 request；\(s\) 是入口 residency。这个量属于每个 SM 的独立出口，
不能把整卡 TMA 吞吐除以 20 得到，也不能用共享 L2 的 1024 B/cycle/GPU 替代。

因此当前模型实际涉及三类不同的 L2 相关容量：

| 参数 | 作用域 | 证据等级 | runner 的作用 |
| --- | --- | --- | --- |
| \(C_{L2,r}^{ub}\) | 整 GPU 共享 | `profiler_model_peak` 条件上界 | runner 只做反证与 sustained 校准，不负责定义该上界 |
| \(C_{L2,w}^{ub}\) | 整 GPU 共享 | `profiler_model_peak` 条件上界 | runner 只做反证与 sustained 校准，不负责定义该上界 |
| \(C_{TMA,SM}(p,d,s)\) | 每 SM 独立 | `measured_sustained` | 必须按 payload、并发与 residency 测量 |

历史 component suite 还测了整卡 `l2.read` 与 `l2.write` sustained rate。它们
不能覆盖上述两个硬件上界，也不能证明 read 与 write 同时发生时互不争用。

## 3. 参数—runner 覆盖矩阵

定义“精确覆盖”为 timed 指令流、payload/ratio、线程与 CTA 合同、作用域、入口
residency 和计时边界均匹配；只匹配一部分称为“邻近点”。

| 性能参数或约束 | 原 runner | 对抗式判定 | 本次处理 |
| --- | --- | --- | --- |
| 每精度、每 MMA M/N atom 的 compute sustained rate | compute campaign，12 精度 × 3 atom | 精确覆盖 | 保留 |
| 整卡 L2 read sustained rate | 16 MiB read-only component | 单向精确；不证明 duplex | 新增 L2 duplex ratio surface |
| 整卡 L2 write sustained rate | 16 MiB write-only component | 单向精确；不证明 duplex | 新增 L2 duplex ratio surface |
| 整卡 DRAM read/write sustained rate | 两个独立 256 MiB case | 单向精确；两者不可相加 | 新增 DRAM-read + L2-write-path proxy ratio surface；外部 write bytes 仍缺 |
| 单 SM TMA ingress | 32 KiB uniform 与 tc5a 16+32 KiB | tc5a 精确，其余 schedule 只有邻近点 | 新增 4–64 KiB payload/residency surface；联合 issue topology 仍需 exact full-kernel 反证 |
| TMA residency | working-set 大小与 warmup | 不充分 | 新 TMA campaign 强制 NCU hit/miss 与 TMA bytes |
| TMEM scale ingress | `32x128b.warpx4` | 对当前 block-scale transport atom 精确 | 保留 |
| TMEM accumulator readback | x8/x16 × 1/4 warps | 对当前 schedule manifest 精确 | 保留 |
| TMA+MMA+readback+store 联合可达性 | full-GEMM candidate + NCU holdout | 能反证完整流水，但不是独立硬件上界 | 保留在第三层，禁止倒灌为条件上界 |
| kernel/grid 固定成本 | `fixed_seconds=0` | 未测 | 经验包络必须标注“零固定成本理想假设”；预测小 GEMM wall time 前需 fixed-cost campaign |
| 所有 12 精度完整 GEMM与同精度 denominator | full-GEMM 仅 5 精度 | 未覆盖 | 继续保持 `all_precisions_closed=false` |

TMA payload sweep 建立的是单路径服务曲线，不能自动把任意 schedule 的多请求 issue
topology 宣称为精确 capacity。`tc5a` 的四 stage/八请求已有精确 case；其它 schedule
若要取得同等级资格，必须增加由 manifest 派生的 exact joint case，或者只让相应
完整 GEMM NCU holdout 作为第三层反证。禁止把 payload sweep 中最快的点无条件
套给所有 schedule。

## 4. 新 duplex 比例矩阵

定义 (r:q) 为 timed kernel 请求的 read bytes 与 write bytes 的最简整数比。
定义 (C_{mem}^{joint}(r,q,s)) 为入口状态 (s) 下，同核同时读写的总服务率，
单位 B/s。它不能由 (C_r+C_w)、\(\max(C_r,C_w)\) 或两个独立 trial 推导。

HBM/LPDDR 冷入口覆盖 `1:1`、`2:1`、`1:2`、`3:8`、`1:4`、`17:64`、
`9:32`。这些比例来自当前十二种 accumulator-output 精度的逻辑唯一输入/输出比。

L2 热入口覆盖 `27:16`、`27:8`、`27:4`、`3:1`、`4:1`、`6:1`、`8:1`、
`12:1`、`16:1`、`24:1`、`32:1`、`48:1`、`64:1`、`96:1`。这些比例不是
手写常量，而是从当前 workload、schedule、precision manifest 在冻结的
`N=1024/2048/4096` shape 上，将重复 TMA request bytes 与有效 accumulator store
bytes 机械化简所得；三个分数依次对应 block-scaled value 加 SFA/SFB transport。

`96:1` 是不可约比例。CUDA binary、run spec、case manifest、runtime output 和
independent auditor 共同冻结 `max_operation_groups=128`，覆盖当前最大需求 96；
不允许删除该点或把它近似成 `64:1`。device loop 是 runtime loop，固定复用 8 个
`uint4` load 暂存值，所以上限从 64 调到 128 不引入 96 组静态寄存器展开。

每个 case 保存 `requested_read_bytes`、`requested_write_bytes`、总 requested bytes、
最早 CTA start、最晚 CTA stop、20-SM coverage、十次 trial、源码/binary/SASS hash
和 NCU 原始报告。独立 auditor 从 trial 字段重新计算比例和 B/s。

## 5. microbenchmark 不能证明什么

定义 (P^\star) 为所有语义等价经典 GEMM 中真实但未知的最好性能；定义
(P_{ub}) 为只使用物理 rate upper 得到的条件上界；定义
\(\widehat P_{env}\) 为使用 measured sustained 参数得到的经验理想包络；三者
单位与 workload 相同，浮点为 FLOP/s，整数为 OP/s。

microbenchmark 的最高实测值只能证明硬件至少做到过这么快，不能证明硬件绝不
可能更快。因此：

- 1024/512 B/cycle/GPU 可作为已知 L2 条件上界；
- TMA、TMEM、compute microbenchmark 的最高 sustained rate 只能进入
  \(\widehat P_{env}\)，不能冒充 (P_{ub})；
- duplex runner 用来修正“独立方向可完美重叠”的经验假设，也不是物理 rate upper；
- 完整 GEMM 超过 \(\widehat P_{env}\) 表示经验参数或适用合同需要重校准；只有
  语义相同的完整 GEMM 超过 (P_{ub}) 才构成上界反证。

## 6. Thor 必跑项

本次修改了 GPU-facing CUDA source、case matrix 和 NCU 证据合同，因此需要 Thor
新跑；旧结果不能重新命名后导入。冻结代码提交后执行：

```bash
RUN_ID=thor-t5000-parameter-supplement-maxn-YYYYMMDD-a
EXPECTED_COMMIT=$(git rev-parse HEAD)

bash microbench/run_sm110_parameter_supplement.sh \
  "$RUN_ID" "$EXPECTED_COMMIT"
```

成功日志必须出现独立一行 `PARAMETER_SUPPLEMENT_COMPLETE`。结果提交应包含：

- `results/sm110_tma_payload_campaign/$RUN_ID-tma-payload`；
- `results/sm110_memory_duplex_campaign/$RUN_ID-memory-duplex`。

在两个独立 auditor 均通过、结果 commit 回传且 model importer 加入精确
ratio/payload applicability 之前，报告必须同时写：

- `payload_duplex_runner_definition_complete=true`；
- `physical_memory_duplex_closed=false`；
- `cold_external_write_bytes_closed=false`；
- `all_performance_parameter_runner_definition_complete=false`；
- `new_parameter_measurement_complete=false`；
- 历史 `all_common_resources_closed=true` 仅适用于旧的独立资源定义。

本地或 Thor 上可随时执行机器覆盖审计：

```bash
python3 scripts/sm110_gemm_model/runner_coverage.py

# 对“所有参数 runner 都已定义”的强声明执行 fail-closed gate；当前应返回 1。
python3 scripts/sm110_gemm_model/runner_coverage.py \
  --require-all-performance-parameters
```

## 7. microbenchmark 来源

| 测量 | runner/source | 独立 auditor |
| --- | --- | --- |
| TMA payload/residency surface | `microbench/sm110_tma_payload_campaign/run_tma_payload_campaign.py`；CUDA source：`microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu` | `microbench/sm110_tma_payload_campaign/audit_campaign.py` |
| hot-L2 duplex + cold-read/write-path proxy surface | `microbench/sm110_memory_duplex_campaign/run_memory_duplex_campaign.py`；CUDA source：`microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu` | `microbench/sm110_memory_duplex_campaign/audit_campaign.py` |
| compute surface | `microbench/sm110_gemm_campaign/run_compute_campaign.py` | `microbench/sm110_gemm_campaign/audit_campaign.py` |
| TMEM/TMA/component | `microbench/sm110_gemm_component_campaign/run_component_campaign.py` | `microbench/sm110_gemm_component_campaign/audit_campaign.py` |
| 完整流水与 holdout | `microbench/sm110_full_gemm_campaign/run_full_gemm_campaign.py` | `microbench/sm110_full_gemm_campaign/audit_campaign.py` |
| 参数—runner 覆盖声明 | `scripts/sm110_gemm_model/runner_coverage.py` | `scripts/sm110_gemm_model/test_runner_coverage.py` |
