# EXP-04：memory duplex surface

## 1. 研究问题

同时发生 read 与 write 时，GPU-wide memory path 的 total service 如何随 issued-byte read:write ratio 变化；独立单向 peak 能否被 ratio-qualified joint point 替代？

## 2. 对应模型参数

- hot-L2：`l2.duplex`；
- cold path：`hbm.duplex.proxy`；
- `applicable_read_write_ratios`；
- `external_write_bytes_proven`。

## 3. Case matrix

runner 从当前 workload/precision/schedule manifest 推导 ratio：

- cold ratio：7；
- hot-L2 ratio：14；
- 共 21 case；
- 20 SM；
- 4 blocks/SM；
- 256 threads/CTA；
- 每个 logical read/write op 对应 128 B；
- read/write groups 在同一 kernel 中交错；
- 每 case 10 external trials。

不可约 96:1 ratio 由 binary 的 128 operation-group 上限直接支持，不能近似或删除。

## 4. 工作量与速率

定义：

\[
Q_R=S\,B\,T\,I\,O_R\,128,
\qquad
Q_W=S\,B\,T\,I\,O_W\,128,
\]

其中 \(S\) 为 SM 数，\(B\) 为 blocks/SM，\(T\) 为 threads/CTA，\(I\) 为 iterations，\(O_R,O_W\) 为 read/write operation groups。

\[
\widehat C_{\mathrm{duplex}}
=\frac{Q_R+Q_W}{T_{\mathrm{globaltimer}}}.
\]

capacity 只适用于 exact reduced ratio \(Q_R:Q_W\)。

## 5. NCU 门禁

所有 case 要求：

- `lts__t_sectors_op_read.sum` 覆盖至少 90% requested read bytes；
- `lts__t_sectors_op_write.sum` 覆盖至少 90% requested write bytes；
- function-scoped `LDG.E.128` 与 `STG.E.128`；
- base-unit raw CSV、`.ncu-rep` 和 independent parser。

hot-L2 额外要求 read hits > misses。

cold case 在 Thor 上没有 direct external write-byte counter，因此只要求：

- L2 read lookup miss sectors 证明至少 60% requested read bytes 到达外部；
- write sectors 只证明 store 进入 L2 write path；
- 固定记录 `external_write_bytes_proven=false`。

## 6. 当前结果

Thor `-i`：21/21 case、每 case 10 trial、NCU 完整、auditor pass。

当前 importer 输出：

- 14 个 `l2.duplex`；
- 7 个 `hbm.duplex.proxy`；
- 0 个 physical `hbm.duplex`。

当前 required L2 ratios 和 cold proxy ratios 均覆盖；physical HBM duplex 仍缺。

## 7. 进入模型

hot-L2 capacity 以 total issued bytes/s 进入经验资源层，并绑定 exact read:write ratio。

cold capacity 只进入 proxy coverage 和诊断，资源 ID 不等于 `hbm.duplex`，所以不能满足 physical external memory empirical demand。

## 8. 不能证明什么

- cold proxy 不证明 external write physical bytes；
- measured duplex point 不是 strict joint outer bound；
- L2 joint service 不证明 per-SM TMA ingress；
- ratio surface 不证明完整 TMA/MMA/readback/store causal pipeline；
- 21/21 campaign complete 不等于最终 GEMM model complete。

## 9. 源码与工件

- CUDA source：[memory_path_bandwidth.cu](../../../../microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu)
- runner：[run_memory_duplex_campaign.py](../../../../microbench/sm110_memory_duplex_campaign/run_memory_duplex_campaign.py)
- auditor：[audit_campaign.py](../../../../microbench/sm110_memory_duplex_campaign/audit_campaign.py)
- contract/readme：[README.md](../../../../microbench/sm110_memory_duplex_campaign/README.md)
- importer：[evidence_import.py](../../../../scripts/sm110_gemm_model/evidence_import.py)
