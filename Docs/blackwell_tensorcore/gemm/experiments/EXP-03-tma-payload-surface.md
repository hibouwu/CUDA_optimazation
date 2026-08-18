# EXP-03：TMA payload/residency surface

## 1. 研究问题

serialized schedule-independent TMA request 在不同 payload 和 residency 下能提供多少 service；payload size 是否显著影响 per-SM L2-hit ingress 与 full-GPU cold path？

## 2. 对应模型参数

```text
tma.smem_ingress.per_sm.payload_<size>
tma.hbm.payload_<size>
```

这些容量用于 payload surface coverage 和缺口审计；它们不是 exact multi-request schedule topology capacity。

## 3. Case matrix

- payload：4、8、16、32、64 KiB；
- residency：hot-L2、cold-HBM；
- 共 10 case；
- destination slots：2；
- threads/CTA：128；
- resident CTA/SM：1；
- hot-L2：单 CTA、单 observed SM；
- cold-HBM：20 CTA、20 SM；
- 每 case 10 external trials。

## 4. 计时与工作量

定义 issued payload \(Q_{\mathrm{payload}}\) 为 blocks × iterations × payload bytes。rate 为：

\[
\widehat C_{\mathrm{TMA,payload}}
=\frac{Q_{\mathrm{payload}}}{T_{\mathrm{globaltimer}}}.
\]

hot-L2 rate 保持 B/s/SM；cold rate 保持 B/s/GPU。

## 5. NCU 与接受门禁

- exact base-unit raw CSV；
- kernel row 唯一匹配；
- hot-L2 read hit dominant；
- cold case 使用 L2 miss sector 证明外部 read；
- NCU report、raw CSV、stderr、summary 全部保留；
- function-scoped TMA SASS；
- trial rate 从 requested bytes/globaltimer 独立重算；
- source/binary/SASS/environment/hash/COMPLETE 通过 independent auditor。

## 6. 当前结果

Thor `-i` campaign：10/10 case、每 case 10 trial、NCU 完整、auditor pass。当前 importer 重放生成 10 个 closure-qualified capacity。

当前 five-point surface 覆盖普通 value request，但不包含 block-scaled 512 B 与 1 KiB scale request，因此：

```text
all_required_tma_payloads_measured = false
```

## 7. 进入模型

每条 capacity 绑定 payload、residency request source、20-SM hardware/mode、threads、destination slots、resident CTA 和 timed scope。

它们不能直接满足 EXP-05 的 exact topology resource；exact schedule 还需要 family、request count、stage、row stride 与 precision 完全匹配。

## 8. 不能证明什么

- serialized payload rate 不证明多 request/stage pipeline；
- 4/8/16/32/64 KiB 不覆盖 512 B/1 KiB scale request；
- cold TMA read path 不证明完整 GEMM output write joint service；
- per-SM rate 与 GPU-wide shared L2 rate不能互换；
- runner 10/10 完成不等于所有 schedule payload 完整。

## 9. 源码与工件

- CUDA source：[tma_gmem_smem_bandwidth.cu](../../../../microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu)
- runner：[run_tma_payload_campaign.py](../../../../microbench/sm110_tma_payload_campaign/run_tma_payload_campaign.py)
- auditor：[audit_campaign.py](../../../../microbench/sm110_tma_payload_campaign/audit_campaign.py)
- runbook：[README.md](../../../../microbench/sm110_tma_payload_campaign/README.md)
- importer：[evidence_import.py](../../../../scripts/sm110_gemm_model/evidence_import.py)
