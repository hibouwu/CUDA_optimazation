# 旧冻结依赖的路径兼容

以下31份文件是旧合同和review绑定的只读兼容副本，来自相邻gh200_sm90_archive，逐字节相同，共约169KiB。旧核验器拒绝symlink，因此使用真实文件；不另维护正文，不修改已冻结的路径或SHA。当前实验结论仍由本目录的EXP/R页面维护，本清单不激活旧任务或授予当前代码的GPU资格。

此次恢复使旧CPU测试能定位历史设计、协议和依赖；两个旧S02/S03整合review仍不为后来变更的core背书。S12/S13快照测试在各自临时repo中显式使用CPU fixture接线，并新增实际旧gate拒绝变更代码的检查；真正结果根的reviews、contracts、raw和旧门槛不改。

| 旧路径文件 | 只读来源 |
|---|---|
| `INTERFACE.md` | [归档原件](../gh200_sm90_archive/INTERFACE.md) |
| `INTERFACE-PREFLIGHT.md` | [归档原件](../gh200_sm90_archive/INTERFACE-PREFLIGHT.md) |
| `EXP-04-memory-baseline-v2.md` | [归档原件](../gh200_sm90_archive/EXP-04-memory-baseline-v2.md) |
| `INTERFACE-CALIBRATION.md` | [归档原件](../gh200_sm90_archive/INTERFACE-CALIBRATION.md) |
| `EXP-03-compute-baseline.md` | [归档原件](../gh200_sm90_archive/EXP-03-compute-baseline.md) |
| `EXP-05-legacy-fma.md` | [归档原件](../gh200_sm90_archive/EXP-05-legacy-fma.md) |
| `EXP-06-legacy-mma.md` | [归档原件](../gh200_sm90_archive/EXP-06-legacy-mma.md) |
| `EXP-07-legacy-wgmma.md` | [归档原件](../gh200_sm90_archive/EXP-07-legacy-wgmma.md) |
| `EXP-09-shared-memory.md` | [归档原件](../gh200_sm90_archive/EXP-09-shared-memory.md) |
| `EXP-08-low-precision.md` | [归档原件](../gh200_sm90_archive/EXP-08-low-precision.md) |
| `EXP-10-matrix-exchange.md` | [归档原件](../gh200_sm90_archive/EXP-10-matrix-exchange.md) |
| `INTERFACE-TIMER-RESOLUTION.md` | [归档原件](../gh200_sm90_archive/INTERFACE-TIMER-RESOLUTION.md) |
| `EXP-11-synchronization.md` | [归档原件](../gh200_sm90_archive/EXP-11-synchronization.md) |
| `EXP-08-lowering-amendment.md` | [归档原件](../gh200_sm90_archive/EXP-08-lowering-amendment.md) |
| `EXP-12-async-copy.md` | [归档原件](../gh200_sm90_archive/EXP-12-async-copy.md) |
| `EXP-11-lowering-amendment.md` | [归档原件](../gh200_sm90_archive/EXP-11-lowering-amendment.md) |
| `EXP-13-global-duplex.md` | [归档原件](../gh200_sm90_archive/EXP-13-global-duplex.md) |
| `EXP-14-tma-bulk.md` | [归档原件](../gh200_sm90_archive/EXP-14-tma-bulk.md) |
| `INTERFACE-FORMAL-B3.md` | [归档原件](../gh200_sm90_archive/INTERFACE-FORMAL-B3.md) |
| `INTERFACE-S12-EARLY-VALIDATION.md` | [归档原件](../gh200_sm90_archive/INTERFACE-S12-EARLY-VALIDATION.md) |
| `EXP-15-descriptor-feasibility.md` | [归档原件](../gh200_sm90_archive/EXP-15-descriptor-feasibility.md) |
| `EXP-15-tma-tensor-2d.md` | [归档原件](../gh200_sm90_archive/EXP-15-tma-tensor-2d.md) |
| `EXP-08-bounded-fp8-input-amendment.md` | [归档原件](../gh200_sm90_archive/EXP-08-bounded-fp8-input-amendment.md) |
| `INTERFACE-S13-FORMAL-B3.md` | [归档原件](../gh200_sm90_archive/INTERFACE-S13-FORMAL-B3.md) |
| `INTERFACE-S11-EARLY-VALIDATION.draft.md` | [归档原件](../gh200_sm90_archive/INTERFACE-S11-EARLY-VALIDATION.draft.md) |
| `EXP-16-tma-stage-request.draft.md` | [归档原件](../gh200_sm90_archive/EXP-16-tma-stage-request.draft.md) |
| `INTERFACE-S14-FORMAL-B3.draft.md` | [归档原件](../gh200_sm90_archive/INTERFACE-S14-FORMAL-B3.draft.md) |
| `INTERFACE-S15-FRESH-QUOTA.draft.md` | [归档原件](../gh200_sm90_archive/INTERFACE-S15-FRESH-QUOTA.draft.md) |
| `EXP-17-cluster-dsm-multicast.draft.md` | [归档原件](../gh200_sm90_archive/EXP-17-cluster-dsm-multicast.draft.md) |
| `EXP-18-auxiliary-resources.draft.md` | [归档原件](../gh200_sm90_archive/EXP-18-auxiliary-resources.draft.md) |
| `EXP-01-compute-audit.md` | [归档原件](../gh200_sm90_archive/EXP-01-compute-audit.md) |
