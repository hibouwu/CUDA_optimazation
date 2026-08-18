# 审计与复现

## 1. 本地规范与模型回归

```bash
python3 -m scripts.sm110_gemm_model.schema_doc_audit
python3 scripts/sm110_gemm_model/runner_coverage.py \
  --require-all-performance-parameters
python3 -m unittest discover \
  -s scripts/sm110_gemm_model -p 'test_*.py'
```

`--require-all-performance-parameters` 当前预期非零退出，因为 exact/joint/physical-HBM/full-precision runner coverage 尚未完整。非零是正确的 gap gate，不是脚本故障。

## 2. Parameter supplement 审计与导入

```bash
RUN_ID=thor-t5000-parameter-plots-maxn-20260817-i
TMA_DIR="results/sm110_tma_payload_campaign/${RUN_ID}-tma-payload"
DUPLEX_DIR="results/sm110_memory_duplex_campaign/${RUN_ID}-memory-duplex"

python3 microbench/sm110_tma_payload_campaign/audit_campaign.py \
  "$TMA_DIR"
python3 microbench/sm110_memory_duplex_campaign/audit_campaign.py \
  "$DUPLEX_DIR"

python3 -m scripts.sm110_gemm_model.cli import-tma-payload-campaign \
  --repo-root . --run-dir "$TMA_DIR"
python3 -m scripts.sm110_gemm_model.cli import-memory-duplex-campaign \
  --repo-root . --run-dir "$DUPLEX_DIR"
```

输出为 mergeable `{"capacities": [...]}` JSON；命令不会修改 base profile。cold rows 必须导入为 `hbm.duplex.proxy`。

## 3. Portable closure suite

对同 commit 的 compute/component/full 三批：

```bash
python3 -m scripts.sm110_gemm_model.cli audit-closure-suite \
  --repo-root . \
  --compute-run-dir results/sm110_gemm_campaign/<suite>-compute \
  --component-run-dir results/sm110_gemm_component_campaign/<suite>-components \
  --full-gemm-run-dir results/sm110_full_gemm_campaign/<suite>-full \
  --expected-commit <40-hex-commit> --require-ncu \
  --base-capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --hardware scripts/sm110_gemm_model/profiles/thor_sm110.json \
  --workloads scripts/sm110_gemm_model/examples/workloads.json \
  --schedules scripts/sm110_gemm_model/examples/schedules.json \
  --pipeline-profiles scripts/sm110_gemm_model/profiles/pipeline_profiles.json \
  --output results/sm110_model_closure/<suite>/portable_suite_report.json
```

旧 multi-commit evidence 必须继续使用 `import-closure` / `import-composite-closure`，不能伪装成单 commit suite。

## 4. Exact resource 与 causal import

```bash
python3 -m scripts.sm110_gemm_model.cli import-resource-capacities \
  --repo-root . --suite-id "$RESOURCE_SUITE_ID" \
  --expected-commit "$EXPECTED_COMMIT" \
  --output results/sm110_model_closure/$RESOURCE_SUITE_ID/resource_capacities.json

python3 -m scripts.sm110_gemm_model.cli import-causal-profile \
  --repo-root . --run-id "$CAUSAL_RUN_ID" \
  --expected-commit "$EXPECTED_COMMIT" \
  --output results/sm110_model_closure/$CAUSAL_RUN_ID/pipeline_profiles.json
```

没有 Thor bundle 时，不运行 finish/import 来制造空 profile；static-only 状态单独报告。

## 5. Campaign 回归

```bash
python3 -m unittest \
  microbench.sm110_tma_payload_campaign.test_campaign \
  microbench.sm110_memory_duplex_campaign.test_campaign \
  microbench.sm110_gemm_resource_campaign.test_campaign \
  microbench.sm110_gemm_causal_campaign.test_campaign

(cd microbench/sm110_full_gemm_campaign && \
 PYTHONPATH=../../ python3 -m unittest test_campaign)
```

## 6. 文档与链接审计

`schema_doc_audit` 扫描 current `gemm/` 规范文档集合；legacy 单体文档不再作为 schema definition source。还应检查：

- 本地 Markdown link 全部存在；
- current 文档不使用旧 empirical `hbm.read/write` 公式；
- current replay 的 model/evidence commit 与首页一致；
- experiment ID 都能映射到 source/runner/auditor；
- `git diff --check` 无问题。

## 7. Thor 运行纪律

- clean tracked worktree；
- exact expected commit；
- MAXN 与 1.575-GHz lock 记录；
- 单一全局 GPU lock；
- detached launcher 与持久 log；
- bounded trial/NCU timeout；
- `SIGTERM → grace → SIGKILL` process-group termination；
- 新合同使用新 RUN_ID；
- 每个 case 保存 result/trials/SASS/NCU/environment/hash；
- independent auditor pass 后才提交 result branch。

完整启动、恢复、结果提交命令见 [microbench README](../../../../microbench/README.md) 与 [Thor closure runbook](../../THOR_CLOSURE_RUNBOOK.md)。
