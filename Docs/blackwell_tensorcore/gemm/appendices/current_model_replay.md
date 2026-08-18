# 当前模型重放

## 1. 重放身份

本页描述当前 f06 模型对现有 evidence 的语义，不表示重新运行 GPU。

| 项目 | 值 |
| --- | --- |
| model code | `f06f2cd917a4cb23806b5e1be06120be9152ed7b` |
| hardware profile | `thor_t5000_sm110_20sm` / 20 SM / MAXN / 1.575 GHz snapshot |
| parameter data | `aa845dd9e70e2c541ae3a7d5293bf8de4bd55092` |
| parameter branch | `thor-results/thor-t5000-parameter-plots-maxn-20260817-i` |
| current target completion | false |

## 2. Parameter import replay

当前 importer 对真实 `-i` result tree 重放：

```text
TMA payload capacities       10
L2 duplex capacities         14
cold HBM duplex proxies       7
physical HBM duplex           0
```

全部 importer 会先运行 canonical independent auditor，再验证 source locator 与 artifact paths；不是读取一个手工汇总表。

## 3. 当前 empirical memory 语义

current code 对 cold schedule 要求：

```text
hbm.duplex  (physical, exact ratio)
l2.duplex   (schedule-issued ratio)
```

对 hot schedule 要求：

```text
l2.duplex   (schedule-issued ratio)
```

已有 cold result 只有 `hbm.duplex.proxy`，所以 current cold empirical layer 没有 physical HBM 数值。已有 hot-L2 ratios 完整，但完整 integrated envelope 还要求 exact TMA resource 与 causal profile。

## 4. 当前 completion replay

将 base capacities、`-i` payload/duplex、24 个 calibration/holdout workloads 和 6 个 schedule 合并后：

| 门禁 | 值 |
| --- | --- |
| all precision contracts present | true |
| compute campaigns planned | true |
| current manifest data paths modeled | true |
| required TMA payloads planned | false |
| required TMA payloads measured | false |
| required cold proxies measured | true |
| required physical HBM duplex measured | false |
| required L2 duplex measured | true |
| all full-GEMM campaigns planned | false |
| all full-GEMM scenarios planned | false |
| dependency span complete | false |
| cache residency complete | false |
| joint overlap complete | false |
| all-precision three-layer closure | false |
| duplex campaign frozen at current basis | false |
| epilogue campaign frozen at current basis | false |
| joint-pipeline campaign frozen | false |
| final source appendix generated | true |
| final complete | false |

## 5. 为什么没有当前 128.436 TFLOP/s headline

128.436 TFLOP/s 是迁移前模型基于独立 read/write component 与 legacy tc5a resource applicability 得到的历史 envelope。current model 增加：

- ratio-qualified duplex；
- physical cold HBM joint requirement；
- exact topology/stride；
- causal profile；
- hardware/mode/clock scope；
- all-legal-schedule fail-closed selection。

在这些当前门禁缺失时，合法输出是 `insufficient_evidence`，不是沿用旧数值。历史采集本身仍有效，见 [historical_results](historical_results.md)。

## 6. 重放命令

具体命令见 [audit_and_reproduction](audit_and_reproduction.md)。当 parameter result branch 被检出到同一 repository tree 后，运行两个 importer，再运行 coverage/target-completion；不要手工把 summary 中位数复制到 `capacities.json`。
