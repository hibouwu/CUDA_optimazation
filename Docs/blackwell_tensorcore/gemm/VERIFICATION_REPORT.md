# Thor/SM110 GEMM 建模体系独立验证报告（草案）

> 验证日期：2026-08-19（会话内独立审计）。本报告不修改任何模型文件，仅记录验证方法与结论。
> 验证对象：`Docs/blackwell_tensorcore/gemm/`（文档）+ `scripts/sm110_gemm_model/`（代码）+ `microbench/`（证据）。

## 0. 总结论

- **数学公式层：自洽，方向正确。** 全部抽查数值独立重算吻合。
- **代码层：忠实实现文档公式。** 149 项单元测试全部通过；关键路径逐行对照一致。
- **证据链：真实可复现。** CSV 源文件存在且 auditor 校验通过；反向篡改测试确认 auditor 能抓错。
- **物理参数：全部 plausible，多数有 NVIDIA 官方来源确认。**
- **无法验证项：** 历史 Thor observed 数据（120.039 / 130.633 TFLOP/s）原始采集树不在当前仓库，只能按文档引用；L2 1024/512 B/cycle 是 NCU profiler model peak，无独立公开规格可核（但文档已诚实标注证据类别）。

---

## 1. 已验证成立（A 类）

### 1.1 数学公式与手算重算

| 检查项 | 文档值 | 独立重算 | 结论 |
|---|---|---|---|
| W_use = 2MNK (2048³) | 17,179,869,184 FLOP | 17,179,869,184 | ✓ |
| 最小输入字节 | 16 MiB | 2×2048²×2 B = 16 MiB | ✓ |
| 最小输出字节 | 16 MiB | 2048²×4 B = 16 MiB | ✓ |
| N_task | 128 | (2048/128)×(2048/256) = 128 | ✓ |
| Q_TMA,issued | 192 MiB | 4096 visits×(16+32) KiB = 192 MiB | ✓ |
| LPDDR 时间下界 | 122.91 µs | 32 MiB / 273 GB/s | ✓ |
| cold 性能上界 | 139.776 TFLOP/s | 17.179e9 / 122.91e-6 | ✓ |
| hot 上界 | 258.5 TFLOP/s | compute 瓶颈 | ✓ |
| L2 read/write 下界 | 10.40 / 20.80 µs | 16 MiB / 1.6128、0.8064 TB/s | ✓ |

### 1.2 因果流水线递推（06 章 vs 代码）

- 递推公式 F_o / L_o / E_o 与 `predict_pipeline_worker_seconds` 逐行一致；
- off-by-one 正确：首任务 o=0，accumulator 复用等待 E_{o-A}（代码 `epilogue_done[task - accumulator_buffers]`）；
- 手算验证（λ_J=10ns, ι_J=2ns, λ_E=5ns, A=2, K_t=3, O=3）：
  F=[10,16,22], L=[14,20,26], E=[19,25,31] ns，与测试锁定的 31e-9 完全一致，且两个 max 分支（22 vs 21）均被覆盖；
- 范围外推被拒绝（`profile_range` → ModelError）。

### 1.3 严格上界与 fail-closed 语义（04/05 章 vs 代码）

- `T_r^LB = Q_r^LB / U_r`，多资源取 `max`，`P_ub = W_use / T_ub^LB` ✓；
- strict compute 用 W_use、empirical compute 用 W_issue ✓；
- 经验层强制交集所有 hard upper（`hard_upper:*` 参与 max）✓；
- 缺容量 → `insufficient_evidence`（保留 resource_seconds 作诊断，不拼"半个包络"）✓；
- 同一资源多个 upper 取最紧者（`min(rate)`）✓；
- GPU-wide L2 不乘 SM 数（有专项测试锁定）✓；
- per-SM TMA ingress 用 slowest-wave makespan，不从整卡 rate 反推 ✓；
- legacy tc5a stride2048 单向 alias，不覆盖 1024/4096 ✓；
- measured rate 不被当作 strict upper（`evidence_kind` 过滤 + 测试锁定）✓。

### 1.4 实测 CLI 行为

- `cli audit` → `{"findings": []}`（全部 capacity 声明与源文件一致）；
- `cli evaluate`（FP16 N=1024 cold）：
  - strict upper = 69.888 TFLOP/s，瓶颈 `hbm.total`，resource_seconds 四项全部与手算一致；
  - empirical_ideal_envelope = `insufficient_evidence`，missing_resources 清单与 08 章缺口声明一致；
- hot-L2 变体：strict upper = 258.5 TFLOP/s，瓶颈 `tensor.fp16` ✓ 与 worked example 结论一致；
- `schema_doc_audit` → PASS；
- 单元测试 149 项全部通过（含 lesson3/4/6/8 手算匹配、double-buffer 递推精确值、upper violation 报错等）。

### 1.5 证据链真实性

- `microbench/mma_compute_only/plots/benchmark_results.csv` 真实存在；实测 FullSM4WarpBlock：
  BF16 258.03 / FP8 516.06 / FP4 1032.11 TFLOP/s，与官方 dense 谱系（258.5/517/1035）吻合到 99.7–99.8%；
- 由实测 258.03 TFLOP/s 反推时钟：258.03e12 / 20 SM / 8192 FLOP/cycle = 1.5749 GHz ≈ 1.575 GHz profile 值（强交叉验证）；
- `results/sm110_*_campaign/` 采集树真实存在（含 run_spec、SASS、binary hash），但为 static 产物，无 Thor 计时数据——与文档 `complete=false` 一致；
- 反向测试：篡改 CSV original_value → auditor 报 `csv_original_value_mismatch`；measured 改名冒充 upper → 报 `measured_mislabeled_as_upper`。

### 1.6 物理参数可信度（web 核验，详见子代理报告）

- SM110 / T5000 / 20 SM / 1.57 GHz MAXN / LPDDR5X 256-bit 273 GB/s：NVIDIA 官方来源直接确认；
- 官方博客 T5000 表格原文 "517 TFLOPs (Dense FP8 | Sparse FP16)"：FP16 dense = 258.5 的推导是官方口径；
- TMEM 512×128×32-bit = 256 KiB/CTA：与 MLC《Modern GPU Programming》及 B200 微基准论文一致；
- 2048³ FP16 Roofline：compute 258.5 / DRAM 139.78 / L2 不瓶颈，全部复算吻合。

---

## 2. 存疑 / 需要额外证据（B 类）

1. **L2 read 1024 / write 512 B/cycle/GPU**（`profiler_model_peak`）：
   换算、数量级、与实机 946.7 B/cycle 的方向关系都对，但 NVIDIA 未公开 SM110 逐周期 L2 总线规格；该 upper 依赖 NCU 模型峰值。文档已诚实标注证据类别为 `profiler_model_peak` 且要求 condition 声明，未冒充硬规格。**建议**：在 Thor 上做一次 L2 端口级压测（read 与 write 分别、混合 ratio sweep）尽量逼近端口峰值，或从 NCU 报告中提取 model peak 出处留档。
2. **历史 observed（tc5a 120.039 / cuBLAS 130.633 TFLOP/s）**：
   数值 ≤ cold upper 139.776，不自相矛盾；但原始采集树不在当前仓库（`results/` 只有 static 产物与更早的 43.9 TFLOP/s 旧实验），无法机械复现。文档已标注 legacy/historical。**建议**：若需要审计链闭合，把 2026-08-14 采集树的 summary/CSV/audit 报告归档进仓库。
3. **cold 场景经验层**：physical `hbm.duplex` 缺失导致 cold empirical layer fail-closed。这不是错误，是证据缺口；文档与代码均诚实输出 `insufficient_evidence`。

---

## 3. 无法验证（C 类）

- Thor 实机采集（causal 182-case timing、E5M2/其余精度 full GEMM、512 B/1 KiB block-scale payload）——需要 Thor 硬件回传，当前仓库与文档均未声称拥有。
- "已观测最好值在未来是否会被某个更好实现超越"——模型只做条件上界与经验包络，不做绝对最优证明（文档已声明）。

---

## 4. 总体判断

这套建模体系在**数学、代码、证据分级**三个层面均通过了独立验证：
公式自洽、方向正确（measured 永不冒充 upper、GPU-wide 不乘 SM 数、max 不求和、缺证据 fail closed）；
代码忠实实现并有 149 项测试锁定关键性质；物理参数有官方来源支撑。**未发现确定错误。**
体系中"最弱"的两处（L2 profiler peak、历史 observed 无法复现）文档均已诚实标注，属于证据缺口而非逻辑缺陷。
