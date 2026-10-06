# S21：如何使用六类实验的条件化观测

当前有 279 条服务观测和 10 个排除项，来自六个已完成 C 审查的正式家族。exporter r4 已完成独立 B 审查；本文及发布来源桥接仍是候选，等待另一未实施者审查。原四份导出保留 draft 标签和原字节，尚不能只凭这个文件宣称参数已发布。

## 文件与资格来源

| 文件 | 用途 |
|---|---|
| `all-six-draft-r4/parameters.json` | 279 条配置的观测值、条件、所有正式样本和统计 |
| `all-six-draft-r4/excluded.json` | 10 个控制、不稳定或不可导出的配置及原因 |
| `all-six-draft-r4/sources.json` | 六个输入归档、C/COMPLETE、冻结源码和完整输入哈希 |
| `s21-publication-bridge-v1/envelope.json` | 连接四输出哈希、exporter B、六份 C 与完成记录的候选封套 |
| `s21-publication-bridge-v1/manifest.json` | 冻结桥接源码、封套、例子、本文和原字节资格副本 |

以上两组目录都位于 suite 的 `implementation/`。参数资格将由原四文件、封套和独立 `publication-bridge-B` 门禁共同提供，不修改历史 summary 或 draft 标签。候选检查不会授权参数：只有 `bridge.py check --published` 在确切独立门禁存在、完整来源哈希一致时，才报告条件化观测合格。

exporter B 的 `gate_files` 含六个 `COMPLETE`，因此不能通过通用 GPU 门禁的可变文件名规则。桥接使用独立的只读来源检查，允许确切完成记录作为冻结输入，并在 `frozen-inputs/` 保存原字节；不改变 `suite_io`，不将 COMPLETE 当作代码或 GPU 启动门禁。

## 先选条件，再取数值

以 `family`、`run_id` 和 `case_id` 定位配置，读取 `conditions.case` 及 `conditions.resolved`。指令、输入与累加类型、依赖链、tile/shape、参与线程、CTA/grid 范围、循环长度、stage、请求数、完成边界必须与将要建模的服务相符。校准存在时采用实际 `resolved_iterations`；固定长度观测不能解释成单指令裸延迟或每 tile 增量。

| 字段 | 读取方式 |
|---|---|
| `unit`、`clock_domain`、`scope` | 保持原单位、原时钟域和执行范围；单 CTA `clock64` 不与跨 SM 时间相减 |
| `value` | 全部同协议正式样本的合并中位数；不是最大样本或硬件峰值 |
| `statistics`、`batches`、`samples` | 保留 n、mean、median、min、max、CV 和全部同协议正式批次；不筛掉不稳定批次后挑最好结果 |
| `conditions.device`、`conditions.environment` | 实际 UUID、架构、驱动与工具链；不同身份不能直接合并 |
| `conditions.resource_observations` | 逐样本实际 blocks/resources、观察到的 SM ID；occupancy 上限与观察覆盖分开 |
| `limitations` | 未证明的缓存驻留和物理 HBM 流量、计数器权限状态；不能补成物理量 |
| `source_run`、`samples[].raw/receipt` | 相对输入 suite 根目录解析；每个记录附 SHA256 |
| `condition_sha256` | 完整条件对象身份，不是只按精度或 case 名称去重 |

例如 `B_transport/clock64_cycle/CTA` 是指定单 CTA 完成窗口中的有效运输量／局部周期。`B/s` 仍须看参数声明的逻辑请求、读写方式和整网格完成边界，不能自动叫作物理 HBM 带宽。FP 浮点 FLOP、整数 OP、请求字节和完成时间不互换。

## 一条 raw 的手算

选 `async_copy/formal-b3-v2` 的 `ca_w4_stages1_one_cta`，`batch_00/trial_00/attempt_00/raw.jsonl`。配置为一个 CTA、128 线程、8192 轮；每线程每轮执行八次 4 B 请求，stage=1。消费者读取相邻线程复制的所有 uint32 并累加，等待和消费者完成属于该服务边界。完整实验见 [S12 异步复制](EXP-12-async-copy.md) 和 [实测结果](EXP-12-async-copy-results-v1.md)。

raw 中 `work_count=33554432`，有效运输量为：

\[
1\times128\times8192\times8\times4=33\,554\,432\ \mathrm{B}.
\]

`read_payload_bytes` 和 `write_payload_bytes` 均为 33,554,432 B，但本配置的 numerator 是 `work_count`，一次运输计一次，不能把两者相加作为运输量。

同一 CTA 的计时字段为 `start_cycle=18467894432170`、`stop_cycle=18467920533509`，所以：

\[
\Delta c=26\,101\,339\ \text{clock64 cycles},\qquad
r=\frac{33\,554\,432}{26\,101\,339}
 =1.285544469576829\ \mathrm{B_{transport}/clock64\_cycle/CTA}.
\]

这是第一条样本。十条正式样本的合并中位数为 **1.2855407510685195**，CV 为 **0.0002303748980315688**。样本值与中位数分别保留，不能用第一条样本替代合并观测。局部周期也不直接转换成整卡 HBM 带宽。

`s21-publication-bridge-v1/example.json` 绑定这条 raw 的 SHA256、整数工作量、周期差、精确分数和中位数；校验脚本按原字段重新计算。

## 复现与校验

在项目根目录复现 exporter，输出路径必须是新目录，且不能放进任何登记的历史 run：

```bash
python3 -B results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-parameter-export-B/exporter-r4.py \
  --suite results/gh200_resource_campaign/20261001-resource-suite-v2 \
  --code microbench/gh200_resource_campaign \
  --output /一个已存在父目录下的全新导出目录
```

只检查桥接候选及全部输入哈希：

```bash
python3 -B results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-publication-bridge-v1/bridge.py check --full
```

独立桥接 B 签署后使用同一命令加 `--published`。未签时会拒绝；普通 `check` 始终报告候选状态，不因 exporter B 已通过而自授资格。输入、来源或输出改变时创建新的桥接身份，不覆盖 v1。

这六类观测覆盖 `memory_baseline`、`legacy_fma`、`legacy_mma`、`legacy_wgmma`、`async_copy` 和 `global_duplex`。FP8/INT8、TMA、DSM、组合实验等后续家族未由本封套取得参数资格。建模时先保持该条服务的条件和完成边界，再由组合模型讨论并行与竞争，不能直接相加或乘以 SM 数后宣称得到完整 GEMM 性能。
