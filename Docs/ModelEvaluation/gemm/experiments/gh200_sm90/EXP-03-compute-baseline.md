# EXP-03：计算基线的旧配置映射

## 映射对象

本节冻结 S05–S07 重测所需的旧计算条件。机器可读结果为 [legacy_mapping.json](../../../../../microbench/gh200_resource_campaign/contracts/legacy_mapping.json)，由 [build_legacy_mapping.py](../../../../../microbench/gh200_resource_campaign/contracts/build_legacy_mapping.py) 从三批归档生成。映射完成不代表重测完成；所有目标案例保留 `pending_remeasurement`。

| 旧批次 | 配置行 | 范围与循环策略 | 正式条件 | 空窗口条件 | 原正式记录数 |
|---|---:|---|---:|---:|---:|
| initial | 32 | 单 CTA；128、512、2048 次循环 | 96 | 0 | 672 |
| sustained | 42 | 单 CTA、全 GPU；每个配置与范围分别用 CUDA event 校准长度 | 84 | 0 | 420 |
| audit | 7 | 单 CTA、全 GPU；8192、32768、65536 次循环 | 42 | 7 | 504 |
| 合计 | 81 | 保留原计时约定 | 222 | 7 | 1596 |

audit 的 7 个空窗口条件均为**单 CTA、0 次循环**，每项 12 条记录，共 84 条；旧源码没有全 GPU 空窗口。229 个映射条件由 222 个正式条件和这 7 个控制条件组成。audit 另外保留 78 条预热、504 条切换预热的原始行号，不计入正式条件。initial 和 sustained 的预热没有独立 raw 记录，只能引用源码中的策略。

## 为什么不能只保留 62 个组合

三批共有 62 个不同的指令与启动组合。这个统计只用于识别计算形式，不作为合并计时实验的依据。

例如 FP16 WGMMA `m64n64k16`、128 线程、2 条累加链、batch=16、wait=0，在单 CTA 下对应 initial 的三个短循环、sustained 的一个动态长度、audit 的三个固定长循环，共七个正式条件。initial 每轮 commit 后 wait0；sustained/audit 在循环结束后还执行一次 wait0。initial 仅记录 clock64；后两批还记录 globaltimer，audit 另记录本地 start/stop cycle。七项共享 `basic_combination_id`，但各有不同的 `mapping_id`。

每个映射项直接保存完整 `key`：PTX 名称及含操作数/scale/transpose 的完整汇编模板、形状、输入和累加类型、操作数来源、布局、threads、chains、batch、wait、输入常量与汇编绑定、计时边界、结果排空、范围、网格策略、循环策略及正式/空窗口角色。`mapping_id` 是该键按字段排序后紧凑 JSON 的 SHA256。修改输入常量、布局或排空边界会改变身份；案例名字不参与去重。

## 长度、计时和输入约定

sustained 先执行 4096 次预热，再用 8192 次循环取得 `pilot_ms`，按以下公式选择同一配置与范围后续使用的长度：

```text
iterations = clamp(int(8192 * 100.0 / max(pilot_ms, 0.01)), 8192, maximum_iterations)
maximum_iterations = 65536（MMA/WGMMA）或 1048576（FMA）
```

映射同时保留该公式和各条件在旧 raw 中实际使用的长度。重测时需要保留动态校准这个实验条件，不能把旧观测长度误写成原先预定的固定长度。

单 CTA 的 clock64 只在本地相减；全 GPU 的窗口为 `max(stop_ns)-min(start_ns)`。audit 的 `sum_per_smid(max(stop_cycle)-min(start_cycle))` 是逐 SM 求覆盖区间再求和，不跨 SM 相减时钟。计时区间包含循环、必要 WGMMA commit/wait、寄存器结果求和、volatile SMEM 写入和 CTA barrier。操作数准备、WGMMA setup fence 在开始计时之前，最终输出写回和主机检查在停止计时之后。CUDA event 还包含 kernel 启动与结束部分，不能替换为纯指令时延。

FMA 从 0 开始执行 `D=D*0.5+0.25`；MMA/WGMMA 的 A/B 为均匀常量 `1/16`，从 0 累加。WGMMA 的 A/B 来自无 swizzle 的 K-major SMEM 描述符；MMA 的寄存器片段保留 `.row.col`。映射记录原始汇编绑定及 WGMMA SMEM 位型，不把这些简单输入扩展为任意输入性能或非均匀正确性结论。

## 来源和完整性

initial 没有归档独立 `cases.json`。生成器读取归档 `summary.json` 的 32 行配置，并逐项核对归档 `probe.cu` 中的 PTX、循环链数、batch、线程/输出数、工作量、主机调用和 wait/计时边界；缺失独立 manifest 的事实继续保留。sustained 和 audit 使用各自归档的 cases 文件，并验证它们与前一批源码的 SHA256 链接。

三批均核对旧 summary 或 SHA256SUMS 中已有的 source/raw/cases 哈希。新映射另记录所读取的 source、raw、summary、cases、生成脚本和运行脚本的路径及 SHA256，配置项记录函数起始行和函数体哈希，条件项记录每条正式记录或空窗口在 raw 中的行号。路径相对仓库根目录，换目录重放不依赖原来的绝对路径。

生成器要求每个条件的 repeat/round 编号完整且不重复，检查范围、长度、数值错误字段和 FLOP 工作量，拒绝未映射的正式条件/空窗口。它不重新判定硬件性能资格，也不代替之后的独立审查；旧结果的资格边界见 [EXP-01](EXP-01-compute-audit.md)。旧 7/5/12 次重复来自单进程，后续重测采用新协议的外部独立进程重复，不能把旧记录直接充作新样本。

## 离线复现

在仓库根目录执行，均不需要 CUDA 或 GPU：

```bash
python3 microbench/gh200_resource_campaign/contracts/build_legacy_mapping.py --check
python3 -m unittest discover -s microbench/gh200_resource_campaign/tests -p 'test_legacy_mapping.py' -v
```

首次生成去掉 `--check`；指定另一个仓库副本可用 `--repo-root /path/to/copied/repository`。`--check` 逐字节比较重新生成的 JSON。测试覆盖完整矩阵、旧 raw 行号无遗漏/无重复、空窗口范围、计时/排空差异、输入身份、损坏记录与源码元数据拒绝、来源哈希拒绝和换目录重放。生成器只写指定的新映射文件，拒绝输出到旧归档目录。
