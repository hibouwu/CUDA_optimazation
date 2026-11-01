# Lean 形式化建模目标（GOAL）

关联文档：[架构规范（ARCH）](ARCH.md) · [推进计划（PLAN）](PLAN.md) · [GEMM 建模总目标](../Docs/ModelEvaluation/gemm/GOALS.md)

## 1. 核心动机与方法来源

前期性能建模实验在 Python 中以单点时间预测为主：当候选模型与实测出现残差时，往往通过局部改结构或扫参数反复试错，容易混淆“某个固定模型结构内部的参数取值问题”与“所声明的约束组合与观测结果在数学上是否相容”。

本子项目借鉴 **Algebraic World Model: From Exploration to Proofs**（Guancheng Pan, Qixin Tan, Yang Yuan, Andrew Chi-Chih Yao，[arXiv:2610.40503](https://arxiv.org/abs/2610.40503)）的思路：将环境探索（微基准测量与程序图分析）提炼出的候选假设写成结构化约束，并用 Lean 4 建立可机检的形式化证明。

引入 Lean 的目标限定为以下两项产出：

1. **在抽象模型内确定条件性上下界包络 $[L, U]$**
   不将单点拟合误差作为唯一交付。给定显式声明的事件依赖结构 $P$、观察终点集 $O$ 与包含源起点区间及边偏移区间的参数假设 $H = [H_{\min}, H_{\max}]$，在抽象模型内严格证明：所有满足双侧规范（$\mathrm{ValidLower}_{H_{\min}}(\tau) \land \mathrm{ValidUpper}_{H_{\max}}(\tau)$）的轨迹，其观察完成时间 $T_O(\tau)$ 均受限于条件性下界 $L$ 与条件性上界 $U$：
   \[
   L(P, O, H_{\min}) \le T_O(\tau) \le U(P, O, H_{\max}),
   \]
   且下极值排程 $\underline{\tau}$ 与上极值排程 $\bar{\tau}$ 自身均满足同一区间模型的双侧规范（即 $L$ 与 $U$ 在模型内精确可达）。在带秩 DAG 上，当所有源起点与全部边偏移均固定（$H_{\min} = H_{\max}$）时，双侧规范确定唯一轨迹，因此 $L = U$。
2. **检验候选假设与观测的相容性（Incompatibility & Pruning Certificates）**
   - **结构无解判据**：证明某类程序控制与资源容量组合（如 $S=1, n \ge 2, d^{\min}_{\mathrm{load}} > 0$）不存在满足下界规范的完整排程；
   - **候选不相容判据**：证明在已声明的候选组合 $(P, O, H_{\min}, H_{\max})$ 下，若观测区间 $I_{\mathrm{obs}}$ 与 $[L, U]$ 不相交，则不存在同时满足双侧规范且 $T_O(\tau) \in I_{\mathrm{obs}}$ 的轨迹，从而发现不相容并限制后续解释（若要排除整类带自由参数的假设或作单一机制归因，仍需覆盖相应参数域的统一证明或独立区分证据）。

## 2. 第一阶段完成标准

第一阶段限定为**一条抽象模型内的完整证明链**，满足以下四项即算达成首轮目标：

1. **内部安全性质可证**：在同一轨迹语义下，仅从下界必要等待规则推出同槽不同代不覆盖、任意时刻在途槽数不超过容量上限 $S$（对任意 $S \ge 1, n \ge 0$ 成立）。
2. **双侧紧上下界与可达性可证**：在同一轨迹语义下，由必要等待规则推出路径下界；在保证图无环的参数条件（$S \ge 2, n \ge 0$）下证明极值排程 $\underline{\tau}, \bar{\tau}$ 均满足 $\mathrm{ValidLower}_{H_{\min}} \land \mathrm{ValidUpper}_{H_{\max}}$，推出 $L \le T_O(\tau) \le U$，并在固定参数实例上给出 $L = U$ 的紧界。
3. **无解与不相容判据可证**：给出“$S=1, n \ge 2, d^{\min}_{\mathrm{load}} > 0$ 因正权环而无解”的形式化反证，给出“$(P, O, H_{\min}, H_{\max})$ 与观测区间 $I_{\mathrm{obs}}$ 不相容”的通用定理，以及改变槽数 $S$ 或增删跨拍控制边时极值上下界严格改变的对比证明。
4. **无未完成证明与额外公理依赖**：所有定理通过 Lean 编译检查，除标准逻辑公理外不含 `sorry` 或自定义硬件公理（硬件、推进与抽象边界均作为定理的显式前提传入）。

## 3. 职责边界：哪些问题不放进 Lean

| 问题类别 | 归属位置 | Lean 子项目的处理方式 |
|---|---|---|
| 源码/SASS 与程序依赖图的对应关系 | [CUTLASS 程序图](../Docs/ModelEvaluation/gemm/model/mechanism/program/cutlass_program_graph.md) 与 `cfg_b_program.py` | 作为结构来源参考；静态程序 DAG 不含动态排队与在途分摊，未完成竞争归约与完整映射前，不宣称抽象包络直接覆盖真实硬件或完整二进制 |
| 局部延迟、服务率与实测端点的获取 | [GH200 实机实验](../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/README.md) | 实测事件差值不等同于合取规则中的边偏移参数；接入前须单独建立参数资格与测量窗口对应关系 |
| 批量配置与动态竞争的数值模拟 | 现有 Python 模型 | 继续由 Python 执行；Lean 仅验证形式化约束结构、条件性上下界与不相容判据 |
