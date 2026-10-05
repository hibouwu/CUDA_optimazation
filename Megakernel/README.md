# Megakernel 研究

从 MPK 和 ForgeMegakernel 两篇论文开始，研究如何把模型推理中的多个算子组织到一个持续运行的 GPU kernel 中，以及如何处理任务调度、数据依赖、同步和正确性验证。

## 从哪里开始

建议先读 [MPK](MPK/阅读笔记.md)，理解 SM 级任务图与 kernel 内部的运行时，再读 [ForgeMegakernel](ForgeMegakernel/阅读笔记.md)，研究 coding agents 如何生成和验证模型的 decode megakernel。

| 论文 | 原文 | 阅读重点 |
| --- | --- | --- |
| MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs | [arXiv:2512.22219](https://arxiv.org/abs/2512.22219) | 编译器如何构造任务图，运行时如何调度任务，如何形成跨算子流水线 |
| ForgeMegakernel: A General Framework for Efficient Auto-Regressive Model Decode Megakernels | [arXiv:2609.12379](https://arxiv.org/abs/2609.12379) | 如何指导代码生成，如何校验中间状态，如何组织依赖与缓冲区 |

已完成第一轮内容核查（2026-09-21）：原文、部分 MPK 冻结源码、ForgeMegakernel 服务表格重算和精度检查的数学边界分析。先读 [内容核查报告](内容核查报告.md)，再进入各论文笔记。**尚未执行两篇论文的 GPU kernel，性能数字仍属于作者报告。**

## 带着这些问题阅读

- 一个完整推理步骤怎样拆成任务？任务与 SM、线程块之间是什么关系？
- 数据依赖如何表达和满足？哪些同步可以局部化，哪些需要全局协调？
- 性能收益来自启动开销减少、访存减少，还是计算与通信重叠？各自适用于什么条件？
- 怎样验证中间结果和最终输出？如何在相同硬件、模型、精度和负载下比较性能？
- 编译器生成与 agent 生成分别依赖什么信息？更换模型或 GPU 时，需要改哪些部分？

两份笔记已记录原文位置、已核实内容和待验证问题。原文保存在各自的 `sources/` 中；本次 CPU 核查代码和结果在 [verification/](verification/)。开始 GPU 复现后，再将代码和实验记录放到对应论文目录。
