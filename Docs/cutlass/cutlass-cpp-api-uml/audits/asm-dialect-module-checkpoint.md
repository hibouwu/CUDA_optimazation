# ASM 方言工作恢复点

检查点日期：2026-09-08。后续任务已切换为 Dense FP16 2SM 端到端模块，因此本检查点后不继续扩展 asm 语法。

## 已完成并冻结

- [ASM01 NVCC 方言复核](phase-1-asm01-nvcc-dialect-recheck.md)：已撤回原“固定 CUDA 源码错误”定性。
- [旧审查](phase-1-bitfield-asm-review.md)顶部已追加显著撤回说明和新证据链接，旧 GNU 结果与正文作为历史比较保留。
- `audits/nvcc-asm01-probe/run-o6wkuhtz/` 是最终矩阵证据。原文/有逗号对照 × SM100a/SM110a × 未调用/实际调用，共 8 cases；实际调用时 12 个 entry 各有一条 PTX `tcgen05.cp` 和一条 SASS `UTCCP`，NVCC 13.0.88 与 PTXAS 均通过。同一架构 raw/control 的完整 PTX 和 NVCC CUBIN 逐字节相同。
- `run-_tpkdhwu/` 保留 Host feature-test 宏不兼容失败；`run-1z5ect5w/` 保留将 1cta/2cta 混入同一个测试 kernel 造成的 PTXAS harness 失败。它们不能代替最终结果。
- 未执行 GPU kernel，未加载 CUDA module，没有安装或修改系统；固定快照和生成器未改。

冻结文件散列：

```text
audits/phase-1-asm01-nvcc-dialect-recheck.md
784766d6474d4fafe1fcc9ad8de14773ffed2af6ab347835ed701d54c29dbe75

audits/nvcc-asm01-probe/run_probe.py
56d54c548688f81ec3ffc854e710c96ed279287b5104739786062b241e25c13e

audits/nvcc-asm01-probe/verify_probe.py
61398e0794cd639b49a595cdff15d4902012c77df8ccd8c33d4fbe6a8dcf9c03
```

可复核命令：

```bash
.venv/bin/python audits/nvcc-asm01-probe/verify_probe.py audits/nvcc-asm01-probe/run-o6wkuhtz
```

## 尚未实现的独立模块

任务切换时，`scripts/declaration_cuda_asm.py` 和 `tests/test_declaration_cuda_asm.py` 均未创建；目前没有可宣称完成的 asm 独立解析模块或 core 接入。仅形成以下待实现设计，不能当作已有功能：

1. 用保留字符串类别与字节区间的词法扫描定位 `asm`/`__asm__` 和平衡的参数括号，只在该作用域的顶层冒号划分 output/input/clobber/label 区域，不能对任意 C++ 字符串调用使用裸正则补逗号。
2. 记录每个 operand 的 constraint、表达式、可选 symbolic name、完整原文范围、原始分隔文本和预处理条件；保留 memory clobber，不把表达式内部的逗号/冒号当作 asm 分隔符。
3. 初始已验证 NVCC 方言样本是原 12 个空 output、两个 input 的 `"r"(dst_addr)  "l"(src_addr)`。parser-only separator edit 与原文/真实分隔符、NVCC 实际编译证据分开；不修改固定快照。
4. 输出相邻操作数、named operand、内部条件指令、宏化 assembly template 等更多形式若没有足够证据，应保留 pending，不因词法上能分段就自动宣布 NVCC 支持。
5. 后续可全 824 文件枚举候选和未解析项，但这轮未执行该 asm 模块扫描，也没有对应统计。

恢复时先读取本检查点和 NVCC 复核，再根据当时端到端模块需求判断是否需要继续。不要自动重开旧的全库语法扩展方向。
