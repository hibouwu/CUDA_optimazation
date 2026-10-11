仅有源码或 SASS 的 PC 顺序，不能授予实际采样时刻的包围资格。本文补充旧 `C-20261011-prefix-observation-review-v1/protocol.txt` 的第 2–4 项；旧协议和诊断结果不覆写。

`asm volatile` 的公开保证涉及生成 PTX 时不删除或移动 asm；`memory` clobber限制周围的内存优化。将三条独立读放进一个 asm、查看 PC 顺序或添加内存栅栏，都没有提供 special-register 采样序列化的公开保证。[CUDA 12.9 Inline PTX](https://docs.nvidia.com/cuda/archive/12.9.1/inline-ptx-assembly/index.html#incorrect-optimization) 与 [PTX memory fence](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar) 分别规定了这些用途。

740156 的实际编译路径已经加强：

| 端点 | GT_pre 与 clock 的依赖 | clock 与 GT_post 的依赖 | 无效路径 |
|---|---|---|---|
| FIRST | `1790` GT 写 R12:R13；`17a0/17b0` 判非零；`17c0` 按结果跳过 `17d0` clock | `17e0/17f0` 消费 clock 的 R14:R15；`1800` 的 GT 读本身受 `P3` 谓词约束 | `1820..1840` 清零三个寄存器对 |
| MAIN | `2450` GT 写 R12:R13；`2460/2470` 判非零；`2480` 按结果跳过 `2490` clock | `24a0/24b0` 消费 clock 的 R14:R15；`24c0` 的 GT 读本身受 `P3` 谓词约束 | `24e0..2500` 清零三个寄存器对 |

这里存在前读结果参与后读执行条件的真实依赖。后读没有被搬到条件之前或取消谓词；不是将读数留到末尾 store 才消费。[PTX 的谓词执行语义](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#predicated-execution) 支持对此代码路径的审查。helper 的 PTX 局部 before/cycles/after 在成功和失败路径都赋值，再无条件移至三个 C++ 输出，因此当前 `=l` 不属于“只在条件分支更新输出却未使用 `+l`”的问题。前置零值使整条记录无效，无重试；运行时仍必须检查零值、顺序、覆盖和 overflow。

这是源码及已编译依赖审查通过，仍没有独立证明实际寄存器采样时刻或连续物理 ns 精度。后续算术必须显式声明两个前提：实际采样遵守审查过的结果依赖/控制路径；同一次调用的 GT 是单调参考计数域。`clock64` 读的是 cycle counter；`globaltimer` 的公开描述还保留了 target-specific 行为限制，不能在此处顺带授予跨目标物理校准。[PTX clock64](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#special-registers-clock64)、[PTX globaltimer](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)。

在上述前提下，事件 e 的 cycle 采样点对应的 GT 值属于 `[n_e^-, n_e^+]`。FIRST→MAIN 的计数域时长属于 `[n_MAIN^-−n_FIRST^+, n_MAIN^+−n_FIRST^-]`，并可与源码保证的非负顺序取交集。原 post-GT 点差的误差界仍为 `max(width_FIRST, width_MAIN)`，配对差的保守误差界仍为两调用误差界之和。公式只传播计数域内的采样不确定性；量化、目标行为或 GT 到连续物理 ns 的额外误差需要另有证据和预算。读数单调或区间很窄都不能反向证明两个前提成立。

新接口只覆盖全部 132 CTA 的 FIRST/MAIN，两个 WG、最多两个输出序号。9504-byte shadow 的 0..7 保存 GT_pre，8 保存 overflow；原 cycle 和 GT_after 的布局槽保留。新 binary 需要自己的冻结身份，旧 prefix 模型不能绕过其 binary 约束复用。此次 C 审查校验了输入 source manifest 的 858 项，原 856 项中 854 项不变；740156 是 CPU 编译，无 GPU 分配。编译资源仍为 168 registers、2 barriers、零 spill；相同资源数量不能证明运行扰动为零。

扰动控制、off/off null、全字段覆盖、正确性和独立留出继续分别验收。当前没有新 GT_pre 实测记录，不能为旧档案补造包围区间。FIRST/MAIN 的界也不能替代 PERMIT/DONE/FINAL、producer_tail、D_visible 或完整 event 的证据。生命周期传播保留这些端点的未知上界，关键 CTA 候选集合保持冻结。
