740161 两组完整控制的 C 独立审查通过；必要局部 gate 仍失败。最终证据为 `review-final-v2/report.json` 和 `review-final-v2/tests.txt`，主线程原分析与汇总未被覆写。

校验了 79a86b8d68de1e6fd8a13e3e0ab018f67d01df35 的冻结与输入包，17 个输入归档文件和 105 个输出归档文件全部一致。提交时间记录为巴黎时间 15:31:45，Slurm 提交为 15:31:46。实际 a058 分配为 288 CPU、4 GPU、OverSubscribe=NO，CUDA UUID 为原 ef 卡；binary 与已审查的 740156 完全相同。

实际 40 个目标进程与 manifest 次序完全一致：先 20 个 off/on，再 20 个 off/off；各组 10 对，各对先后方向交替。off/off 使用不同 arm 标识，两个真实 flag 均为 0；producer-tail 始终为 0。每次固定 16 个 observer-off 前序调用。没有丢弃、替换、重试或跨组重新配对。

从原生槽位独立构造每进程全部 6372 条系列，逐项等于冻结 evaluator 的输入。独立重算了全部有符号配对差、64/256ns 判定、event/CV、SM 变化和开启读取诊断；与主线程完整结果相等。163840 个数值样本由独立整数乘积参考验证；21120 条分项累加恒等式、69120 条全调用原点恒等式通过。冻结提交中的新旧 reader/control 26 项 CPU 检查通过。

| 原冻结指标 | off/on | off/off |
|---|---:|---:|
| 原 6372 系列失败数 | 4912 | 4729 |
| L 系列失败数 | 432/432 | 432/432 |
| L 各系列绝对配对差中位数范围 | 64..416ns | 80..448ns |
| L 各系列绝对配对差最大值范围 | 320..1056ns | 320..960ns |
| event 绝对比例差中位数 | 0.732086% | 0.375104% |
| event 绝对比例差最大值 | 1.612346% | 2.188322% |
| 参考 arm0 event 总体 CV | 0.582265% | 1.059387% |
| CTA→SM 映射变化 | 0 | 0 |

event 的 2%/5% 和 CV 的 5% 条件均通过；局部系列条件均未整体通过。TAIL 仍定义为 consumer FINAL−last DONE，不是 producer_tail。没有按全事件通过替换局部失败。

on 的 8640 stamp 宽度为 `{0:670,32:7240,64:718,96:12}`；每个窗口的十个误差界及两个端点宽度均与原始槽位一致。这些量只描述条件 GT 参考域内的读取宽度。off 没有 GT_pre，不为其补零，不构造双方配对区间，不计算 on/on gate，不从两批次中心相减，不重评模型或 V09 候选。

局部配对差与读取误差的区别可直接写成等式。假设采样遵守已审查的依赖顺序，令 T 为 FIRST、MAIN 两个 cycle 采样点对应 GT 值之差，η_e 为事件 e 的 post-GT 相对 cycle 采样点的参考计数偏移，则记录量 `N=T+η_MAIN−η_FIRST`。两次调用之差包含 `ΔT` 以及两个调用采样偏移的变化。开启记录只能在声明前提下界定自身 η；关闭记录缺少 GT_pre，两个调用的 T 也没有相等证据。因此本批的局部差不能定量分摊为执行变化、GT 参考行为或采样效应。

off/off 的失败表明开启 GT_pre/helper 不是本协议下局部差出现的必要条件。它不证明观察代码无扰动，也不证明全部差异是读取误差。off/on 的 W 有 203/432 失败、off/off 为 1/432，因两批为先后执行的独立组，也不能将二者相减解释为观察代码的因果贡献。所有采样包围、物理 ns、观察、读取精度、局部预测、关键 CTA 与完整 GEMM 资格继续为 false。

```sh
python3 review.py \
  --run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls \
  --ordering-review /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-build/reanalysis/C-20261011-ordering-review-v1/report.json \
  --output NEW_C_REVIEW_DIRECTORY
```

`review-final-v1` 保留本轮加固公共报告前后 SHA 守卫之前的 CPU 中间重放；最终重放使用 v2。本审查没有执行新 GPU 进程、改动公共代码或申请重复控制。
