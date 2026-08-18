# 常见错误模型与反证方法

## 1. 把 measured rate 当 upper

错误：

\[
P_{\mathrm{ub}}=W/(Q/\widehat C_{\mathrm{measured}}).
\]

measured rate 只证明硬件至少达到该速率，不能证明任何实现都无法更快。真实 GEMM 超过它时会产生伪“上界矛盾”。

修正：strict 层只用 rate upper；measured point 进入 empirical layer。

## 2. 把 L2 乘 SM 数

错误：

```text
1024 B/cycle/GPU × 20 SM
```

1024 B/cycle 是共享 GPU-wide L2 read model peak。乘 SM 会把同一 bus 复制 20 次。

修正：L2 用一次 GPU-wide time；per-SM TMA 另算 task waves。

## 3. 用 aggregate TMA / 20 推 per-SM

full-grid TMA measurement 可能已经被共享 L2 限速。aggregate/20 不是独立 SM 出口证据。

修正：单 CTA、单 observed SM、hot-L2 隔离 per-SM ingress。

## 4. 把 independent read/write peak 同时达到

错误：分别取最快 read 和最快 write，再假设同时成立。

修正：经验层使用 exact ratio `l2.duplex` / physical `hbm.duplex`；strict 层只有在有 joint outer proof 时才加入 normalized region。

## 5. 把 cold proxy 当 physical HBM duplex

Thor cold proxy 证明 external read miss 和 L2 write issue，但 `external_write_bytes_proven=false`。

修正：资源 ID 固定为 `hbm.duplex.proxy`，不能满足 `hbm.duplex`。

## 6. 把 value 与 scale 合成一条 request

错误：把 4096-B value + 512-B scale 写成 4608-B TMA request。

修正：A/B value 与 A/B scale 四类 payload 独立计账；当前 scale surface 缺 512 B/1 KiB。

## 7. 用 stage 数猜 TMA rate

stage、request count、payload、barrier、SMEM layout 和 threads 共同决定 topology。相同 stage 不等价。

修正：使用 exact family/stride resource ID；缺 capacity 时 fail closed。

## 8. 忽略一个无数值的合法 schedule

错误：只从已有预测的 schedule 中取最大 performance，并称为 manifest upper。

修正：任一合法 schedule 缺数值，整个 manifest layer `insufficient_evidence`。

## 9. 用完整 GEMM 替代 causal profile

full-GEMM total time 可以反证错误 component model，但不能唯一分解 startup、joint interval、accumulator reuse 和 drain。

修正：使用 exact event DAG runner；full-GEMM 保留为第三层 validation。

## 10. 把 static closure 当 Thor runtime

编译、PTX、SASS、host self-test 和 synthetic auditor 只证明各自证据域。

修正：runtime/numerical/performance 状态单独标记；缺 raw Thor trial 时保持 `NOT_RUN` 或 `insufficient_evidence`。

## 11. 把 cross-precision ratio 当库胜负

FP4 candidate / FP16 cuBLAS 可以是诊断比值，但不是 same-precision comparison。

修正：独立记录 correctness reference、performance denominator 和 relation。

## 12. 反证顺序

当 observation 超过模型时，依次检查：

1. workload/precision/output 是否同合同；
2. numerator 是否处于正确资源边界；
3. rate 的单位与 scope；
4. hardware/mode/clock；
5. residency 与 timed scope；
6. capacity 是 upper 还是 measured；
7. exact topology/ratio；
8. artifact/importer arithmetic；
9. manifest 是否遗漏 schedule；
10. empirical joint/causal 是否需要重校准。
