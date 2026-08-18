# FP16 N=2048 current-model 手算

本例演示一条完整论证链。目标不是复刻旧 128.436 TFLOP/s，而是按 current f06 模型判断哪些层能给数值、哪些层必须 fail closed。

## 1. Workload 与 schedule

workload：

```text
M=N=K=2048
precision=fp16_f32
beta=0
epilogue=none
output=fp32 accumulator
```

schedule：

```text
tc5a_m128n256k64_stage4
BM=128, BN=256, BK=64
stages=4
threads=192
readback_warps=4
cta_group=1
split_k=1
```

## 2. Useful compute

\[
W_{\mathrm{use}}
=2\times2048^3
=17{,}179{,}869{,}184\ \mathrm{FLOP}.
\]

shape 可整除，所以：

\[
W_{\mathrm{issue}}=W_{\mathrm{use}}.
\]

## 3. Minimum 与 unique bytes

A 和 B 各有 \(2048^2\) 个 FP16 元素：

\[
Q_{\mathrm{input,min}}
=2\times2048^2\times2
=16\ \mathrm{MiB}.
\]

输出为 FP32：

\[
Q_D
=2048^2\times4
=16\ \mathrm{MiB}.
\]

当前无 scale、\(\beta=0\)，因此 ideal cold-entry：

```text
read  = 16 MiB
write = 16 MiB
ratio = 1:1
total = 32 MiB
```

## 4. Task 与 K tiles

\[
N_M=2048/128=16,
\qquad
N_N=2048/256=8,
\]

\[
N_{\mathrm{task}}=16\times8=128.
\]

\[
N_K=2048/64=32.
\]

20 SM、one resident CTA/SM 时，最慢 SM 需要：

\[
\left\lceil\frac{128}{20}\right\rceil=7
\]

个 output task wave。

## 5. 单 K tile 的独立 TMA payload

A value：

\[
q_A=128\times64\times2=16\ \mathrm{KiB}.
\]

B value：

\[
q_B=64\times256\times2=32\ \mathrm{KiB}.
\]

每个 output task 有 32 个 K tile，因此每 task：

\[
Q_{\mathrm{TMA/task}}
=32\times(16+32)\ \mathrm{KiB}
=1.5\ \mathrm{MiB}.
\]

## 6. Schedule-issued TMA/L2 bytes

tile visits：

\[
N_{\mathrm{visit}}=128\times32=4096.
\]

A issued：

\[
Q_{A,\mathrm{issued}}
=4096\times16\ \mathrm{KiB}
=64\ \mathrm{MiB}.
\]

B issued：

\[
Q_{B,\mathrm{issued}}
=4096\times32\ \mathrm{KiB}
=128\ \mathrm{MiB}.
\]

所以：

\[
Q_{\mathrm{TMA,issued}}=192\ \mathrm{MiB}.
\]

这不是 DRAM 一定读取 192 MiB；它表示 schedule 向 TMA/L2 请求了 192 MiB。ideal cold-entry unique input 仍是 16 MiB。

L2 empirical read/write ratio：

```text
read  = 192 MiB
write =  16 MiB
ratio = 12:1
```

EXP-04 `-i` surface 包含 12:1 `l2.duplex`。

## 7. Cold strict upper

### 7.1 LPDDR total

\[
T_{\mathrm{ext}}^{\mathrm{LB}}
=\frac{32\ \mathrm{MiB}}{273\ \mathrm{GB/s}}
\approx122.91\ \mu s.
\]

\[
P_{\mathrm{ext,ub}}
=\frac{17.179869184\ \mathrm{GFLOP}}
       {122.91\ \mu s}
\approx139.776\ \mathrm{TFLOP/s}.
\]

### 7.2 Tensor Core conditional upper

使用 FP16 conditional Tensor Core upper：

\[
T_{\mathrm{compute}}^{\mathrm{LB}}
=\frac{17.179869184\ \mathrm{GFLOP}}
       {258.5\ \mathrm{TFLOP/s}}
\approx66.46\ \mu s.
\]

### 7.3 Strict L2

minimum L2 read：

\[
T_{\mathrm{L2,R}}^{\mathrm{LB}}
=\frac{16\ \mathrm{MiB}}{1.6128\ \mathrm{TB/s}}
\approx10.40\ \mu s.
\]

minimum L2 write：

\[
T_{\mathrm{L2,W}}^{\mathrm{LB}}
=\frac{16\ \mathrm{MiB}}{0.8064\ \mathrm{TB/s}}
\approx20.80\ \mu s.
\]

因此 cold strict time lower bound 由 LPDDR total 主导：

\[
T_{\mathrm{ub}}^{\mathrm{LB}}
=\max(122.91,66.46,10.40,20.80)\ \mu s,
\]

\[
P_{\mathrm{ub}}\approx139.776\ \mathrm{TFLOP/s}.
\]

这是方向正确的条件上界，不是预测 candidate 会达到 139.776 TFLOP/s。

## 8. Hot-L2 strict upper

hot-L2 去掉 LPDDR total，剩余主要 upper 为 compute 258.5 TFLOP/s；L2 minimum read/write 时间更短。因此：

\[
P_{\mathrm{ub,hot}}=258.5\ \mathrm{TFLOP/s}
\]

在当前条件集合下成立。它仍是宽松外上界。

## 9. Current empirical layer

### 9.1 已有证据

- L2 12:1 duplex：已有 `-i` capacity；
- tc5a FP16 stride2048 historical exact TMA ingress：已有 scoped alias；
- TMEM readback：已有 component capacity；
- shared hard uppers：已有。

### 9.2 缺失证据

cold 场景：

- 1:1 cold result 只有 `hbm.duplex.proxy`；
- physical `hbm.duplex` 缺失。

hot/cold 集成包络：

- closure-qualified Thor causal profile 缺失。

所以 current output 应是：

```text
cold empirical resource layer = insufficient_evidence
causal pipeline layer         = insufficient_evidence
integrated empirical envelope = insufficient_evidence
```

即使 hot resource layer 的独立 component 可能已有数值，缺 causal profile 时 integrated envelope 仍没有数值。

## 10. Historical observation

2026-08-14 历史 Thor：

```text
tc5a median          120.039 TFLOP/s
same-precision cuBLAS 130.633 TFLOP/s
candidate/reference    91.89%
```

这些 observation 仍有效。旧 128.436 TFLOP/s 是 legacy independently-composed envelope，不是 current f06 integrated envelope。

## 11. 结论

本例同时得到：

- cold strict upper：139.776 TFLOP/s；
- hot strict upper：258.5 TFLOP/s；
- current integrated empirical envelope：`insufficient_evidence`；
- historical observed candidate：120.039 TFLOP/s；
- historical same-precision cuBLAS：130.633 TFLOP/s。

“严格上界有数值”与“经验包络缺证据”可以同时成立；这正是三层模型必须分开的原因。
