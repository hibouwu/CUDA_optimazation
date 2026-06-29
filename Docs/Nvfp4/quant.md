# Quantization

## NVFP4

[https://arxiv.org/pdf/2509.25149](https://arxiv.org/pdf/2509.25149)

## FP4

| sign | exponent bits | mantissa bit |  value |
| ---- | ------------- | ------------ | ------ |
| 0    | 00            | 0            |   0 |
| 0    | 00            | 1            | 0.5 |
| 0    | 01            | 0            |   1 |
| 0    | 01            | 1            | 1.5 |
| 0    | 10            | 0            |   2 |
| 0    | 10            | 1            |   3 |
| 0    | 11            | 0            |   4 |
| 0    | 11            | 1            |   6 |

取值范围在 -6 到 6 之间，靠近 0 的值更密集，靠近 6 的值更稀疏。

## INT4

| bit pattern | 十进制值 |
| ----------- | ---: |
| 0000        |    0 |
| 0001        |    1 |
| 0010        |    2 |
| 0011        |    3 |
| ...         | ...  |
| 1110        |   14 |
| 1111        |   15 |

均匀分布在 0 到 15 之间。

## FP4 packed payload / PTX bit storage

| 格式       | 含义             | 总 bit 数 | 承载它的 PTX bit 类型 |
| ---------- | ---------------- | --------: | --------------------- |
| `e2m1`     | 1 个 FP4 值       | 4 bit     | 通常不能作为普通独立寄存器变量自然使用 |
| `e2m1x2`   | 2 个 FP4 值打包   | 8 bit     | `.b8`                 |
| `e2m1x4`   | 4 个 FP4 值打包   | 16 bit    | `.b16`                |
| `e2m1x8`   | 8 个 FP4 值打包   | 32 bit    | `.b32`                |

`.b8`、`.b16`、`.b32` 是 PTX 层的 bit 类型。它们本身只是 raw bits；只有在具体 FP4 指令语义下，payload 才会被解释成 `e2m1x2/e2m1x4/e2m1x8`。

例如：

```asm
.reg .b8 a;   // a 是 8-bit raw payload，可被相关 FP4 指令解释为 e2m1x2
.reg .b32 x;  // x 是 32-bit raw payload，可被相关 FP4 指令解释为 e2m1x8
```

## PTX register / PTX virtual register

PTX 里的 `.reg` 声明的是 PTX virtual register。PTX 不是最终硬件 ISA，而是会被 ptxas/driver 翻译到目标 GPU 的 SASS/native ISA。

PTX virtual register 后续会经过寄存器分配，映射到 SASS 层可见的寄存器，例如 `R0/R1/R2...`。

如果两个 PTX virtual registers 的生命周期不重叠，ptxas 可能把它们复用到同一个 SASS register slot。

## SASS register / register slot

SASS 里的 `R0/R1/R2...` 是硬件指令层可见的寄存器编号。对 CUDA 性能分析来说，通常把一个 R register slot 理解成一个 32-bit register slot。

`.b64`、`f64`、`u64` 这类 64-bit 数据通常会占用两个 32-bit register slots。

## register operand / 寄存器操作数

register operand 是一条指令参数列表里引用的寄存器。

例如：

FFMA R4, R1, R2, R3;

这里 R4 是目标寄存器操作数（destination register operand），R1/R2/R3 是源寄存器操作数（source register operand）。

寄存器操作数不是寄存器 payload 里的某个 byte 或某个 FP4 字段。

## MMA / Tensor Core operand

MMA/Tensor Core 指令通常读取一组 register operand 作为 fragment。

这些 register operand 的 payload 可以是 packed 格式。例如一个 .b32 payload 可以承载 e2m1x8，也就是 8 个 FP4 value。

Tensor Core 按指令规定的 layout 解释这些 bit，而不是把每个 4-bit FP4 当成独立寄存器逐个读取。

## 寄存器读写依赖

```asm
FFMA R4, R1, R2, R3;   // R4 = R1 * R2 + R3
FADD R5, R4, R6;       // 立刻读取 R4
```

| 类型  | 名字            | 例子               | 是否重要                     |
| --- | ------------- | ---------------- | ------------------------ |
| RAW | write 后 read  | 先写 `R4`，后面读 `R4` | 最重要，直接导致等待               |
| WAW | write 后 write | 两条指令都写 `R4`      | 有顺序约束，但通常由编译器处理          |
| WAR | read 后 write  | 先读 `R4`，后面写 `R4` | 通常不如 RAW 明显，编译器/调度会避免出问题 |
| RAR | read 后 read   | 两条指令都读 `R4`      | 没有数据依赖，但可能有寄存器端口/bank 压力 |


