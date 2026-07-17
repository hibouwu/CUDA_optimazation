# 学习编程模型

## 硬件和软件模型

一个 Grid 是一次 Kernel 启动所创建的全部线程任务的逻辑集合，它本身不是 GPU 分配的硬件资源，而是由许多 CTA 组成，并由 GPU 调度器分批分配到不同 SM 上执行。

CTA Cluster 是若干 CTA 组成的可选协作组，其中所有 CTA 会被保证同时调度到同一个 GPC 内，从而能够进行跨 CTA 同步和访问分布式共享内存。

CTA 与 Thread Block 是同一个概念，它由一组可以共享 Shared Memory 并进行同步的线程组成；一个 CTA 只能完整地驻留在一个 SM 上，不能跨越多个 SM，但一个 SM 可以根据线程数、寄存器和 Shared Memory 占用同时驻留多个 CTA。

CTA 中的线程会按照每 32 个划分为一个 Warp，Warp 是 SM 进行线程调度和指令发射的基本单位。Thread 是 Warp 中最小的逻辑执行实例；

Warp 发射指令后，每个活跃 Thread 作为一个 lane 在相应的硬件执行单元上执行。对于普通 FP32 或 INT32 运算，可以近似理解为一个 CUDA Core 处理一个 Thread 的数据，但 Thread 不会固定绑定某个 CUDA Core；执行访存或矩阵指令时，使用的则是 Load/Store Unit 或 Tensor Core。SM 内部的硬件资源是共享的，所有线程都可以访问这些资源，但每个线程只能访问自己分配到的寄存器和 Shared Memory 的一部分。

## 硬件参数

| 层级        | Thor T5000/DRIVE AGX Thor | RTX 5070 Laptop | RTX 5070 Desktop |
| ------------------- | ---------: | ------: | ---------------: |
| Compute Capability  |     `11.0` |          `12.0` |           `12.0` |
| GPU                 |             1 |               1 |                1 |
| GPC                 |                 3 |             约 3 |                5 |
| TPC                 |          10 |              18 |               24 |
| SM                  |              20 |              36 |               48 |
| SM/TPC              |            2 |               2 |                2 |
| SM processing block |           4/SM，共 80 |      4/SM，共 144 |       4/SM，共 192 |
| Warp Scheduler      |         4/SM，共 80 |      4/SM，共 144 |       4/SM，共 192 |
| CUDA Core           |    128/SM，共 2560 |   128/SM，共 4608 |    128/SM，共 6144 |
| Tensor Core执行单元 |    4/SM，理论共 80* |      4/SM，共 144 |       4/SM，共 192 |
| RT Core             |      未公开/不强调 |       1/SM，共 36 |        1/SM，共 48 |
| Texture Unit        |      4/SM，共 80 |      4/SM，共 144 |       4/SM，共 192 |

## 软件参数

| 执行单位              |       Thor | 5070 Laptop | 5070 Desktop |
| ----------------- | ---------: | ----------: | -----------: |
| Warp 大小           | 32 threads |          32 |           32 |
| 最大驻留 warp/SM      |         48 |          48 |           48 |
| 最大驻留 thread/SM    |       1536 |        1536 |         1536 |
| 最大驻留 CTA/SM       |         24 |          24 |           24 |
| 全 GPU 最大驻留 warp   |        960 |        1728 |         2304 |
| 全 GPU 最大驻留 thread |     30,720 |      55,296 |       73,728 |
| 单 CTA 最大 thread   |       1024 |        1024 |         1024 |

