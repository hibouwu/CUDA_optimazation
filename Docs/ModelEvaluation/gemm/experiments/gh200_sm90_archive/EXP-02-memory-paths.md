# EXP-02：GH200 SMEM 与全局访问路径

## 参数与矩阵

读取实际设备容量和驻留属性；测量单 CTA 的 SMEM scalar 服务，以及全 GPU 的请求数据吞吐。经验服务不作为物理上界。

| 家族 | 配置 | 范围 |
|---|---|---|
| SMEM read | stride=1/2/4/8/16/32 | 一 CTA、256 线程、32 KiB 分配，每轮每线程 8 次 scalar 读取 |
| SMEM write | stride=1 | 同上，每线程写不同位置 |
| global read | 请求 8 MiB，`.ca` / `.cg` 对照 | grid=4×SM数，256 线程/CTA，16 B/线程访问 |
| global stream read | 请求 256 MiB，`.cg` | 同上 |
| global stream write | 请求 256 MiB，`.wb` | 同上，结束前 device fence |
| global duplex | 输入、输出各请求 128 MiB，1:1 读写 | 同上，结束前 device fence |

12 个配置，每配置 10 次独立进程，轮内打乱次序。全局分配向上对齐完整 CTA 访问轮次，以实际 allocation 字段计算流量。SMEM 地址复用与 bank 映射随 stride 改变，不把分配大小当作每轮访问量。

## 计时与正确性

各 CTA 在操作准备后取 globaltimer/clock64，结束前同步；global 写路径包含 device fence。fence 约束访问顺序，不单独保证消费者可见，也不证明强制刷新到 HBM。初始化和输出检查排除在设备窗口外。

SMEM 报告请求 B/clock64 cycle/CTA；global 报告请求总字节除以最早开始至最晚完成的全网格区间。保留 CUDA event 对照、逐 CTA 时间戳和实际 SM 覆盖。

SMEM 使用 PTX `ld/st.volatile.shared`，外层迭代禁止展开，保留每轮 8 次访存；SASS 检查计时窗口内的向后分支与实际循环体。读取使用全部加载 lane 参与 checksum，由 host 按地址公式独立核对；global 写入另做全数组逐元素检查，SMEM 写入则核对每线程已写区域的 checksum。global 输入 seed 随外部 trial 改变；SMEM 使用固定的按地址变化的模式，写测试预填不同值以检测漏写。每次进程先做两次同路径预热。

## 接受条件

- GH200/SM90a、有效单 GPU Slurm 分配；
- 配置、源码、二进制、SASS、trial 哈希一致；
- 12×10 记录完整，数值、字节数、时间、覆盖与统计重算通过；
- 目标函数计时循环内保留 LDS/STS、LDG/STG；SMEM 循环体恰有 8 次对应访存；
- 完成记录绑定 summary，NCU 尝试和返回状态保留。

结果等级为 `timing_validated`。两次预热不证明热稳态；无计数器时不宣称缓存命中、物理 HBM 流量或饱和端口。device fence 也不等于强制将缓存写回 HBM。

## 运行与文件

| run ID | Slurm 作业 | 结果与用途 |
|---|---:|---|
| `20261001-memory-a` | 729292 | 编译失败，保留诊断，无性能数据 |
| `20261001-memory-b` | 729348 | 旧审查器通过；后续发现 SMEM load 被移出循环，整批不用于建模 |
| `20261001-memory-c` | 729365 | 修正后重测，12×10 记录通过含 SASS 循环检查的审查 |

`memory-b` 的原始完成状态保留；[后审判定](../../../../../results/gh200_resource_campaign/20261001-memory-b/post_audit.json)明确否决建模使用。具体证据是 shared-read 函数的 LDS 位于 `0x480–0x500`，实际重复的 `0x510–0x730` 只有整数累加，没有 LDS。数值正确、速率算术正确仍不足以证明目标访存执行了预定次数。

正式结果采用 `memory-c`；不把被否决批次与正式样本混合。

- [runner](../../../../../microbench/gh200_resource_campaign/run_campaign.py)
- [CUDA 探针](../../../../../microbench/gh200_resource_campaign/resource_probe.cu)
- [独立审查器](../../../../../microbench/gh200_resource_campaign/audit_campaign.py)
- [绘图脚本](../../../../../microbench/gh200_resource_campaign/plot_results.py)

新增结果保存到 `results/gh200_resource_campaign/<run-id>/`，失败目录保留，不覆盖历史运行。

## 当前实测与使用边界

正式 run：`20261001-memory-c`，作业 729365，`romeo-a057`，CUDA 12.9，132 SM；远端运行 44 秒且正常退出。本地再次核对原始 trial、冻结工件、计时循环、进度和 COMPLETE 绑定，summary 哈希不变。

设备 Runtime 返回 L2 cache **62,914,560 B（60 MiB）**、全局内存 **95 GiB**、每 SM SMEM **228 KiB**、CTA opt-in SMEM **227 KiB**。设备属性里的 peak clock 是峰值属性，不是本轮实际固定频率；原始遥测保留动态频率。

| 配置 | 请求吞吐中位数 | 单位 | 使用边界 |
|---|---:|---|---|
| smem_read_stride1 | 115.633 | B/clock64 cycle/CTA | CV 12.41%，暂用范围，待稳定性复测 |
| smem_read_stride2 | 63.981 | B/clock64 cycle/CTA | 给定访问组织的经验服务 |
| smem_read_stride4 | 31.996 | B/clock64 cycle/CTA | 给定访问组织的经验服务 |
| smem_read_stride8 | 15.999 | B/clock64 cycle/CTA | 给定访问组织的经验服务 |
| smem_read_stride16 | 8.000 | B/clock64 cycle/CTA | 给定访问组织的经验服务 |
| smem_read_stride32 | 4.000 | B/clock64 cycle/CTA | 给定访问组织的经验服务 |
| smem_write_stride1 | 127.919 | B/clock64 cycle/CTA | 给定访问组织的经验服务 |
| global_read_ca_8m | 13127.090 | GB/s requested payload/GPU | 给定访问组织的经验服务 |
| global_read_cg_8m | 7739.437 | GB/s requested payload/GPU | 给定访问组织的经验服务 |
| global_read_cg_256m | 3514.280 | GB/s requested payload/GPU | 给定访问组织的经验服务 |
| global_write_256m | 3824.902 | GB/s requested payload/GPU | 给定访问组织的经验服务 |
| global_duplex_128m | 3392.663 | GB/s requested payload/GPU | 给定访问组织的经验服务 |

SMEM stride=1 的范围为 **88.048–115.633 B/clock64 cycle/CTA**，样本波动较大，不能只拿中位数固定到通用模型。其余配置保留各自条件与分布，也不作为物理上界。全局请求吞吐按读写请求之和计量；duplex 的结果不是读和写分别各有该速率。

8 MiB `.ca` 请求吞吐超过 HBM 产品标称值，并不矛盾：计的是重复请求，不是物理外存流量；命中层级仍未用计数器证明。所有 global 配置均不命名为 L1/L2/HBM 物理带宽。循环、地址运算、整数 checksum 和必要同步也在计时内。

- [正式测量报告](../../../../../results/gh200_resource_campaign/20261001-memory-c/REPORT.md)
- [机器可读结果](../../../../../results/gh200_resource_campaign/20261001-memory-c/summary.json)
- [SMEM 与 global 图表](../../../../../results/gh200_resource_campaign/20261001-memory-c/plots/index.md)
- [Slurm 作业记录](../../../../../results/gh200_resource_campaign/slurm_accounting_20261001.txt)

## 语义依据

[PTX cache operators](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#cache-operators)定义 `.ca/.cg/.wb` 的缓存策略；策略本身不证明命中率。[CUDA memory fence](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#memory-fence-functions)区分顺序与可见性。本实验没有消费者握手或物理 HBM 流量证明。
