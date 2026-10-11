固定 prepare3 的 CUDA12.9.41 返回物已核对：857 个 source 文件与返回 manifest 一致，两个 binary/SASS/resources/log 均与返回归档逐字匹配。job740240 的记录分配为 4 CPU、16 GiB、无 GRES，exit code 0。B 只读本地返回物，没有 GPU、SSH/Slurm 操作。

| stages | H | 首次try | 再次try | 慢路径入口/返回 | B | collector末尾 |
|---|---|---|---|---|---|---|
| 4 | 0x1e10 | 0x1e40 | 0x1e60 | 0x5830 / 0x5890→0x1e80 | 0x1ea0 | 0x20f0 |
| 6 | 0x1d10 | 0x1d40 | 0x1d60 | 0x5770 / 0x57d0→0x1d80 | 0x1da0 | 0x1ff0 |

H 后才计算 barrier 地址/phase 并 try；首次成功直接跳到汇合点，再次失败进入慢路径。慢路径执行 try、必要时 NANOSLEEP/PHASECHK，失败重试，成功返回汇合点；选中线程随后读取 B。H 的谓词包含目标线程/输出、开关以及 unsigned(group−2)<15，即 local2～16。两条路径都未把 barrier 检查提前到 H 之前，累计算术均在 B 之后。原始指令见 [stage4](stage4-instructions.txt)、[stage6](stage6-instructions.txt)。

B 至 collector 末尾有 38 条静态指令，包含 B；校验谓词和溢出分支会改变实际执行条数，不能把 38 当固定周期成本。该区域没有 global/local/shared store，在下一次 WARPGROUP.ARRIVE 前汇合；Q 仍含地址/phase 准备、try/wait 控制、汇合和时钟影响。record 与汇合开销归到残余，但会改变后续等待和整个 WG 推进。

两组 GEMM 均 REG168、STACK0、LOCAL0；构建日志中的所有 kernel spill 均为零，静态 HGMMA 各16条。资源一致和 SASS 顺序只能支持编译布局与源码分段，不能证明实际延迟、无扰动、旧 binary 等价或模型资格。主线的指令地址审查已独立匹配；详见 [报告](B-q-early-sass-review.json)。
