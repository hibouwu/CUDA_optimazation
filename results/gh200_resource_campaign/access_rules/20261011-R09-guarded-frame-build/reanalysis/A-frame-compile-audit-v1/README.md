# 740070实际CUTLASS编译接口：A独立只读审查

编译接口的有界检查通过。审查未执行任何二进制或CUDA调用；runtime_qualified和observation_qualified均false，不能据此启动正确性、pilot或30进程。

逐一核对860份源文件、冻结脚本/命令/接口、返回包身份、四个二进制与SASS SHA；原三个overlay与20261010失败控制源码逐字节相同。四个实际CUTLASS kernel模板符号相同，都是cfg_b/stage6。

| 实际kernel | 编译寄存器 | barrier | stack/spill | 编译静态shared | 动态寄存器池 |
|---|---:|---:|---:|---:|---|
| plain | 168 | 2 | 0/0 B | 1024 B | consumer232/producer40 |
| matched_dual | 168 | 2 | 0/0 B | 1024 B | consumer232/producer40 |
| guarded_bracket | 168 | 2 | 0/0 B | 1024 B | consumer232/producer40 |
| plain_prefix | 168 | 2 | 0/0 B | 1024 B | consumer232/producer40 |

四kernel HGMMA16/TMALD3/TMAST8/arrival6/wait2一致。这些是静态指令数量，不代表动态次数或同步延迟相同。

FIRST实际读取PC为1a60/1af0/1ba0，MAIN为24e0/2570/2620。独立从原SASS提取：CLOCK与GT的高低两半比较、成功BRA、初始化为零的计数器、GE2判断/回边、失败全零块及跳过最后读的出口全部符合原候选；没有SEL替代控制。最终CLOCK不重试，若为零须由reader保留为无效。FIRST三个64位存储相距0/48/80B；MAIN为136/184/216B，对应原clock、GT及新after字段，保留字段10/11。

SASS R5来自TID.X，P0=(R5&127)!=0或R5<128，FIRST/MAIN均以P0跳过；384线程下仅128/256进入。两处之间未改写该TID或写者谓词。原同步和工作overlay字节相同；观察增加的执行路径仍可能扰动设备，静态通过不证明计时有效。

动态资源输入另外核对：overlay与plain的SharedStorage定义逐字节相同，四kernel模板类型相同；启动和occupancy源码均使用GemmKernel::SharedStorageSize、384线程。编译报告的1024B是静态shared，不能当动态启动字节数。本阶段没有求出动态SharedStorageSize的数值、执行cudaFuncSetAttribute或查询运行时occupancy，相关字段显式false。相同类型与寄存器池只是编译输入相同的证据。

完整来源和逐指令断言见audit.json、audit.py。该程序只适用于本次冻结二进制和PC，不是通用实际kernel SASS检查器；重复运行只复验同一JSON，不修改主集成文件。

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-R09-guarded-frame-build/reanalysis/A-frame-compile-audit-v1/audit.py
```

源码、二进制、前序模块、raw SHA到最终reader的完整拒绝链另由主集成接入；本审查未将packet验证视作归档身份绑定，也没有把cfg_b编译结果迁成cfg_a的观察资格。
