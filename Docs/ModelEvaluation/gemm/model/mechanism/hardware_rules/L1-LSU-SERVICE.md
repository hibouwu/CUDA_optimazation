# L1 cache与普通LSU服务

更新：2026-10-17。阶段8，有限范围与预算见[PLAN](MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)，旧窗口、新探针与现行开发筛查见[L1-01](../../../experiments/gh200_sm90/Q13-l1-lsu/L1-01-replay.md)。

## 1. 实际程序需求

冻结cfg_b的普通global／local及参数读取见[程序图 §2](../program/cutlass_program_graph.md#2-对象操作与依赖)。`cfg_b_program.ordinary_lsu_demands(program)`保存直接alpha标量条件、未激活LDG分支、零local需求、静态LDC／ULDC和覆盖关系。请求级组合的`ordinary_lsu=True`仅添加该需求报告；普通global／local请求为0，参数读取继续使用已有启动、坐标／控制和复合服务时间。

A/B输入和D输出沿[TMA服务](TMA-SHARED-SERVICE.md)；普通L1探针的加载价格没有加到这两条矩阵路径。EXP-18的local／spill样本保留其实际生成代码与窗口来源。

## 2. 条件证据与工作假设

| 路径 | 已有可用关系 | 当前规则或待判别响应 |
|---|---|---|
| 普通global依赖返回 | R01的.cg单lane链和4warp的1／4条流，保持原消费者与循环窗口 | 每条链的下一地址来自上一次加载；多链响应约束发出／返回重叠，使用原条件窗口 |
| 普通global cache对照 | EXP-13的read.ca／read.cg最终SASS同体，仅加载修饰不同；small／large完整窗口 | 旧完整窗口保留；新探针已取得工作集／请求carveout与多链的条件响应，实际分区容量与独立价格仍未辨识 |
| local与spill | EXP-18显式local／spill压力的32／128 word完整窗口 | 窗口含控制、输出排空与CTA会合，保留条件价格；cfg_b无local请求 |
| 参数／常量 | cfg_b主kernel的c[0]读取与程序位置 | 保留已有复合时间覆盖，分项服务待匹配观测 |

新探针固定总逻辑读取量`256×W`，同一kernel内先遍历W，再计时。工作假设为：在可复用的小工作集，.ca获得局部缓存服务；扩大工作集或增加共享内存carveout后，这一收益减小；.cg作为绕过L1的配对路径。四条分开的链用于检验依赖返回受限时的重叠响应，SASS核对实际加载组织。现行开发实测已完成：小工作集.ca窗口低于.cg，大工作集差异低于原描述门槛；请求carveout变化主要作用于.ca，四链在固定工作量下均缩短窗口。具体范围与资格见[L1-01](../../../experiments/gh200_sm90/Q13-l1-lsu/L1-01-replay.md#2-新探针条件与观察量)；随后以同一程序的12开发点冻结`T=a_J+G*tau_class,J`，其中`G=256W/(16×128×J)`；每个J共用固定项与快／慢服务。原8个128 KiB留出全部通过时间及响应判据，支持hint0的.ca仍属快复用，hint100及.cg属慢服务；[冻结参数、完整预测与留出判定](../../../experiments/gh200_sm90/Q13-l1-lsu/L1-01-replay.md#4-原8条件独立留出)保留。规则限该程序、W、请求carveout及J，group服务包含地址、发出、返回与消费。

上述cache含义依据[PTX ISA 8.8 cache operators](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#cache-operators)。L1准备放在每次目标kernel内部；跨kernel暖机不作为L1已暖的证据。Hopper的L1／texture／SMEM容量关系及carveout属性见[Hopper Tuning Guide](https://docs.nvidia.com/cuda/archive/12.9.1/hopper-tuning-guide/index.html#unified-shared-memory-l1-texture-cache)；探针记录请求的carveout百分比和实际资源，实际分区容量尚未观测。

## 3. 接入与剩余缺口

当前交付为cfg_b实际需求、旧条件窗口、可采样探针及静态构建。六个小程序启用需求报告后原请求、节点时刻及端点保持。原开发清单已完成目标实测、完整checksum及独立评分，目标工具链的资源／加载组织已核对。原8留出已验证条件group服务时间，最小验证收口。实际cache分区容量、物理流量与单load价格仍缺证据；cfg_b零普通global／local需求保持。
