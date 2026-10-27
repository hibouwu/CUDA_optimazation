# COMPOSITION-01：cfg_b同版请求级系统

更新：2026-10-17。有限登记见[PLAN](../../../model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)，模块来源与覆盖的唯一入口为[组合](../../../model/mechanism/composition.md#2-模块与资格)。该同版离线交付轮新增GPU作业、参数拟合和模型候选为0；后续模块实测状态见下文。

## 1. 现象与固定接线

九方向已有独立证据及接口，但原12矩阵的状态化源、地址、频率和调用协议尚未由共同入口统一导出。直接把source／output回调放进参数时，`json.dumps(report)`报`Object of type method is not JSON serializable`。本轮只修接线和交付，原服务规则、参数、分配／放置假设、P0及plain映射固定。

`cfg_b_system.predict(input, profile, protocol)`从case／setup、A/B/D基址和分配、CTA分组、冻结服务参数及运行前频率模型推进。10次暖机和1次目标共用状态化源；目标经共同请求入口导出逐CTA／Ktile节点、请求就绪、计算／释放、输出、寄存器配额和活动窗。该归档版沿零额外控制时间核对full／empty的phase与计数。后续L2和A–E有限实测已完成，资格与结果分别维护于[L2-01](../Q03-source-service/L2-01-saturation.md)和[COMPUTE-01](COMPUTE-01-resident-chain.md)；尚无新服务价格替换到此版。

## 2. 回归与原评分

| 条件 | 本轮回归 | plain预测／实测 | 对照来源 |
|---|---|---|---|
| G3_base | 132CTA、50688请求、11调用 | 47.434986／49.376 µs | 原前缀版完整CPU时间线 |
| G3_stage4 | 同上 | 49.286443／54.544 µs | 原前缀版完整CPU时间线 |
| G3_A16 | 同上 | 47.479335／63.664 µs | 原完整时间线与固定前缀投影 |

三条件共152064个目标请求，原服务节点、请求完成、输出和11次调用内部窗口精确保持；新增配额节点沿零价启动假设。其余九条件复用旧开发成绩，G6保留原组级时间线及逐atom绑定0，未重新求解。结果、输入、源码和比较见[交付包](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261017-cfg-b-system-001/README.md)。

原12行、240个选中成功进程及12次失败尝试的身份与覆盖已核对。保存的前后两个版本共1440个逐进程分项重聚合，与旧评分精确相同；本轮没有重新解码raw。规格与V09对照保留在同一评分证据中。plain绝对误差中位／最大仍为10.224758%／25.422005%，关键CTA命中1/120；stage响应3.903147%对10.466624%，A16响应0.093495%对28.937137%，B80预测0对18.131614%。原5%／10%门槛均未通过，独立C2/C3没有新增证据。

保存源码的67项CPU检查通过，包括聚合逐位复现、六图接口同开、等价SMEM写块、HBM读miss替换和实际回调导出。首轮三项的plain与观测端点保持，但零价同步的前缀加减使内部请求相差1 ULP；修复后追加三项11调用，精确回归通过。首轮源码及输出保留在`initial/`。另一处入口修复将声明的A/B/D基址完整传入源；平移A基址的CPU反例通过，原12行A=0保持。

## 3. 复现与最小缺口

运行和校核命令统一见[交付包](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261017-cfg-b-system-001/README.md)。预测只读保存的input／profile／protocol；评分读取旧观测，两条路径分别执行。

当前结论：结构化的请求级工作假设可以完整运行并重放，原反例仍反证其对全部条件的充分性。后续L2六形式停止、A驻留窗口的条件循环量通过长度留出，B/C零额外控制具体形式未通过开发响应，D/E只有有限开发数值／路径证据。B/C及D/E的trace端点不满足向plain迁移条件，仍缺可辨识的返回／供给／输出价格；原12条件数值与门槛失败保持。L1的[36开发筛查](../Q13-l1-lsu/L1-01-replay.md#2-新探针条件与观察量)已完成，尚未导出通用价格；频率及C1支线保持暂停。

[Q05完成规则的G3迁移](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261018-Q14-compute-completion-transfer-001/EXECUTION.md)完成33调用，默认完整JSON精确复现。显式d95使S6／S4 plain为47.490720／49.342177 µs，stage响应3.898566%对实测10.466624%；每条件仅264个输出尾释放后移，普通释放及MAIN空间跨度保持。本轮收口，未扩大12条件或替换默认服务，原独立C2/C3缺口保持。
