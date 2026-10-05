# 全库逐模块边界：固定文件归属与跨模块关系

本次建议基于固定提交 `8f50b052e1099fb982392a622caab69b97b63128` 的 `data/scope.json`、实际 include 目录、当前 Dense 四份模块数据和定点源码核对。随后按 root 授权新增独立登记脚本及测试；未修改快照、既有 Dense 数据、根构建脚本、模板或 PROGRESS。

结论是采用两套互不替代的划分：**824 个物理文件各有一个主归属模块；Dense 是穿过这些模块的一条配置验证路径。** 不把“被 Dense 引用”解释为“归 Dense 所有”，也不把一个代表配置已经闭合的调用链解释为整文件、整个 API 家族或全模块已经覆盖。

## 1. 固定分母及可复核分类规则

登记输入为 `data/scope.json`，SHA-256 `2b8a0094ff40dcd0a47c4da64e309fda8af98dc1ecdeb9b28a6e8a8692c2970f`。它包含 CUTLASS 712、CuTe 112，共 824 文件。后缀分布为 `.h` 494、`.hpp` 290、`.inl` 40。

本机只读复查发现，默认 `rg --files snapshot/include/cutlass snapshot/include/cute` 只列出 784 文件，遗漏恰好是全部 40 个 `.inl`。使用 `rg --files -uuu` 后路径集合与 manifest 的 824 项一致。因此登记脚本不以 ignore-aware 文件搜索、头文件后缀、Dense include 可达性或某个架构激活结果作为分母。它只读取 manifest；这些搜索行为不构成重新建立阶段 0 的理由。

规则的可执行定义及每文件命中的 `rule_id` 已写入 `data/module-registry.json`。匹配不是“谁排前面谁赢”：每条规则独立计算 include/exclude，恰好命中一条才接受。未知目录、非法路径、重复路径、重叠规则及删除文件缩小固定 712+112 分母均报错。以下是实际生成结果，不是 API 数量或完成率。

| 规则 | 主模块 ID / 归属 | 文件数 | 范围规则摘要 |
| --- | --- | ---: | --- |
| R01 | `sync_resource` / Pipeline、Barrier、资源生命周期 | 11 | pipeline 3 文件，加两个顶层barrier/semaphore、arch/barrier与grid_dependency_control、detail/cluster、CuTe两个cluster文件和TMEM allocator |
| R02 | `runtime_bridge` / Host 与 CUDA 启动桥接 | 7 | device_kernel、kernel_launch、cluster_launch、cuda_host_adapter、两个kernel_hardware_info文件、workspace |
| R03 | `arch_primitives` / 架构原语与启用条件 | 30 | cutlass/arch，扣除R01例外；加cute/arch的config、util、simd_sm100 |
| R04 | `collective_support` / 共享 Collective 支持 | 11 | cutlass/detail，扣除cluster和下述3个基础文件，包含collective子目录 |
| R05 | `gemm_interface` / 问题、枚举与调度类型 | 4 | cutlass/gemm直接子文件 |
| R06 | `gemm_device` / Device 接口 | 30 | cutlass/gemm/device |
| R07 | `gemm_kernel` / Kernel 与 Tile Scheduler | 115 | cutlass/gemm/kernel |
| R08 | `gemm_collective` / Mainloop 与 Builder | 84 | cutlass/gemm/collective，包含30个.inl |
| R09 | `gemm_tiles` / Thread、Warp、Threadblock组件 | 83 | gemm/thread、warp、threadblock；名称不作代际判定 |
| R10 | `epilogue_collective` / 输出 Collective 与 Builder | 24 | epilogue/collective，包含6个.inl |
| R11 | `epilogue_fusion` / Fusion 与 EVT | 13 | epilogue/fusion |
| R12 | `epilogue_tiles` / 输出 tile 组件 | 95 | epilogue直接子文件及thread、warp、threadblock |
| R13 | `cute_mma` / MMA分派、Atom、Traits、包装 | 34 | cute/arch/mma*、atom/mma*，加algorithm/gemm及cooperative_gemm |
| R14 | `cute_copy` / Copy、TMA、TMEM搬运 | 24 | cute/arch/copy*、atom/copy*，加copy、cooperative_copy、prefetch算法 |
| R15 | `cute_core` / Tensor、Layout与基础设施 | 48 | CuTe直接子文件、容器/数值/util、其余通用algorithm及atom/partitioner |
| R16 | `convolution` / Convolution | 98 | cutlass/conv，包含4个.inl |
| R17 | `transform` / Transform | 34 | cutlass/transform |
| R18 | `reduction` / Reduction | 11 | cutlass/reduction |
| R19 | `distributed` / Experimental Distributed | 8 | cutlass/experimental/distributed |
| R20 | `cutlass_foundation` / 数值、容器、布局、兼容基础 | 60 | 其余CUTLASS直接子文件、layout/platform/thread；detail/helper_macros、dependent_false、layout |
| 合计 | **20 个主模块** | **824** | 未归属0、重复归属0，仅为文件归属闭合 |

这是稳定归档边界，不要求一次处理完115文件的Kernel模块。模块内部仍可按接口、源码区间与候选ID形成小工作包；架构、执行粒度、数值路径、API代际是经核查后附加的导航维度，不拆出互相重复的文件分母。若以后决定进一步拆分大模块，应版本化规则并重新验证824项单归属，不按“已经读过”与“还没读过”划模块。

## 2. 为什么这些边界需要源码例外

- `gemm/device/gemm_universal_adapter.h:69–78` 明确说明同时支持2.x/3.x；`:123` 和 `:630` 是不同特化。该文件主归属始终为 `gemm_device`，但实体、特化和条件分别登记。`.h` 不能据此等于legacy，单独通过3.x路径也没有覆盖另一特化。
- `gemm/kernel/sm100_tile_scheduler.hpp:135–160` 有GEMM形状的参数转换，而 `:163–171` 又声明卷积专用重载。因此文件主归属 `gemm_kernel`，卷积消费者仍跨模块连接到这个精确重载，不能再复制到卷积模块伪造第二个文件所有者。
- `detail/layout.hpp:53–191` 包含RowMajor/ColumnMajor到Stride的映射和TensorNHWC等卷积布局特化。它归基础布局，GEMM/卷积的 `template_binds`、`aliases` 或 `type_uses` 保留跨模块关系。`detail/mma.hpp:42–83` 则判定稀疏/block-scaled操作类，归共享Collective支持。
- `detail/sm100_tmem_helper.hpp:58–73` 决定累加器形状、overlap条件和stride，Dense Mainloop `:475` 实际使用。它归共享支持，TMEM分配器归资源模块，MMA fragment类归CuTe MMA；三者通过真实类型和资源关系连接，不因为都涉及TMEM而把声明合成一个API。
- `cute/atom/partitioner.hpp:48–97` 的 `TV_Tiler`/`TV_Partitioner` 是泛型线程—值布局划分，不是MMA专用包装，故归 `cute_core`。反过来 `cute/arch/mma_sm100_desc.hpp` 的真实UMMA descriptor类型归 `cute_mma`，不是因为只有数据字段就降级成“非API结构”。

主归属只回答“哪个文件模块负责把这个源文件完整记账”。它不改C++ namespace、成员owner、friend注入位置、链接属性或实体身份。宏定义的源文件、宏调用点所在文件、生成声明的物理调用span必须各保留原身份；定义模块和调用模块不同不是重复文件归属。

## 3. 承接 Dense 的最短扩展路径

已有Dense数据应作为跨模块、固定配置的证据覆盖层继续复用，而不是另抄一份接口图。`module.json` 已将其声明为 `representative_design_path_not_global_coverage`，且没有runtime执行声明。

建议按以下顺序形成下一批小工作包；这是工作优先级，不是全模块通过顺序。

1. **先闭合已有源接口的明确缺口。** `gemm_device` 与 `runtime_bridge` 可复用 `host.json` 的生命周期、重载、Arguments→Params、launch桥接和外部CUDA边界；优先补该文件已列明的其余重载/特化与可用分支，不能把occupancy查询当成kernel发射。随后以 `gemm_kernel` 为owner补现有Scheduler接口的参数下沉与工作状态，不先横扫所有Kernel模板。
2. **把当前TMA搬运链按Copy模块贯通。** `cute_copy` 可从当前contracts中的TMA load/store、TMEM load和descriptor绑定继续。比如 `copy_traits_sm90_tma.hpp:283–301` 的两个 `.with` 重载保留不同descriptor来源；`:313–319` 删除的 `copy_unpack` 说明“未绑定对象”不能当作可执行copy。需要把原始Traits、绑定后的Traits、Copy_Atom、算法rank分派及架构wrapper分别记录。
3. **资源协议复用后按同一资源向外扩展。** `sync_resource` 已有Dense输入SMEM、累加器TMEM、输出SMEM及分配释放的有限资源契约；继续补未选分支、明确参与者数量、phase/stage、事务完成主体和测试分支。跨 `cute_copy`/`cute_mma` 的提交原语与等待/释放不是同一事件。现有 `pipeline_init_wait` 全局owner问题提示：复用前要核对物理namespace与完整签名，不能仅复用一个已经生成的entity_id。
4. **完成输出算子与CuTe底层参数边界。** `epilogue_collective` 的 `to_underlying_arguments` 在 `sm100_epilogue_tma_warpspecialized.hpp:315` 实际下沉到FusionCallbacks，`:323–331` 又转发workspace接口；这是进入 `epilogue_fusion` 的具体边界。`cute_mma` 则已由 `gemm_dispatch.json` 提供本例83→275→142→190→Atom94→friend2072与实际Tensor类型，不需重新猜测。下一包可核对该文件其它原始重载与被选择条件之外的源候选，而不是复编译同一正例。
5. **从现有差异点扩到其它操作域。** Convolution有独立 `conv::collective::CollectiveBuilder`（`conv/collective/collective_builder.hpp:82`），其StageCount/KernelSchedule标签位于conv namespace，不能合并到同名GEMM标签；Transform和Reduction有独立Host入口；Distributed引入Graph和跨设备协议。先选择一个明确入口及必要跨模块依赖，不把它们仅当Dense的名字替换。

即使其他文件仍存在声明诊断，也能开展这些范围内的关系、协议和离线表达工作。进入一个工作包前只需核对它使用的实体、签名、候选、条件和直接依赖；若相关声明确实解析错误，修这个实际阻断点或保留明确pending。模块整体声明/关系通过仍要求其全部所属文件义务对账，但不是启动下一工作包的先决条件。

## 4. 冷文件、兼容层和.inl不得消失

“当前Dense未用到”不是“无接口”。反例包括：

- `experimental/distributed/device/dist_gemm_universal_wrapper.hpp:53` 声明 `DistributedGemmUniversalAdapter`；`:491` 的 `construct_graph` 是有CUDA版本条件的Graph构建；`:624–628` 与 `:677–691` 分别连接底层DeviceGemm和Graph发射。主归属distributed，跨Host runtime与GEMM接口，不能套用普通Adapter::run语义。
- `transform/device/transform_universal_adapter.hpp:54–59` 明确在 `cutlass::transform::device` 声明独立Adapter；`:206–209` 绑定TransformKernel的device_kernel并调用ClusterLauncher。它include GEMM或注释提到GEMM并不能改变API的实际归属。
- `reduction/device/reduce_split_k.h:51` 的 `ReduceSplitK` 拥有自己的Arguments和workspace；`:180–201` 为独立run/launch/error路径。它可消费GEMM Split-K产生的工作区，但不是Mainloop内部实现。
- `cutlass/thread/matrix.h:47–95` 声明每线程Matrix模板、成员类型和别名；是否能在特定配置编译需要另验，不能因为冷门、可能存在源码问题或没有入当前调用链而删除候选。
- `cute/util/print_svg.hpp:48–87` 有color对象和 `print_svg_mma` 模板，且依赖MMA/Copy Atom。它属于范围内的Host/Device工具接口，而非仓库外图集工具；同理print_latex、debug和兼容宏都保留主归属。

40个.inl全部归拥有Builder入口的操作域：GEMM30、Epilogue6、Conv4。它们的主模板/偏特化/条件/帮助函数照常进入声明和关系账本，不因为被其他header include就把文件折叠或只计入口。源码中的 `#if 0`、旧CUDA/Host compiler兼容分支以及平台替代namespace保留variant，不以当前工具链是否激活做裁剪。

冷文件队列建议按尚未对账的文件/候选ID建立，而不是按当前Dense调用闭包定义。当前四份Dense数据的**节点path和边evidence.path**在manifest中相交得到33个源文件，另有791个没有出现在这两个字段集合中；这只是特定引用口径，不包含所有compile_evidence和include闭包，既不是覆盖率，也不能直接等同“791个完全未接触的文件”。

## 5. 已有资产如何复用、哪里不能盲信

- `scope.json`、阶段0检查与固定snapshot可直接作为分母、内容指纹和源码锚点；登记模块不重复生成快照。
- `declarations.json` 中带physical span、raw signature、variant/宏来源的记录可作为模块候选实体索引。当前机械报告明确只通过身份/位置自洽，不等于语义完整；636项blocking诊断仍留在原模块/文件范围，不能因新导航归属而消失。
- `candidates.json` 与 `candidate-mapping.json` 保留原始470518候选及各项义务。当前summary有12062项pending，并单列6479项文件include义务、192353项表达式关系义务；已经有依据判为表达式的项目转入关系工作，不能为了模块启动重新当成声明失败，也不能当作全库关系已经完成。
- Dense `host.json`、`type_path.json`、`contracts.json`、`gemm_dispatch.json` 的现有API/边ID、源码锚点和配置证据复用；同一个API跨视图引用，不以归属模块新建同名拷贝。`run-334xibhm` 和最终dispatch probe是固定配置静态证据，不能推出其他Template绑定或runtime结论。
- 当前离线源码页、签名面板、逐边PlantUML/SVG、链接/DOM回归可复用为展示设施；实际浏览器验收未完成这一边界原样保留，导航登记完成不改变它。

建议后续跨模块关系至少保留source/target实体或参数端点、callsite/occurrence、物理文件owner、condition、配置bindings、证据和缺口。边由源调用/声明出现位置的文件模块负责对账，目标只链接其规范身份；资源契约另用配置与资源字段确定实例身份。跨文件声明/定义属于同一实体时，不为满足单文件owner拆成不同实体。

## 6. 登记表交付与验证

新增文件为 `scripts/build_module_registry.py`、`tests/test_module_registry.py`、生成的 `data/module-registry.json` 和本报告。可重生成：

```bash
.venv/bin/python scripts/build_module_registry.py
.venv/bin/python -m unittest discover -s tests -p test_module_registry.py -v
```

schema v1 的 `modules` 提供 `module_id/title/role/rule_ids/primary_file_count/status/layer`；`files` 提供 `path/primary_module_id/rule_id/source_sha256/git_blob/git_mode/navigation/coverage`。`navigation`原样保留manifest的directory、architecture hints和pending API-generation标签，另以顶层版本信息提供导航；这些字段不判断接口代际。`rules`公开每个精确/前缀/直接子文件selector及排除项，`counts`公开模块、库、后缀与.inl分布。

9项测试通过，覆盖824唯一主归属、20模块计数、40个.inl、detail/同步例外、冷门兼容接口、混合2.x/3.x adapter、冲突规则拒绝、未知/非法路径拒绝、删除/重复manifest文件拒绝、导航标签不提升为语义结论。生成器以源manifest及自身指纹原子发布；没有运行全库提取器或关系重建。

登记表只声明文件归属已经分配，`api_coverage_claimed=false`、`relationship_coverage_claimed=false`，每文件声明/关系/协议状态均为 `not_asserted`。这不是文件/API关系覆盖完成报告，也不改变全库未通过的结论。
