# Scheduler 到 Kernel/Collective 的任务交接检查点

## 范围与实际推进

上一轮完成三类方案差异走查并运行了Host布局检查，属于有实质状态和证据变化的推进。本轮没有继续扩大案例库，而是补齐主线此前只有概述的任务交接接口。

固定提交不变。通过已有声明账本关联了 `sm100_tile_scheduler.hpp` 的13个独立声明节点：WorkTileInfo别名、初始任务、CTA坐标、两种fetch_next_work、K迭代/计数/起点、两种compute_epilogue、requires_fixup、六参数fixup及continue_current_work。全部明确使用字节范围、限定名和种类选择，未使用手工伪造声明身份，也没有重跑全库提取。

新增17条关系（16个Kernel直接调用点及1条类型关系）；原 `contract.edge.kernel_fixup` 的身份和调用点保持，目标从历史占位转到具体六参数方法，并给出选中完整K分支的源码依据。没有为同一个物理调用另外增加重复边。总计619节点、994关系；原824文件及全库未完成义务不变。

三张过程总图分别覆盖工作描述/坐标/K、各角色的下一任务交接、Epilogue判定/空fixup。旧逐边详情仍保留；新默认入口不是每两条边切片。设备交接总图也改为汇总同一operator()实体的两个图角色，避免只按某个节点ID遗漏新增调用。

## 对方案验证有影响的事实

1. **WorkTileInfo到CTA坐标不是元素寻址。** 输出为M/N/L的CTA级Tile索引，K位置为占位符。下游还要结合CTA形状、布局与步长；不能把它直接当作矩阵元素坐标或未拆分的MMA Tile坐标。
2. **本选择按完整K计算。** get_work_k_tile_count返回size(ceil_div(K,TileK))，没有读取WorkTileInfo来拆分K；get_work_k_tile_start为0。主线K=128、CTA TileK=64，据源码公式推导2个K块，不是两个独立归约任务。
3. **五个下一任务调用点分别存在。** Mainloop输入、Scheduler角色、MMA、Epilogue输入、Epilogue输出分别在664、701、741、819、877行调用三参数fetch。单参数重载不消费CLC，未给它虚构当前Kernel调用。
4. **当前与下一任务必须区分。** MMA和Epilogue输出先取得next，仍完成current，随后才替换工作描述并检查有效性。next无效不能据此跳过最后一个current。
5. **Epilogue输入的更新时机不同。** 它先保存旧compute_epilogue判定，再取next并提前替换work_tile_info；真正输入搬运仍使用旧cta_coord_mnkl，本轮完成后才更新坐标。没有将所有角色套成同一条状态更新时间线。
6. **fixup不等于一定做归约或等待。** 这里实际调用的六参数方法仅原样返回累加器消费者状态。两个compute_epilogue重载都返回true，分别用于输入/输出角色；这也不是WorkTileInfo或元素边界检查。

## 寻找反例与修复

本轮是实现者的源码反向自查和回归，不宣称独立使用者已经完成验收。

- 初次将双参数compute_epilogue视为未调用兼容入口。沿Kernel全部同名调用反查后，在816行找到Epilogue输入分支的真实调用，已单独纳入，并保留910行单参数调用。
- 声明选择器初次去掉了函数体左花括号前的空格，使12个方法与账本的签名字节区间差一字节。构建保留为明确未解析，没有改成symbolic放行；按固定源码/账本的范围语义修正后，13项全部关联到已有规范声明。
- 对Epilogue输入源码反查，发现其work_tile_info赋值早于实际load。已纠正调用条件与文字，不把MMA/Epilogue输出的更新顺序泛化给输入角色。
- 原operator()设备图只收集`contract.api.kernel`角色。新增调用使用同一声明的`host.kernel.operator`角色，现按entity_id收集已发布出边，同时保留所有原边ID；图中两个角色明确标注为同一声明，不伪装成两个重载。
- 复用原fixup调用关系消除了本轮最初重复记录同一物理调用的风险；旧占位节点只保留为历史边界，不再作为该调用的当前目标。

## 验证与未完成项

Scheduler专用12项检查覆盖声明范围/身份、重载参数、五个独立fetch位置、两个Epilogue重载、fixup原调用身份、各角色current/next顺序、完整K公式和过程图覆盖。23项总图回归和原页面/参数/主线回归继续执行。三张新增SVG已做静态图像检查，行端点、文件相对路径、调用位置和类型线可辨认。

这些证据只覆盖已列声明与Kernel直接交接；CLC发布、完成、无效响应、swizzle/raster内部与GDC尚未完整提取或验证。没有证明全工作唯一性、任意形状尾块安全、其他调度器、GPU运行或性能。真实浏览器和独立使用审查继续未完成；全库声明阻断项仍保留，不能将此检查点视为全库完成。
