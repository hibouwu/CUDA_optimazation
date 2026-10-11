# 主集成独立复核cfg_c完整rank图

接入C d6422c9为a23bdcaa，审查源/full接合、两WG尾部wait/release、连续四stage代次、producer tail与输出可见性分离。独立运行44项相关测试全部通过；重新构造B保存的12份源模板，完整图gzip与C正式v2逐字节相同。

覆盖26静态条件、1556有效物理cluster、7148rank输出；核对320归档B输入，连同模板、源码与当前模块共341输入SHA全部匹配。总报告除三份模块的工作树路径外逐字段相同，三份模块SHA也一致。V09仅映射既有静态几何，不重评分。

从接入前提交e7f862e直接加载旧output_model，cfg_a S6/K64/T1,T2,T8完整图、cfg_c T1,T4,T12本地输出图以及cfg_a arrival routes全部逐字段相同，见regression.json。

cfg_c唯一开放入口为cluster_initialized；源接口原空前驱通过并集合并，不覆盖acquire、expect_tx、acquire_return依赖。本地B迟延可经四个empty arrival影响伙伴四stage后的复用。最后输入producer tail等待全部槽；两个WG独立tail_wait0/release→MAIN，不将角色端点强制同时。source_safe_exit与D_visible/cuda_complete分别保留。

本产物是固定256×128×64、cluster1×2、S4、Ktiles>=2的逻辑图；没有服务价格、实际entry时间、观测资格或新预测分数。Ktiles=1和idle cluster未补造图，不改变原cfg_a/b/c完整任务范围；超出接口覆盖仍须显式处理。下一步供给候选可接到真实源/目的节点做条件检验，但不能把未定入口或服务当零成本预测。
