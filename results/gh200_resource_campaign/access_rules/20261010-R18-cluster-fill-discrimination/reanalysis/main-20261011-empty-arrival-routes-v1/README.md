# 主线复核 cfg_a empty arrival 端点

C d91e65e合入为4a31199。主线独立编译并执行冻结spread_arrivals_to_warpgroup函数，确认每源rank的实际发送线程128/256→dest0、168/296→dest1。8条cluster emit routes、源码/host程序/二进制SHA、归档SASS位置及旧图投影报告与C-v6完整相同。

新增节点只细分emit、目的可见和完整producer_acquire返回，不能将完整调用返回戳解释成裸empty wait返回。原T1/T2节点投影不变；B的12份原子源组合图也独立检查，新增节点投影后旧依赖全部保留，其余旧字段完全相同。

28项output dependency/lifecycle/cfg_a供给测试通过。CPU函数执行和SASS位置只证明静态映射，尚无cfg_a sender观测或局部时延资格，不借用cfg_b四warp周期门槛。
