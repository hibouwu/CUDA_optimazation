# cfg_c producer-tail 增量扰动对照准备

固定2304×3072×4096、S4、cluster1×2、swizzle1、紧凑lda/alloc-lda4096、ldb3072、dyadic seed17。复用已通过全D检查且设备SASS未改变的binary；指定既有参考GPU-ef。此输入分配不同于旧738496的padding，不据此迁移旧计时。

manifest列出10对、20个目标进程，交替off/on与on/off顺序。每个进程先执行16次同buffer、同dual binary的observer-off前序。采样要求独占节点、实际分配设备与串行执行，不覆盖CUDA_VISIBLE_DEVICES选择未分配卡。

全部132CTA、两WG、全部输出参与判定：逐角色L，以及entry到producer-start、FIRST/MAIN/PERMIT/DONE/FINAL。每条时线的十个配对差先取绝对值，再要求中位≤64ns、最大≤256ns；不跨角色池化或减共同漂移。event配对绝对相对差中位≤2%、最大≤5%，off event总体CV≤5%。这些是沿用已有对照的工程上限，未从740088单对时长估计，不能解释为timer精度。增加共同端点检查防止L相消掩盖整条时线平移。

SM变化逐对保存，ns判定保留所有记录；原生cycle只作同一记录内差的诊断，不以clock拟合校正门槛。全部记录还需固定身份、严格解码、数值抽样检查。失败不剔除、补抽或调阈值。

本对照只能约束在现有dual基础上新增tail采样的增量扰动。off无tail戳，不能直接评价tail端点偏差；也不能授予dual基础观测、硬件包围、D可见或整kernel完成资格。已存在的观察失败不被覆盖。

当前已完成具体命令、串行采集器和逐系列判定器，尚未冻结或执行。判定器从两份旧smoke包各提取2556系列/132CTA并验证数值，但未用单对评价扰动。14项相关测试通过，包含端点共同平移但L不变、单个超限尾值、正负相消、SM变化保留、缺失/重排/错坐标/bool flag拒绝及event/CV失败。采集器先以setup核对实际CUDA设备，再按清单运行20目标进程，不重试；分析器要求完整冻结链、归档回执和完成标记。运行前仍须完成独立规格审查、远端独占执行脚本和完整冻结包；没有正式结果。正确性审查见前一correctness目录的main-producer-tail-correctness-review-v1。
