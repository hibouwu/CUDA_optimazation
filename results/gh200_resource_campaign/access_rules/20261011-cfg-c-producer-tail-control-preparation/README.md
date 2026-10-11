# cfg_c producer-tail 增量扰动对照准备

固定2304×3072×4096、S4、cluster1×2、swizzle1、紧凑lda/alloc-lda4096、ldb3072、dyadic seed17。复用已通过全D检查且设备SASS未改变的binary；指定既有参考GPU-ef。此输入分配不同于旧738496的padding，不据此迁移旧计时。

manifest列出10对、20个目标进程，交替off/on与on/off顺序。每个进程先执行16次同buffer、同dual binary的observer-off前序。采样要求独占节点、实际分配设备与串行执行，不覆盖CUDA_VISIBLE_DEVICES选择未分配卡。

全部132CTA、两WG、全部输出参与判定：逐角色L，以及entry到producer-start、FIRST/MAIN/PERMIT/DONE/FINAL。每条时线的十个配对差先取绝对值，再要求中位≤64ns、最大≤256ns；不跨角色池化或减共同漂移。event配对绝对相对差中位≤2%、最大≤5%，off event总体CV≤5%。这些是沿用已有对照的工程上限，未从740088单对时长估计，不能解释为timer精度。增加共同端点检查防止L相消掩盖整条时线平移。

SM变化逐对保存，ns判定保留所有记录；原生cycle只作同一记录内差的诊断，不以clock拟合校正门槛。全部记录还需固定身份、严格解码、数值抽样检查。失败不剔除、补抽或调阈值。

本对照只能约束在现有dual基础上新增tail采样的增量扰动。off无tail戳，不能直接评价tail端点偏差；也不能授予dual基础观测、硬件包围、D可见或整kernel完成资格。已存在的观察失败不被覆盖。

当前已完成具体命令、串行采集器和逐系列判定器，尚未冻结或执行。判定器从两份旧smoke包各提取6372系列/132CTA并验证数值，但未用单对评价扰动。16项相关测试通过，包含端点共同平移但L不变、单个超限尾值、正负相消、SM变化保留、缺失/重排/错坐标/bool flag拒绝及event/CV失败。采集器先以setup核对实际CUDA设备，再按清单运行20目标进程，不重试；分析器要求完整冻结链、归档回执和完成标记。远端独占脚本和分配检查已写好，验证a058/288CPU/4GPU及OverSubscribe=NO；用旧合法分配记录和三种篡改副本验证拒绝。运行前仍须完成独立规格审查和完整冻结包；没有正式结果。正确性审查见前一correctness目录的main-producer-tail-correctness-review-v1。

冻结前按C审查补入派生窗口：W=PERMIT−MAIN、E=DONE−PERMIT、GAP=FIRST−previous DONE（首输出减producer-start）、TAIL=FINAL−last DONE，均逐系列应用同一64/256ns门槛。MAIN变化−40ns且PERMIT变化+40ns时，两个entry相对端点各自可通过64ns，但W变化80ns必须失败；已加入整数反例测试。该修改未读取新对照结果、未增加硬件观察点或放宽门槛。

C审查指出跨CTA整条时线平移还可能被各自entry归零隐藏；冻结前增加2256条共同原点系列，使用每调用min(CTA entry)的globaltimer原点，检查entry、producer-start、FIRST/MAIN/PERMIT/DONE和FINAL。全部应用相同64/256ns门槛；clock诊断设为null，禁止跨SM原生clock相减。整条CTA +80ns而自身局部窗口不变的反例必须失败。共同原点不能观测整调用相对launch的平移，因此仍不授kernel完成或launch边界资格。
