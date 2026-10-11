# 请求路径完整图主线重算

接入B提交fad0d6b1，在主线重新执行analyze_r10_input_path.py。16次完整图、384192需求的所有模型结果与B逐字段相同；1017共同输入SHA一致，6个路径变化的模块按文件名核对SHA一致，两份报告各1023输入均重新核对。179项R13测试通过。没有重新运行GPU或数值样本。

另以旧043d06b8 Git blob和最终交付源码进行[独立代码检查](../../../20261010-R10-producer-delay/reanalysis/main-input-path-code-audit-v1/README.md)：16个合成场景的默认/显式零价格共32个完整结果字典相同；用分数前缀和的闭式FIFO表达式验证1152需求完成时刻，检查目的到达、输入工作账本及stage释放依赖。该检查不替代实测验证。

σ固定为前缀必要区间中点0.8080982817329745 ns/16KiB，未因完整结果重新调价。B0理想root512的S响应684.238ns出现，但B32 L响应仍约0；AlongM/single126/persistent首窗的B32响应和空间跨度缺失，R13 S4预等early方向也与已有十对记录相反。因此该固定点不采用，不能据此排除整个区间或队列家族。

进一步逐CTA核对：十个R10条件的1116个首轮窗口全部post-FIRST计算空闲为0，L均在1e-9ns内等于16×288=4608ns。这定位了固定点的失败：请求路径延迟尚未穿透为首轮内部的计算停顿。下一步先检查原必要区间内的供给余量和分支变化，避免仅换一个价格点反复评分；端点限制、旧观察资格和cfg_a/c/跨卡缺口保留。

完整重算命令：`python microbench/gh200_resource_campaign/access_rules/analyze_r10_input_path.py --data-root results/gh200_resource_campaign/access_rules --output <新输出目录>`。
