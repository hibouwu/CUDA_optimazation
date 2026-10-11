# 主线重放局部时钟相容性

A的4f39ab6合入为af06c62；65项R09测试通过。当前公共解析器保持不变，普通20进程仍走公共replay，旧740048 smoke仅走绑定两份原始SHA的专属历史诊断，current_prefix_protocol_qualified始终false。上一目录v1的身份拒绝记录保留。

本次21进程报告与A-local-clock-consistency-current-reader-v1完整JSON相同。stage4/6的常数L周期率必要端点余量中位数为658.42/671.22ns；允许各CTA独立、非负且不超过原声明1.98GHz的可变有效率后，局部必要端点余量中位数为8.40/4.78ns，最大14.65/13.55ns。旧smoke分别为常数529.73ns、可变中位4.78ns/最大11.53ns。

这是由目标端点构造的存在性见证，不能作为自由预测输入、物理DVFS轨迹或新价。结果只说明常数率失败不迫使局部timer有数百ns误差；跨SM共同轨迹及观测资格问题不因此解决。
