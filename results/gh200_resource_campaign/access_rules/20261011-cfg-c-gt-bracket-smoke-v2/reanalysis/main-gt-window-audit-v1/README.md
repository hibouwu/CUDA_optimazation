A c3759a7接入主线c37b8caa。使用主线模块重跑740160真实GT三元组分析，三份完整分析文件和独立verification.json均逐字节相同。17输入SHA、14冻结文件、858构建source及15/30输入/输出archive文件核对；864stamp、432WG窗口、216共同窗口、252首/次输出配对全部精确复算，4个错误标签/边界/误差符号/相容判定反例拒绝。10项A交付产物SHA核对。

共同窗口先在同CTA/SM原生clock域选FIRST最小、MAIN最大端点，GT保持该端点标签，相等时选WG0。84个两输出CTA的共同窗口率区间及168个同WG率区间均分离；first.rate_low/second.rate_high−1共同窗口最小/中位/最大为9.745247%/10.195148%/10.969350%。仅在采样包围和e=0额外GT坐标误差前提下拒绝精确常数平均率，不把它命名为已测物理频率，更不直接否定允许误差的5%/10%时间预测。

记录点读数Npost相对选定cycle采样时长T的误差范围为[-width_begin,+width_end]；保留非对称端点归属。没有拟合额外误差预算、价格或规则，没有复用旧binary校准。全D正确性只绑定原主线证明；本独立区间重放没有重新计算全D。

本次只用一个enabled smoke调用，不能覆盖后来740161两个必要控制的局部失败或独立留出/跨卡要求。全部观察/预测资格false，无新GPU。

复放命令同A原README，分析入口使用主线r09_gt_window_audit.py，inputs使用本目录inputs.json，output使用新目录；随后使用本目录verify.py独立核对。analysis与verification保留完整输出。
