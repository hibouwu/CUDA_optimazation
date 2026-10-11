# 主线复核四 leader 条件判读

c2db50f合入50898b8后，84项B专用测试通过。当前主线decoder重新读取740054四份原始gz，以两份相同归档run.sh中的common-sw1-16声明绑定，逐进程重算4096个输出检查值。与B-20261011-release-warp-smoke-v2相比，完整processes、暖机绑定、门槛、条件区间和全部不合格原因相同；仅input_sha256中的代码位置与当前decoder版本有差别。

stage4上游差368cycle，stage6 local差82cycle；每stage仅一对、非独占GPU432、不是预列ef研究。interpretation_qualified=false，未由小event差消除局部失败。统计采用median(abs(delta))，abs(median(delta))不参与通过判定。

v1四leader条件区间仍依赖整warp原arrival被包围的假设；此报告不证明该前提。主线正在准备128线程各自前后戳的v2，不将四leader字段静默当作完整128线程观测。
