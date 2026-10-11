# 反馈探针八目标正确性

固定job740367的stage4/6两个binary，每stage依次运行OFF、ALL15、ONLY2、ONLY16各一次，均先16次同buffer/同stage binary的OFF前序。只验证记录和每目标4096个独立精确抽样值，不估计反馈/观察扰动。

首次51f8244冻结后，job740368在setup的subprocess.Popen之前因binary mode0644而PermissionError；没有setup.stdout或records.json，0/8目标执行。CPU构建包内mode0755，复制时使用copyfile丢失执行位；两个binary内容SHA始终未变。原包`2137a5ab…bdf38`、receipt=1及失败原样保存。

新attempt/executable-bit-fix恢复同byte二进制mode0755，并将执行位纳入freeze及采集前校验，0f40fb3预提交后job740369完成原八目标。没有重编译或重采已执行目标。回收包SHA `d62f551672504d53ce56118064ae24b0a4637d108f2fbac39a9814784837a4dc`；manifest SHA `72e291a53d2b79ab293938f0f38f2dbca5ec40d560c5e38802c55641afa00f52`，freeze SHA `57dfa2d5aef69f3e37ad9db009326fd1a651dfc3b8cdac37f9940c515e385aca`。

11输入身份、binary执行位、全部原始记录/同调用16prefix、模式mask/count/包含链及32768个抽样值精确核对通过，观察线程均SM124，实际a058/ef排他4GPU/288CPU单流。未做全D检查。

| stage/模式 | D / cycle | 选中等待和 | Q2 | Q16 | 完整Q/R |
|---|---:|---:|---:|---:|---|
| 4/OFF | 11000 | 未观测 | 未观测 | 未观测 | 未观测 |
| 4/ALL15 | 11000 | 4021 | 191 | 164 | 4021 / 6979 |
| 4/ONLY2 | 10924 | 157 | 157 | 未观测 | 未观测 |
| 4/ONLY16 | 11123 | 331 | 未观测 | 331 | 未观测 |
| 6/OFF | 7890 | 未观测 | 未观测 | 未观测 | 未观测 |
| 6/ALL15 | 9798 | 2288 | 141 | 163 | 2288 / 7510 |
| 6/ONLY2 | 7963 | 129 | 129 | 未观测 | 未观测 |
| 6/ONLY16 | 8196 | 286 | 未观测 | 286 | 未观测 |

每条件一次，不从本表推反馈大小。single的完整Q/R为None，不能把D−Q_i当等待外时间；组件ns也为None。完整复算见[verification.json](verification.json)，入口为`microbench/gh200_resource_campaign/access_rules/verify_r13_wait_feedback.py`，需要显式传入上述三份SHA与实际成功attempt路径。所有观察、服务、预测和留出资格仍未通过。
