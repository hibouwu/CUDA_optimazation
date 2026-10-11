# 原始记录到前序预测输入的严格投影

新增r09_extract_prefix_input.py，连接既有采集记录和固定共同规则。调用者提供raw SHA及记录中的binary SHA；原始包必须只有一份setup/call/check，重复/缺失事件拒绝。工具仅选静态配置、前序event及缓冲区/调用次数组织字段，输出只含setup和prefix_event_us，另在stdout回执中绑定raw、binary、提取器和输入SHA。输出路径已存在时拒绝覆盖。

提取器不读取目标trace、elapsed_us、周期、残差或check的数值内容，不因目标正确性决定是否产生预测。缓冲区身份必须为正整数，其是否相同由固定规则验收。协议配置和16次正有限前序的最终接受仍由原固定预测入口完成；之后才单独检查目标数值/trace及观测门槛。预测成功不能替代目标验收。

20个已知raw的严格只读映射检查通过：除允许字段外任何读取都会抛错，将全部目标字段替换成不可读取对象后输入仍相同。每个记录的异buffer、无效pointer和缺历史路径均拒绝；CLI原始SHA不符与覆盖也拒绝。提取后调用原固定规则，结果与直接前序输入相同，未重评分目标误差。

示例：`python microbench/gh200_resource_campaign/access_rules/r09_extract_prefix_input.py --raw <raw.jsonl.gz> --raw-sha256 <采集记录SHA> --binary-sha256 <采集manifest中的binary SHA> --output <新prefix.json>`；再将prefix.json交给原r09_prefix_clock_rule.py及固定rule SHA。提取器和规则都应在未来新采集前绑定到该次清单，目前尚未完成新实验冻结。

A后续补充的serialization-boundary及输出SHA已重核，规则与预测源码未变；旧主线规则复放目录保留原验证版本。这是采集后的字段投影，不声称数值预提交，旧观察失败和全部资格保持。C继续独立观测审查，无新GPU。

复验本目录`python verify.py`。
