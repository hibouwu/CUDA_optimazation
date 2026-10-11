# 前序投影的静态协议一致性修复

A独立审查指出，v1忽略call中重复的静态执行声明，导致setup保持原值而call声称adaptive warmup、swizzle8、observer1或两份source时仍接受。原20个合法记录没有这种矛盾，既有评分未改变；但这会影响未来新记录的输入契约。

修复提取器：18个setup/call同名协议字段要求严格类型和值相同，target_swizzle映射setup.swizzle；另要求release_warp_warmup、epilogue_warmup_defer、second_prefill_warmup为整数0。这22项属于执行组织，不是目标计时。缺失也拒绝，不能以原文件SHA已固定代替自洽检查。

20个原记录逐字段变异，共440个矛盾和440个缺失输入均拒绝；异buffer、无效pointer、缺前序、raw SHA和覆盖检查保持。目标elapsed/trace/check值不参与预测，毒值检查仍20/20。动态加载原255e93a Git blob，与新提取器比较全部20个合法前序输入，逐值相同。共同规则SHA仍200c1461...，没有重拟或重新评分目标误差。

复验`python verify.py`。旧v1为255e93a时的历史审查，对旧协议语义的重放应使用该Git版本；新采集准备须固定v2提取器SHA368c2ecb...和原规则SHA，仍需独立观测验收。没有GPU或新留出/资格。
