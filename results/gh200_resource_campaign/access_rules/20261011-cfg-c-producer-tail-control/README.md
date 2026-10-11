# cfg_c producer-tail固定增量对照740091

冻结提交b11fae8，20个目标进程（10对交替off/on顺序），参考GPU-ef8692f3、a058独占288CPU/4GPU。全部使用同一已数值验证的binary、紧凑输入分配及16次同buffer同dual observer-off前序。归档回执returncode=0；冻结17文件、实际分配、binary、20原raw身份及完整标记均通过。81920数值样本、全部132CTA/两WG/输出严格解码通过，配对CTA到SM映射无变化。

冻结增量资格未通过：6372时线中4771失败。432条L全部失败，逐条十对绝对差中位范围32..608ns，最大范围320..1280ns；预设门槛中位64ns/最大256ns。entry相对及共同原点MAIN各432条全部失败，FINAL各264条全部失败。W两条失败、E279条失败、GAP158条失败；TAIL264条全部通过。各类型完整逐对差保存在analysis.json，分类汇总在summary.json。

整调用event绝对相对差中位0.445643%、最大1.135973%，off总体CV0.310071%，通过整调用门槛，不能替代局部资格。GLOBAL_ENTRY全部通过也不能解释后续变化的物理原因。

当前结果只表明固定协议未建立增量扰动资格。不能把ns差全归因于尾采样开销、服务变化或物理时钟，也不能用共同漂移/拟合换算抵消失败。未剔除配对、补抽或改门槛。tail精度、基础dual、D可见/整kernel边界和独立跨卡性能资格均未授予；V09失败保持。

重算：`python analyze.py`（目录需保留原input/output归档解包后的build和returned文件）。receipt.json绑定返回归档，freeze.json绑定输入。没有新的冻结留出预测。
