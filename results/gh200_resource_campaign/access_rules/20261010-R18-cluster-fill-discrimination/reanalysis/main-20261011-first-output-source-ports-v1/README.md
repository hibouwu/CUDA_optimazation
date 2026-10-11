# 首输出源码切点独立复核

主线ee27859集成C的a338360；6项专用测试通过。执行 `python microbench/gh200_resource_campaign/access_rules/analyze_r18_first_output.py --input results/gh200_resource_campaign/access_rules/20261010-R09-R13-R18-failure-controls --output results/gh200_resource_campaign/access_rules/20261010-R18-cluster-fill-discrimination/reanalysis/main-20261011-first-output-source-ports-v1`。已完成的复放使用上述路径的绝对形式；first-output-source-ports.json与C报告逐字节一致。

固定12个CTA及两位WG leader，共24writers。FIRST、COMMIT_G0/G16/G32/G63、MMA_RETURN、MAIN定义六段本地窗口，逐CTA/role精确守恒；五个新增点的时间戳payload为1920bytes。g16/g32/g63的slot/generation/fullphase分别为4/2/0、2/5/1、3/10/0。MMA_RETURN含最终wait1/release62，tail wait0/release63在return之后。

仅证明源码端点和准备接口；尚未实现公共probe并通过独立控制pilot，不能宣称实际ns观察有效或局部留出通过。
