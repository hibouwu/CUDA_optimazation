# 跨SM共同变化时钟独立复核

主线9ec0a87集成A的8715452；8项专用测试通过。执行 `python microbench/gh200_resource_campaign/access_rules/r09_shared_clock.py --base results/gh200_resource_campaign/access_rules --output results/gh200_resource_campaign/access_rules/20261010-R09-R13-R18-failure-controls/reanalysis/main-20261011-shared-clock-consistency-v1`，完整JSON与A报告一致。

另直接核验84份证书：可行势函数逐边满足整数差约束，不可行环闭合且边权和为负，不调用原求解器。入口核对24份输入SHA；每个分支33264点。所有21进程即使不设频率上限也拒绝32ns端点预算。stage4 capped/uncapped最小整数预算中位228.5/177ns，stage6为247/233ns，旧smoke为231/209ns。

模型允许各SM独立未知offset与共同任意非负F(t)，不直接相减跨SM cycle。这是假设端点预算的可行性边界，不是实测打点精度或物理DVFS证据；旧smoke资格保持false，不提供预测价格。

完整报告以shared-clock-consistency.json.gz无损保存；解压后与复放JSON逐字节相同。
