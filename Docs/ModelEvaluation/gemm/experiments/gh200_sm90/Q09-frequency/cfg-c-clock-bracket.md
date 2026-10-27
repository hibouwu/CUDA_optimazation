# cfg_c 双时钟观察

7 个目录（10-11）：用 globaltimer–clock–globaltimer 三元读取约束同 CTA 首、次输出的换算率。

## 结论

- 在条件 e=0 的算术下，84 个共同窗口配对互不相交。这只验证了读数域的区间接口，没有证明物理采样包围关系，观察资格未取得（740160）。
- GT 观察与 off/off 空对照：局部系列 4912/6372、4729/6372 与 L 系列 432/432 均未过门；GT 前读不是差异出现的必要条件，不再按合格观测推进（740161）。
- 构建失败、预检失败与 MMA 参考准备保留在结果目录。

## 轮次

按日期排列；全部结果目录见[Q09-frequency/cfg-c-clock-bracket](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/cfg-c-clock-bracket/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| <a id="round-gt-bracket"></a>740160：真实GT-C-GT记录：三元读取自身的跨度能否让同CTA首/次输出共享精确常数换算率？ | 固定cfg_c S4、同buffer的16次off前序及新binary；各一次off/on，读取432WG窗口的864三元组。 | 条件e=0算术下84共同窗口配对均不相交；只验证读数域区间接口，不证明物理采样包围或误差，也未取得观察资格。 | [原结果目录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/cfg-c-clock-bracket/20261011-cfg-c-gt-bracket-smoke-v2)、[原页判定](R09-inkernel-clock-stages.md#真实gt三元组怎样约束同cta首次输出) |
| 740161：GT观察与off/off null控制：GT前读是否为局部配对差出现的必要条件，当前读数能否进入预测留出？ | 同新binary、cfg_c固定几何与16次off前序；10对off/on及10对off/off，共40新进程，producer-tail始终关闭。 | 局部系列分别4912/6372、4729/6372失败，L均432/432失败；GT前读不是差异的必要条件，停止按合格观测推进，不以整event通过覆盖。 | [原结果目录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/cfg-c-clock-bracket/20261011-cfg-c-gt-controls)、[原页判定](R09-inkernel-clock-stages.md#真实gt三元组怎样约束同cta首次输出) |
