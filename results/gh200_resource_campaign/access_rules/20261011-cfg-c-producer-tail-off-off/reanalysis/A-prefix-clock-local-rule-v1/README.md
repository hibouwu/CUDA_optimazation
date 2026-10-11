# R09共同系数候选的独立预测入口

本包把已经开发评分的简单共同系数候选固定为可复放规则，SHA为`200c1461f5fdd065d826d51be5d408ac728285b2bc3a83b9147b48ad7d0e16e3`。冻结内容包括校准setup及共同系数、740095三份长参考的原生周期和工作量、两个Python源码SHA、八份来源文件SHA。没有新增采集，原观察、新独立留出、完整组合和跨卡资格均为false。

预测只需要`rule.json`、两个源码和一份前序输入文件，无需读取归档目标raw或校准结果目录。输入顶层只能有`setup`和`prefix_event_us`；setup字段集合必须完整且与校准协议一致，前序必须是16个正有限event值。当前目标trace、cycle、ns、event、残差及配对臂无论放在顶层或setup都会拒绝。算法固定使用共同系数，不允许在评分时切换到两序号方案。

原CUDA源码已经证明末次前序event在目标启动前同步、返回主机。未来局部验证可以在采样前固定规则SHA和允许的历史输入，采样后从原记录提取这份前序输入，再生成条件于前序的预测；数值运算的执行时点不改变其输入依赖。这个安排没有演示在线runner调用，也不替代实验负责人采样前冻结完整条件与有效观察标准。现有binary、计时器顺序、前序与目标之间的准备流程均须保持原绑定；新增host hook或观察协议必须重新绑定和验证。

当前规则仍只覆盖GPU-ef8692f3、binary9fdbf10a、cfg_c2304×3072×4096、S4/cluster1×2、compact/dyadic17/sw1、16次同buffer同dual observer-off前序与首个observer-off目标。原生周期是工作量缩放后的1049379/16cycle；局部时间为`(1049379/16)×E_prefix_ns/kappa_common`，kappa_common≈155671.491839cycle。不支持其它K、cfg_a/b、输入/源/顺序历史或跨卡；全组目标没有缩到此组件范围。

复验只使用已知740092的20份前序，生成的40个序号预测与既有共同候选逐值相同，不重新评分目标时间。23种输入、历史、设备与源码负例均拒绝；CLI另拒绝调用者固定SHA不符和输出覆盖。在仅含规则、源码与前序输入的新临时目录中，预测成功；准备脚本重放得到逐字节相同规则。所有示例都是已知开发记录，不能当新留出。

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/A-prefix-clock-local-rule-v1/verify.py
```

示例输入格式见`example-prefix.json`，原记录身份见`verification.json`。生成预测时传入固定SHA和一个不存在的输出路径：

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/A-prefix-clock-local-rule-v1/r09_prefix_clock_rule.py \
  --rule /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/A-prefix-clock-local-rule-v1/rule.json \
  --rule-sha256 200c1461f5fdd065d826d51be5d408ac728285b2bc3a83b9147b48ad7d0e16e3 \
  --prefix-input /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/A-prefix-clock-local-rule-v1/example-prefix.json \
  --output /tmp/r09-prefix-local-prediction.json
```

输出保存规则SHA、输入文件SHA、工作量、静态周期、两序号换算率和局部预测时间；全部资格固定false。每次预测须用新输出路径。正式新局部采集仍由主集成组织，需新原始身份、有效观察和预冻结独立性证据；本包不授予GPU权限或沿用完整V09留出的历史批准。
