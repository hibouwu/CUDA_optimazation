# R09前序投影的静态协议一致性审查

只从原raw提取静态setup和前序event，仍须核对call重复声明的执行协议。旧提取器44a01fdd把call中的前序策略、swizzle、observer flag和源副本数等均当成可忽略字段；它们与setup冲突后，仍生成同一预测。raw SHA绑定只固定文件内容，不证明两个声明一致。

对740092全部20份原记录，本审查构造10种协议值矛盾与3种严格类型矛盾，共260个软件反例。旧快照全部接受，主集成3233d2e0的修复快照368c2ecb全部拒绝。13种反例涉及warmup_policy/swizzle/fixed_calls、target_swizzle、producer_tail_observer/prefix_flag、history_protocol/source、source_copies和mirror_target_slot；另含True冒充整数1、False冒充整数0，避免仅用Python值相等判定一致性。

| 独立检查 | 旧44a01fdd | 修复368c2ecb |
|---|---:|---:|
| 原始有效记录可生成原共同候选预测 | 20/20 | 20/20 |
| 协议矛盾反例被拒绝 | 0/260 | 260/260 |
| 目标trace/event毒值不影响前序投影 | 20/20 | 20/20 |
| CLI接受保留前序的数值检查失败记录 | 是 | 是 |

最后一项仅证明预测输入生成不依赖目标结果：合成raw将目标trace/event改成非法值，check.status改为failed并保留setup及前序，CLI仍提取相同输入。该合成包只用于软件负例，不得进入实测评分；真实新数据的数值失败、原身份和观察失败仍由后续校验／评分保留。这里没有重复评分目标时间，也没有GPU操作或资格提升。

主集成的修复还覆盖18个同名协议字段、target_swizzle别名和3个前序强制零处理。主线440个矛盾及440个缺失检查另见main-prefix-extraction-v2；本审查独立证明表中260个反例，不把两个检查范围混算。

原／修复提取器快照、report.json和fixed-report.json均保留，复验读取既有冻结共同规则，SHA仍为200c1461f5fdd065d826d51be5d408ac728285b2bc3a83b9147b48ad7d0e16e3。两快照的预测接口、原binary及前序协议没有改变；新增的是输入投影前的静态一致性拒绝。

新目录重放，输出路径须不存在：

```sh
r09_projection_output=$(mktemp -d /tmp/r09-projection-review-XXXXXX)
cp /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/A-prefix-projection-review-v1/{review.py,extractor-snapshot.py,extractor-fixed-snapshot.py} "$r09_projection_output/"
python "$r09_projection_output/review.py"
python "$r09_projection_output/review.py" --extractor "$r09_projection_output/extractor-fixed-snapshot.py" --output "$r09_projection_output/fixed-report.json" --expect-rejection
```

未来局部冻结需同时固定共同规则SHA与提取器SHA，保留raw身份→prefix输入身份→预测身份的对应；当前200c规则只绑定两份预测源码，未自动绑定后来增加的提取器。静态协议一致性与前序值可得性的来源证明仍属于输入身份边界；目标完成后投影生成的条件预测不宣称数值预提交，也不提供有效观察、新独立留出、完整组合或跨卡成绩。
