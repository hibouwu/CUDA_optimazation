此目录把同一次调用的 FIRST/MAIN GT 包围记录接入既有 cfg_a/c 生命周期偏序。采样顺序和单调 GT 参考仍是调用方显式声明的前提；源码/SASS 审查 SHA 只绑定证据身份，工具不将声明升级为硬件证明。未改公共 probe、reader、runner 或 `output_model.py`。

`lifecycle_brackets.py` 检查 binary、GPU、调用、节点、CTA、物理线程、坐标、SM 范围及同 CTA 的 SM 一致性；需要完整覆盖所给两 rank 图的所有 FIRST/MAIN 标记。缺失 GT_pre、零值、overflow、重复节点、混调用、原生 cycle 次序倒退和不可行源码约束均拒绝。所有 GT 先减同一调用的共享整数原点，再交给固定模型的零服务下界偏序传播；不跨 SM 相减 raw cycles，不估计服务价格。

最后者接口要求相同调用、GPU、binary 和共享原点的端点上下文，并显式区分 consumer_FINAL、source_safe_exit、cuda_complete。区间“可能最后”集合只是保守超集；只有该超集完全位于原冻结候选内，才给出有条件的覆盖证书。相同上/下界的并列保留，未知上界不排除，候选列表不扩张。这里的证书不授观察、关键 CTA 或预测资格；完整端点全集与语义仍由调用方绑定。

最终 CPU 检查见 `tests-final.txt`：18 项通过，包括 cfg_a/c 的 K64、T1/T2 真正源图，约 1.79e18 GT 附近的一 tick 精度和整数平移，混调用/SM/坐标拒绝，未知输出完成时间以及冻结最后者的并列、唯一分离和漏选下界。合成测试的 10ns 边延迟只是可行性见证，不是校准参数。

`smoke-review-final-v1` 是 740160 的最终离线适配结果：使用 B 已归档的固定 cfg_c K64 aligned 静态坐标，不从计时挑选 CTA；手工按原生 trace/shadow 槽重取 864 stamp，与主线 `stamps.json.gz` 全字段相等。66 cluster、132 CTA、432 WG 输出窗口都满足已有源图约束。全部 132 个 consumer_FINAL、66 个 source_safe_exit、66 个 cuda_complete 上界仍未知，故没有从本批缩小最后者集合，也没有加载、改变或重评 V09 候选。

包围宽度 `{0:58,32:729,64:75,96:1,128:1}` 仅为 GT 计数域内的描述。绑定清单和共享原点均在 smoke 后构造，只用于事后诊断，不构成预提交数值预测。所有观察、扰动、null、局部预测、关键 CTA 和完整 GEMM 资格为 false。此批没有 FINAL、producer_tail 或 D_visible 的 GT_pre，不补造这些读数。

重放时输出到新目录：

```sh
python3 -m unittest -v test_lifecycle_brackets.py
python3 review_smoke.py \
  --run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-bracket-smoke-v2 \
  --ordering-review ../C-20261011-ordering-review-v1/report.json \
  --output NEW_C_REVIEW_DIRECTORY
```

`smoke-review-final-v1/cli-example` 保留一个真实 cluster 的图、绑定清单、packet 和命令行投影；其结果与批处理投影相等。`smoke-review-v1` 与 `tests.txt` 保留本轮 SM 范围及统一原点接口加固前的 CPU 中间结果，最终重放使用 `smoke-review-final-v1` 与 `tests-final.txt`。本目录未执行任何 GPU 进程。
