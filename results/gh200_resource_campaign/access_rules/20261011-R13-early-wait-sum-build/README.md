# R13 早段累计 full 等待探针

固定 cfg_b 2816×1536×4096，stage4/6、AlongN、132 CTA、dyadic/seed17。仅 CTA0 的物理线程256、WG本地首输出（CTA输出j1）累计local K组2..16的15次full try/wait路径，起点在try_wait前，终点在consumer_wait成功返回后。

同调用 `D=C16−C1`，`Q=Σ(B_i−H_i)`，等待外路径为 `D−Q`。仅保留原生cycle；两个分量的ns为None，不以整个窗口的平均率分摊。关闭新增开关时Q及等待外分量都为None，不解释为零。新增sum/count/mask/H2/B16/invalid/flag写入原空槽34..40，原progress字段保留。

累加对象在WG调度循环外构造，仅目标输出更新，全循环结束后统一flush；循环内不写GMEM。B之后的累加和校验属于等待外观察路径，也可能改变后续等待的到达时刻。新同binary off/on只量化flag增量，不能证明off等于原progress binary。16次前序始终关闭新增观测且使用同buffer；stage4和stage6分别运行自身kernel，不能据此称跨stage历史相同。

## 已完成

- prepare3逐项核对857个源文件；剔除新增取时/累加块后collective与原档案逐字相同，kernel与pipeline未变。实际trace头统一为probes/v06_trace.hpp，overlay仅forward。
- CPU-only job740240使用CUDA12.9.41，stage4/6编译成功：168寄存器、2barrier、零spill、16条静态HGMMA。分配为4CPU/16GiB，无GPU。
- SASS选中路径：stage4 H/test/wait/B为0x1e10/0x1e40/0x1e60/0x1ea0；stage6为0x1d10/0x1d40/0x1d60/0x1da0。慢等待分支回到B之前；B至collector结束各38条静态指令，无GMEM/local store。静态指令数不等于观测耗时。
- 6项CPU测试覆盖真实C++累加器算术/坏序列、重复缺组、采样包含关系、off未观测、同SM/工作/前序身份、独立4096数值检查。A发现H2晚于旧H16仍被接收，解析器已修复为`C1≤H2≤H16≤B16≤C16`并增加stage4/6负例。
- B限定源审查在[主线独立目录](reanalysis/main-B-boundary-review-v1/README.md)复放，完整报告与128行group覆盖表逐字一致。

[build-audit.json](build-audit.json)绑定成功二进制、源manifest、输出包SHA与SASS地址。prepare3/contract.json保留源码准备时compiled=false快照，后续构建结论由build-audit单列。失败记录见[attempts.json](attempts.json)：首次准备锚点不唯一、两次本地toolchain不可用、740238双头重复定义都保留；不覆盖为成功。

## 复放与后续边界

源码生成入口为 `microbench/gh200_resource_campaign/access_rules/prepare_r13_wait_sum.py --base results/gh200_resource_campaign/access_rules/20261010-R13-stage-progress --output NEW_DIRECTORY`。编译命令在该输出的build/commands.json；本次成功采用prepare3/source_hashes.json固定来源和CUDA12.9.41。测试运行 `python -m unittest discover -s microbench/gh200_resource_campaign/access_rules -p test_r13_wait_sum.py -v`。

严格decoder为decode_r13_wait_sum.py，要求caller固定raw SHA和GPU UUID；数值检查按同次4096个唯一样本独立重算，尚不代表全D正确。当前无实际Q采样、无扰动/null通过、无新增服务价、无完整预测资格。trace word23仅退出SMID，不含FINAL时间；CUDA event包含末次flush，不能输出最后CTA证书。

下一步仅[四目标正确性smoke](../20261011-R13-early-wait-sum-correctness/manifest.json)，每stage×flag一次，先验证数值和trace是否按设计产生；它不执行64次中位数草案、不判定性能或局部观察等价。任何后续性能结论仍需另行冻结有效观察标准。
