# 740071：A独立记录及条件数学复核

三臂原raw记录均通过独立复核：共12288个不同位置的dyadic数值精确正确；plain无trace，matched/guarded各264个角色。完整源/二进制/模块/冻结snapshot/返回包/raw身份重新核对；未调用会改写主报告的verify.py，没有执行设备代码或提交GPU。

真实trace的entry/producer均为正且顺序正常，SM在132SM设备范围内；各原角色与end字段通过额外检查。三target的动态smem=215040B、epilogue=18432B、API max_active_ctas_per_sm=1、实际与scheduler132SM一致。这里记录的是API资源输入，不能叫执行期间occupancy采样。

原生CLOCK括号宽度min/median/max：FIRST73/77/130cycle，MAIN73/74/138cycle。以Fraction从原begin/end直接重算132个CTA的两窗口交集、最小总区间容差及最小点可行性，与主集成条件报告逐项一致；不调用common_rate_interval求这些数值。全部窗口满足3200ns下限。原128ns总区间预算下0/132可行；所需最小总区间容差min/median/max为1149.598/1212.218/1242.163ns。

这是以GLOBALTIMER硬件取样落在前后CLOCK之间为前提的算术结果。它不证明前提，不修改128ns预算，也不认证原现象或物理频率。每臂只有一个进程，原扰动分布与空间资格未取得；observation_qualified=false、original_phenomenon_inference=inconclusive。

冻结packet reader另有可复现拒绝缺口：分别把CTA0 entry cycle/ns、producer cycle/ns改为0，或把head[2]/[11]/[12]改为133（零基SM132越界），matched和guarded共10个合成复制负例仍全部被接受为264角色、errors为空。这些仅在内存复制上测试，没有写原raw。真实记录没有这些异常，数值结论不撤回；但后续完整parser需补齐head端点与SM范围拒绝。原snapshot和负例结果保留，已反馈主集成，不补写原协议。

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/reanalysis/A-smoke-boundary-audit-v1/audit.py
```

完整输入SHA、132个独立交集、10个复制负例见audit.json。重复入口只复验同一JSON，不改公共文件。
