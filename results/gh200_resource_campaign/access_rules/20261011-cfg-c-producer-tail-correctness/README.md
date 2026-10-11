# 740088 producer-tail全D与记录正确性

冻结提交f35ce88，复用740087 binary9fdbf10afe162a92f44a889a1fe3c87b89f57a0e73a5ec385abde32035742b55，设备完整SASS与740082 byte相同。740088在a057获配单GPU、4CPU/16GiB；实际GPU-fd5317a4-afa5-b1e6-4376-135af6a193bb，保留原CUDA_VISIBLE_DEVICES和分配上下文，不改卡编号。

一次check-producer-tail在相同A/B/D分配上分别off/on执行，7077888个D元素逐位相同且无非有限值。导出完整D再用独立period17/seed17参考逐元素核对，7077888值全部精确；full-D SHA保存verification.json。此正确性模式不执行16次前序，不当作正式计时。

另各一次warm off/on，均16次同dual binary、新增观察关闭的前序；每次独立4096样本复核，总8192。新decoder完整132CTA×两WG/原工作坐标/SM/端点顺序通过；off头14/15为零，on全部132个elected producer tail有效。source/build/freeze/binary/snapshot/raw返回身份链在verify.py重放，原源码及分析快照可从input归档恢复。

仅数值和packet smoke通过，没有开/关扰动分布、空间资格、动态服务价或kernel完成结论。不同GPU身份不视作独立跨卡性能评估；紧凑分配也不与旧padded数据混为同条件。observation_qualified=false。下一步需先冻结针对新增观察的有效性门槛，再考虑相应最小对照，不能用本次单对成绩代替。
