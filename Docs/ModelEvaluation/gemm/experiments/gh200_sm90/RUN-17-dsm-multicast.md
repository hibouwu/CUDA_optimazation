# S17：DSM、多播与 cluster 同步批量测量方案

测量同一 GPU 内的远端 SMEM 访问、TMA 多播及 cluster 同步，保留本地访问和单目标搬运作为对照。

## 固定配置

cluster 大小 2/4/8；七种形式为 local_read、dsm_read、local_write、dsm_write、cluster_sync、bulk_single_target、bulk_all_targets；one_cluster / 全 GPU 两个范围，共 42 名义坐标。

运行初始查询全部 21 个 form/cluster 目标的 cluster 支持、资源和 active capacity。occupancy 给出的上限不称为实际驻留；能力或容量不支持时保存原因，不硬启动。旧最大点的适用证据可复用，其余覆盖仍需完成。

## 一次运行

提交前把当前 v2 数值参考和正式采样路径接齐。一次编译后查询全能力，再连续执行支持的短点；先测最大点和跨 CTA 退出/缓冲复用边界。所有访问者存活期间保持 cluster 生命周期，最后完成 cluster 会合才允许退出。

通过冻结规则后，脚本连续对支持配置校准、预热和十进程采样，不按 cluster 大小拆三次人工验收。共享二进制与参考，完整远端结果和参与者状态逐点保存。任一死锁、值错误或未知清理停止相关实验。

## 工作量与分析

DSM 访问按实际参与 CTA、迭代、请求量计字节。cluster_sync 计完成同步服务，采用自己的单位，不硬转换为带宽。多播分别保留源请求字节、总接收字节和每目标有效写入；不得把接收量当作 HBM 读取量。

单 cluster 的计时只能使用明确定义的同一计时者或 globaltimer 边界，不能跨 SM 相减 clock64。全 GPU 用 globaltimer 包络。远端与本地对照要求相同请求量和完成边界；同步偏斜与同步指令自身服务分别解释。

## 当前需要补的代码与交付

已有 [cluster_dsm_multicast_v1.cu](../../../../../microbench/gh200_resource_campaign/probes/cluster_dsm_multicast_v1.cu)、能力查询和有限短矩阵规划。剩余短验证、正式 host 循环与采样尚未完成。

最终交付 42 坐标终态、DSM 本地/远端对照、多播两类字节口径和 cluster 同步结果，附完整条件与复现命令。流程见[批量执行约定](RUN-REMAINING.md)。
