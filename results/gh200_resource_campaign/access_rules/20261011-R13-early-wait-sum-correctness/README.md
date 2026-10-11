# R13 累计等待四目标正确性检查

固定构建job740240的stage4/6两个binary，每stage的observer0/1各一次，四个独立进程顺序为s4/off、s4/on、s6/off、s6/on；每次目标前有16次同buffer、同stage binary的observer0前序。

只核对指定ef GPU、声明参数、各目标4096个唯一dyadic值、padding/nonfinite、完整同调用记录、15组累计mask/count及端点序；并不验证全D，也没有性能通过门槛。失败停止并保留，禁止替换样本。时长和Q只作为smoke诊断保存，不能估计stage效应、观测扰动或服务价格。全部观察、预测与留出资格仍为false。

manifest和freeze在提交前保存，二进制/脚本/解析器均固定SHA；a058排他分配计实际4GPU/288CPU，仅一个计时流。collect在Slurm设置的可见设备上先用setup核实际GPU，不覆盖CUDA_VISIBLE_DEVICES。命令与结果在本目录保存。
