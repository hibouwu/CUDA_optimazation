# cfg_c：物理源绑定与尚未闭合的full接口

26静态条件；其中150原双时钟进程、18960个CTA和614400个D抽样值重新复核。V09仅静态几何。
cfg_a的22份规范化压缩绑定逐字节不变。

每rank A16KiB投给两rank；B0/B1各8KiB投给本地。每cluster每K-group六源atom/64KiB、八目的atom/96KiB。
每rank full取本地producer arrival、两rank的A16及本地B0/B1；名义48KiB，valid/fill分开。
输出OOB rank仍可能提供伙伴必需的A；C本地输出图保留每tile八个名义子tile和源buffer代次。

源接口图在arrive_expect_tx/acquire_return开放；未接入cfg_c的empty、消费安全释放或有限stage约束。
这些根节点表示尚未给出的前驱，不可当作零成本完成或正式时间预测。C的cfg_a rank图未迁入。
所有source/full模板、本地输出模板、静态绑定与逐输出有效面积已保存；没有新价格或GPU运行。
