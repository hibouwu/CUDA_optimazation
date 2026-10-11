# cfg_c elected producer-tail：740082目标构建

初版thread0在collective.load_tail调用后采样，不能仅由源码保证等待了实际elected lane。v2改为在原collective if(lane_predicate)分支内、pipeline.producer_tail返回后采样，同原选中lane执行，不增加同步；原cooperative与pingpong kernel保持原字节。版本cfg-c-producer-tail-v2明确拒绝初版协议，初版归档保留。

740082在a057以4CPU/16GiB完成CUDA12.9 ARM目标构建，AllocTRES无GPU，未执行kernel。856源SHA、构建冻结及返回包身份通过检查。kernel168寄存器、2barrier、零stack/spill；1024B为静态shared，动态分配尚未运行核对。

具体SASS已保存并按PC复核：5ab0选举、5ac0非选中lane退出（分歧选举路径在5fb0..6030返回同门）；四槽TRYWAIT在5b40/5bc0/5c50/5ce0，失败分别转入带阻塞PHASECHK和重试的慢路径，最后回5d00。其后读取观察flag，关闭则5d20退出；开启在5d30/5d40读取CLOCK/globaltimer，5db0/5dc0写head14/15（0x70/0x78），stride0x2880对应1296word。

binary SHA97e3475b3ad3dd8cdb1612923646e29b262ebfb7186676dc377eb3b709576b7d；SASS SHA81cd006cc1fbbdc7ab045b93a9c317d7a62c24ec5e6fbb52ec9d7eefebfaba62。audit-source.py/audit-build.py为本次源投影与目标指令检查入口；input.tar.gz和output.tar.gz保留实际提交与返回内容。

新decoder仍7项合成测试通过，额外明确拒绝v1版本。当前没有设备数值/全D/扰动证据，尚不能采正式机制样本。端点是原elected mainloop producer四槽tail返回，不是D可见或kernel完成；指令先后不建立CLOCK/GT硬件包围。后续先补全D验证入口并核对host旗标/trace，再考虑最小数值检查。
