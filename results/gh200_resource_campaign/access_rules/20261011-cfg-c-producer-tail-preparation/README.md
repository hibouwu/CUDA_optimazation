# cfg_c producer-tail观察端点：源码准备

基于当前run_r18独立派生，固定cfg_c 2304×3072×4096、256×128×64、cluster1×2、stage4、grid66×2×1、swizzle1、dyadic/seed17，A/B紧凑布局。仅setup/warm模式，拒绝其他几何、干预和历史。此布局不冒充旧738496带额外分配行距的同一输入身份。

在主输入producer的collective_mainloop.load_tail返回后调用新helper，只由thread0在observer=1时读取clock/globaltimer并写原trace头[14]/[15]。版本cfg-c-producer-tail-v1；开/关共用同一目标二进制，16次common-sw1前序的新增observer固定关闭，目标才设置请求flag。前序仍是该dual binary，不能叫共享plain模块前序。

原trace每CTA1296word，132CTA分配1368576B不变；新增有效戳2112B，off保留零字段。源级对照确认撤销唯一hook即可恢复原cooperative kernel，mainloop collective与pingpong逐字节相同，canonical header避免重复定义。856源SHA已验证，prepared-source.tar.gz保存源和构建命令。

此端点测量load_tail返回，不是D全局可见、全部线程退出或kernel完成。两个clock读也不建立硬件包围；后续只能按各自时钟域和明确边界分析。

当前仅通过Python语法和源投影审查，未执行目标编译/GPU。仍需目标SASS和资源审查、严格新版本decoder、全D数值、冻结开/关扰动对照；未声明观测资格、完整时间预测、新留出或跨卡结果。旧runner/probe原文件未修改，未启动V10。
