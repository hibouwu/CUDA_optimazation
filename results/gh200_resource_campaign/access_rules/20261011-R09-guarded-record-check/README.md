# guarded三臂记录检查

复用740070三个target和同一plain-prefix模块，每臂一进程、16次共同模块前序后一个目标调用。仅数值抽样、完整trace协议与runtime资源信息检查，不是全D、计时或观察资格。只使用Slurm分配的一张GPU，无跨卡预测结论，不做30进程采样。二进制/模块/源码清单/编译报告/原始数据/分析snapshot逐层绑定，不重新编译。


740071完成，a057/GPU-43269fbc-449d-3e0f-908a-9c81229546d3，单GPU非独占；实际资源与耗时见accounting.txt。三臂各4096个D抽样值与独立解析参考精确一致，共12288值。plain无设备trace，matched/guarded各264角色，全部packet有效。源码清单逐文件、三个binary及plain模块、freeze、raw、分析snapshot SHA均复核。

三target运行时smem=215040bytes、smem_epilogue=18432bytes，API max_active_ctas_per_sm=1，实际device和scheduler均132SM；这不是执行期间occupancy采样。guarded FIRST括号cycle宽度min/median/max=73/77/130，MAIN=73/74/138；仅原生整数记录及顺序检查，不证明GLOBALTIMER硬件包围或ns精度。没有全D验证、正式扰动或机制采样，observation_qualified=false。

复放将output.tar.gz解到returned、analysis_snapshot.tar.gz解到本目录，保留原740070 build目录（含source）后运行python verify.py。复放入口核对采样前manifest绑定的source manifest/build audit与每份source、binary/module、raw身份，再做数值及packet校验；不可仅拿一段trace宣称原实验身份成立。
