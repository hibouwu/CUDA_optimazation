# 共用 plain 前序模块：构建检查

作业740042在romeo-a057完成，退出0:0、用时69秒，AllocTRES为4CPU/16GiB/1节点，无GPU分配。CUDA12.9.41编译共享模块及原plain程序；未创建有效GEMM上下文或启动GPU kernel，不构成正确性、计时或观察器验收。

共享库可通过RTLD_LOCAL加载；create的空参数测试返回失败并清空handle。设备GEMM的1280条SASS指令（含16条HGMMA）在共享模块、同批plain程序和冻结V09的cfg_b_plain之间逐条一致，指令SHA256为`0b882f219276e78f6df026bf44e05e004c8b5c252c60c2598367f35ab2b46dd9`。完整SASS文件含其它辅助kernel，不能用整文件差异替代GEMM比较。

[验证记录](verification.json)列出文件与指令哈希；[构建脚本](run.sh)包含精确编译选项和不触发GPU的接口负例；[Slurm启动参数](command.json)、[输入身份](input-sha256.json)、[回传身份](receipt.json)保留复核链。input.tar.gz保存CUTLASS和probe源快照，output.tar.gz及returned/保存ARM二进制、SASS、日志和分配文本，均在本地结果目录，不以临时远端目录作为唯一副本。

本地先前失败分别是CUDA13拒绝GCC16、GCC15缺cc1plus，以及绕过检查后的GCC16/glibc头文件错误；没有为通过构建修改系统头文件。受支持的远端构建解决了这一准备阻碍。

下一步仍需接入host的加载/创建/16次前序调用，并在正式观察前核对相同缓冲区、分配与模块加载时机、正确性及观察扰动。A的精确夹取算法与扰动筛选独立验收；本记录不批准V10。
