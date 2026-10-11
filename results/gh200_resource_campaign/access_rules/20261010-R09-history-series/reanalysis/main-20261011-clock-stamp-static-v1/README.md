# 有界时钟读取独立静态复核

fab91fd合入主线841f5d0，6项针对性测试通过。直接从r09_clock_stamp.cuh提取一处/两处inline PTX，经本地CUDA13.0 ptxas -arch=sm_90a -v编译，再由cuobjdump --dump-sass反汇编和audit_fixture检查，两种均通过，原PTX/SASS/编译日志保留。

每处三次64bit特殊寄存器读取；前两次结果的高低半部比较控制后续分支，零读数最多重试两次，耗尽返回全零。此处只确认独立fixture的编译控制流。尚未在目标CUDA12.9/CUTLASS正文构建，不证明硬件读取包围关系、实际ns精度、资源占用或扰动资格；不启用到公共probe。
