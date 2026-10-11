# 原始R10最后K16的工作与释放语义

每个K组仍调用8个m64n128k16 atom并commit一次。最后K16只有2个atom包含有效K，另6个仍执行零填充区域。
每输出32次full等待、32个commit组、31次普通wait1/前组释放、1次tail wait0/末组释放；每组期待32768B。
全调用16128组、129024次atom调用；有效数学量31200000000FLOP，名义MMA量33822867456FLOP。

S6/Kt32输出切换：后输出q0–4的acquire等待前输出q26–30普通释放，q5等待前输出q31的tail释放。
下一WG的MMA顺序通知位于上一mma返回之后、mma_tail之前；不能给首组额外添加前输出tail依赖。
empty完成需要拥有该组的一个WG共128次到达，不是两个WG共256次；到达延迟未定。

既有图可复用持续ring、普通/末组释放、MMA与epilogue两级顺序。旧full-tile guard仍拒绝该几何，零源fill/D生命周期与服务价格未补齐。
