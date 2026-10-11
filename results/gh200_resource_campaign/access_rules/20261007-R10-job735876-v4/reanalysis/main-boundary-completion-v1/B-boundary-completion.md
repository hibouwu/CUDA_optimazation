# 原始R10零源与目的完成端口

每组full合流保留arrive_expect_tx与A/B0/B1三个逻辑目的完成义务。valid/fill只划分逻辑完成条件，不假设独立硬件队列。
每条件48384个目的盒，47712个非空有效源义务，3558个fill义务。
672个零源B1没有valid分支，但保留submit→8192B fill完成→destination完成→full合流，完成时间未定。
按有效源量缩减full预算会在1868组少计26882304B；删除零源目的盒会在672组少计5505024B。
抹去地址/stride后，完整工作、ring/完成拓扑与源身份等价类在三种行距间完全一致；相同参数/初态且只读取这些量的模型不能区分行距。
接口未闭合empty、consumer释放、producer tail或D生命周期；原时间图与guard不变。
