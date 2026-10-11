# R09实际CUTLASS guarded读取构建

仅cfg_b/stage6目标CUDA12.9 CPU构建，三个target和独立plain-prefix共享库；不执行setup、数值、pilot或留出。三target同1552word/CTA、1638912byte清零布局，共同模块前序。原tile0..9保留，FIRST/MAIN新after字段10/11；matched写零，guarded调用原候选。此阶段未完成reader及实际读取包围/扰动资格。
