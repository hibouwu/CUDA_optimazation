# guarded读取设备C++独立复核

0c9afd合入2a53633后直接用未修改candidate header，分别生成一处/两处设备C++调用，经NVRTC13.0、ptxas sm_90a和cuobjdump独立复核，audit_fixture均通过。包含真实cstdint类型及24byte/8byte对齐断言。此为隔离设备前端/控制流证据；不执行CUDA runtime、driver或GPU，不证明实际CUTLASS帧资源或硬件读取包围。
