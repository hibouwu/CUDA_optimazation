B的0b17f8b2接入主线92ce1aac。主线重跑三种ldb账本，82共同输入SHA与6迁移同内容模块SHA核对；全部非身份报告字段相同，三份完整gzip逐字节相同。15份归档源码绑定、60旧记录、120个保存CTA计数、48384组ring检查通过；193项R13测试通过。除新增静态绑定函数外，供给模块AST与f521a01f相同，原完整tile/time API未变。

归档默认SS pingpong、swizzle1/AlongM和grid(132,1,1)给出linear=CTA+132j，坐标(linear%21,linear//21)。CTA0–107有4输出，108–131有3输出，WG0/1分别处理偶/奇j。旧trace的CTA0–3保存最后WG0窗口，即j2、坐标(12,12)..(15,12)，完整M/N、只有K尾部；每窗有效源1024000B、目的fill24576B。其global group64..95，S6首slot4/gen10/full-empty参数0/1，末slot5/gen15/1/0，不是从slot0重新开始。

`verify_independent.py`未调用schedule或granule模型：逆向遍历矩阵全部坐标分配CTA/ordinal，核对每条件504输出、16128组状态；再枚举有效FP16元素地址，核对每条件256个A/B覆盖cohort及254个邻组交集。A最后K组到下一输出首K组仍有1950个128B单元重叠，不能仅在输出边界重置共享源状态。几何单元重叠不证明实际in-flight合并、驻留或所有CTA同时发起。

保存的count2只与上述绑定兼容，并不能单独区分CTA总T3/T4；这些坐标是源码推导而非旧trace观测。原局部周期支持的是四个M/N内部j2窗口，不直接覆盖首输出或M/N尾部服务。完整GEMM时间覆盖全部边界，仍需独立的ready/release/D生命周期和价格，不能从这四窗口补齐。首次full等待仍在旧trace窗口外。未改观察资格、未重评预测、未拟合时间或运行GPU。

复放主线analyze_r10_boundary_binding.py时run为原735876-v4、prior为B-boundary-ledger-v4、output为新目录；随后运行本目录verify_independent.py。main-verification.json和independent-binding-verification.json分别保存完整重放比对和独立枚举结果。
