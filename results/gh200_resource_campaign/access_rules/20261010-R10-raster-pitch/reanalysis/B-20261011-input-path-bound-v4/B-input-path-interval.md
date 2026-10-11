# single126 B32：必要价格闭区间内首窗无空闲证书

价格区间 [0.698601908678, 0.917594654788] ns/16KiB；精确有理数计算。
全区间证明通过：True。126个首输出L均为4608ns。

| 保守余量 | ns |
|---|---:|
| cohort_span_below_subscription_ns | 671.684379077 |
| cohort_order_gap_ns | 205.429419195 |
| producer_next_slot_gap_ns | 257.284377691 |
| initial_producer_free_slack_ns | 1216.078448875 |
| source_capacity_headroom_ns | 101.619607019 |
| request_capacity_headroom_ns | 56.766146993 |
| destination_capacity_headroom_ns | 144.000000000 |
| initial_source_cleared_ns | 282.089824844 |
| initial_request_cleared_ns | 12.969064692 |
| initial_destination_induction_slack_ns | 290.773498504 |
| refill_TC_slack_ns | 83.940606305 |
| D_after_all_input_issues_ns | 1501.429419195 |

初始g1～g5最小余量：56.766146993ns。
这是一项固定模型的条件排除，不是新硬件规则或独立留出通过。
