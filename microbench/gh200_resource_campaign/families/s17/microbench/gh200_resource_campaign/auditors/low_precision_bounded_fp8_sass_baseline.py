"""Compile-only 730679 candidate baseline; independent source-B/B2 still required.

All PC/predicate/operand/control text retained; changed lowering requires review.
This identity is not proof of a hardware retired-instruction count.
"""
BASELINE = {'_Z27lp_bounded_v1_wgmma_e4m3_g1ijbPN2gh5StampEPd': {'instruction_count': 504,
                                                      'instructions_sha256': '5392b264c07544493d4491b501e1f9dd6e0ac4035d8d7f7f84fa9122b1642049'},
 '_Z27lp_bounded_v1_wgmma_e4m3_g2ijbPN2gh5StampEPd': {'instruction_count': 504,
                                                      'instructions_sha256': '5648dce00ba57231a4e7eef7606aff8bd183b8f7ed6b25c7d7f67e3e8b01639e'},
 '_Z27lp_bounded_v1_wgmma_e5m2_g1ijbPN2gh5StampEPd': {'instruction_count': 504,
                                                      'instructions_sha256': '339b5e646738da66fabc4324e6ebed23c8f8cd677628a2d76a6a7d7da12bf6d6'},
 '_Z27lp_bounded_v1_wgmma_e5m2_g2ijbPN2gh5StampEPd': {'instruction_count': 504,
                                                      'instructions_sha256': '06fc32228df82f16e88de55143b7925bd912b1a0527423b57750a75f9d3ae69c'}}
