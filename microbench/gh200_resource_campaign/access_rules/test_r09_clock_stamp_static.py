from pathlib import Path
import unittest

from r09_clock_stamp_static import audit_fixture,inline_fixture


SASS = 'Function : r09_clock_stamp_ptx_fixture\n/*0000*/                   LDC R1, c[0x0][0x28];\n/*0010*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ;\n/*0020*/                   ULDC.64 UR4, c[0x0][0x208];\n/*0030*/                   CS2R R2, SR_CLOCKLO;\n/*0040*/                   ISETP.NE.U32.AND P0, PT, R2, RZ, PT;\n/*0050*/                   ISETP.NE.U32.AND.EX P0, PT, R3, RZ, PT, P0;\n/*0060*/               @P0 BRA 0xb0;\n/*0070*/                   VIADD R0, R0, 0x1;\n/*0080*/                   ISETP.GE.U32.AND P0, PT, R0, 0x2, PT;\n/*0090*/              @!P0 BRA 0x30;\n/*00a0*/                   BRA 0x130;\n/*00b0*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ;\n/*00c0*/                   CS2R R4, SR_GLOBALTIMERLO;\n/*00d0*/                   ISETP.NE.U32.AND P0, PT, R4, RZ, PT;\n/*00e0*/                   ISETP.NE.U32.AND.EX P0, PT, R5, RZ, PT, P0;\n/*00f0*/               @P0 BRA 0x170;\n/*0100*/                   VIADD R0, R0, 0x1;\n/*0110*/                   ISETP.GE.U32.AND P0, PT, R0, 0x2, PT;\n/*0120*/              @!P0 BRA 0xc0;\n/*0130*/                   CS2R R2, SRZ;\n/*0140*/                   CS2R R4, SRZ;\n/*0150*/                   CS2R R6, SRZ;\n/*0160*/                   BRA 0x180;\n/*0170*/                   CS2R R6, SR_CLOCKLO;\n/*0180*/                   LDC.64 R8, c[0x0][0x210];\n/*0190*/                   STG.E.64 desc[UR4][R8.64], R2;\n/*01a0*/                   STG.E.64 desc[UR4][R8.64+0x8], R4;\n/*01b0*/                   STG.E.64 desc[UR4][R8.64+0x10], R6;\n/*01c0*/                   EXIT;\n'
SELECT_LOWERING = 'Function : r09_clock_stamp_ptx_fixture\n/*0000*/                   LDC R1, c[0x0][0x28];\n/*0010*/                   ULDC.64 UR4, c[0x0][0x208];\n/*0020*/                   CS2R R4, SR_CLOCKLO;\n/*0030*/                   ISETP.NE.U32.AND P0, PT, R4, RZ, PT;\n/*0040*/                   ISETP.NE.U32.AND.EX P0, PT, R5, RZ, PT, P0;\n/*0050*/                   CS2R R2, SR_GLOBALTIMERLO;\n/*0060*/                   SEL R6, R2, RZ, P0;\n/*0070*/                   SEL R7, R3, RZ, P0;\n/*0080*/                   ISETP.NE.U32.AND P0, PT, R6, RZ, PT;\n/*0090*/                   ISETP.NE.U32.AND.EX P0, PT, R7, RZ, PT, P0;\n/*00a0*/                   CS2R R8, SR_CLOCKLO;\n/*00b0*/                   LDC.64 R2, c[0x0][0x210];\n/*00c0*/                   SEL R8, R8, RZ, P0;\n/*00d0*/                   SEL R9, R9, RZ, P0;\n/*00e0*/                   STG.E.64 desc[UR4][R2.64], R4;\n/*00f0*/                   STG.E.64 desc[UR4][R2.64+0x8], R6;\n/*0100*/                   STG.E.64 desc[UR4][R2.64+0x10], R8;\n/*0110*/                   EXIT;\n'


class ClockStampStatic(unittest.TestCase):
    def test_compiled_control_chain_is_retained(self):
        result=audit_fixture(SASS)
        self.assertEqual(result['special_reads'],3)
        self.assertEqual(len(result['dependent_exit_branches']),2)
        self.assertFalse(result['runtime_qualified'])

    def test_actual_select_lowering_does_not_count_as_control_dependency(self):
        with self.assertRaises(ValueError):audit_fixture(SELECT_LOWERING)

    def test_missing_half_or_branch_cannot_pass(self):
        for old,new in [('R3, RZ, PT, P0','R2, RZ, PT, P0'),('@P0 BRA 0xb0','BRA 0xb0'),
                        ('@!P0 BRA 0x30','@!P0 BRA 0xc0'),('R0, 0x2, PT','R0, 0x3, PT')]:
            self.assertIn(old,SASS)
            with self.subTest(old=old),self.assertRaises(ValueError):audit_fixture(SASS.replace(old,new,1))

    def test_zero_exit_and_retry_initialization_are_required(self):
        for old,new in [('CS2R R2, SRZ','CS2R R2, SR_CLOCKLO'),('IMAD.MOV.U32 R0, RZ, RZ, RZ','IMAD.MOV.U32 R0, RZ, RZ, 0x1')]:
            self.assertIn(old,SASS)
            with self.subTest(old=old),self.assertRaises(ValueError):audit_fixture(SASS.replace(old,new,1))

    def test_scope_is_not_a_general_cutlass_checker(self):
        with self.assertRaises(ValueError):audit_fixture(SASS.replace('r09_clock_stamp_ptx_fixture','cutlass_kernel'))
        with self.assertRaises(ValueError):audit_fixture(SASS,2)

    def test_inline_body_is_extracted_with_explicit_output_bindings(self):
        header=Path(__file__).with_name('r09_clock_stamp.cuh').read_text()
        single,double=inline_fixture(header,1),inline_fixture(header,2)
        self.assertEqual(single.count('mov.u64 before,%clock64;'),1)
        self.assertEqual(double.count('mov.u64 before,%clock64;'),2)
        self.assertNotIn('%%clock64',single)
        self.assertNotIn('%0',single)
        with self.assertRaises(ValueError):inline_fixture(header,3)


if __name__=='__main__':
    unittest.main()
