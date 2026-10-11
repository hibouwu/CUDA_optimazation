from copy import deepcopy
import unittest

from r09_plain_prefix_audit import describe_prefix


def fixture():
    setup=dict(config='cfg_b',m=2816,n=1536,k=4096,stages=6,grid=[1,132,1],swizzle=1,
        lda=4096,ldb=1536,ldd=1536,target_a_ptr=100,target_b_ptr=200,
        prefix_a_ptr=100,prefix_b_ptr=200,scratch_bytes=1368576)
    call=dict(plain_prefix_abi=1,plain_prefix_calls=16,warmup_calls=16,plain_prefix_same_buffers=True,
        warmup_policy='common-sw1-16',target_a_ptr=100,target_b_ptr=200,prefix_a_ptr=100,prefix_b_ptr=200,
        warmup_us=[50.]*16,plain_prefix_inter_call_us=[20.]*16)
    return setup,call


class PlainPrefixAudit(unittest.TestCase):
    def test_plain_module_identity_is_distinct_from_legacy_label(self):
        setup,call=fixture()
        description=describe_prefix(setup,call)
        self.assertEqual(description['protocol'],'plain-module-16-sw1/abi1')
        self.assertAlmostEqual(description['prefix_between_gap_fraction'],300/1100)
        self.assertEqual(description['last_prefix_to_target_event_gap_us'],20)
        del call['plain_prefix_abi']
        with self.assertRaises(ValueError):describe_prefix(setup,call)

    def test_equal_kernel_counts_do_not_hide_different_prepare_bytes(self):
        setup,call=fixture()
        other=deepcopy(setup);other['scratch_bytes']=827904
        a,b=describe_prefix(setup,call),describe_prefix(other,call)
        self.assertEqual(a['prefix_kernel_event_sum_us'],b['prefix_kernel_event_sum_us'])
        self.assertNotEqual(a['scratch_clear_bytes_per_invocation'],b['scratch_clear_bytes_per_invocation'])

    def test_transposed_grid_and_pointer_changes_fail(self):
        setup,call=fixture()
        with self.assertRaises(ValueError):describe_prefix(dict(setup,grid=[132,1,1]),call)
        with self.assertRaises(ValueError):describe_prefix(setup,dict(call,prefix_a_ptr=101))

    def test_incomplete_or_negative_event_history_fails(self):
        setup,call=fixture()
        for change in (dict(plain_prefix_inter_call_us=[20.]*15),dict(warmup_us=[50.]*15),
                       dict(plain_prefix_inter_call_us=[-1.]+[20.]*15)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                describe_prefix(setup,dict(call,**change))


if __name__=='__main__':unittest.main()
