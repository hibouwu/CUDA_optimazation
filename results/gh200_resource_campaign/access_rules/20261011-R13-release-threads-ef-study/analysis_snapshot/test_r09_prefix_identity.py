import unittest
from analyze_r18 import validate_plain_prefix


class PrefixIdentityTests(unittest.TestCase):
    def fixtures(self):
        protocol = 'r09-plain-module-sw1-16/v1'
        row = dict(plain_prefix_protocol=protocol, warmup_policy='common-sw1-16', plain_prefix_clear_bytes=1368576)
        setup = dict(plain_prefix_protocol=protocol, plain_prefix_abi=1, scratch_bytes=1368576,
                     plain_prefix_clear_bytes=1368576, plain_prefix_preparation='trace-memset-device-sync-per-call')
        call = dict(setup, plain_prefix_calls=16, plain_prefix_same_buffers=True, plain_prefix_inter_call_us=[20.]*16)
        return setup, call, row

    def test_legacy_history_is_unchanged(self):
        validate_plain_prefix({}, {}, {})

    def test_undeclared_and_old_smoke_identity_cannot_be_pooled(self):
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            validate_plain_prefix({}, {'plain_prefix_abi':1}, {})
        setup, call, row = self.fixtures()
        with self.assertRaises(ValueError):
            validate_plain_prefix({}, call, row)

    def test_explicit_preparation_and_gaps_are_required(self):
        setup, call, row = self.fixtures()
        validate_plain_prefix(setup, call, row)
        for key, value in [('plain_prefix_clear_bytes',827904), ('plain_prefix_abi',2),
                           ('plain_prefix_preparation','unknown')]:
            changed = dict(call, **{key:value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plain_prefix(setup, changed, row)
        for gaps in ([0.]*15, [float('nan')]*16, [-1.]*16):
            with self.subTest(gaps=gaps), self.assertRaises(ValueError):
                validate_plain_prefix(setup, dict(call,plain_prefix_inter_call_us=gaps), row)


if __name__ == '__main__':
    unittest.main()
