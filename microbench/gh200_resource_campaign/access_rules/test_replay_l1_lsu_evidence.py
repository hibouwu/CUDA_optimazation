"""原184指令纯读同体对照及cache语义之外变化的拒绝检查。"""
import re
import unittest

import cfg_b_composition as aggregate
import replay_l1_lsu_evidence as replay


class CacheSassMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = aggregate.DEFAULT_ARCHIVE.parents[1] / (
            '20261001-resource-suite-v2/global_duplex/formal-b3-v1/build/sass.txt')
        cls.text = path.read_text()
        cls.parts = cls.text.split('Function : ')
        cls.ca_index = next(i for i, part in enumerate(cls.parts)
                            if part.startswith('gd_read_ca_r1_w0\n'))

    def test_actual_184_instruction_body_differs_only_in_cache_semantics(self):
        result = replay.compare_cache_sass(self.text)
        self.assertTrue(result['matched_body'])
        self.assertEqual(result['instruction_count'], 184)
        self.assertEqual(result['differing_instructions'], 1)
        self.assertEqual(result['difference_pc'], '0x410')
        self.assertEqual(result['high64_xor'], '0x4000')

    def test_unrelated_instruction_change_is_rejected(self):
        original = self.parts[self.ca_index]
        changed = re.sub(r'(/\*\s*0000\s*\*/\s*)([A-Z][A-Z0-9_.]*)',
                         r'\1NOP', original, count=1)
        self.assertNotEqual(changed, original)
        parts = self.parts[:]
        parts[self.ca_index] = changed
        with self.assertRaisesRegex(ValueError, 'cache策略之外的机器码差异'):
            replay.compare_cache_sass('Function : '.join(parts))

    def test_extra_encoding_bit_is_rejected(self):
        original = self.parts[self.ca_index]
        changed = original.replace('000ea2000c1ebd00', '000ea2000c1ebd01', 1)
        self.assertNotEqual(changed, original)
        parts = self.parts[:]
        parts[self.ca_index] = changed
        with self.assertRaisesRegex(ValueError, 'cache语义/编码差异不符'):
            replay.compare_cache_sass('Function : '.join(parts))


if __name__ == '__main__':
    unittest.main()
