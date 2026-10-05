import random
import struct
import unittest
from common import auxiliary_reference_v1 as ref


class AuxiliaryReferenceTests(unittest.TestCase):
    def test_affine_against_scalar_modular_recurrence(self):
        random_source = random.Random(1801)
        for bits in (32, 64):
            mask = (1 << bits) - 1
            for _ in range(100):
                initial, multiplier, addend = [random_source.getrandbits(bits) for _ in range(3)]
                count = random_source.randrange(200)
                expected = initial
                for _ in range(count):
                    expected = (expected * multiplier + addend) & mask
                self.assertEqual(ref.affine(initial, multiplier, addend, count, bits), expected)

    def test_all_1024_conversion_patterns_against_host_ieee_encoding(self):
        for q in range(1024):
            exact = 1 + q / 1024
            half_bits = int.from_bytes(struct.pack('<e', exact), 'little')
            float_bits = int.from_bytes(struct.pack('<f', exact), 'little')
            self.assertEqual(half_bits, 0x3c00 | q)
            self.assertEqual(float_bits, 0x3f800000 | (q << 13))
            self.assertEqual((float_bits >> 13) & 1023, half_bits & 1023)

    def test_service_feedback_and_overflow(self):
        for seed in (0, 3, 0xffffffff):
            for thread in (0, 31, 127):
                for stream in range(4):
                    for count in (1, 2, 5, 33, 257):
                        x = ref.initial(seed, thread, stream)
                        a = ref.initial(seed ^ 0xa5a5a5a5, thread, stream) | 1
                        c = (ref.initial(seed ^ 0x5a5a5a5a, thread, stream) << 32) | ref.initial(seed ^ 0xc3c3c3c3, thread, stream)
                        for _ in range(count):
                            last = (x * a + c) & ref.MASK64
                            x = last & ref.MASK32
                        self.assertEqual(ref.service_result('mad_wide_u32', seed, thread, stream, count), last)
                        q = ref.initial(seed, thread, stream) & 1023
                        salt = (ref.initial(seed ^ 0x9e3779b9, thread, stream) & 1023) | 1
                        for i in range(count):
                            q = (q + salt + i) & 1023
                        self.assertEqual(ref.service_result('cvt_rn_f16_f32', seed, thread, stream, count), 0x3c00 | q)
                        self.assertEqual(ref.service_result('cvt_f32_f16', seed, thread, stream, count), 0x3f800000 | (q << 13))

    def test_atomic_requests_are_distinct_from_added_value(self):
        self.assertEqual(sum(1 + t % 7 for t in range(128)), 507)
        self.assertEqual(ref.atomic_results(0, 128, True, 5), [65794 + 507 * 5])
        self.assertEqual(len(ref.atomic_results(0, 128, False, 5)), 128)
        self.assertEqual(ref.atomic_results(0xffffffff, 128, True, 1)[0], (65793 + 507) & ref.MASK32)

    def test_reject_invalid_domains(self):
        for seed in (-1, 1 << 32, True, 0.5):
            with self.assertRaises(ValueError):
                ref.initial(seed, 0, 0)
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ref.word_result(0, 0, 0, n)
        with self.assertRaises(ValueError):
            ref.service_result('unknown', 0, 0, 0, 1)


if __name__ == '__main__':
    unittest.main()
