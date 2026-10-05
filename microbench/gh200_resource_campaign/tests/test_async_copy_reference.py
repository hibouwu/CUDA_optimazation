"""Independent enumeration checks for the S12 modular-address host oracle."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORDS = 8388608 // 4


def enumerate_checksum(width, blocks, block, consumer, steps, seed):
    total = 0
    producer = (consumer + 1) % 128
    for step in range(steps):
        byte_offset = ((block * 128 + producer + step * blocks * 128) * width) % (WORDS * 4)
        for byte in range(0, width, 4):
            total += 17 * ((byte_offset + byte) // 4) + seed
    return total % (1 << 32)


def check_lifecycle(stages, steps, *, fixed_wait=False, omit_release=False):
    pending = list(range(min(stages, steps)))
    slots = {step % stages: step for step in pending}
    for step in range(steps):
        remaining = len(pending)
        wait = stages - 1 if fixed_wait else min(stages - 1, steps - step - 1)
        # A wait proves at least this many oldest groups have completed.
        completed = remaining - wait
        if completed <= 0 or pending[0] != step:
            raise ValueError('current producer group is not proved complete')
        if slots[step % stages] != step:
            raise ValueError('slot overwritten before consumption')
        pending.pop(0)
        if step + stages < steps:
            if omit_release:
                raise ValueError('cross-thread consumer may still read the slot')
            slots[step % stages] = step + stages
            pending.append(step + stages)
    if pending:
        raise ValueError('pending groups after drain')


class AsyncCopyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('g++ required for the real host reference')
        cls.temp = tempfile.TemporaryDirectory()
        source = Path(cls.temp.name) / 'reference.cpp'
        source.write_text('#include "' + str(ROOT / 'common/async_copy_reference.hpp') + '"\n'
            '#include <iostream>\nint main(){unsigned w,g,b,t,n,s;'
            'while(std::cin>>w>>g>>b>>t>>n>>s) std::cout<<async_copy_reference::checksum(w,g,b,t,n,s)<<"\\n";}\n')
        cls.binary = Path(cls.temp.name) / 'reference'
        subprocess.run([compiler, '-std=c++17', '-O2', str(source), '-o', str(cls.binary)], check=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_shapes_and_long_modular_wrap(self):
        coordinates = []
        for width in (4, 8, 16):
            for blocks in (1, 3, 132, 528):
                for consumer in (0, 31, 32, 127):
                    for steps in (1, 8, 16, 65536):
                        coordinates.append((width, blocks, blocks - 1, consumer, steps, 4294967291))
        text = ''.join(' '.join(map(str, row)) + '\n' for row in coordinates)
        result = subprocess.run([str(self.binary)], input=text, text=True, capture_output=True, check=True, timeout=10)
        values = list(map(int, result.stdout.splitlines()))
        self.assertEqual(len(values), len(coordinates))
        for row, value in zip(coordinates, values):
            self.assertEqual(value, enumerate_checksum(*row), row)

    def test_pipeline_tail_and_release(self):
        for stages in (1, 2, 4):
            for steps in (1, 2, 3, 4, 7, 8, 16, 65536):
                check_lifecycle(stages, steps)
        for stages in (2, 4):
            with self.assertRaises(ValueError):
                check_lifecycle(stages, 16, fixed_wait=True)
        for stages in (1, 2, 4):
            with self.assertRaises(ValueError):
                check_lifecycle(stages, 16, omit_release=True)

    def test_frozen_contract_work_and_alignment(self):
        contract = json.loads((ROOT / 'contracts/async_copy.json').read_text())
        self.assertEqual(len(contract['cases']), 24)
        self.assertEqual(len({c['id'] for c in contract['cases']}), 24)
        for case in contract['cases']:
            p = case['parameters']; width = p['request_bytes_per_thread']; stages = p['stages']
            self.assertEqual(p['shared_payload_bytes'], stages * 128 * width)
            self.assertEqual(p['global_array_bytes'] % width, 0)
            # This full-GPU payload already exceeds uint32; never compute it in unsigned int.
            work = 528 * 128 * 8192 * 8 * width
            self.assertGreater(work, (1 << 32) - 1)


if __name__ == '__main__':
    unittest.main()
