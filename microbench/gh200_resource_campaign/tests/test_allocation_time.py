"""Allocation duration policy with the saved S14 Slurm response as fixture."""
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from runners import environment

FIXTURE = CODE.parents[1] / 'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s14-preflight-deployment-a/allocation-731147.json'


class AllocationTimeTests(unittest.TestCase):
    def test_finite_partition_bounds(self):
        for partition, limit in [('instant', '00:10:00'), ('instant', '01:00:00'),
                                 ('short', '00:10:00'), ('short', '03:00:00'),
                                 ('short', '0-03:00:00')]:
            with self.subTest(partition=partition, limit=limit):
                environment.check_allocation_time(f'Partition={partition} TimeLimit={limit}')

    def test_reject_invalid_or_excessive_limits(self):
        for fields in ['Partition=instant TimeLimit=01:00:01',
                       'Partition=short TimeLimit=03:00:01',
                       'Partition=short TimeLimit=1-00:00:00',
                       'Partition=short TimeLimit=00:00:00',
                       'Partition=short TimeLimit=UNLIMITED',
                       'Partition=short TimeLimit=Unknown',
                       'Partition=short TimeLimit=03:60:00',
                       'Partition=short TimeLimit=-01:00:00',
                       'Partition=long TimeLimit=00:10:00',
                       'Partition=instant,short TimeLimit=00:10:00',
                       'Partition=short', 'TimeLimit=03:00:00',
                       'Partition=short Partition=instant TimeLimit=00:10:00',
                       'Partition=short TimeLimit=03:00:00 TimeLimit=00:10:00']:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                environment.check_allocation_time(fields)

    def test_real_allocation_interface_and_preserved_identity(self):
        saved = json.loads(FIXTURE.read_text())
        responses = {
            ('scontrol', 'show', 'config'): saved['slurm_config'],
            ('scontrol', 'show', 'hostnames', 'romeo-a057'): 'romeo-a057\n',
            ('nvidia-smi', '-i', saved['cuda_visible_devices'], '--query-gpu=uuid,name,driver_version', '--format=csv,noheader,nounits'):
                ','.join(saved[key] for key in ('uuid', 'name', 'driver')) + '\n',
            ('/fixture/nvcc', '--version'): saved['compiler'],
        }
        observed = []
        for info in [saved['job_info'], saved['job_info'].replace('Partition=instant', 'Partition=short').replace('TimeLimit=00:10:00', 'TimeLimit=03:00:00')]:
            responses[('scontrol', 'show', 'job', '-o', saved['job'])] = info
            with patch.dict(environment.os.environ, {'SLURM_JOB_ID': saved['job'], 'CUDA_VISIBLE_DEVICES': saved['cuda_visible_devices']}), \
                 patch.object(environment, 'capture', side_effect=lambda args: responses[tuple(args)]), \
                 patch.object(environment.os, 'uname', return_value=types.SimpleNamespace(nodename=saved['host'])), \
                 patch.object(environment.os, 'geteuid', return_value=saved['execution_uid']), \
                 patch.object(environment.shutil, 'which', side_effect=lambda name: '/fixture/' + name), \
                 patch.object(environment, 'sha', return_value='CPU-fixture-tool-hash'):
                result = environment.inspect_allocation()
            self.assertEqual(set(result), set(saved))
            self.assertEqual(result['job_info'], info)
            observed.append(environment.environment_identity(result))
        self.assertEqual(observed[0], observed[1])
        for key in ('uuid', 'name', 'driver', 'compiler', 'execution_uid'):
            self.assertEqual(observed[0][key], saved[key])

    def test_job_budget_still_enforces_point_timeout(self):
        with patch.object(environment, 'remaining_seconds', return_value=139):
            with self.assertRaises(environment.Checkpoint):
                environment.budget({'job': '1'}, 120)
        with patch.object(environment, 'remaining_seconds', return_value=140):
            environment.budget({'job': '1'}, 120)


if __name__ == '__main__':
    unittest.main()
