import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('module_registry',ROOT/'scripts/build_module_registry.py')
REG=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(REG)

class ModuleRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=(ROOT/'data/scope.json').read_bytes();cls.scope=json.loads(raw)
        cls.data=REG.build_registry(cls.scope,scope_sha256=hashlib.sha256(raw).hexdigest(),generator_sha256='fixture')

    def test_fixed_824_single_owner_and_20_modules(self):
        self.assertEqual(self.data['counts']['files'],824)
        self.assertEqual(self.data['counts']['modules'],20)
        self.assertEqual(sum(self.data['counts']['files_by_module'].values()),824)
        self.assertEqual(len({x['path'] for x in self.data['files']}),824)
        for f in self.data['files']:
            self.assertEqual(REG.classify(f['path'])['rule_id'],f['rule_id'])

    def test_all_40_inl_files_are_owned_without_discovery_filters(self):
        self.assertEqual(self.data['counts']['files_by_extension']['.inl'],40)
        self.assertEqual(self.data['counts']['inl_by_module'],{'convolution':4,'epilogue_collective':6,'gemm_collective':30})

    def test_detail_and_protocol_exceptions_are_explicit(self):
        expected={'include/cutlass/detail/layout.hpp':'cutlass_foundation',
                  'include/cutlass/detail/collective/moe_stride_utils.hpp':'collective_support',
                  'include/cutlass/detail/sm100_tmem_helper.hpp':'collective_support',
                  'include/cutlass/arch/barrier.h':'sync_resource',
                  'include/cutlass/arch/grid_dependency_control.h':'sync_resource',
                  'include/cute/arch/tmem_allocator_sm100.hpp':'sync_resource',
                  'include/cute/atom/partitioner.hpp':'cute_core'}
        for path,owner in expected.items():self.assertEqual(REG.classify(path)['module_id'],owner)

    def test_cold_and_compatibility_files_not_excluded(self):
        for path,owner in [('include/cute/util/print_svg.hpp','cute_core'),
            ('include/cutlass/thread/matrix.h','cutlass_foundation'),
            ('include/cutlass/platform/platform.h','cutlass_foundation'),
            ('include/cutlass/experimental/distributed/device/dist_gemm_universal_wrapper.hpp','distributed')]:
            self.assertEqual(REG.classify(path)['module_id'],owner)

    def test_mixed_adapter_keeps_one_file_owner_not_one_generation(self):
        record=next(x for x in self.data['files'] if x['path']=='include/cutlass/gemm/device/gemm_universal_adapter.h')
        self.assertEqual(record['primary_module_id'],'gemm_device')
        self.assertEqual(record['navigation']['api_generation'],'pending_entity_review')
        self.assertFalse(self.data['policy']['api_coverage_claimed'])
        self.assertFalse(self.data['policy']['relationship_coverage_claimed'])

    def test_overlap_fails_instead_of_silently_using_order(self):
        rules=copy.deepcopy(REG.RULES);extra=copy.deepcopy(rules[0]);extra['rule_id']='conflicting';rules.append(extra)
        with self.assertRaisesRegex(ValueError,'Expected one owner'):
            REG.classify('include/cutlass/arch/barrier.h',rules)

    def test_unknown_domain_and_invalid_path_fail(self):
        for path in ['include/cutlass/new_domain/unreviewed.hpp','include/cute/new_domain/unreviewed.hpp','../include/cute/tensor.hpp','include/cute/../tensor.hpp']:
            with self.assertRaises(ValueError):REG.classify(path)

    def test_duplicate_or_removed_manifest_file_fails(self):
        changed=copy.deepcopy(self.scope);changed['files'][-1]=copy.deepcopy(changed['files'][0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):REG.build_registry(changed,scope_sha256='fixture',generator_sha256='fixture')
        changed=copy.deepcopy(self.scope);changed['files'].pop();changed['file_count']-=1
        with self.assertRaisesRegex(ValueError,'fixed 824'):REG.build_registry(changed,scope_sha256='fixture',generator_sha256='fixture')

    def test_navigation_hints_are_preserved_not_promoted_to_api_labels(self):
        original={x['path']:x for x in self.scope['files']}
        for item in self.data['files']:
            self.assertEqual(item['navigation']['architecture_name_hints'],original[item['path']]['architecture_name_hints'])
            self.assertEqual(item['navigation']['api_generation'],original[item['path']]['api_generation'])
            self.assertEqual(item['source_sha256'],original[item['path']]['sha256'])
            self.assertEqual(item['coverage']['declarations'],'not_asserted')

if __name__=='__main__':unittest.main()
