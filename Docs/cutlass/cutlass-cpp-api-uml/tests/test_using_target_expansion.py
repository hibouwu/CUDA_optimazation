"""Independent exact-source and adversarial macro-target spelling checks."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from using_target_expansion import expand_using_target, PROVIDERS
from scan_candidates import lex


class UsingTargetExpansionTests(unittest.TestCase):
    def test_isolated_absolute_import_without_sys_path_in_tmp(self):
        code = '''
import importlib.util
from pathlib import Path
import sys
file=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location("isolated_using_target_review",file)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
path="include/cute/util/type_traits.hpp"
source=(module.ROOT/"snapshot"/path).read_bytes()
needle=b"CUTE_STL_NAMESPACE::remove_cv_t"
start=source.index(needle)
result=module.expand_using_target(path,source,{"start_byte":start,"end_byte":start+len(needle)})
assert not result["pending"],result["pending"]
assert {v["expanded_spelling"] for v in result["variants"]}=={"std::remove_cv_t","cuda::std::remove_cv_t"}
assert str(file.parent) not in sys.path,sys.path
print("ISOLATED_IMPORT_AND_EXPANSION_OK")
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code,
                                 str(ROOT / 'scripts/using_target_expansion.py')],
                                cwd='/tmp', text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'ISOLATED_IMPORT_AND_EXPANSION_OK')

    def expand(self, source, expression, *, path='include/target_review.hpp', files=None):
        raw = source.encode() if isinstance(source, str) else source
        needle = expression.encode() if isinstance(expression, str) else expression
        start = raw.index(needle)
        def reader(p):
            if files is not None and p in files:
                return files[p]
            file = ROOT / 'snapshot' / p
            return file.read_bytes() if file.is_file() else None
        return expand_using_target(path, raw, {'path': path, 'start_byte': start,
                                              'end_byte': start + len(needle)}, reader)

    def actual(self, path, expression):
        return self.expand((ROOT / 'snapshot' / path).read_bytes(), expression, path=path)

    def spellings(self, result):
        return {''.join(t.text for t in lex(v['expanded_spelling'].encode())[0])
                for v in result['variants'] if v['validation_status'] == 'source_proven'}

    def assert_pending(self, result, reason=None):
        self.assertTrue(result['pending'])
        self.assertFalse(result['completeness']['all_expansion_spellings_proven'])
        if reason:
            self.assertIn(reason, {p['reason'] for p in result['pending']})

    def test_actual_cute_target_has_both_real_provider_branches(self):
        result = self.actual('include/cute/util/type_traits.hpp', 'CUTE_STL_NAMESPACE::remove_cv_t')
        self.assertEqual(self.spellings(result), {'std::remove_cv_t', 'cuda::std::remove_cv_t'})
        self.assertFalse(result['pending'])
        self.assertFalse(result['completeness']['cpp_lookup_performed'])
        self.assertEqual(result['source_range']['start_line'], 92)
        for variant in result['variants']:
            definition = variant['macro_definition_chain'][0]
            self.assertEqual(definition['source_sha256'], PROVIDERS['CUTE_STL_NAMESPACE'][1])
            self.assertEqual(definition['include_chain'][-1]['target_path'], 'include/cute/config.hpp')
            self.assertTrue(variant['conditions'])
            self.assertNotIn('entity_id', variant)

    def test_actual_cmath_empty_replacement_is_defined_and_root_qualified(self):
        result = self.actual('include/cutlass/functional.h', 'CUTLASS_CMATH_NAMESPACE :: isnan')
        self.assertEqual(self.spellings(result), {'::isnan', 'std::isnan'})
        self.assertFalse(result['pending'])
        empty = next(v for v in result['variants'] if v['macro_bindings']['CUTLASS_CMATH_NAMESPACE'] == '')
        self.assertTrue(empty['expanded_root_qualified'])
        deletion = next(m for m in empty['source_mapping'] if m.get('macro_name'))
        self.assertEqual(deletion['virtual_start_byte'], deletion['virtual_end_byte'])
        self.assertLess(deletion['source_range']['start_byte'], deletion['source_range']['end_byte'])
        self.assertEqual(empty['macro_definition_chain'][0]['command'], 'define')

    def test_actual_cutlass_stl_keeps_external_override_as_third_pending_branch(self):
        result = self.actual('include/cutlass/platform/platform.h', 'CUTLASS_STL_NAMESPACE::integral_constant')
        self.assertEqual(self.spellings(result), {'std::integral_constant', 'cuda::std::integral_constant'})
        self.assertEqual(len(result['variants']), 3)
        self.assert_pending(result, 'external_override_binding_required')
        external = next(v for v in result['variants'] if v['validation_status'] == 'pending')
        self.assertIsNone(external['expanded_spelling'])
        self.assertEqual(external['conditions'][0]['expression'], 'defined(CUTLASS_STL_NAMESPACE)')
        self.assertTrue(external['missing_external_bindings'])
        self.assertEqual(external['source_contract_status'], 'parameterized_external_binding')
        self.assertEqual(result['source_contract_status'], 'source_proven_with_parameterized_external_binding')
        self.assertEqual(result['diagnostics'], [])
        self.assertTrue(result['completeness']['all_source_branches_accounted_for'])
        for v in result['variants']:
            self.assertTrue(any('CUTLASS_STL_NAMESPACE' in c['expression'] for c in v['conditions']))

    def test_include_must_precede_target_not_merely_exist_later(self):
        result = self.expand('using CUTE_STL_NAMESPACE::T;\n#include <cute/config.hpp>\n', 'CUTE_STL_NAMESPACE::T')
        self.assert_pending(result, 'provider_not_proven_before_target')
        self.assertFalse(self.spellings(result))

    def test_pragma_once_does_not_resurrect_macro_after_transitive_undef(self):
        files = {'include/prelude.hpp': b'#include <cutlass/cutlass.h>\n#undef CUTLASS_CMATH_NAMESPACE\n'}
        source = '#include "prelude.hpp"\n#include <cutlass/detail/helper_macros.hpp>\nusing CUTLASS_CMATH_NAMESPACE :: isnan;'
        result = self.expand(source, 'CUTLASS_CMATH_NAMESPACE :: isnan', files=files)
        self.assert_pending(result, 'macro_undefined_or_redefined_in_include_prefix')
        self.assertFalse(self.spellings(result))
        self.assertEqual(sum(e['command'] == 'undef' for e in result['proof']['events']), 1)

    def test_conditionally_included_provider_is_not_unconditional_proof(self):
        source = '#if PROJECT_CUTE\n#include <cute/config.hpp>\n#endif\nusing CUTE_STL_NAMESPACE::T;'
        result = self.expand(source, 'CUTE_STL_NAMESPACE::T')
        self.assert_pending(result, 'conditional_provider_visibility')
        self.assertTrue(result['proof']['events'][0]['include_conditions'])

    def test_conditional_pragma_once_state_cannot_hide_other_branch_undef(self):
        files = {'include/mutator.hpp': b'#pragma once\n#if !defined(__CUDACC_RTC__)\n#undef CUTE_STL_NAMESPACE\n#endif\n'}
        source = '#include <cute/config.hpp>\n#if defined(__CUDACC_RTC__)\n#include "mutator.hpp"\n#endif\n#include "mutator.hpp"\nusing CUTE_STL_NAMESPACE::T;'
        result = self.expand(source, 'CUTE_STL_NAMESPACE::T', files=files)
        self.assert_pending(result, 'macro_undefined_or_redefined_in_include_prefix')
        self.assertFalse(self.spellings(result))

    def test_provider_tail_include_mutation_is_not_ignored(self):
        path = 'include/cute/util/debug.hpp'
        changed = (ROOT / 'snapshot' / path).read_bytes() + b'\n#undef CUTE_STL_NAMESPACE\n'
        result = self.expand('#include <cute/config.hpp>\nusing CUTE_STL_NAMESPACE::T;',
                             'CUTE_STL_NAMESPACE::T', files={path: changed})
        self.assert_pending(result, 'macro_undefined_or_redefined_in_include_prefix')

    def test_source_after_target_does_not_retroactively_undefine_macro(self):
        source = '#include <cute/config.hpp>\nusing CUTE_STL_NAMESPACE::T;\n#undef CUTE_STL_NAMESPACE\n'
        result = self.expand(source, 'CUTE_STL_NAMESPACE::T')
        self.assertEqual(self.spellings(result), {'std::T', 'cuda::std::T'})
        self.assertFalse(result['pending'])

    def test_changed_fixed_provider_or_local_override_remains_pending(self):
        path = 'include/cute/config.hpp'
        changed = (ROOT / 'snapshot' / path).read_bytes().replace(b'#  define CUTE_STL_NAMESPACE std', b'#  define CUTE_STL_NAMESPACE vendor')
        result = self.expand('#include <cute/config.hpp>\nusing CUTE_STL_NAMESPACE::T;',
                             'CUTE_STL_NAMESPACE::T', files={path: changed})
        self.assert_pending(result, 'fixed_provider_source_changed')
        result = self.expand('#define CUTLASS_STL_NAMESPACE SELECTED_STL\n#include <cutlass/platform/platform.h>\n#define SELECTED_STL vendor\nusing CUTLASS_STL_NAMESPACE::T;',
                             'CUTLASS_STL_NAMESPACE::T')
        self.assert_pending(result, 'macro_undefined_or_redefined_in_include_prefix')
        self.assertFalse(self.spellings(result))

    def test_spliced_comment_does_not_invent_an_undef(self):
        source = b'#include <cute/config.hpp>\n// continuation \\\n#undef CUTE_STL_NAMESPACE\nusing CUTE_STL_NAMESPACE::T;'
        result = self.expand(source, 'CUTE_STL_NAMESPACE::T')
        self.assertFalse(result['pending'])
        self.assertEqual(self.spellings(result), {'std::T', 'cuda::std::T'})
        self.assertFalse(any(e['command'] == 'undef' for e in result['proof']['events']))

    def test_utf8_and_spliced_macro_token_mapping_use_physical_bytes(self):
        source = '// 中文前缀\n#include <cute/config.hpp>\nusing CUTE_STL_\\\nNAMESPACE::T;'
        expression = 'CUTE_STL_\\\nNAMESPACE::T'
        result = self.expand(source, expression)
        self.assertEqual(self.spellings(result), {'std::T', 'cuda::std::T'})
        raw = source.encode()
        self.assertEqual(result['source_range']['start_byte'], raw.index(expression.encode()))
        for v in result['variants']:
            encoded = v['expanded_spelling'].encode()
            for part in v['source_mapping']:
                s = part['source_range']; physical = raw[s['start_byte']:s['end_byte']]
                self.assertEqual(hashlib.sha256(physical).hexdigest(), s['raw_sha256'])
                actual = encoded[part['virtual_start_byte']:part['virtual_end_byte']]
                if part['mapping'] == 'exact_source_slice':
                    self.assertEqual(actual, physical)
                else:
                    self.assertEqual(actual.decode(), part['expanded_replacement'])
                    self.assertEqual(physical, b'CUTE_STL_\\\nNAMESPACE')

    def test_repeated_macro_is_one_correlated_choice_and_strings_are_not_replaced(self):
        expression = 'Box<CUTE_STL_NAMESPACE::T, CUTE_STL_NAMESPACE::U, decltype("CUTE_STL_NAMESPACE")>'
        result = self.expand('#include <cute/config.hpp>\nusing X=' + expression + ';', expression)
        self.assertEqual(len(result['variants']), 2)
        for variant in result['variants']:
            self.assertIn('"CUTE_STL_NAMESPACE"', variant['expanded_spelling'])
            self.assertEqual(sum(p['mapping'] == 'macro_token_replacement' for p in variant['source_mapping']), 2)

    def test_source_rtc_guard_rejects_impossible_other_target_branch(self):
        source = '#include <cute/config.hpp>\n#if defined(__CUDACC_RTC__)\nusing CUTE_STL_NAMESPACE::T;\n#endif'
        result = self.expand(source, 'CUTE_STL_NAMESPACE::T')
        self.assertEqual(self.spellings(result), {'cuda::std::T'})
        self.assertEqual(len(result['excluded_condition_combinations']), 1)

    def test_changed_condition_macro_cannot_reuse_historical_branch_as_current(self):
        source = '#include <cute/config.hpp>\n#define __CUDACC_RTC__ 1\nusing CUTE_STL_NAMESPACE::T;'
        self.assert_pending(self.expand(source, 'CUTE_STL_NAMESPACE::T'), 'condition_macro_state_changed_in_source_prefix')
        source = '#include <cutlass/platform/platform.h>\n#ifdef CUTLASS_STL_NAMESPACE\nusing CUTLASS_STL_NAMESPACE::T;\n#endif'
        result = self.expand(source, 'CUTLASS_STL_NAMESPACE::T')
        self.assert_pending(result, 'condition_macro_state_changed_in_source_prefix')
        self.assertEqual(len(result['variants']), 3)  # Do not delete defaults as a fake contradiction.

    def test_if_zero_undef_is_retained_as_inactive_evidence_not_applied(self):
        source = '#include <cute/config.hpp>\n#if 0\n#undef CUTE_STL_NAMESPACE\n#endif\nusing CUTE_STL_NAMESPACE::T;'
        result = self.expand(source, 'CUTE_STL_NAMESPACE::T')
        self.assertFalse(result['pending'])
        self.assertTrue(next(e for e in result['proof']['events'] if e['command'] == 'undef')['inactive'])

    def test_unknown_or_function_macro_and_replacement_rescan_are_pending(self):
        result = self.expand('#define TARGET(X) X\nusing TARGET(Base)::T;', 'TARGET(Base)::T')
        self.assert_pending(result, 'unsupported_target_macro')
        result = self.expand('#include <cute/config.hpp>\n#define std vendor\nusing CUTE_STL_NAMESPACE::T;', 'CUTE_STL_NAMESPACE::T')
        self.assert_pending(result, 'replacement_token_requires_macro_rescan')

    def test_generated_or_missing_library_include_is_not_fake_availability(self):
        for include, reason in [('"cutlass/review_missing_header.h"', 'missing_library_include_source'),
                                ('REVIEW_SELECTED_HEADER', 'unresolved_generated_include')]:
            result = self.expand('#include <cute/config.hpp>\n#include ' + include + '\nusing CUTE_STL_NAMESPACE::T;',
                                 'CUTE_STL_NAMESPACE::T')
            self.assert_pending(result, reason)
            self.assertEqual(result['source_contract_status'], 'source_contract_pending')
            self.assertTrue(result['diagnostics'])

    def test_cuda_std_include_macro_name_without_definition_proof_is_not_a_boundary(self):
        files = {'include/hidden_mutator.hpp': b'#undef CUTE_STL_NAMESPACE\n'}
        source = '#include <cute/config.hpp>\n#define CUDA_STD_HEADER(x) "hidden_mutator.hpp"\n#include CUDA_STD_HEADER(type_traits)\nusing CUTE_STL_NAMESPACE::T;'
        result = self.expand(source, 'CUTE_STL_NAMESPACE::T', files=files)
        self.assert_pending(result, 'unresolved_generated_include')
        self.assertFalse(self.spellings(result))
        self.assertTrue(result['diagnostics'])
        source = '#include <cutlass/cutlass.h>\n#include <cute/config.hpp>\n#undef CUDA_STD_HEADER\n#include CUDA_STD_HEADER(type_traits)\nusing CUTE_STL_NAMESPACE::T;'
        self.assert_pending(self.expand(source, 'CUTE_STL_NAMESPACE::T'), 'unresolved_generated_include')

    def test_fixed_cuda_std_include_macro_has_actual_definition_evidence(self):
        result = self.actual('include/cute/util/type_traits.hpp', 'CUTE_STL_NAMESPACE::remove_cv_t')
        boundaries = [b for b in result['proof']['boundaries'] if b['classification'] == 'external_generated_standard_header']
        self.assertTrue(boundaries)
        for boundary in boundaries:
            proof = boundary['include_macro_definition_evidence']
            self.assertEqual(len(proof), 1)
            self.assertEqual(proof[0]['source_range']['path'], 'include/cutlass/cutlass.h')
            self.assertEqual(proof[0]['body_spelling'], '<cuda/std/header>')

    def test_plain_target_is_not_resolved_to_a_new_entity(self):
        result = self.expand('namespace dst { using ::src::f; }', '::src::f')
        self.assertEqual(self.spellings(result), {'::src::f'})
        self.assertTrue(result['root_qualified_in_source'])
        self.assertEqual(result['variants'][0]['macro_definition_chain'], [])
        self.assertFalse(result['completeness']['cpp_lookup_performed'])

    def test_all_120_actual_using_sites_of_the_three_fixed_families(self):
        counts = {name: 0 for name in PROVIDERS}
        for path in ('include/cute/util/type_traits.hpp', 'include/cute/numeric/int.hpp',
                     'include/cute/numeric/integer_sequence.hpp', 'include/cute/container/array.hpp',
                     'include/cutlass/platform/platform.h', 'include/cutlass/functional.h'):
            source = (ROOT / 'snapshot' / path).read_bytes()
            tokens, _, _ = lex(source)
            for index, token in enumerate(tokens):
                if token.text != 'using' or tokens[index + 1].text not in PROVIDERS:
                    continue
                end = index + 1
                while tokens[end].text != ';':
                    end += 1
                name = tokens[index + 1].text
                result = expand_using_target(path, source, {'start_byte': tokens[index + 1].start,
                                                            'end_byte': tokens[end - 1].end})
                counts[name] += 1
                self.assertEqual(len(self.spellings(result)), 2, (path, token.start, result['pending']))
                self.assertEqual({p['reason'] for p in result['pending']},
                                 {'external_override_binding_required'} if name == 'CUTLASS_STL_NAMESPACE' else set())
        self.assertEqual(counts, {'CUTE_STL_NAMESPACE': 75, 'CUTLASS_CMATH_NAMESPACE': 5, 'CUTLASS_STL_NAMESPACE': 40})

    def test_invalid_physical_range_is_rejected_and_results_do_not_mutate_cache(self):
        source = b'using CUTE_STL_NAMESPACE::T;'
        with self.assertRaises(ValueError):
            expand_using_target('include/x.hpp', source, {'start_byte': 7, 'end_byte': 26})
        source = '#include <cute/config.hpp>\nusing CUTE_STL_NAMESPACE::T;'
        first = self.expand(source, 'CUTE_STL_NAMESPACE::T')
        pristine = copy.deepcopy(first)
        first['variants'][0]['macro_definition_chain'][0]['conditions'][0]['expression'] = 'BROKEN'
        second = self.expand(source, 'CUTE_STL_NAMESPACE::T')
        self.assertEqual(second, pristine)


if __name__ == '__main__':
    unittest.main()
