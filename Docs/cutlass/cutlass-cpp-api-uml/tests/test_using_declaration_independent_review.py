"""Bounded source inventory and Clang oracles, not a production using resolver."""
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from scan_candidates import lex


def written_using_forms(source):
    """Physical spellings only; never infer the owner or imported entity kind.

    This fixed-snapshot inventory does not expand macros or resolve C++ lookup.
    It distinguishes an alias's optional attribute from an operator= import.
    """
    tokens, _, issues = lex(source)
    if issues:
        raise AssertionError(issues)
    line_starts = [0] + [m.end() for m in re.finditer(b'\n', source)]
    result = []
    for index, token in enumerate(tokens):
        if token.text != 'using':
            continue
        stop = index + 1
        while stop < len(tokens) and tokens[stop].text != ';':
            stop += 1
        if stop == len(tokens):
            raise AssertionError('unterminated written using candidate')
        tail = tokens[index + 1:stop]
        words = [t.text for t in tail]
        if words[0] == 'namespace':
            kind = 'using_directive'
        elif words[0] == 'enum':
            kind = 'using_enum'
        else:
            after_name = 1
            while words[after_name:after_name + 2] == ['[', '[']:
                depth = 0
                while after_name < len(words):
                    word = words[after_name]
                    depth += (word == '[') - (word == ']')
                    after_name += 1
                    if depth == 0:
                        break
            kind = ('alias_declaration' if tail[0].kind == 'identifier'
                    and words[after_name:after_name + 1] == ['='] else 'using_import')
        end = tokens[stop].end
        result.append({'kind': kind, 'words': words, 'start_byte': token.start,
                       'end_byte': end, 'start_line': bisect_right(line_starts, token.start),
                       'raw': source[token.start:end].decode()})
    return result


class FixedUsingSourceInventory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = json.loads((ROOT / 'data/scope.json').read_text())
        cls.rows = []
        assert len(scope['files']) == 824
        for entry in scope['files']:
            source = (ROOT / 'snapshot' / entry['path']).read_bytes()
            assert hashlib.sha256(source).hexdigest() == entry['sha256']
            cls.rows.extend(dict(row, path=entry['path']) for row in written_using_forms(source))

    def test_physical_inventory_preserves_all_forms_and_inactive_branches(self):
        self.assertEqual(Counter(r['kind'] for r in self.rows), {
            'alias_declaration': 59136, 'using_import': 499, 'using_directive': 184})
        nonalias = [r for r in self.rows if r['kind'] != 'alias_declaration']
        self.assertEqual(len({r['path'] for r in nonalias}), 162)
        self.assertEqual(Counter(r['words'][1] for r in nonalias
                                 if r['kind'] == 'using_directive'),
                         {'cute': 172, 'detail': 11, 'SM90': 1})
        typename = [r for r in nonalias if r['words'][0] == 'typename']
        self.assertEqual([r['start_line'] for r in typename], list(range(73, 79)))
        self.assertEqual({r['path'] for r in typename},
                         {'include/cutlass/gemm/kernel/gemm_grouped_per_group_scale.h'})

    def test_attribute_alias_is_not_import_and_equal_operator_is_not_alias(self):
        result = written_using_forms(b'''// using fake::x;
const char* p = "using fake::y;";
using red [[deprecated("message = ;")]] = atomic_add<T>;
using Base::operator=;
using typename Base<T>::type;
using Base<T>::Base;
using namespace N;
using enum Color;
''')
        self.assertEqual([r['kind'] for r in result], [
            'alias_declaration', 'using_import', 'using_import', 'using_import',
            'using_directive', 'using_enum'])
        self.assertEqual(result[0]['start_line'], 3)
        actual = next(r for r in self.rows if r['path'] == 'include/cutlass/functional.h'
                      and r['start_line'] == 990)
        self.assertEqual(actual['kind'], 'alias_declaration')
        self.assertIn('[[deprecated("use atomic_add instead")]]', actual['raw'])

    def test_source_spelling_keeps_macro_root_scope_and_inherited_constructor_distinct(self):
        index = {(r['path'], r['start_line']): r for r in self.rows}
        for path, line, raw in [
            ('include/cute/util/type_traits.hpp', 92, 'using CUTE_STL_NAMESPACE::remove_cv_t;'),
            ('include/cutlass/fast_math.h', 58, 'using ::cuda::std::swap;'),
            ('include/cute/pointer.hpp', 88, 'using iter_adaptor<P, gmem_ptr<P>>::iter_adaptor;'),
            ('include/cutlass/pipeline/sm90_pipeline.hpp', 161, 'using ArrivalToken::ArrivalToken;'),
            ('include/cutlass/pipeline/sm90_pipeline.hpp', 165, 'using ArrivalToken::ArrivalToken;'),
        ]:
            self.assertEqual(index[path, line]['raw'], raw)
            self.assertEqual(index[path, line]['kind'], 'using_import')
        # Repeated terminal spelling is merely an inventory candidate, not a resolver.
        candidates = [r for r in self.rows if r['kind'] == 'using_import'
                      and r['words'][-1] in r['words'][:-1]]
        self.assertEqual(len(candidates), 83)


class UsingClangSemanticOracles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clang = shutil.which('clang++')
        if not cls.clang:
            raise unittest.SkipTest('Independent using oracle requires local Clang')

    def compile(self, source, *args, ast=False):
        result = subprocess.run([self.clang, '-std=c++17', '-fsyntax-only',
                                 *(['-Xclang', '-ast-dump=json'] if ast else []),
                                 *args, '-x', 'c++', '-'], input=source,
                                text=True, capture_output=True, timeout=30)
        return result

    def accept(self, source, *args, ast=False):
        result = self.compile(source, *args, ast=ast)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout) if ast else result

    def reject(self, source, diagnostic, *args):
        result = self.compile(source, *args)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(diagnostic, result.stderr)

    def test_one_using_import_retains_two_overloads_and_alias_is_different_decl(self):
        tree = self.accept('''
namespace src { void f(int); void f(double); struct T {}; }
namespace dst { using src::f; using src::T; using X = src::T; }
static_assert(__is_same(dst::T,src::T) && __is_same(dst::X,src::T));
static_assert(static_cast<void(*)(int)>(&dst::f) == static_cast<void(*)(int)>(&src::f));
static_assert(static_cast<void(*)(double)>(&dst::f) == static_cast<void(*)(double)>(&src::f));
''', ast=True)
        dst = next(n for n in tree['inner'] if n.get('name') == 'dst')
        shadows = [n['target'] for n in dst['inner'] if n['kind'] == 'UsingShadowDecl'
                   and n['target']['kind'] == 'FunctionDecl']
        self.assertEqual({n['type']['qualType'] for n in shadows}, {'void (int)', 'void (double)'})
        self.assertEqual(len({n['id'] for n in shadows}), 2)
        self.assertEqual([n['name'] for n in dst['inner'] if n['kind'] == 'TypeAliasDecl'], ['X'])

    def test_macro_target_branches_keep_one_local_lookup_name_and_reject_fake_qname(self):
        source = '''
namespace impl_a { struct Thing {}; } namespace impl_b { struct Thing {}; }
#if USE_B
#define TARGET impl_b
#else
#define TARGET impl_a
#endif
namespace owner { using TARGET::Thing; }
static_assert(__is_same(owner::Thing,TARGET::Thing));
'''
        for branch in (0, 1):
            self.accept(source, f'-DUSE_B={branch}')
            self.reject(source + 'using Wrong = owner::TARGET::Thing;',
                        f"no member named 'impl_{'b' if branch else 'a'}' in namespace 'owner'",
                        f'-DUSE_B={branch}')

    def test_directive_has_no_shadow_declarations_and_does_not_escape_local_block(self):
        source = '''
namespace src { constexpr int x=1; struct T {}; }
namespace dst { using namespace src; constexpr int before=x;
constexpr int x=2; constexpr int after=x; }
static_assert(dst::before==1 && dst::after==2 && dst::x==2);
static_assert(__is_same(dst::T,src::T));
void local_ok() { { using namespace src; (void)x; } }
'''
        tree = self.accept(source, ast=True)
        dst = next(n for n in tree['inner'] if n.get('name') == 'dst')
        directive = next(n for n in dst['inner'] if n['kind'] == 'UsingDirectiveDecl')
        self.assertNotIn('name', directive)
        self.assertEqual(directive['nominatedNamespace']['name'], 'src')
        self.assertFalse(any(n['kind'] == 'UsingShadowDecl' for n in dst['inner']))
        self.reject(source + 'void bad() { (void)x; }', "use of undeclared identifier 'x'")

    def test_inherited_member_and_constructor_access_do_not_create_fake_members(self):
        self.accept('''
struct Base { int member; explicit Base(int); };
struct Derived : Base { using Base::member; using Base::Base; };
static_assert(__is_same(decltype(&Derived::member),int Base::*));
template<class T> struct B { using type=T; explicit B(int); };
template<class T> struct D:B<T> { using typename B<T>::type; using B<T>::B; };
D<int> d(1); static_assert(__is_same(D<int>::type,int));
''')
        prefix = '''
enum class BarrierStatus { WaitAgain, WaitDone };
class ArrivalToken { public: ArrivalToken(BarrierStatus) {} ArrivalToken()=delete; };
class ProducerToken : public ArrivalToken { using ArrivalToken::ArrivalToken; };
'''
        self.accept(prefix + 'ProducerToken token(BarrierStatus::WaitDone);')
        self.reject(prefix + 'ProducerToken token;', 'implicitly-deleted default constructor')
        self.reject(prefix.replace('public: ArrivalToken(', 'protected: ArrivalToken(')
                    .replace('{ using ArrivalToken::', '{ public: using ArrivalToken::')
                    + 'ProducerToken token(BarrierStatus::WaitDone);', 'protected constructor')


if __name__ == '__main__':
    unittest.main()
