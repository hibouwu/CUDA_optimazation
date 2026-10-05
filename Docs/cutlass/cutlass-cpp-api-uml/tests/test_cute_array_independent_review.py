"""Source-only CUTE array contracts and independent Host Clang oracles."""
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
ARRAY = ROOT / 'snapshot/include/cute/container/array.hpp'


class CuteArraySourceInventory(unittest.TestCase):
    def test_written_callable_inventory_and_zero_storage_specialization(self):
        text = ARRAY.read_text()
        primary, zero = text.split('template <class T>\nstruct array<T, 0>', 1)
        zero, free = zero.split('template <class T, size_t N>\nCUTE_HOST_DEVICE constexpr\nbool operator==', 1)
        self.assertEqual(primary.count('CUTE_HOST_DEVICE constexpr'), 22)
        self.assertEqual(zero.count('CUTE_HOST_DEVICE constexpr'), 22)
        self.assertEqual(free.count('CUTE_HOST_DEVICE constexpr'), 7)
        self.assertEqual(len(re.findall(r'\busing \w+\s*=', primary)), 10)
        self.assertEqual(len(re.findall(r'\busing \w+\s*=', zero)), 10)
        self.assertIn('element_type __elems_[N];', primary)
        self.assertNotIn('__elems_', zero)
        self.assertNotRegex(text, r'\brbegin\s*\([^)]*\)\s*(?:const\s*)?\{')

    def test_three_get_overloads_and_tuple_element_have_different_bound_contracts(self):
        text = ARRAY.read_text()
        self.assertEqual(text.count('static_assert(I < N, "Index out of range")'), 3)
        self.assertIn('T& get(array<T,N>& a)', text)
        self.assertIn('T const& get(array<T,N> const& a)', text)
        self.assertIn('T&& get(array<T,N>&& a)', text)
        self.assertNotIn('get(array<T,N> const&&', text)
        self.assertEqual(text.count('struct tuple_element<I, cute::array<T,N>>\n{\n  using type = T;\n};'), 2)

    def test_tuple_namespace_conditions_remain_distinct(self):
        text = ARRAY.read_text()
        self.assertIn('#include CUDA_STD_HEADER(tuple)', text)
        self.assertIn('namespace CUTE_STL_NAMESPACE', text)
        self.assertIn('#ifdef CUTE_STL_NAMESPACE_IS_CUDA_STD\nnamespace std', text)
        self.assertIn('#if (__CUDACC_VER_MAJOR__ >= 13)', text)
        self.assertIn('#include <cuda/std/__tuple_dir/structured_bindings.h>', text)
        self.assertIn('template <class... _Tp>\n  struct tuple_size;', text)
        config = (ROOT / 'snapshot/include/cute/config.hpp').read_text()
        self.assertIn('#if defined(__CUDACC_RTC__)\n#  define CUTE_STL_NAMESPACE cuda::std', config)
        self.assertIn('#  define CUTE_STL_NAMESPACE std', config)


class CuteArrayClangContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clang = shutil.which('clang++')
        cls.cuda = Path('/usr/local/cuda-13.0/targets/x86_64-linux/include')
        if not cls.clang or not (cls.cuda / 'cccl/cuda/std/utility').is_file():
            raise unittest.SkipTest('Independent Clang oracle requires local Clang and CUDA/CCCL headers')

    def compile(self, body):
        prefix = '#include "cute/container/array.hpp"\n#include <type_traits>\n#include <utility>\n#include <tuple>\n'
        return subprocess.run([self.clang, '-std=c++17', '-I', str(ROOT / 'snapshot/include'),
            '-I', str(self.cuda), '-I', str(self.cuda / 'cccl'), '-x', 'c++', '-fsyntax-only', '-'],
            input=prefix + body, text=True, capture_output=True, timeout=30)

    def accepts(self, body):
        result = self.compile(body)
        self.assertEqual(result.returncode, 0, result.stderr)

    def rejects(self, body, diagnostic):
        result = self.compile(body)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(diagnostic, result.stderr)
        self.assertNotIn('file not found', result.stderr)
        return result

    def test_aliases_const_access_and_get_value_categories(self):
        self.accepts('''
using A=cute::array<int,2>; using CV=cute::array<const volatile int,2>;
static_assert(std::is_same_v<CV::element_type,const volatile int>);
static_assert(std::is_same_v<CV::value_type,int>);
static_assert(std::is_same_v<CV::reference,const volatile int&>);
static_assert(std::is_same_v<CV::pointer,const volatile int*>);
static_assert(std::is_same_v<decltype(std::declval<A&>().begin()),int*>);
static_assert(std::is_same_v<decltype(std::declval<A const&>().begin()),int const*>);
static_assert(std::is_same_v<decltype(std::declval<A&>().cbegin()),int const*>);
static_assert(std::is_same_v<decltype(std::declval<A&>().cend()),int const*>);
static_assert(std::is_same_v<decltype(cute::get<0>(std::declval<A&>())),int&>);
static_assert(std::is_same_v<decltype(cute::get<0>(std::declval<A const&>())),int const&>);
static_assert(std::is_same_v<decltype(cute::get<0>(std::declval<A&&>())),int&&>);
static_assert(std::is_same_v<decltype(cute::get<0>(std::declval<A const&&>())),int const&>);
static_assert(std::is_same_v<decltype(cute::get<0>(std::declval<cute::array<const int,2>&&>())),int const&&>);
''')

    def test_zero_has_null_sentinels_no_element_storage_or_mutation_requirements(self):
        self.accepts('''
struct alignas(64) NoOps {
  NoOps()=delete; NoOps(int)=delete; NoOps(NoOps const&)=delete;
  NoOps& operator=(NoOps const&)=delete;
};
using Z=cute::array<NoOps,0>;
static_assert(std::is_empty_v<Z> && std::is_copy_constructible_v<Z>);
constexpr bool zero() { Z a{},b{}; a.clear(); a.swap(b); cute::swap(a,b);
  auto r=cute::reverse(a);
  return a.empty() && a.size()==0 && a.max_size()==0 && a.data()==nullptr
      && a.begin()==nullptr && a.end()==nullptr && a.cbegin()==nullptr && a.cend()==nullptr;
}
static_assert(zero());
void fill_without_assignment(NoOps const& v) { Z a; a.fill(v); cute::fill(a,v); }
static_assert(std::is_same_v<decltype(std::declval<Z&>().front()),NoOps&>);
''')

    def test_zero_free_clear_still_requires_T_zero_construction(self):
        result = self.rejects('''
struct NoZero { NoZero()=default; NoZero(int)=delete; };
void f() { cute::array<NoZero,0> a; cute::clear(a); }
''', 'deleted')
        self.assertIn('array.hpp:355', result.stderr)

    def test_get_bound_is_not_a_signature_constraint_and_tuple_element_is_unchecked(self):
        self.accepts('''
using A=cute::array<int,1>;
static_assert(std::is_same_v<decltype(cute::get<99>(std::declval<A&>())),int&>);
static_assert(std::is_same_v<std::tuple_element_t<99,cute::array<int,0>>,int>);
static_assert(std::tuple_size_v<cute::array<int,0>> == 0);
''')
        self.rejects('void f(){cute::array<int,1> a{}; (void)cute::get<1>(a);}', 'Index out of range')

    def test_zero_front_is_not_a_safe_value_access(self):
        self.rejects('''
constexpr int invalid(){cute::array<int,0> a{};return a.front();}
static_assert(invalid()==0);
''', 'constant expression')

    def test_reverse_is_a_new_array_not_in_place_and_clear_keeps_size(self):
        self.accepts('''
constexpr bool f() { cute::array<int,3> a{{1,2,3}}; auto r=cute::reverse(a);
  if(a[0]!=1 || r[0]!=3 || r[2]!=1) return false;
  a.clear(); return a.size()==3 && !a.empty() && a[0]==0 && a[2]==0; }
static_assert(f());
''')

    def test_reverse_positive_N_needs_copy_assignment_from_const_source(self):
        self.rejects('''
struct MoveOnly { MoveOnly()=default; MoveOnly(MoveOnly&&)=default;
  MoveOnly(MoveOnly const&)=delete; MoveOnly& operator=(MoveOnly const&)=delete; };
void f(){cute::array<MoveOnly,1> a{};(void)cute::reverse(a);}
''', 'deleted')

    def test_swap_uses_adl_but_wrapper_is_not_noexcept(self):
        self.accepts('''
namespace element { struct E { int v; };
constexpr void swap(E& a,E& b) noexcept {a.v=40;b.v=90;} }
constexpr bool f(){cute::array<element::E,1> a{{{1}}},b{{{2}}};a.swap(b);return a[0].v==40&&b[0].v==90;}
static_assert(f());
using A=cute::array<element::E,1>;
static_assert(!noexcept(std::declval<A&>().swap(std::declval<A&>())));
''')

    def test_array_equality_uses_element_not_equal_and_zero_does_not_erase_wellformedness(self):
        self.accepts('''
struct E {int v;}; bool operator==(E const&,E const&)=delete;
constexpr bool operator!=(E a,E b){return a.v!=b.v;}
constexpr cute::array<E,1> a{{{3}}},b{{{3}}};static_assert(a==b);
''')
        self.rejects('struct NoCompare{}; bool f(){cute::array<NoCompare,0> a,b;return a==b;}', 'invalid operands')

    def test_structured_binding_uses_cute_get_not_std_get(self):
        self.accepts('''
constexpr bool f(){cute::array<int,2> a{{1,2}};auto& [x,y]=a;x=7;return a[0]==7&&y==2;}
static_assert(f());
''')
        self.rejects('void f(){cute::array<int,2> a{};(void)std::get<0>(a);}', 'no matching function')


if __name__ == '__main__':
    unittest.main()
