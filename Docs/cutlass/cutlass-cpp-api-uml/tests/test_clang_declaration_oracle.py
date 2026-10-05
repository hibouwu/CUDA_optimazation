"""A compiler-backed cross-check for the independently discovered identity bugs.

These reduced programs test the extractor's rules, not whole-library coverage.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
PIN='8f50b052e1099fb982392a622caab69b97b63128'
sys.path.insert(0,str(ROOT/'scripts'))
from extract_declarations import Extractor


def ast(source):
    p=subprocess.run(['clang++','-x','c++','-std=c++17','-fsyntax-only','-Xclang','-ast-dump=json','-'],
        input=source,text=True,capture_output=True,check=True)
    return json.loads(p.stdout)


def walk(node):
    yield node
    for child in node.get('inner',[]):yield from walk(child)


@unittest.skipUnless(shutil.which('clang++'),'clang++ is required for this independent oracle')
class ClangDeclarationOracle(unittest.TestCase):
    def test_reference_return_overloads_are_two_methods(self):
        source='struct S { S& f(int) { return *this; } S& f(double) { return *this; } };'
        compiler=[n for n in walk(ast(source))if n.get('kind')=='CXXMethodDecl' and n.get('name')=='f']
        self.assertEqual(len(compiler),2)
        self.assertEqual({n['type']['qualType']for n in compiler},{'S &(int)','S &(double)'})
        ex=Extractor(PIN);ex.extract('oracle.hpp',source.encode())
        ours=[e for e in ex.entities.values()if e['qualified_name']=='S::f']
        self.assertEqual(len(ours),2)
        self.assertTrue(all(e['kind']=='method'for e in ours))

    def test_friend_is_namespace_function_and_redeclared(self):
        source='namespace n { struct S { friend int f(S const&); }; int f(S const& x) { return 1; } }'
        nodes=list(walk(ast(source)))
        f=[n for n in nodes if n.get('name')=='f' and n.get('kind')=='FunctionDecl']
        self.assertEqual(len(f),2)
        self.assertEqual(f[1]['previousDecl'],f[0]['id'])
        self.assertFalse([n for n in nodes if n.get('name')=='f' and n.get('kind')=='CXXMethodDecl'])
        ex=Extractor(PIN);ex.extract('oracle.hpp',source.encode())
        ours=[e for e in ex.entities.values()if e['qualified_name']=='n::f']
        self.assertEqual(len(ours),1)
        self.assertEqual(ours[0]['kind'],'function')

    def test_compound_namespace_reuses_identity(self):
        source='namespace a::b { int f(); } namespace a { namespace b { int f(); } }'
        f=[n for n in walk(ast(source))if n.get('name')=='f' and n.get('kind')=='FunctionDecl']
        self.assertEqual(len(f),2)
        self.assertEqual(f[1]['previousDecl'],f[0]['id'])
        ex=Extractor(PIN);ex.extract('oracle.hpp',source.encode())
        ours=[e for e in ex.entities.values()if e['qualified_name']=='a::b::f']
        self.assertEqual(len(ours),1)


if __name__=='__main__':unittest.main()
