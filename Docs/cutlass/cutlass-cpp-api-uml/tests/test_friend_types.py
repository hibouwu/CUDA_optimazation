import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from extract_declarations import Extractor


class FriendTypeTests(unittest.TestCase):
    def extract(self,source):
        result=Extractor('friend-type-regression');result.extract('fixture.hpp',source.encode());return result

    def test_forward_class_and_struct_are_one_type(self):
        e=self.extract('class F; struct F {};')
        self.assertEqual(len(e.entities),1)
        self.assertEqual(len(e.occurrences),2)

    def test_friend_class_and_struct_definition_are_one_type(self):
        e=self.extract('namespace n {struct S {private:friend class F;};struct F {}; }')
        fs=[o for o in e.occurrences if o['name']=='F']
        self.assertEqual(len(fs),2)
        self.assertEqual(len({o['entity_id']for o in fs}),1)
        self.assertEqual({o['qualified_name']for o in fs},{'n::F'})
        self.assertFalse(e.diagnostics)

    def test_existing_nested_type_is_not_namespace_injection(self):
        e=self.extract('namespace n {struct S {struct F {};friend struct F;};}')
        fs=[o for o in e.occurrences if o['name']=='F']
        self.assertEqual(len(fs),2)
        self.assertEqual(len({o['entity_id']for o in fs}),1)
        self.assertEqual({o['qualified_name']for o in fs},{'n::S::F'})

    def test_qualified_and_simple_type_friends_are_references(self):
        e=self.extract('namespace n {struct F {};struct S {friend class n::F;friend F;};}')
        refs=[o for o in e.occurrences if o['kind']=='friend_type_reference']
        self.assertEqual({o['target_type']for o in refs},{'n::F','F'})
        self.assertTrue(all(not o['is_definition']for o in refs))
        self.assertFalse(e.diagnostics)

    def test_existing_outer_namespace_type_is_not_redeclared_in_inner_namespace(self):
        e=self.extract('struct F {}; namespace n {struct S {friend struct F;};}')
        fs=[o for o in e.occurrences if o['name']=='F']
        self.assertEqual(len(fs),2)
        self.assertEqual({o['qualified_name']for o in fs},{'F'})
        self.assertEqual(len({o['entity_id']for o in fs}),1)

    def test_friend_template_owns_only_its_own_template_parameters(self):
        e=self.extract('template<class T> struct S {template<class U> friend struct F;};template<class V> struct F {};')
        fs=[o for o in e.occurrences if o['name']=='F']
        self.assertEqual(len(fs),2)
        self.assertEqual(len({o['entity_id']for o in fs}),1)
        self.assertEqual(len(fs[0]['template_parameters']),1)
        self.assertEqual(fs[0]['template_parameters'][0]['parameters'][0]['name'],'U')
        self.assertFalse(e.diagnostics)


if __name__=='__main__':unittest.main()
