"""Preprocessor expressions stay in the ledger without becoming fake APIs."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from extract_declarations import Extractor


class MacroContextTests(unittest.TestCase):
    def test_include_and_condition_are_not_declaration_macros(self):
        source = b'#include CUDA_STD_HEADER(tuple)\n#if CUDA_ARCH_FAMILY(1000)\nMAKE_DECL(X)\n#endif\n'
        e = Extractor('macro-context')
        e.extract('fixture.hpp', source)
        classified = e.files[0]['non_declaration_macro_uses']
        self.assertEqual({m['name']:m['classification'] for m in classified}, {
            'CUDA_STD_HEADER':'include_operand', 'CUDA_ARCH_FAMILY':'preprocessor_condition'})
        pending = [d.get('macro_name') for d in e.diagnostics if d['category']=='declaration_macro_expansion_pending']
        self.assertEqual(pending, ['MAKE_DECL'])
        for item in classified:
            self.assertEqual(source[item['start_byte']:item['end_byte']].decode(),item['expression'])


if __name__ == '__main__': unittest.main()
