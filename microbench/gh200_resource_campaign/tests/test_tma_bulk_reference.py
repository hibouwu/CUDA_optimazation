"""CPU checks for the64-bit address ring, full destination oracle and work unit."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class TmaBulkReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which('g++')
        if not compiler:raise unittest.SkipTest('g++ unavailable')
        cls.tmp=tempfile.TemporaryDirectory(); cls.binary=Path(cls.tmp.name)/'reference'
        source=Path(cls.tmp.name)/'reference.cpp'
        source.write_text('''#include "tma_bulk_reference.hpp"
#include <iostream>
int main(){using namespace tma_bulk_reference;u64 B,Q,I,b,s,w;u32 seed;
while(std::cin>>B>>Q>>I>>b>>s>>w>>seed){try{
std::cout<<allocation_bytes(B,Q)<<" "<<completed_payload(B,I,Q)<<" "
<<slot_offset(b,I-1,B,Q)<<" "<<global_input_word(b,I-1,w,B,Q,seed)<<" "
<<destination_word(b,s,w,B,Q,I,seed)<<"\\n";
}catch(const std::exception&){std::cout<<"rejected\\n";}}}
''')
        subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-I',str(ROOT/'common'),str(source),'-o',str(cls.binary)],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_ring_boundary_full_word_coverage_and_modular_seed(self):
        cases=[]
        for Q in (1024,4096,8192,16384,32768,65536):
            for I,seed in ((1,0),(2,3),(33,2**32-1),(32,3),(128,3),(65536,2**32-1)):
                for B,b in ((1,0),(528,527),(4097,4096)):
                    for s in (0,1,31):
                        for w in (0,Q//4-1):cases.append((B,Q,I,b,s,w,seed))
        rows=subprocess.run([str(self.binary)],input=''.join(' '.join(map(str,c))+'\n' for c in cases),text=True,capture_output=True,check=True).stdout.splitlines()
        self.assertEqual(len(rows),len(cases))
        for c,row in zip(cases,rows):
            B,Q,I,b,s,w,seed=c
            last=(I-1)%32
            offset=(b*32+last)*Q
            source=(17*((b*32+last)*(Q//4)+w)+seed)%(2**32)
            value=(29*(b*(Q//4)+w)+seed)%(2**32)
            # Explicit visit enumeration is independent of the C++ coverage rule.
            visited=set(i%32 for i in range(min(I,33)))
            output=value if s in visited else value^(2**32-1)
            self.assertEqual(list(map(int,row.split())),[B*32*Q+32,B*I*Q,offset,source,output],c)
        self.assertGreater((4096*32+31)*65536,2**32)

    def test_domain_and_overflow_rejected(self):
        cases=[(0,1024,1,0,0,0,0),(1,2048,1,0,0,0,0),
               (2**63,65536,1,0,0,0,0),(1,1024,0,0,0,0,0),
               (1,1024,65537,0,0,0,0),(1,1024,1,1,0,0,0),
               (1,1024,1,0,32,0,0),(1,1024,1,0,0,256,0),
               (2**40,65536,65536,0,0,0,0)]
        rows=subprocess.run([str(self.binary)],input=''.join(' '.join(map(str,c))+'\n' for c in cases),text=True,capture_output=True,check=True).stdout.splitlines()
        self.assertEqual(len(rows),len(cases))
        self.assertTrue(all(row.endswith('rejected') for row in rows))

    def test_contract_matrix_and_single_transport_count(self):
        c=json.loads((ROOT/'contracts/tma_bulk.json').read_text())
        self.assertEqual(len(c['cases']),24)
        self.assertEqual({x['parameters']['payload_bytes'] for x in c['cases']},{1024,4096,8192,16384,32768,65536})
        for x in c['cases']:
            self.assertEqual(x['threads'],128)
            self.assertEqual(x['parameters']['global_slots_per_cta'],32)
            self.assertEqual(x['work_unit'],'byte')


if __name__=='__main__':unittest.main()
