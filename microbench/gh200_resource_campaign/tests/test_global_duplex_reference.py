"""Compile the real C++ oracle; compare against independent vector enumeration."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class GlobalDuplexReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which('g++')
        if compiler is None:raise unittest.SkipTest('g++ required')
        cls.temp=tempfile.TemporaryDirectory();source=Path(cls.temp.name)/'reference.cpp';cls.binary=Path(cls.temp.name)/'reference'
        source.write_text('#include "'+str(ROOT/'common/global_duplex_reference.hpp')+'"\n#include <iostream>\n'
            'int main(){unsigned long long bytes,T,t;unsigned r,I,seed;'
            'while(std::cin>>bytes>>T>>t>>r>>I>>seed)std::cout<<global_duplex_reference::read_checksum(bytes,T,t,r,I,seed)<<"\\n";}\n')
        subprocess.run([compiler,'-std=c++17','-O2',str(source),'-o',str(cls.binary)],check=True,timeout=30)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_closed_form_against_enumeration(self):
        cases=[]
        for T in (1,32,256,768):
            for groups in (1,3,17):
                for reads in (0,1,2,4):
                    for iterations in (1,2,16):
                        cases.append((T*groups*16,T,T-1,reads,iterations,4294967295))
        result=subprocess.run([str(self.binary)],input=''.join(' '.join(map(str,row))+'\n' for row in cases),
            text=True,capture_output=True,check=True,timeout=10)
        outputs=list(map(int,result.stdout.splitlines()));self.assertEqual(len(outputs),len(cases))
        for (size,T,thread,reads,iterations,seed),actual in zip(cases,outputs):
            values=[];vectors=size//16
            for group in range(vectors//T):
                for request in range(reads):
                    index=((group*reads+request)*T+thread)%vectors
                    for lane in range(4):values.append((17*(4*index+lane)+seed)%(1<<32))
            expected=sum(values)*iterations%(1<<32)
            self.assertEqual(actual,expected)

    def test_address_ownership_and_multiplicity(self):
        from collections import Counter
        for T in (1,32,256):
            for groups in (1,3,17):
                V=T*groups
                for ratio in (1,2,4):
                    counts=Counter()
                    for thread in range(T):
                        for group in range(groups):
                            for request in range(ratio):
                                index=((group*ratio+request)*T+thread)%V
                                self.assertEqual(index%T,thread);counts[index]+=1
                    self.assertEqual(set(counts),set(range(V)))
                    self.assertEqual(set(counts.values()),{ratio})

    def test_contract_finite_matrix(self):
        contract=json.loads((ROOT/'contracts/global_duplex.json').read_text());cases=contract['cases']
        self.assertEqual(len(cases),18);self.assertEqual(len({c['id'] for c in cases}),18)
        for size in ('small','large'):
            rows=[c for c in cases if c['parameters']['working_set_class']==size]
            self.assertEqual(len(rows),9)
            ratios={(c['parameters']['read_requests_per_group'],c['parameters']['write_requests_per_group']) for c in rows}
            self.assertEqual(ratios,{(1,0),(0,1),(1,1),(2,1),(4,1),(1,2),(1,4)})
        self.assertEqual(251658240*16*(2+1),12079595520)


if __name__=='__main__':unittest.main()
