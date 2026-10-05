"""S10 CPU oracle, finite matrix, byte accounting and negative checks."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from auditors.matrix_exchange import validate_contract,work,reference,fragment_coordinate,audit_sass


def function_body(source,signature):
    start=source.index(signature);brace=source.index('{',start);depth=1;end=brace+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]


class MatrixExchangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.contract=json.loads((BASE/'contracts/matrix_exchange.json').read_text())

    def test_finite_matrix_and_declared_work(self):
        validate_contract(self.contract)
        for case in self.contract['cases']:
            p=case['parameters'];w=work(case)
            if p['mode']=='shuffle':self.assertEqual(w['work'],65536*p['streams']*(case['threads']//32))
            else:
                direction=8192*8*p['matrices']*128
                self.assertEqual(w['work'],direction*(2 if p['mode']=='roundtrip' else 1))
                self.assertEqual(w['read']+w['write'],w['work'])
        self.assertEqual(work(next(c for c in self.contract['cases'] if c['id']=='load_x2_normal'))['work'],16777216)

    def test_contract_mutations(self):
        for field,value in [('work_unit','FLOP'),('threads',64),('iterations',8193)]:
            d=copy.deepcopy(self.contract);d['cases'][0][field]=value
            with self.assertRaises(ValueError):validate_contract(d)
        d=copy.deepcopy(self.contract);d['cases'][-1]['parameters']['membermask']=65535
        with self.assertRaises(ValueError):validate_contract(d)

    def test_lane_half_bijections_and_poison_bounds(self):
        for trans in (False,True):
            coords=[fragment_coordinate(t,h,trans) for t in range(32) for h in range(2)]
            self.assertEqual(set(coords),{(r,c) for r in range(8) for c in range(8)})
            self.assertEqual(len(set(coords)),len(coords))
        for case in self.contract['cases'][:18]:
            n=case['parameters']['matrices']
            row_addresses=[q*512+((lane//8)%n)*128+(lane%8)*16 for q in range(8) for lane in range(32)]
            self.assertTrue(all(a%16==0 and 0<=a<=4096-16 for a in row_addresses))
            output=reference(case,2,4294967295);self.assertEqual(len(output),32+256*n+2048)
            tail=output[-2048:]
            self.assertEqual(sum(x!=0xdead for x in tail),0 if case['parameters']['mode']=='load' else n*8*64)

    def test_actual_cpp_reference_against_independent_python(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('host C++ compiler unavailable')
        source=(BASE/'probes/matrix_exchange.cu').read_text()
        header='''#include <vector>\n#include <string>\n#include <iostream>\nstruct MXCase {std::string id;void* kernel;int matrices,mode,threads,streams;bool transpose;};\n'''
        functions=function_body(source,'unsigned short mx_value(')+'\n'+function_body(source,'std::vector<unsigned> mx_reference(')
        main='''\nint main(){MXCase c;int length;unsigned seed;while(std::cin>>c.matrices>>c.mode>>c.threads>>c.streams>>c.transpose>>length>>seed){auto values=mx_reference(c,length,seed);for(auto v:values)std::cout<<v<<" ";std::cout<<"\\n";}}'''
        rows=[];expects=[]
        for case in self.contract['cases']:
            p=case['parameters'];shuffle=p['mode']=='shuffle'
            for seed in (3,193,4294967295):
                for length in ((1,3,8,33,65536) if shuffle else (1,2,8192)):
                    rows.append(f"{p.get('matrices',0)} {dict(load=0,store=1,roundtrip=2,shuffle=3)[p['mode']]} {case['threads']} {p.get('streams',0)} {int(p.get('transpose',False))} {length} {seed}")
                    expects.append(reference(case,length,seed))
        with tempfile.TemporaryDirectory(prefix='gh200-matrix-oracle-') as tmp:
            root=Path(tmp);(root/'test.cpp').write_text(header+functions+main)
            subprocess.run([compiler,'-std=c++17','-O2',str(root/'test.cpp'),'-o',str(root/'test')],check=True,capture_output=True,timeout=30)
            result=subprocess.run([str(root/'test')],input='\n'.join(rows)+'\n',text=True,capture_output=True,check=True,timeout=30)
            actual=[list(map(int,line.split())) for line in result.stdout.splitlines()]
            self.assertEqual(actual,expects)
            # The same compiled host routine with one corrupted column coefficient must fail.
            (root/'test.cpp').write_text(header+functions.replace('11*col','10*col')+main)
            subprocess.run([compiler,'-std=c++17','-O2',str(root/'test.cpp'),'-o',str(root/'test')],check=True,capture_output=True,timeout=30)
            result=subprocess.run([str(root/'test')],input=rows[0]+'\n',text=True,capture_output=True,check=True,timeout=30)
            self.assertNotEqual(list(map(int,result.stdout.split())),expects[0])

    def test_instruction_direction_count_transpose_and_spill_rejected(self):
        case=self.contract['cases'][0];body=[(16*i,'LDSM.16.M88 R0, [R2];') for i in range(8)]
        text='Function : matrix_exchangeILi1ELb0ELi0_suffix\n/*0000*/ CS2R R0, SR_GLOBALTIMERLO;\n/*0010*/ BAR.SYNC 0;\n/*0200*/ STS [R0], R1;\n/*0210*/ BAR.SYNC 0;\n/*0220*/ CS2R R0, SR_GLOBALTIMERLO;\n'
        with patch('auditors.matrix_exchange.timed_loops',return_value=[(32,160,body)]):self.assertEqual(len(audit_sass(text,{'cases':[case]})),1)
        for bad in (body[:-1],body+[(150,'STSM.16.M88 [R2], R0;')],[(pc,op.replace('M88','MT88')) for pc,op in body],body+[(150,'LDS R0, [R2];')]):
            with patch('auditors.matrix_exchange.timed_loops',return_value=[(32,160,bad)]):
                with self.assertRaises(ValueError):audit_sass(text,{'cases':[case]})
        for corrupted in (text.replace('/*0010*/ BAR.SYNC 0;',''),text.replace('/*0210*/ BAR.SYNC 0;',''),text.replace('/*0200*/ STS [R0], R1;','')):
            with patch('auditors.matrix_exchange.timed_loops',return_value=[(32,160,body)]):
                with self.assertRaises(ValueError):audit_sass(corrupted,{'cases':[case]})
        with self.assertRaises(ValueError):audit_sass(text+'LDL R0,[R2];',{'cases':[case]})

    def test_shuffle_requires_eight_step_main_and_separate_remainder(self):
        case=next(c for c in self.contract['cases'] if c['id']=='shuffle_t32_streams1')
        text='Function : warp_exchangeILi32ELi1_suffix\n/*0000*/ CS2R R0, SR_GLOBALTIMERLO;\n/*0010*/ BAR.SYNC 0;\n/*0200*/ STS [R0], R1;\n/*0210*/ BAR.SYNC 0;\n/*0220*/ CS2R R0, SR_GLOBALTIMERLO;\n'
        main=[(32+16*i,'SHFL.IDX R0, R0, R1, 31;') for i in range(8)]
        tail=[(256,'SHFL.IDX R0, R0, R1, 31;')]
        loops=[(32,176,main),(256,272,tail)]
        with patch('auditors.matrix_exchange.timed_loops',return_value=loops):
            result=audit_sass(text,{'cases':[case]})
            self.assertEqual([x['counts']['SHFL'] for x in result[0]['loops']],[8,1])
        for bad in ([(32,176,main[:1]),(256,272,tail)],[(32,176,main)],[(32,176,main),(256,288,tail*2)]):
            with patch('auditors.matrix_exchange.timed_loops',return_value=bad):
                with self.assertRaises(ValueError):audit_sass(text,{'cases':[case]})

if __name__=='__main__':unittest.main()
