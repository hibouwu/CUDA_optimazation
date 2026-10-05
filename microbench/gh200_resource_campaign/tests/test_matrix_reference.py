"""Execute generated MMA operand setup and the C++ matrix oracle on the CPU.

These checks validate packing and reference arithmetic, not device MMA semantics.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('matrix_generator', BASE/'probes/generate_legacy_compute.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class MatrixReferenceTests(unittest.TestCase):
    def compile_run(self, source, stdin=''):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('host compiler unavailable')
        with tempfile.TemporaryDirectory(prefix='gh200-matrix-reference-') as directory:
            p=Path(directory);(p/'test.cpp').write_text(source)
            subprocess.run([compiler,'-std=c++17','-O2','-I',str(BASE/'common'),str(p/'test.cpp'),'-o',str(p/'test')],check=True,capture_output=True,timeout=30)
            return subprocess.run([str(p/'test')],input=stdin,text=True,capture_output=True,check=True,timeout=30).stdout

    @staticmethod
    def a_coordinate(kind,lane,e):
        # Independent decomposition by register pair, row half and K half.
        if kind in ('f16','bf16'):
            return lane//4+8*((e//2)&1),2*(lane%4)+(e&1)+8*(e//4)
        if kind=='tf32':return lane//4+8*(e%2),lane%4+4*(e//2)
        return lane//4,lane%4

    @staticmethod
    def b_coordinate(kind,lane,e):
        if kind in ('f16','bf16'):return 2*(lane%4)+(e%2)+8*(e//2),lane//4
        if kind=='tf32':return lane%4+4*e,lane//4
        return lane%4,lane//4

    def check_packing(self,kind,shape,seed,records):
        m,n,k=shape;seen={'A':set(),'B':set()}
        for name,lane,e,value in records:
            coordinate=self.a_coordinate(kind,lane,e) if name=='A' else self.b_coordinate(kind,lane,e)
            self.assertNotIn(coordinate,seen[name]);seen[name].add(coordinate)
            row,col=coordinate
            expected=(1+row+2*col+seed%3)/256 if name=='A' else (1+col+3*row+seed%5)/256
            self.assertEqual(value,expected,(kind,name,lane,e))
        self.assertEqual(seen['A'],{(r,c) for r in range(m) for c in range(k)})
        self.assertEqual(seen['B'],{(r,c) for r in range(k) for c in range(n)})

    def test_execute_actual_generated_mma_operand_setup(self):
        contract=json.loads((BASE/'contracts/legacy_mma.json').read_text())
        for kind in ('f16','bf16','tf32','f64'):
            case=next(c for c in contract['cases'] if c['parameters']['input_type']==kind)
            setup=generator.mma_body(case)[0]
            if kind in ('f16','bf16'):
                decode='decode_half' if kind=='f16' else 'decode_bf16'
                output=r'''for(int e=0;e<8;++e)std::cout<<"A "<<tid<<" "<<e<<" "<<DECODE((a[e/2]>>(16*(e%2)))&65535)<<"\n";
for(int e=0;e<4;++e)std::cout<<"B "<<tid<<" "<<e<<" "<<DECODE((b[e/2]>>(16*(e%2)))&65535)<<"\n";'''.replace('DECODE',decode)
            elif kind=='tf32':
                output=r'''for(int e=0;e<4;++e)std::cout<<"A "<<tid<<" "<<e<<" "<<decode_float(a[e])<<"\n";
for(int e=0;e<2;++e)std::cout<<"B "<<tid<<" "<<e<<" "<<decode_float(b[e])<<"\n";'''
            else:output=r'std::cout<<"A "<<tid<<" 0 "<<a[0]<<"\nB "<<tid<<" 0 "<<b[0]<<"\n";'
            # Host intrinsics only encode exactly representable dyadic test values.
            source=r'''
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
unsigned __float_as_uint(float x){unsigned u;std::memcpy(&u,&x,4);return u;}
float decode_float(unsigned u){float x;std::memcpy(&x,&u,4);return x;}
unsigned short __float2bfloat16_rn(float x){return __float_as_uint(x)>>16;}
unsigned short __bfloat16_as_ushort(unsigned short x){return x;}
unsigned short __float2half_rn(float x){auto u=__float_as_uint(x);return (((u>>23)&255)-112)*1024+((u&8388607)>>13);}
unsigned short __half_as_ushort(unsigned short x){return x;}
double decode_half(unsigned u){return std::ldexp(double(1024+(u&1023)),int(u>>10)-25);}
double decode_bf16(unsigned u){return decode_float(u<<16);}
struct {unsigned x;} threadIdx;
int main(){unsigned seed;std::cin>>seed;bool nonuniform=true;std::cout<<std::setprecision(17);
for(unsigned tid=0;tid<32;++tid){threadIdx.x=tid;
'''+setup+output+'\n}}\n'
            for seed in (3,193,4294967295):
                lines=self.compile_run(source,str(seed)).splitlines()
                rows=[(s[0],int(s[1]),int(s[2]),float(s[3])) for s in map(str.split,lines)]
                self.check_packing(kind,case['parameters']['shape'],seed,rows)
                bad=list(rows);other=next(i for i,r in enumerate(rows) if r[0]=='A' and r[3]!=rows[0][3])
                a,b=bad[0],bad[other];bad[0]=(*a[:3],b[3]);bad[other]=(*b[:3],a[3])
                with self.assertRaises(AssertionError):self.check_packing(kind,case['parameters']['shape'],seed,bad)

    def test_actual_cpp_matrix_and_output_coordinates(self):
        source=r'''
#include <iostream>
#include <iomanip>
#include "legacy_compute_reference.hpp"
int main(){int m,n,k,w,it,groups,chains;unsigned seed;bool varied;
std::cout<<std::setprecision(17);
while(std::cin>>m>>n>>k>>w>>it>>groups>>chains>>seed>>varied){
auto values=legacy_reference::matrix(m,n,k,groups,chains,it,16,seed,varied);
for(int g=0;g<groups;++g)for(int c=0;c<chains;++c)for(int t=0;t<w;++t)for(int e=0;e<m*n/w;++e){
auto rc=legacy_reference::output_coordinate(w,m,t,e);
std::cout<<g<<" "<<c<<" "<<t<<" "<<e<<" "<<rc.first<<" "<<rc.second<<" "
<<values[((g*chains+c)*m+rc.first)*n+rc.second]<<"\n";
}}
}
'''
        for m,n,k,width in ((8,8,4,32),(16,8,8,32),(16,8,16,32),(64,64,16,128)):
            for it,varied in ((1,True),(2,True),(65536,False)):
                seed=4294967295;groups=2;chains=2
                rows=self.compile_run(source,f'{m} {n} {k} {width} {it} {groups} {chains} {seed} {int(varied)}').splitlines()
                seen=set()
                for line in rows:
                    g,c,t,e,row,col=map(int,line.split()[:6]);value=float(line.split()[6]);seen.add((g,c,row,col))
                    lane=t%32
                    expected_row=(t//32)*16+lane//4+8*((e//2)%2) if width==128 else lane//4+(8*(e//2) if m==16 else 0)
                    expected_col=2*(lane%4)+(e%2)+(8*(e//4) if width==128 else 0)
                    self.assertEqual((row,col),(expected_row,expected_col))
                    product=sum((1+row+2*j+seed%3)*(1+col+3*j+seed%5) for j in range(k))
                    expected=product*it*16/65536+(1+g+c)/64 if varied else it*16*k/256
                    self.assertEqual(value,expected)
                self.assertEqual(len(seen),groups*chains*m*n)

    def test_wgmma_groups_share_one_input_tile(self):
        contract=json.loads((BASE/'contracts/legacy_wgmma.json').read_text())
        case=next(c for c in contract['cases'] if c['threads']==256)
        setup=generator.wgmma_body(case)[0]
        self.assertIn('unsigned short as[1024]',setup)
        self.assertIn('unsigned short bs[1024]',setup)
        self.assertIn('lc_descriptor(as)',setup)
        self.assertNotIn('[group]',setup)
        touched=[i for tid in range(256) for i in range(tid,1024,256)]
        self.assertEqual(sorted(touched),list(range(1024)))

if __name__=='__main__':unittest.main()
