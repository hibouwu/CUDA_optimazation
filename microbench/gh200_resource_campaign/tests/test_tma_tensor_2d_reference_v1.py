"""S15 CPU coordinate/swizzle/overflow and uint16 archive semantics."""
import hashlib,struct,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CPP=r'''
#include "REFERENCE_PATH"
#include "WRITER_PATH"
#include <iostream>
#include <set>
int main(int argc,char** argv){namespace r=tma_tensor_2d_reference;
 if(argc==2){std::vector<std::uint16_t> words(65537);for(unsigned n=0;n<words.size();++n)words[n]=std::uint16_t(n*31+65535);std::cout<<tensor_artifacts::write(argv[1],words);return 0;}
 unsigned layouts=0;for(unsigned q:{1024u,4096u,8192u,16384u,32768u,65536u})for(unsigned mode=0;mode<3;++mode){if(q==65536&&mode==2)continue;const unsigned w=q==65536?128:64;auto l=r::layout(q,2*w+(mode==1?16:0),mode==2);++layouts;
  std::set<unsigned> addresses;for(unsigned y=0;y<l.h;++y)for(unsigned x=0;x<l.w;++x){unsigned at=r::physical_index(x,y,l);addresses.insert(at);if(r::physical_index(at%l.w,at/l.w,l)!=y*l.w+x)return 1;}
  if(addresses.size()!=q/2||*addresses.rbegin()!=q/2-1)return 2;
  if(r::work(2,33,l)!=2ull*33*q||r::allocation_bytes(2,l)!=2ull*32*l.h*l.p+256)return 3;
  auto a=r::global_element(0,31,l.w-1,l.h-1,2,l),b=r::global_element(1,0,0,0,2,l);if(a>=b)return 4;
  for(unsigned seed:{0u,3u,4294967295u})for(unsigned x=0;x<l.w;++x){auto v=r::input(1,31,x,l.h-1,seed,true);if(r::poison(v)==v)return 5;}
 }
 unsigned rejected=0;auto reject=[&](auto f){try{f();}catch(const std::exception&){++rejected;}};
 reject([]{r::layout(65536,256,true);});reject([]{r::layout(1024,140,false);});reject([]{r::layout(1024,144,true);});reject([]{r::layout(2048,128,false);});reject([]{r::allocation_bytes(~r::u64(0),r::layout(1024,128,false));});reject([]{r::global_rows(1ull<<32,r::layout(1024,128,false));});reject([]{r::global_element(2,0,0,0,2,r::layout(1024,128,false));});reject([]{r::work(1,65537,r::layout(1024,128,false));});
 if(layouts!=17||rejected!=8)return 6;std::cout<<layouts<<" "<<rejected;return 0;
}
'''
class TensorReference(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.binary=Path(cls.temp.name)/'test';source=Path(cls.temp.name)/'test.cpp';source.write_text(CPP.replace('REFERENCE_PATH',str(ROOT/'common/tma_tensor_2d_reference.hpp')).replace('WRITER_PATH',str(ROOT/'common/tensor_artifacts.hpp')))
  subprocess.run(['g++','-std=c++17','-O2',str(source),'-o',str(cls.binary)],check=True,capture_output=True,timeout=30)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_all17_layouts_bijection_address_ownership_and_eight_rejections(self):
  run=subprocess.run([str(self.binary)],capture_output=True,text=True,timeout=30);self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(run.stdout,'17 8')
 def test_uint16_little_endian_sha_buffer_boundary_and_exclusive_create(self):
  path=Path(self.temp.name)/'words';run=subprocess.run([str(self.binary),str(path)],capture_output=True,text=True,timeout=30);self.assertEqual(run.returncode,0,run.stderr)
  expected=b''.join(struct.pack('<H',(n*31+65535)&65535) for n in range(65537));self.assertEqual(path.read_bytes(),expected);self.assertEqual(run.stdout,hashlib.sha256(expected).hexdigest())
  repeat=subprocess.run([str(self.binary),str(path)],capture_output=True,text=True,timeout=30);self.assertNotEqual(repeat.returncode,0)

if __name__=='__main__':unittest.main()
