"""Independent hashlib/struct checks for host-only word artifacts; no GPU."""
from pathlib import Path
import hashlib,os,struct,subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]


class WordArtifactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();src=Path(cls.temp.name)/'artifacts.cpp';cls.binary=Path(cls.temp.name)/'artifacts'
        src.write_text('#include "'+str(ROOT/'common/word_artifacts.hpp')+'"\n#include <iostream>\n#include <iterator>\n'
            'int main(int argc,char**argv)try{std::string mode=argv[1];if(mode=="hash"){word_artifacts::Sha256 h;'
            'std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(std::cin)),{});size_t stride=std::stoull(argv[2]);'
            'for(size_t i=0;i<bytes.size();i+=stride)h.update(bytes.data()+i,std::min(stride,bytes.size()-i));std::cout<<h.finish();}'
            'else if(mode=="write"){std::vector<std::uint32_t> words;unsigned long long x;while(std::cin>>x)words.push_back(x);std::cout<<word_artifacts::write_words(argv[2],words);}'
            'else{std::vector<std::uint64_t> x;unsigned long long v;while(std::cin>>v)x.push_back(v);auto words=word_artifacts::split_u64(x);std::cout<<word_artifacts::write_words(argv[2],words);}return 0;}'
            'catch(const std::exception& e){std::cerr<<e.what();return 2;}')
        subprocess.run(['g++','-std=c++17','-O3','-Wall','-Wextra',str(src),'-o',str(cls.binary)],capture_output=True,check=True,timeout=30)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_hashlib_standard_vectors_and_padding_chunk_boundaries(self):
        values=[b'',b'abc',b'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq',b'a'*1000000]
        values += [bytes((i*17+3)%256 for i in range(n)) for n in (1,55,56,57,63,64,65,127,128,129,65535,65536,65537)]
        for data in values:
            for chunk in (1,7,64,65536):
                run=subprocess.run([str(self.binary),'hash',str(chunk)],input=data,capture_output=True,check=True,timeout=15)
                self.assertEqual(run.stdout.decode(),hashlib.sha256(data).hexdigest(),(len(data),chunk))

    def test_uint32_le_real_files_and_no_overwrite(self):
        for count in (0,1,16,16383,16384,16385):
            words=[(0xf0000001+29*i)&0xffffffff for i in range(count)]
            dest=Path(self.temp.name)/f'words-{count}'
            run=subprocess.run([str(self.binary),'write',str(dest)],input=' '.join(map(str,words)).encode(),capture_output=True,check=True,timeout=15)
            expected=b''.join(struct.pack('<I',x) for x in words)
            self.assertEqual(dest.read_bytes(),expected)
            self.assertEqual(run.stdout.decode(),hashlib.sha256(expected).hexdigest())
            again=subprocess.run([str(self.binary),'write',str(dest)],input=b'0',capture_output=True)
            self.assertEqual(again.returncode,2);self.assertEqual(dest.read_bytes(),expected)
        target=Path(self.temp.name)/'must-not-create';link=Path(self.temp.name)/'dangling';link.symlink_to(target)
        result=subprocess.run([str(self.binary),'write',str(link)],input=b'1',capture_output=True)
        self.assertEqual(result.returncode,2);self.assertFalse(target.exists())

    def test_uint64_pairs_preserve_high_bits_and_uintmax(self):
        values=[0,1,2**32-1,2**32,2**53+1,2**63,2**64-1];dest=Path(self.temp.name)/'pairs'
        run=subprocess.run([str(self.binary),'pairs',str(dest)],input=' '.join(map(str,values)).encode(),capture_output=True,check=True)
        expected=b''.join(struct.pack('<Q',x) for x in values)
        self.assertEqual(dest.read_bytes(),expected);self.assertEqual(run.stdout.decode(),hashlib.sha256(expected).hexdigest())


if __name__=='__main__':unittest.main()
