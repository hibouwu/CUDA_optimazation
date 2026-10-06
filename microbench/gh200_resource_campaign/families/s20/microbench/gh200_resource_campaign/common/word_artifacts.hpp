#pragma once
// Host-only full-value artifacts: portable uint32 LE and streaming SHA256.
// This helper performs no GPU calls and introduces no runner/protocol behavior.
#include <array>
#include <cstdint>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace word_artifacts {
class Sha256 {
  std::array<std::uint32_t,8> state{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
  std::array<unsigned char,64> tail{};
  std::uint64_t bytes=0;unsigned used=0;
  static std::uint32_t rotate(std::uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));}
  void block(const unsigned char* input) {
    static constexpr std::uint32_t k[64]={
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::uint32_t w[64];
    for(unsigned i=0;i<16;++i)w[i]=(std::uint32_t(input[4*i])<<24)|(std::uint32_t(input[4*i+1])<<16)|(std::uint32_t(input[4*i+2])<<8)|input[4*i+3];
    for(unsigned i=16;i<64;++i){auto a=w[i-15],b=w[i-2];w[i]=w[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10));}
    auto a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for(unsigned i=0;i<64;++i){const auto first=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+k[i]+w[i];const auto second=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+first;d=c;c=b;b=a;a=first+second;}
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
  }
public:
  void update(const unsigned char* input,std::size_t size) {
    bytes+=size;
    if(used){while(size&&used<64){tail[used++]=*input++;--size;}if(used==64){block(tail.data());used=0;}}
    while(size>=64){block(input);input+=64;size-=64;}
    while(size){tail[used++]=*input++;--size;}
  }
  std::string finish() {
    const std::uint64_t bits=bytes*8;tail[used++]=0x80;
    if(used>56){while(used<64)tail[used++]=0;block(tail.data());used=0;}
    while(used<56)tail[used++]=0;
    for(unsigned i=0;i<8;++i)tail[56+i]=static_cast<unsigned char>(bits>>(56-8*i));
    block(tail.data());std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto word:state)out<<std::setw(8)<<word;return out.str();
  }
};
inline std::string write_words(const std::string& path,const std::vector<std::uint32_t>& words) {
  // Exclusive creation preserves immutable prior evidence and refuses symlinks.
  const int descriptor=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
  if(descriptor<0)throw std::runtime_error("cannot exclusively create word artifact: "+path);
  struct Close {int fd;~Close(){if(fd>=0)::close(fd);}} output{descriptor};
  Sha256 hash;std::array<unsigned char,65536> buffer{};std::size_t offset=0;
  while(offset<words.size()) {
    const std::size_t count=words.size()-offset<buffer.size()/4?words.size()-offset:buffer.size()/4;
    for(std::size_t i=0;i<count;++i)for(unsigned byte=0;byte<4;++byte)buffer[4*i+byte]=static_cast<unsigned char>(words[offset+i]>>(8*byte));
    std::size_t written=0;
    while(written<count*4) {
      const auto amount=::write(descriptor,buffer.data()+written,count*4-written);
      if(amount<0&&errno==EINTR)continue;
      if(amount<=0)throw std::runtime_error("word artifact write failed: "+path);
      written+=std::size_t(amount);
    }
    hash.update(buffer.data(),count*4);offset+=count;
  }
  const int close_status=::close(descriptor);output.fd=-1;
  if(close_status!=0)throw std::runtime_error("word artifact close failed: "+path);return hash.finish();
}
inline std::vector<std::uint32_t> split_u64(const std::vector<std::uint64_t>& values) {
  std::vector<std::uint32_t> words;words.reserve(values.size()*2);
  for(auto value:values){words.push_back(std::uint32_t(value));words.push_back(std::uint32_t(value>>32));}return words;
}
} // namespace word_artifacts
