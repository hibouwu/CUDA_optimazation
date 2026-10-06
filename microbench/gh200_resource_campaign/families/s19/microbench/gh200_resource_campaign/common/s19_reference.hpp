#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace s19 {
constexpr unsigned poison = 0x7fc0abcd;
struct Case { int mode, stages, tiles; std::string id; };
inline Case parse(const std::string& id) {
  const char* modes[] = {"compute", "transport", "serial", "overlap", "output"};
  for(int m=0;m<5;++m) for(int s:{1,2,4}) for(int k:{1,2,4,8,16,32,64}) {
    std::string expected=std::string(modes[m])+"_s"+std::to_string(s)+"_k"+std::to_string(k);
    if(id==expected) return {m,s,k,id};
  }
  throw std::runtime_error("unknown S19 finite case");
}
inline unsigned bits(float x) { unsigned result; std::memcpy(&result,&x,4); return result; }
inline int numerator(int operand,int tile,int row,int col,unsigned seed,bool tagged) {
  int tag=tagged?tile+1:0;
  return operand==0?(3*row+5*col+int(seed%17)+7*tag)%15-7:
                    (7*row+3*col+int(seed%19)+11*tag)%15-7;
}
inline std::vector<unsigned> input(const Case& c,unsigned seed,bool tagged) {
  if(c.mode==0 && tagged) throw std::runtime_error("compute supports periodic input only");
  std::vector<unsigned> out(c.tiles*2048);
  for(int t=0;t<c.tiles;++t) for(int operand=0;operand<2;++operand)
    for(int r=0;r<32;++r) for(int n=0;n<32;++n)
      out[t*2048+operand*1024+r*32+n]=bits(float(numerator(operand,t,r,n,seed,tagged))/16);
  return out;
}
struct Expected { std::vector<unsigned> output,digest,slots,trace_input,trace_c; };
inline Expected expected(const Case& c,int iterations,unsigned seed,bool tagged,bool trace) {
  auto values=input(c,seed,tagged);
  Expected e;
  e.output.resize(1024); e.digest.resize(128); e.slots.assign(8192,poison);
  if(trace) {e.trace_input.resize(iterations*c.tiles*2048);e.trace_c.resize(iterations*c.tiles*1024);}
  std::vector<int> sums(1024,0);
  for(int t=0;t<c.tiles;++t) {
    int slot=t%c.stages;
    if(c.mode!=0 || t<c.stages)
      for(int i=0;i<2048;++i)e.slots[slot*2048+i]=values[t*2048+i];
    if(c.mode!=1) for(int m=0;m<32;++m) for(int n=0;n<32;++n)
      for(int k=0;k<32;++k)
        sums[m*32+n]+=numerator(0,t,m,k,seed,tagged)*numerator(1,t,k,n,seed,tagged);
    if(c.mode==1) for(int tid=0;tid<128;++tid)
      for(int q=0;q<4;++q) for(int j=0;j<4;++j)
        e.digest[tid]+=unsigned(iterations)*values[t*2048+((tid+1)%128)*4+q*512+j];
    if(trace) for(int rep=0;rep<iterations;++rep) {
      for(int i=0;i<2048;++i)e.trace_input[(rep*c.tiles+t)*2048+i]=values[t*2048+i];
      for(int i=0;i<1024;++i)e.trace_c[(rep*c.tiles+t)*1024+i]=bits(float(sums[i])/256);
    }
  }
  for(int i=0;i<1024;++i) {
    float value=float(sums[i])/256;
    if(c.mode==4)value=value*0.5f+float(i%32-16)/16;
    e.output[i]=bits(value);
  }
  return e;
}
}
