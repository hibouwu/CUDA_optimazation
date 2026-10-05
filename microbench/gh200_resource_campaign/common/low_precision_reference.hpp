#pragma once
// Host-only logical matrix oracle. No device packing or GPU result is reused.
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace low_precision_reference {
// kind: 0 E4M3, 1 E5M2, 2 S8, 3 U8. FP8 values use denominator 8.
inline int input_numerator(int kind, int outer, int inner, unsigned seed, bool a) {
  if(kind<0 || kind>3)throw std::runtime_error("unknown low precision type");
  std::uint64_t index=std::uint64_t(outer)+(a?2:3)*std::uint64_t(inner)+seed;
  return kind<2?int(index%7)-3:kind==2?int(index%5)-2:int(index%3);
}
inline double element(int kind,int row,int col,int group,int chain,int iterations,
                      unsigned seed,bool nonuniform) {
  if(iterations<1 || iterations>8192)throw std::runtime_error("reference iteration range");
  if(!nonuniform)return double(iterations)*16*32/(kind<2?256:1);
  if(iterations>2)throw std::runtime_error("nonuniform reference is bounded to 1/2 iterations");
  std::int64_t dot=0;
  for(int k=0;k<32;++k)
    dot+=std::int64_t(input_numerator(kind,row,k,seed,true))*input_numerator(kind,col,k,seed,false);
  const auto repeated=dot*iterations*16;
  return kind<2?double(repeated)/64.0+double(1+group+chain)/8.0
               :double(repeated+1+group+chain);
}
// Full logical matrices are packed for comparison only after independent dot products.
inline std::vector<double> outputs(int kind,bool warpgroup,int groups,int iterations,
                                   unsigned seed,bool nonuniform) {
  const int width=warpgroup?128:32,m=warpgroup?64:16,n=warpgroup?64:8;
  const int per_thread=m*n/width;
  std::vector<double> logical(std::size_t(groups)*2*m*n),packed(std::size_t(groups)*width*2*per_thread);
  for(int g=0;g<groups;++g)for(int c=0;c<2;++c)
    for(int row=0;row<m;++row)for(int col=0;col<n;++col)
      logical[((g*2+c)*m+row)*n+col]=element(kind,row,col,g,c,iterations,seed,nonuniform);
  for(int t=0;t<groups*width;++t)for(int c=0;c<2;++c)for(int e=0;e<per_thread;++e) {
    const int local=t%width,lane=local%32;
    const int row=warpgroup?16*(local/32)+lane/4+8*((e/2)%2):lane/4+8*(e/2);
    const int col=2*(lane%4)+e%2+(warpgroup?8*(e/4):0);
    packed[(t*2+c)*per_thread+e]=logical[(((t/width)*2+c)*m+row)*n+col];
  }
  return packed;
}
}
