#pragma once
// Host-only numerical reference. No CUDA fragment loading or device result is used.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace legacy_reference {
using u128 = unsigned __int128;
inline u128 round_significand(u128 value,int precision) {
  if(!value)return 0;
  int bits=0;for(auto v=value;v;v>>=1)++bits;
  int drop=bits-precision;if(drop<=0)return value;
  auto q=value>>drop,rem=value-(q<<drop),half=u128(1)<<(drop-1);
  if(rem>half || (rem==half && (q&1)))++q;
  return q<<drop;
}
inline int precision(const std::string& kind) {
  if(kind=="f64")return 53;if(kind=="f32")return 24;
  if(kind=="f16"||kind=="f16x2")return 11;
  if(kind=="bf16"||kind=="bf16x2")return 8;
  throw std::runtime_error("unknown FMA format");
}
// Values are exact nonnegative numerators over 2^64. The bounded inputs never
// become subnormal; retained precision is <=53, so the division by two is exact.
inline double fma(const std::string& kind,int thread,int chain,int lane,
                  unsigned seed,int iterations,int batch,bool nonuniform) {
  int p=precision(kind);
  u128 x=nonuniform?u128((std::uint64_t(thread)+chain+lane+seed)%7)<<60:0;
  u128 add=nonuniform?u128(1+(std::uint64_t(thread)+3*chain+lane+seed)%7)<<59:u128(1)<<62;
  for(std::uint64_t i=0,n=std::uint64_t(iterations)*batch;i<n;++i) {
    auto next=round_significand((x>>1)+add,p);
    if(next==x)break;x=next;
  }
  return std::ldexp(static_cast<double>(x),-64);
}
inline std::vector<double> matrix(int m,int n,int k,int groups,int chains,
                                  int iterations,int batch,unsigned seed,bool nonuniform) {
  std::vector<double> out(std::size_t(groups)*chains*m*n);
  for(int g=0;g<groups;++g)for(int c=0;c<chains;++c)
    for(int row=0;row<m;++row)for(int col=0;col<n;++col) {
      std::uint64_t numerator=0;
      if(nonuniform)for(int inner=0;inner<k;++inner)
        numerator+=std::uint64_t(1+row+2*inner+seed%3)*(1+col+3*inner+seed%5);
      double value=nonuniform?std::ldexp(double(numerator)*iterations*batch,-16)
                                  +double(1+g+c)/64.0
                             :double(iterations)*batch*k/256.0;
      out[((g*chains+c)*m+row)*n+col]=value;
    }
  return out;
}
// Convert logical row-major CPU reference to the documented output fragment.
// This is independent from the generated operand packing code.
inline std::pair<int,int> output_coordinate(int width,int m,int thread,int element) {
  int lane=thread%32;
  if(width==128)return {16*(thread/32)+lane/4+8*((element/2)%2),
                       8*(element/4)+2*(lane%4)+element%2};
  if(m==8)return {lane/4,2*(lane%4)+element};
  return {lane/4+8*(element/2),2*(lane%4)+element%2};
}
}
