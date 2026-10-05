#pragma once
// Logical host oracle, independent of shared packing; original short math stays frozen.
#include "low_precision_reference.hpp"
namespace low_precision_bounded_fp8_reference_v1 {
inline double element(int kind,int row,int col,int group,int chain,int iterations,
                      unsigned seed,bool nonuniform) {
  if(kind<0||kind>1||group<0||group>1||chain<0||chain>1||iterations<1||iterations>8192)
    throw std::runtime_error("bounded FP8 reference domain");
  if(nonuniform)return low_precision_reference::element(kind,row,col,group,chain,iterations,seed,true);
  // Exact signed integer products in units of 1/256, not a device-derived checksum.
  std::int64_t dot=0;for(int k=0;k<32;++k)dot+=(k%2==0?1:-1);
  return double(32*(1+2*group+chain)+std::int64_t(iterations)*16*dot)/256;
}
inline std::vector<double> outputs(int kind,bool warpgroup,int groups,int iterations,unsigned seed,bool nonuniform) {
  if(!warpgroup||(groups!=1&&groups!=2))throw std::runtime_error("bounded FP8 participants");
  std::vector<double> logical(groups*2*4096),packed(groups*128*2*32);
  for(int g=0;g<groups;++g)for(int c=0;c<2;++c)for(int m=0;m<64;++m)for(int n=0;n<64;++n)
    logical[((g*2+c)*64+m)*64+n]=element(kind,m,n,g,c,iterations,seed,nonuniform);
  for(int t=0;t<groups*128;++t)for(int c=0;c<2;++c)for(int f=0;f<32;++f) {
    int local=t%128,lane=local%32,m=16*(local/32)+lane/4+8*((f/2)%2),n=2*(lane%4)+f%2+8*(f/4);
    packed[(t*2+c)*32+f]=logical[(((t/128)*2+c)*64+m)*64+n];
  }
  return packed;
}
}
