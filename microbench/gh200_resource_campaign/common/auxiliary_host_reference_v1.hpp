#pragma once
// Short host checks use direct recurrences. Python oracle uses closed forms.
#include <cstdint>
namespace auxiliary_host_reference {
inline std::uint32_t initial(std::uint32_t seed,unsigned thread,unsigned word) {
  return seed+65537u*(thread+1u)+257u*(word+1u);
}
inline std::uint32_t word(std::uint32_t seed,unsigned thread,unsigned index,unsigned iterations) {
  auto value=initial(seed,thread,index);
  for(unsigned i=0;i<iterations;++i)value=1664525u*value+1013904223u;
  return value;
}
inline std::uint64_t service(unsigned operation,std::uint32_t seed,unsigned thread,unsigned stream,unsigned iterations) {
  std::uint64_t state=initial(seed,thread,stream);
  const auto increment=((std::uint64_t(seed)<<32)+65537ull*(thread+1)+257ull*(stream+1))|1ull;
  const auto factor=initial(seed^0xa5a5a5a5u,thread,stream)|1u;
  const auto addend=(std::uint64_t(initial(seed^0x5a5a5a5au,thread,stream))<<32)|initial(seed^0xc3c3c3c3u,thread,stream);
  const auto salt=(initial(seed^0x9e3779b9u,thread,stream)&1023u)|1u;
  if(operation==0)state=(state<<32)|initial(seed,thread,stream+17);
  if(operation>=2)state&=1023u;
  for(unsigned i=0;i<iterations;++i) {
    if(operation==0)state+=increment;
    else if(operation==1)state=std::uint64_t(std::uint32_t(state))*factor+addend;
    else {
      unsigned q=unsigned(state);
      if(i)q=operation==2?q&1023u:(q>>13)&1023u;
      q=(q+salt+i)&1023u;
      state=operation==2?(0x3c00u|q):(0x3f800000u|(q<<13));
    }
  }
  return state;
}
inline std::uint32_t atomic(std::uint32_t seed,unsigned address,bool same,unsigned iterations) {
  unsigned increment=0;
  if(same)for(unsigned t=0;t<128;++t)increment+=1u+t%7u;
  else increment=1u+address%7u;
  return initial(seed,0,address)+iterations*increment;
}
}
