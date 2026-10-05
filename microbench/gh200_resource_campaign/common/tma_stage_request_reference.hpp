#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace tma_stage_reference {
using u64=std::uint64_t;using u32=std::uint32_t;
constexpr u64 Q=16384,W=4096,G=32;
inline void domain(unsigned s,unsigned r){if((s!=1&&s!=2&&s!=4)||(r!=1&&r!=2&&r!=4))throw std::invalid_argument("S16 finite S/R");}
inline u64 mul(u64 a,u64 b){if(b&&a>std::numeric_limits<u64>::max()/b)throw std::overflow_error("S16 product");return a*b;}
inline u64 shared(unsigned s,unsigned r){domain(s,r);return s*r*Q+8*s+32;}
inline u64 allocation(u64 b,unsigned r){domain(1,r);if(!b)throw std::invalid_argument("S16 blocks");u64 n=mul(mul(b,G*r),Q);if(n>~u64(0)-32)throw std::overflow_error("S16 guards");return n+32;}
inline u64 payload(u64 b,u64 i,unsigned r){allocation(b,r);if(!i||i>65536)throw std::invalid_argument("S16 item count");return mul(mul(b,i*r),Q);}
inline u64 global_offset(u64 b,u64 i,unsigned request,unsigned r){domain(1,r);if(request>=r)throw std::invalid_argument("S16 request");return ((b*G+i%G)*r+request)*W;}
inline u32 g2s(u64 b,u64 i,unsigned request,u64 word,unsigned r,u32 seed){if(word>=W)throw std::invalid_argument("S16 word");return 17u*u32(global_offset(b,i,request,r)+word)+seed;}
inline u32 s2g(u64 b,unsigned slot,unsigned request,u64 word,unsigned s,unsigned r,u32 seed){domain(s,r);if(slot>=s||request>=r||word>=W)throw std::invalid_argument("S16 source coordinate");return 29u*u32(((b*s+slot)*r+request)*W+word)+seed;}
inline u64 last_item(u64 slot,u64 i,u64 ring){if(!i||slot>=ring)throw std::invalid_argument("S16 last item");return slot<i?slot+((i-1-slot)/ring)*ring:~u64(0);}
inline u32 ring(bool g2s,u64 b,unsigned slot,unsigned request,u64 word,unsigned s,unsigned r,unsigned i,u32 seed){if(g2s)return tma_stage_reference::g2s(b,slot,request,word,r,seed);const auto last=last_item(slot,i,G);return last==~u64(0)?~s2g(b,slot%s,request,word,s,r,seed):s2g(b,last%s,request,word,s,r,seed);}
inline u32 final(bool g2s,u64 b,unsigned slot,unsigned request,u64 word,unsigned s,unsigned r,unsigned i,u32 seed){if(!g2s)return s2g(b,slot,request,word,s,r,seed);const auto last=last_item(slot,i,s);return last==~u64(0)?~tma_stage_reference::g2s(b,slot,request,word,r,seed):tma_stage_reference::g2s(b,last,request,word,r,seed);}
}
