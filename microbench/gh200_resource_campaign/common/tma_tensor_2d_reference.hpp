#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace tma_tensor_2d_reference {
using u64=std::uint64_t;using u16=std::uint16_t;
inline u64 multiply(u64 a,u64 b){if(b&&a>std::numeric_limits<u64>::max()/b)throw std::overflow_error("S15 uint64 product");return a*b;}
struct Layout {unsigned q,w,h,p;bool sw128;};
inline Layout layout(unsigned q,unsigned p,bool sw128){
 unsigned w=q==65536?128:64;unsigned h=q/(2*w);
 if(q!=1024&&q!=4096&&q!=8192&&q!=16384&&q!=32768&&q!=65536)throw std::invalid_argument("S15 finite payload");
 if(p!=2*w&&p!=2*w+16)throw std::invalid_argument("S15 finite stride");
 if(sw128&&(w!=64||p!=128||q>32768))throw std::invalid_argument("S15 SW128 shape/pitch");
 if(w>256||h>256||p%16)throw std::invalid_argument("S15 descriptor dimensions");return {q,w,h,p,sw128};
}
inline u64 global_rows(u64 blocks,Layout l){if(!blocks)throw std::invalid_argument("S15 zero blocks");auto rows=multiply(multiply(blocks,32),l.h);if(rows>std::numeric_limits<std::int32_t>::max())throw std::overflow_error("S15 signed32 tensor coordinates");return rows;}
inline u64 allocation_bytes(u64 blocks,Layout l){auto bytes=multiply(global_rows(blocks,l),l.p);if(bytes>std::numeric_limits<u64>::max()-256)throw std::overflow_error("S15 guard allocation");return bytes+256;}
inline u64 global_element(u64 b,u64 slot,u64 x,u64 y,u64 blocks,Layout l){global_rows(blocks,l);if(b>=blocks||slot>=32||x>=l.w||y>=l.h)throw std::invalid_argument("S15 element coordinate");return ((b*32+slot)*l.h+y)*(l.p/2)+x;}
inline unsigned physical_index(unsigned x,unsigned y,Layout l){if(x>=l.w||y>=l.h)throw std::invalid_argument("S15 shared coordinate");return l.sw128?y*64+((x/8)^(y%8))*8+x%8:y*l.w+x;}
inline u16 input(unsigned b,unsigned slot,unsigned x,unsigned y,unsigned seed,bool g2s){return u16(17ull*x+31ull*y+(g2s?73ull*slot:0ull)+151ull*b+seed);}
inline u16 poison(u16 value){return u16(value^0xffffu);}
inline u16 padding(u64 global_row,unsigned column){return u16(0x7d00u+global_row*11+column);}
inline u16 guard(unsigned index){if(index>=128)throw std::invalid_argument("S15 guard coordinate");return u16(0xa900u+index);}
inline u64 work(u64 blocks,u64 length,Layout l){allocation_bytes(blocks,l);if(!length||length>65536)throw std::invalid_argument("S15 length");return multiply(multiply(blocks,length),l.q);}
}
