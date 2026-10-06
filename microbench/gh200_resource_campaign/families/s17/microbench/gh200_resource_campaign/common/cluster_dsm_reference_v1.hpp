#pragma once
#include <cstdint>
#include <stdexcept>
namespace cluster_reference {
using U=std::uint32_t;using V=std::uint64_t;
inline U dsm(bool write,bool remote,unsigned C,unsigned block,unsigned item,unsigned word,U seed){
 unsigned rank=block%C,base=block/C*C;
 unsigned source=remote?(write?(rank+C-1)%C:(rank+1)%C):rank;
 return (write?29u:17u)*((base+source)*1024u+word)+seed+(write?31u*item:0u);
}
inline U bulk(unsigned cluster,unsigned item,unsigned word,U seed){return 17u*((cluster*32u+item%32u)*4096u+word)+seed;}
inline U mask(bool all,unsigned C){return all?(1u<<C)-1:1u<<(C-1);}
inline U life(unsigned form,unsigned C,unsigned block,unsigned item,unsigned field){
 unsigned rank=block%C;bool multicast=form>=5;unsigned m=multicast?mask(form==6,C):0;
 bool receiver=multicast?bool(m&(1u<<rank)):form!=4;
 const U values[]={rank,C,item,m,receiver?1u:0u,multicast&&receiver?1u:0u,multicast&&receiver?16384u:0u,multicast&&rank==0?1u:0u,multicast&&receiver?1u:0u,form==4?0u:1u,1,1};
 if(field>=12)throw std::runtime_error("S17 lifecycle field");return values[field];
}
inline V requested(unsigned form,unsigned C,unsigned clusters,unsigned items){return form<4?V(clusters)*C*items*4096:form>=5?V(clusters)*items*16384:0;}
}
