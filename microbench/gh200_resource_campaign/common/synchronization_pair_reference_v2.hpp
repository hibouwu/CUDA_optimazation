#pragma once
#include <cstdint>
#include <stdexcept>
namespace synchronization_pair_reference_v2 {
inline std::uint32_t initial(unsigned tid,unsigned seed){if(tid>=32)throw std::invalid_argument("S11 pair lane");return std::uint32_t(seed)+17u*tid;}
inline std::uint32_t peer_value(unsigned tid,unsigned phase,unsigned seed){const unsigned peer=tid^16u;return (initial(peer,seed)+(peer>=16?324508639u:0u))^((phase+1u)*2246822519u);}
inline std::uint32_t checksum(unsigned tid,unsigned iterations,unsigned seed){if(iterations!=1&&iterations!=2&&iterations!=33)throw std::invalid_argument("S11 pair short length");std::uint32_t value=0;for(unsigned phase=0;phase<iterations*8;++phase)value+=peer_value(tid,phase,seed);return value;}
}
