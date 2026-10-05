#pragma once
// New S15 host-only lossless uint16/uint8 writer; shared uint32 helper unchanged.
#include "word_artifacts.hpp"
#include <algorithm>
namespace tensor_artifacts {
template<class T> inline std::string write(const std::string& path,const std::vector<T>& values){
 static_assert(sizeof(T)==1||sizeof(T)==2,"S15 byte/uint16 artifacts only");
 int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("S15 artifact open failed: "+path);
 struct Close{int fd;~Close(){if(fd>=0)::close(fd);}} file{fd};word_artifacts::Sha256 hash;std::array<unsigned char,65536> buffer{};
 std::size_t offset=0;while(offset<values.size()){
  const std::size_t count=std::min(values.size()-offset,buffer.size()/sizeof(T));
  for(std::size_t i=0;i<count;++i){buffer[i*sizeof(T)]=static_cast<unsigned char>(values[offset+i]);if constexpr(sizeof(T)==2)buffer[i*2+1]=static_cast<unsigned char>(values[offset+i]>>8);}
  const std::size_t bytes=count*sizeof(T);std::size_t sent=0;while(sent<bytes){auto n=::write(fd,buffer.data()+sent,bytes-sent);if(n<0&&errno==EINTR)continue;if(n<=0)throw std::runtime_error("S15 artifact write failed");sent+=std::size_t(n);}hash.update(buffer.data(),bytes);offset+=count;
 }
 const int status=::close(fd);file.fd=-1;if(status)throw std::runtime_error("S15 artifact close failed");return hash.finish();
}
}
