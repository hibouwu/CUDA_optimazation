// CPU-only budget precheck. This is not a CUDA validation probe.
#include "../../common/word_artifacts.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using u64=std::uint64_t;
using Clock=std::chrono::steady_clock;
constexpr u64 Q=16384,W=4096,R=4,S=1;
struct Part {const char* leaf;u64 words;};
static u64 integer(const char* s) {
  char* end=nullptr;const auto x=std::strtoull(s,&end,10);
  if(!*s||*end||x==0||x>1024)throw std::runtime_error("CPU writer blocks1..1024");return x;
}
int main(int argc,char** argv) try {
  if(argc!=4)throw std::runtime_error("CPU writer OUT NEW_BLOCKS BUDGET_SECONDS");
  const std::filesystem::path output=argv[1];const u64 blocks=integer(argv[2]);
  char* end=nullptr;const double limit=std::strtod(argv[3],&end);
  if(!*argv[3]||*end||!std::isfinite(limit)||limit<=0||limit>30)throw std::runtime_error("CPU writer budget0<seconds<=30");
  if(std::filesystem::exists(output)||!std::filesystem::create_directory(output))
    throw std::runtime_error("CPU writer output must be a new directory");
  std::cout<<std::setprecision(17);
  const unsigned lengths[]={1,2,5,33};
  const u64 expected=blocks*(41*R*Q+4*S*R*Q+4*32*R*Q+41*64+4*96+4*40)+128;
  double write_total=0,prepare_total=0;u64 bytes=0,files=0;
  for(unsigned launch=0;launch<4;++launch) {
    const u64 i=lengths[launch];
    const Part parts[]={{"trace",blocks*i*R*W},{"final_slots",blocks*S*R*W},
      {"ring_guards",blocks*32*R*W+8},{"lifecycle",blocks*i*16},
      {"counts",blocks*24},{"stamps",blocks*10}};
    for(const auto& part:parts) {
      const auto prepare_start=Clock::now();std::vector<std::uint32_t> data(part.words);
      // Runtime data prevents constant-message hash specialization. Only shape
      // and actual write_words cost matter; these are not GPU numeric outputs.
      for(u64 word=0;word<part.words;++word)data[word]=17u*std::uint32_t(word)+3u+launch;
      if(std::string(part.leaf)=="ring_guards")for(unsigned guard=0;guard<4;++guard) {
        data[guard]=0xd15ea5e0u+guard;data[data.size()-4+guard]=0xd15ea5e4u+guard;
      }
      const double prepare=std::chrono::duration<double>(Clock::now()-prepare_start).count();
      const std::string name="budget_"+std::to_string(launch)+"_"+part.leaf+".u32le";
      const auto write_start=Clock::now();
      const std::string hash=word_artifacts::write_words((output/name).string(),data);
      const double elapsed=std::chrono::duration<double>(Clock::now()-write_start).count();
      write_total+=elapsed;prepare_total+=prepare;bytes+=part.words*4;++files;
      std::cout<<"{\"type\":\"CPU_writer_file\",\"launch_index\":"<<launch
        <<",\"iterations\":"<<i<<",\"leaf\":\""<<part.leaf<<"\",\"path\":\""<<name
        <<"\",\"bytes\":"<<part.words*4<<",\"sha256\":\""<<hash<<"\",\"writer_seconds\":"<<elapsed
        <<",\"prepare_seconds\":"<<prepare<<",\"cumulative_writer_seconds\":"<<write_total<<"}\n"<<std::flush;
      if(write_total>limit) {
        std::cout<<"{\"type\":\"CPU_writer_budget\",\"status\":\"writer_budget_exceeded\",\"full_shape_completed\":false,\"files\":"<<files
          <<",\"completed_bytes\":"<<bytes<<",\"expected_full_bytes\":"<<expected
          <<",\"writer_seconds\":"<<write_total<<",\"prepare_seconds\":"<<prepare_total
          <<",\"budget_seconds\":"<<limit<<",\"GPU_execution\":false}\n";
        return 3;
      }
    }
  }
  if(files!=24||bytes!=expected)throw std::runtime_error("CPU writer fixed24 shape byte count");
  std::cout<<"{\"type\":\"CPU_writer_budget\",\"status\":\"writer_budget_necessary_check_pass\",\"full_shape_completed\":true,\"files\":24,\"completed_bytes\":"<<bytes
    <<",\"expected_full_bytes\":"<<expected<<",\"writer_seconds\":"<<write_total
    <<",\"prepare_seconds\":"<<prepare_total<<",\"budget_seconds\":"<<limit<<",\"GPU_execution\":false}\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
