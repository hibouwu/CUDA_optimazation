#pragma once
#include "r09_plain_prefix.h"
#include <dlfcn.h>
#include <stdexcept>
#include <string>

// Host-only owner; load/create before input initialization. No instrumentation
// or CUTLASS types are linked from the observing executable into this module.
class R09PlainPrefixHost {
  void* library_ = nullptr;
  void* handle_ = nullptr;
  decltype(&r09_plain_prefix_run) run_ = nullptr;
  decltype(&r09_plain_prefix_destroy) destroy_ = nullptr;
public:
  R09PlainPrefixInfo info{};
  R09PlainPrefixHost(const std::string& path, const R09PlainPrefixSpec& spec) {
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error(std::string("plain prefix dlopen: ")+dlerror());
    try {
      auto create = reinterpret_cast<decltype(&r09_plain_prefix_create)>(dlsym(library_,"r09_plain_prefix_create"));
      run_ = reinterpret_cast<decltype(run_)>(dlsym(library_,"r09_plain_prefix_run"));
      destroy_ = reinterpret_cast<decltype(destroy_)>(dlsym(library_,"r09_plain_prefix_destroy"));
      if (!create || !run_ || !destroy_) throw std::runtime_error("plain prefix module is missing ABI symbols");
      char error[512]{};
      if (create(&spec,&handle_,&info,error,sizeof(error))) throw std::runtime_error(error);
      if (!handle_ || info.abi_version!=1 || info.stages!=6 || info.grid_x!=1 || info.grid_y!=132 || info.grid_z!=1)
        throw std::runtime_error("plain prefix module metadata mismatch");
    } catch (...) {
      if (handle_ && destroy_) destroy_(handle_);
      dlclose(library_);
      throw;
    }
  }
  R09PlainPrefixHost(const R09PlainPrefixHost&) = delete;
  R09PlainPrefixHost& operator=(const R09PlainPrefixHost&) = delete;
  ~R09PlainPrefixHost() { if (handle_) destroy_(handle_); if (library_) dlclose(library_); }
  void run() {
    char error[512]{};
    if (run_(handle_,error,sizeof(error))) throw std::runtime_error(error);
  }
};
