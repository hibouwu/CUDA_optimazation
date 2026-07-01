#pragma once

#include <string>

struct BenchOptions {
  std::string case_name = "all";
  int stride = 1;
  int offset = 0;
  int num_iters = 100000;
  int num_warmups = 5;
  int num_repeats = 20;
};

constexpr int kLanes = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kLanes * kWarps;
