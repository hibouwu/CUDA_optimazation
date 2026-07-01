#pragma once

#include <string>

struct BenchOptions {
  std::string case_name = "all";
  int stride = 1;
  int offset = 0;
  int num_iters = 100000;
};

constexpr int kLanes = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kLanes * kWarps;
constexpr int kWarmups = 5;
constexpr int kRepeats = 20;

