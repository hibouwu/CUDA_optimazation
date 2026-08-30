// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::dense_mixed_tma_cpasync_fp8_2sm_moe::Config>(
      std::cout, "dense_mixed_tma_cpasync_fp8_2sm_moe");
  return 0;
}
