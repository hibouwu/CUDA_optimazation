// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::fast_fp32_bf16x9_2sm_smem::Config>(
      std::cout, "fast_fp32_bf16x9_2sm_smem");
  return 0;
}
