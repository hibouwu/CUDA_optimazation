// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::blockwise_fp8_2sm_g64x4x32::Config>(
      std::cout, "blockwise_fp8_2sm_g64x4x32");
  return 0;
}
