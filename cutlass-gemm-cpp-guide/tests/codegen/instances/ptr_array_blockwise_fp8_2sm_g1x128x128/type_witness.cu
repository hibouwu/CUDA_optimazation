// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::ptr_array_blockwise_fp8_2sm_g1x128x128::Config>(
      std::cout, "ptr_array_blockwise_fp8_2sm_g1x128x128");
  return 0;
}
