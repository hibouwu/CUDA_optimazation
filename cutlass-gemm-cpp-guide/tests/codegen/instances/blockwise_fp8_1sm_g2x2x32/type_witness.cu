// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::blockwise_fp8_1sm_g2x2x32::Config>(
      std::cout, "blockwise_fp8_1sm_g2x2x32");
  return 0;
}
