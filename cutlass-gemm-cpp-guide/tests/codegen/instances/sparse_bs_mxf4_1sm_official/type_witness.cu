// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::sparse_bs_mxf4_1sm_official::Config>(
      std::cout, "sparse_bs_mxf4_1sm_official");
  return 0;
}
