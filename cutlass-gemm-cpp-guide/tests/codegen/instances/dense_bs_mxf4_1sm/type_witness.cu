// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"

#include <iostream>

int main() {
  guide::write_codegen_type_report<guide::codegen::dense_bs_mxf4_1sm::Config>(
      std::cout, "dense_bs_mxf4_1sm");
  return 0;
}
