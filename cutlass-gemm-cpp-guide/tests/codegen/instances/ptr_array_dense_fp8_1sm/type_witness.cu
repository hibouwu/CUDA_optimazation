// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::ptr_array_dense_fp8_1sm::Config>(
      std::cout, "ptr_array_dense_fp8_1sm");
  return 0;
}
