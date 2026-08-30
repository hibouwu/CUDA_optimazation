// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::sparse_i8_2sm_official::Config>(
      std::cout, "sparse_i8_2sm_official");
  return 0;
}
