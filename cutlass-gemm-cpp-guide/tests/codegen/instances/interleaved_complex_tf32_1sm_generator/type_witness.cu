// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::interleaved_complex_tf32_1sm_generator::Config>(
      std::cout, "interleaved_complex_tf32_1sm_generator");
  return 0;
}
