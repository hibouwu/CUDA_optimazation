// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::mixed_input_int4_bf16_2sm_convert::Config>(
      std::cout, "mixed_input_int4_bf16_2sm_convert");
  return 0;
}
