// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"

#include <iostream>

int main() {
  guide::write_codegen_type_report<guide::codegen::dense_f16_2sm::Config>(
      std::cout, "dense_f16_2sm");
  return 0;
}
