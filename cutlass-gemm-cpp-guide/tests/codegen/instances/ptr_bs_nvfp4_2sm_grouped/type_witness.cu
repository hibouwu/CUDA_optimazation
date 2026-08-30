// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {
  guide::write_codegen_type_report<guide::codegen::ptr_bs_nvfp4_2sm_grouped::Config>(
      std::cout, "ptr_bs_nvfp4_2sm_grouped");
  return 0;
}
