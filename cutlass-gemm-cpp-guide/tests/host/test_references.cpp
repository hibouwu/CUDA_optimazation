// SPDX-License-Identifier: BSD-3-Clause
#include "guide/block_scale_reference.hpp"
#include "guide/dense_reference.hpp"
#include "guide/scale_layout.hpp"
#include "guide/sparse_reference.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_dense_reference() {
  std::vector<float> a{1, 2, 3, 4, 5, 6};       // M=2, K=3
  std::vector<float> b{7, 8, 9, 1, 2, 3};       // N=2, K=3
  auto d = guide::dense_reference_mnk(a, b, 2, 2, 3);
  require(d.size() == 4, "dense reference output size");
  require(d[0] == 50.0f && d[1] == 14.0f, "dense reference row 0");
  require(d[2] == 122.0f && d[3] == 32.0f, "dense reference row 1");
}

void test_scale_layout() {
  require(guide::sm1xx_scale_storage_size(128, 128, 32) == 512,
          "one M128 x SF4 tile occupies 512 bytes");
  require(guide::sm1xx_scale_offset(0, 0, 128, 128, 32) == 0,
          "m0 sf0 offset");
  require(guide::sm1xx_scale_offset(0, 32, 128, 128, 32) == 1,
          "m0 sf1 offset");
  require(guide::sm1xx_scale_offset(32, 0, 128, 128, 32) == 4,
          "m32 sf0 offset");
  require(guide::sm1xx_scale_offset(1, 0, 128, 128, 32) == 16,
          "m1 sf0 offset");
  require(guide::sm1xx_scale_offset(127, 127, 128, 128, 32) == 511,
          "last scale in a basic chunk");
  require(guide::sm1xx_scale_storage_size(129, 129, 32) == 2048,
          "M and SF tails each add a padded block");
}

void test_block_scaled_reference() {
  std::vector<float> a{1, 2, 3, 4};
  std::vector<float> b{5, 6, 7, 8};
  std::vector<float> sfa{2, 3};
  std::vector<float> sfb{4, 5};
  auto d = guide::block_scaled_reference_mnk(a, b, sfa, sfb, 1, 1, 4, 2);
  // (1*2)*(5*4) + (2*2)*(6*4) + (3*3)*(7*5) + (4*3)*(8*5)
  require(d.size() == 1 && d[0] == 931.0f, "block-scaled logical oracle");
}

void test_sparse24() {
  auto compressed = guide::compress_sparse24({1.0f, 0.0f, -2.0f, 0.0f});
  auto dense = guide::decompress_sparse24(compressed);
  require(dense[0] == 1.0f && dense[1] == 0.0f && dense[2] == -2.0f && dense[3] == 0.0f,
          "sparse 2:4 round trip");
  bool rejected = false;
  try {
    (void)guide::compress_sparse24({1.0f, 2.0f, 3.0f, 0.0f});
  } catch (std::invalid_argument const&) {
    rejected = true;
  }
  require(rejected, "invalid 3:4 group must be rejected");
}

}  // namespace

int main() {
  try {
    test_dense_reference();
    test_scale_layout();
    test_block_scaled_reference();
    test_sparse24();
    std::cout << "HOST_REFERENCE_TESTS_PASS\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "HOST_REFERENCE_TESTS_FAIL: " << error.what() << '\n';
    return 1;
  }
}
