#include "../common/gmem_to_smem_demo.cuh"

int main(int argc, char** argv) {
  return gmem_to_smem::run_demo_entry(
      argc, argv, gmem_to_smem::DemoKind::kTmaScaffold);
}
