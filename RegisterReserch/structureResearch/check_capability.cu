#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  int device;
  cudaGetDevice(&device);
  
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  
  printf("GPU: %s\n", prop.name);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("Max Registers per Block: %d\n", prop.regsPerBlock);
  printf("Max Registers per Thread: %d\n", prop.regsPerMultiprocessor / 32);
  printf("Register Bank Width: %d bytes (typical)\n", 256);  // Typical for modern GPUs
  printf("\nNote: Register file is typically organized in 2-4 banks\n");
  printf("      for port contention management.\n");
  
  return 0;
}
