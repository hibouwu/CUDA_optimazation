#!/bin/bash

# Enable NVIDIA hardware performance counter collection
# for register file bank analysis

echo "=== NVIDIA Performance Counter Enumeration ==="
echo

# Method 1: Check available counters via environment
echo "Method 1: GPU counter enumeration"
echo "=================================="

export CUDA_PROFILER_LOG=profiler.log
export CUDA_PROFILE=1

# Run a simple kernel to trigger profiler
cat > test_profiler.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void simple_kernel(int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    for (int i = 0; i < 100; i++) {
      asm volatile("nop;");
    }
  }
}

int main() {
  simple_kernel<<<1, 32>>>(32);
  cudaDeviceSynchronize();
  return 0;
}
EOF

nvcc -o test_prof test_profiler.cu 2>/dev/null
echo "Running kernel with profiler enabled..."
./test_prof > /dev/null 2>&1

if [ -f profiler.log ]; then
  echo "✓ Profiler log generated"
  echo
  echo "=== Profiler Output (first 100 lines) ==="
  head -100 profiler.log
  
  # Search for bank-related counters
  echo
  echo "=== Bank/RF related counters found ==="
  grep -i "bank\|conflict\|stall\|rf_" profiler.log || echo "(none found)"
else
  echo "✗ Profiler log not generated"
fi

echo
echo "=== Method 2: Query via nvidia-smi ==="
echo "======================================"
nvidia-smi --query-gpu=name,driver_version,vbios_version 2>/dev/null || echo "nvidia-smi info unavailable"

echo
echo "=== Method 3: CUDA Capability Check ==="
echo "========================================"
cat > check_capability.cu << 'EOF'
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
EOF

nvcc -o check_cap check_capability.cu 2>/dev/null
./check_cap
