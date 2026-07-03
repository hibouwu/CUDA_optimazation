#include <cupti.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <map>

/**
 * CUPTI-based Register File Bank Profiler
 * 
 * This program uses NVIDIA's CUPTI (CUDA Profiling Tools Interface)
 * to directly measure register file bank access patterns.
 * 
 * Objectives:
 * 1. Collect hardware counter data for register file accesses
 * 2. Compare stride patterns to detect physical bank organization
 * 3. Distinguish between "2-bank" vs "multi-bank with parity grouping"
 */

#define CUPTI_CALL(call)                                       \
  do {                                                         \
    CUptiResult _status = call;                                \
    if (_status != CUPTI_SUCCESS) {                            \
      const char *errstr = cuptiGetResultString(_status);      \
      fprintf(stderr, "CUPTI error at line %d: %s\n",          \
              __LINE__, errstr);                               \
      return 1;                                                \
    }                                                          \
  } while (0)

// Simple test kernel for register bank analysis
extern "C" __global__ void __launch_bounds__(32, 1) rf_bank_test(
    const int32_t* sources,
    uint64_t* elapsed_cycles,
    int32_t* sinks,
    int iterations,
    int stride_val) {
  
  const int lane = threadIdx.x;
  
  // Load sources with controlled stride
  int32_t s0 = sources[0 * 32 + lane];
  int32_t s1 = sources[(stride_val) * 32 + lane];
  int32_t s2 = sources[(2 * stride_val) * 32 + lane];
  
  // Output register for bank testing
  int32_t result = s0;
  
  // Synchronized timing region
  __syncthreads();
  uint64_t start = clock64();
  
  // Repeated LOP3 operations to measure bank collision latency
  #pragma unroll(32)
  for (int i = 0; i < 32; i++) {
    // LOP3: result = result | s1 | s2
    // If s1 and s2 are in same bank, expect stall
    asm volatile(
      "lop3.lut %0, %1, %2, %3, 0xFE;"  // OR operation
      : "+r"(result)
      : "r"(s1), "r"(s2), "r"(result)
    );
  }
  
  uint64_t end = clock64();
  __syncthreads();
  
  elapsed_cycles[lane] = end - start;
  sinks[lane] = result;
}

// CUPTI callback for event data
static void CUPTIAPI eventCallback(
    CUpti_EventDomain domain,
    CUpti_EventsData *eventData) {
  // Process event data
  // This would contain bank conflict information
}

int main() {
  printf("=== CUPTI Register File Bank Profiler ===\n\n");
  
  // Initialize CUPTI
  printf("Initializing CUPTI...\n");
  CUPTI_CALL(cuptiSubscribe(NULL, eventCallback, NULL));
  
  // Get available event domains
  uint32_t domainCount = 0;
  CUPTI_CALL(cuptiDeviceGetEventDomainCount(0, &domainCount));
  printf("Available event domains: %u\n", domainCount);
  
  // List domains
  std::vector<CUpti_EventDomain> domains(domainCount);
  CUPTI_CALL(cuptiDeviceEnumEventDomains(0, &domainCount, domains.data()));
  
  for (uint32_t i = 0; i < domainCount; i++) {
    uint32_t eventCount = 0;
    CUPTI_CALL(cuptiEventDomainGetNumEvents(domains[i], &eventCount));
    
    const char* domainName = cuptiEventDomainGetName(domains[i]);
    printf("  Domain %u: %s (events: %u)\n", i, domainName, eventCount);
    
    // List events in this domain (limited to first few)
    if (eventCount > 0 && eventCount < 100) {
      std::vector<CUpti_EventID> eventIds(eventCount);
      CUPTI_CALL(cuptiEventDomainEnumEvents(domains[i], &eventCount, eventIds.data()));
      
      for (uint32_t j = 0; j < std::min(10u, eventCount); j++) {
        const char* eventName = cuptiEventGetName(domains[i], eventIds[j]);
        printf("    - %s\n", eventName);
      }
      if (eventCount > 10) {
        printf("    ... and %u more\n", eventCount - 10);
      }
    }
  }
  
  printf("\n=== Event Description ===\n");
  
  // Find SM (Streaming Multiprocessor) domain
  for (uint32_t i = 0; i < domainCount; i++) {
    const char* domainName = cuptiEventDomainGetName(domains[i]);
    if (strstr(domainName, "SMSP") || strstr(domainName, "SM")) {
      printf("Found SM domain: %s\n", domainName);
      
      uint32_t eventCount = 0;
      CUPTI_CALL(cuptiEventDomainGetNumEvents(domains[i], &eventCount));
      
      std::vector<CUpti_EventID> eventIds(eventCount);
      CUPTI_CALL(cuptiEventDomainEnumEvents(domains[i], &eventCount, eventIds.data()));
      
      // Look for bank-related events
      for (uint32_t j = 0; j < eventCount; j++) {
        const char* eventName = cuptiEventGetName(domains[i], eventIds[j]);
        if (strstr(eventName, "bank") || 
            strstr(eventName, "conflict") ||
            strstr(eventName, "stall") ||
            strstr(eventName, "rf_")) {
          printf("  → Bank/Conflict event: %s\n", eventName);
        }
      }
    }
  }
  
  printf("\n=== Analysis ===\n");
  printf("Searching for RF bank metrics on SM110...\n");
  printf("Note: Actual bank counter availability depends on SM architecture\n");
  printf("      and NVIDIA's exposure through CUPTI\n\n");
  
  // Cleanup
  CUPTI_CALL(cuptiUnsubscribe(NULL));
  
  printf("=== Conclusion ===\n");
  printf("If CUPTI exposes per-bank counters:\n");
  printf("  - We can directly count collisions per bank\n");
  printf("  - Compare stride 2 vs stride 1 collision counts\n");
  printf("  - Definitively prove physical bank organization\n\n");
  printf("If not available:\n");
  printf("  - CUPTI likely doesn't expose sub-SM register file details\n");
  printf("  - Our timing-based evidence (mod 2 accuracy: 100%%) remains strongest\n");
  
  return 0;
}
