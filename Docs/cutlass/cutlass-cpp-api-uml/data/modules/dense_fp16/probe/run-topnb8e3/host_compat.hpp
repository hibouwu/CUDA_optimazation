// Probe-local libstdc++14 compatibility for CUDA13.0 + current glibc.
// _GNU_SOURCE is disabled by the established NVCC/rsqrt workaround. GCC's
// installed c++config still advertises these optional GNU pthread overloads,
// although pthread.h then hides their declarations. Disable only those Host
// overload paths before <mutex>; no CUTLASS/CUDA/system header is edited.
#pragma once
#include <bits/c++config.h>
#undef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
#undef _GLIBCXX_USE_PTHREAD_MUTEX_CLOCKLOCK
#define _GLIBCXX_USE_PTHREAD_MUTEX_CLOCKLOCK 0
