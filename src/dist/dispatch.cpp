#include "dispatch.h"
#include "l2.h"
#include "l1.h"
#include "cosine.h"

#include <iostream>

DistFunc getDistanceFunction(DistanceType type) {
    if (type == DistanceType::L2) {
#if defined(__AVX2__)
        return &l2_avx;
#elif defined(__ARM_NEON__)
        return &l2_neon;
#else
        return &l2_scalar;
#endif
    }

    if (type == DistanceType::L1) {
#if defined(__AVX2__)
        return &l1_avx;
#elif defined(__ARM_NEON__)
        return &l1_neon;
#else
        return &l1_scalar;
#endif
    }

    if (type == DistanceType::COSINE) {
#if defined(__AVX2__)
        return &cosine_avx;
#elif defined(__ARM_NEON__)
        return &cosine_neon;
#else
        return &cosine_scalar;
#endif
    }

    return &l2_scalar;
}

void printSimdInfo() {
#if defined(__ARM_NEON__)
    std::cout << "[SIMD] Compiled with NEON\n";
#elif defined(__AVX2__)
    std::cout << "[SIMD] Compiled with AVX2\n";
#elif defined(__SSE2__)
    std::cout << "[SIMD] Compiled with SSE2\n";
#else
    std::cout << "[SIMD] Scalar fallback\n";
#endif
}
