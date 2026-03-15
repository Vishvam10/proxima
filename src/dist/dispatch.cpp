#include "dispatch.h"
#include "cosine.h"
#include "inner_product.h"
#include "l2.h"
double computeDistance(
    DistanceType type,
    const float *a,
    const float *b,
    std::size_t dim,
    bool forceScalar
) {
    switch (type) {
    case DistanceType::L2:
        if (forceScalar)
            return l2_scalar(a, b, dim);
#if defined(__AVX2__)
        return l2_avx(a, b, dim);
#elif defined(__ARM_NEON__)
        return l2_neon(a, b, dim);
#else
        return l2_scalar(a, b, dim);
#endif
    case DistanceType::INNER_PRODUCT:
        if (forceScalar)
            return ip_scalar(a, b, dim);
#if defined(__AVX2__)
        return ip_avx(a, b, dim);
#elif defined(__ARM_NEON__)
        return ip_neon(a, b, dim);
#else
        return ip_scalar(a, b, dim);
#endif
    case DistanceType::COSINE:
        if (forceScalar)
            return cosine_scalar(a, b, dim);
#if defined(__AVX2__)
        return cosine_avx(a, b, dim);
#elif defined(__ARM_NEON__)
        return cosine_neon(a, b, dim);
#else
        return cosine_scalar(a, b, dim);
#endif
    }
    return l2_scalar(a, b, dim);
}
#include "dispatch.h"
#include <iostream>
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