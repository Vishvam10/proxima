#include "proxima/dist/dispatch.h"

#include "proxima/dist/cosine.h"
#include "proxima/dist/inner_product.h"
#include "proxima/dist/l2.h"

#include <iostream>

namespace proxima {

double computeDistance(
    DistanceType type,
    const float *a,
    const float *b,
    std::size_t dim,
    bool force_scalar
) {
    switch (type) {
    case DistanceType::L2:
        if (force_scalar) {
            return l2_scalar(a, b, dim);
        }

#if defined(__AVX2__)
        return l2_avx(a, b, dim);
#elif defined(__ARM_NEON)
        return l2_neon(a, b, dim);
#else
        return l2_scalar(a, b, dim);
#endif

    case DistanceType::INNER_PRODUCT:
        if (force_scalar) {
            return ip_scalar(a, b, dim);
        }

#if defined(__AVX2__)
        return ip_avx(a, b, dim);
#elif defined(__ARM_NEON)
        return ip_neon(a, b, dim);
#else
        return ip_scalar(a, b, dim);
#endif

    case DistanceType::COSINE:
        if (force_scalar) {
            return cosine_scalar(a, b, dim);
        }

#if defined(__AVX2__)
        return cosine_avx(a, b, dim);
#elif defined(__ARM_NEON)
        return cosine_neon(a, b, dim);
#else
        return cosine_scalar(a, b, dim);
#endif
    }

    return l2_scalar(a, b, dim);
}

void printSimdInfo() {
#if defined(__ARM_NEON)
    std::cout << "[SIMD] Compiled with NEON\n";
#elif defined(__AVX2__)
    std::cout << "[SIMD] Compiled with AVX2\n";
#elif defined(__SSE2__)
    std::cout << "[SIMD] Compiled with SSE2\n";
#else
    std::cout << "[SIMD] Scalar fallback\n";
#endif
}

} // namespace proxima