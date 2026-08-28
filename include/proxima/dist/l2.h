#pragma once

#include <cstddef>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace proxima {

inline double l2_scalar(const float *a, const float *b, std::size_t dim) {
    double sum = 0.0;

    for (std::size_t i = 0; i < dim; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);

        sum += d * d;
    }

    return sum;
}

#if defined(__AVX2__)

inline double l2_avx(const float *a, const float *b, std::size_t dim) {
    __m256 sum = _mm256_setzero_ps();

    std::size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);

        const __m256 diff = _mm256_sub_ps(va, vb);
        const __m256 sq = _mm256_mul_ps(diff, diff);

        sum = _mm256_add_ps(sum, sq);
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);

    double total = 0.0;

    for (float value : tmp) {
        total += static_cast<double>(value);
    }

    for (; i < dim; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);

        total += d * d;
    }

    return total;
}

#endif

#if defined(__ARM_NEON) || defined(__aarch64__)

inline double l2_neon(const float *a, const float *b, std::size_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    std::size_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);

        const float32x4_t diff = vsubq_f32(va, vb);

        sum = vmlaq_f32(sum, diff, diff);
    }

    float tmp[4];
    vst1q_f32(tmp, sum);

    double total = 0.0;

    for (float value : tmp) {
        total += static_cast<double>(value);
    }

    for (; i < dim; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);

        total += d * d;
    }

    return total;
}

#endif

} // namespace proxima