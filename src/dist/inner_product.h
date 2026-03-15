#pragma once

#include <cstddef>
#include <iostream>

inline double ip_scalar(const float *a, const float *b, std::size_t dim) {
    double sum = 0.0;
    std::cout << "ip_scalar called\n";

    for (std::size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }

    return sum;
}

#if defined(__AVX2__)
#include <immintrin.h>

inline double ip_avx(const float *a, const float *b, std::size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);

    double dot = static_cast<double>(tmp[0]) + static_cast<double>(tmp[1]) +
                 static_cast<double>(tmp[2]) + static_cast<double>(tmp[3]) +
                 static_cast<double>(tmp[4]) + static_cast<double>(tmp[5]) +
                 static_cast<double>(tmp[6]) + static_cast<double>(tmp[7]);

    for (; i < dim; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }

    return dot;
}
#endif

#if defined(__ARM_NEON__)
#include <arm_neon.h>

inline double ip_neon(const float *a, const float *b, std::size_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    std::cout << "ip_neon called\n\n";

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);
    }

    float tmp[4];
    vst1q_f32(tmp, sum);
    double dot = static_cast<double>(tmp[0]) + static_cast<double>(tmp[1]) +
                 static_cast<double>(tmp[2]) + static_cast<double>(tmp[3]);

    for (; i < dim; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }

    return dot;
}
#endif
