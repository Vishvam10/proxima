#pragma once

#include <cstddef>

// Inner product distance: 1.0f - dot(a, b)
// Lower value = more similar (matches hnswlib "ip" space convention).
inline double ip_scalar(const float* a, const float* b, std::size_t dim) {
    double dot = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return 1.0 - dot;
}

#if defined(__AVX2__)
#include <immintrin.h>

inline double ip_avx(const float* a, const float* b, std::size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);

    double dot = tmp[0] + tmp[1] + tmp[2] + tmp[3]
               + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < dim; ++i) {
        dot += a[i] * b[i];
    }

    return 1.0 - dot;
}
#endif

#if defined(__ARM_NEON__)
#include <arm_neon.h>

inline double ip_neon(const float* a, const float* b, std::size_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);
    }

    float tmp[4];
    vst1q_f32(tmp, sum);
    double dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < dim; ++i) {
        dot += a[i] * b[i];
    }

    return 1.0 - dot;
}
#endif
