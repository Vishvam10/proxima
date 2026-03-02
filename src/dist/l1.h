#pragma once

#include <cstddef>

// L1 (Manhattan) distance variants.
inline float l1_scalar(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += (d >= 0.0f) ? d : -d;
    }
    return sum;
}

#if defined(__AVX2__)
#include <immintrin.h>

inline float l1_avx(const float* a, const float* b, std::size_t dim) {
    const __m256 zero = _mm256_setzero_ps();
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 neg = _mm256_sub_ps(zero, diff);
        __m256 absDiff = _mm256_max_ps(diff, neg);
        sum = _mm256_add_ps(sum, absDiff);
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);

    float total = tmp[0] + tmp[1] + tmp[2] + tmp[3]
                + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += (d >= 0.0f) ? d : -d;
    }

    return total;
}
#endif

#if defined(__ARM_NEON__)
#include <arm_neon.h>

inline float l1_neon(const float* a, const float* b, std::size_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        float32x4_t absDiff = vabsq_f32(diff);
        sum = vaddq_f32(sum, absDiff);
    }

    float tmp[4];
    vst1q_f32(tmp, sum);
    float total = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += (d >= 0.0f) ? d : -d;
    }

    return total;
}
#endif

