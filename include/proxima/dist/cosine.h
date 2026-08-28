#pragma once

#include <cmath>
#include <cstddef>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace proxima {

// Cosine similarity variants.
// Returns raw cosine similarity in [-1, 1], larger is more similar.

inline double cosine_scalar(const float *a, const float *b, std::size_t dim) {
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (std::size_t i = 0; i < dim; ++i) {
        const double va = static_cast<double>(a[i]);
        const double vb = static_cast<double>(b[i]);

        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }

    if (normA == 0.0 || normB == 0.0) {
        return 0.0;
    }

    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

#if defined(__AVX2__)

inline double cosine_avx(const float *a, const float *b, std::size_t dim) {
    __m256 dotVec = _mm256_setzero_ps();
    __m256 normAVec = _mm256_setzero_ps();
    __m256 normBVec = _mm256_setzero_ps();

    std::size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);

        dotVec = _mm256_fmadd_ps(va, vb, dotVec);
        normAVec = _mm256_fmadd_ps(va, va, normAVec);
        normBVec = _mm256_fmadd_ps(vb, vb, normBVec);
    }

    float dotTmp[8];
    float normATmp[8];
    float normBTmp[8];

    _mm256_storeu_ps(dotTmp, dotVec);
    _mm256_storeu_ps(normATmp, normAVec);
    _mm256_storeu_ps(normBTmp, normBVec);

    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int j = 0; j < 8; ++j) {
        dot += static_cast<double>(dotTmp[j]);
        normA += static_cast<double>(normATmp[j]);
        normB += static_cast<double>(normBTmp[j]);
    }

    for (; i < dim; ++i) {
        const double va = static_cast<double>(a[i]);
        const double vb = static_cast<double>(b[i]);

        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }

    if (normA == 0.0 || normB == 0.0) {
        return 0.0;
    }

    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

#endif

#if defined(__ARM_NEON__)

inline double cosine_neon(const float *a, const float *b, std::size_t dim) {
    float32x4_t dotVec = vdupq_n_f32(0.0f);
    float32x4_t normAVec = vdupq_n_f32(0.0f);
    float32x4_t normBVec = vdupq_n_f32(0.0f);

    std::size_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);

        dotVec = vmlaq_f32(dotVec, va, vb);
        normAVec = vmlaq_f32(normAVec, va, va);
        normBVec = vmlaq_f32(normBVec, vb, vb);
    }

    float dotTmp[4];
    float normATmp[4];
    float normBTmp[4];

    vst1q_f32(dotTmp, dotVec);
    vst1q_f32(normATmp, normAVec);
    vst1q_f32(normBTmp, normBVec);

    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int j = 0; j < 4; ++j) {
        dot += static_cast<double>(dotTmp[j]);
        normA += static_cast<double>(normATmp[j]);
        normB += static_cast<double>(normBTmp[j]);
    }

    for (; i < dim; ++i) {
        const double va = static_cast<double>(a[i]);
        const double vb = static_cast<double>(b[i]);

        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }

    if (normA == 0.0 || normB == 0.0) {
        return 0.0;
    }

    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

#endif

} // namespace proxima