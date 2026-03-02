#pragma once

#include <cstddef>
#include <cmath>

// Cosine similarity variants.
// Returns raw cosine similarity in [-1, 1], larger is more similar.
inline float cosine_scalar(const float* a, const float* b, std::size_t dim) {
    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for (std::size_t i = 0; i < dim; ++i) {
        float va = a[i];
        float vb = b[i];
        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }

    if (normA == 0.0f || normB == 0.0f) {
        return 0.0f;
    }

    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

#if defined(__AVX2__)
#include <immintrin.h>

inline float cosine_avx(const float* a, const float* b, std::size_t dim) {
    __m256 dotVec = _mm256_setzero_ps();
    __m256 normAVec = _mm256_setzero_ps();
    __m256 normBVec = _mm256_setzero_ps();
    std::size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

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

    float dot = dotTmp[0] + dotTmp[1] + dotTmp[2] + dotTmp[3]
              + dotTmp[4] + dotTmp[5] + dotTmp[6] + dotTmp[7];
    float normA = normATmp[0] + normATmp[1] + normATmp[2] + normATmp[3]
                + normATmp[4] + normATmp[5] + normATmp[6] + normATmp[7];
    float normB = normBTmp[0] + normBTmp[1] + normBTmp[2] + normBTmp[3]
                + normBTmp[4] + normBTmp[5] + normBTmp[6] + normBTmp[7];

    for (; i < dim; ++i) {
        float va = a[i];
        float vb = b[i];
        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }

    if (normA == 0.0f || normB == 0.0f) {
        return 0.0f;
    }

    return dot / (std::sqrt(normA) * std::sqrt(normB));
}
#endif

#if defined(__ARM_NEON__)
#include <arm_neon.h>

inline float cosine_neon(const float* a, const float* b, std::size_t dim) {
    float32x4_t dotVec = vdupq_n_f32(0.0f);
    float32x4_t normAVec = vdupq_n_f32(0.0f);
    float32x4_t normBVec = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

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

    float dot = dotTmp[0] + dotTmp[1] + dotTmp[2] + dotTmp[3];
    float normA = normATmp[0] + normATmp[1] + normATmp[2] + normATmp[3];
    float normB = normBTmp[0] + normBTmp[1] + normBTmp[2] + normBTmp[3];

    for (; i < dim; ++i) {
        float va = a[i];
        float vb = b[i];
        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }

    if (normA == 0.0f || normB == 0.0f) {
        return 0.0f;
    }

    return dot / (std::sqrt(normA) * std::sqrt(normB));
}
#endif

