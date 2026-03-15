#pragma once
#include <cstddef>
enum class DistanceType { L2, INNER_PRODUCT, COSINE };
double computeDistance(
    DistanceType type,
    const float *a,
    const float *b,
    std::size_t dim,
    bool forceScalar = false
);
void printSimdInfo();