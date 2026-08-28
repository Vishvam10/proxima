#pragma once

#include <cstddef>

namespace proxima {

enum class DistanceType { L2, INNER_PRODUCT, COSINE };

double computeDistance(
    DistanceType type,
    const float *a,
    const float *b,
    std::size_t dim,
    bool force_scalar = false
);

void printSimdInfo();

} // namespace proxima