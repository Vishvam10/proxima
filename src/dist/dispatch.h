#pragma once

#include <cstddef>

enum class DistanceType { L2, INNER_PRODUCT, COSINE };

using DistFunc = double (*)(const float *, const float *, std::size_t);

DistFunc getDistanceFunction(DistanceType type, bool forceScalar = false);
const char *getSimdLabel();
void printSimdInfo();
