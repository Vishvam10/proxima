#pragma once

#include <cstddef>

enum class DistanceType {
    L2,
    L1,
    COSINE
};

using DistFunc = float (*)(const float*, const float*, std::size_t);

DistFunc getDistanceFunction(DistanceType type);
void printSimdInfo();
