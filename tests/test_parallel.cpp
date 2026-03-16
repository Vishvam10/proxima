#include "hnsw.h"
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::cerr << "Starting parallel test...\n";

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    size_t N = 1000;
    size_t DIM = 128;
    std::vector<std::vector<float>> data(N, std::vector<float>(DIM));
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < DIM; ++j)
            data[i][j] = dist(gen);

    std::cerr << "Data generated, building index with addParallel...\n";

    HnswCPU index(16, 200, 42, DistanceType::L2, false);
    index.addParallel(data, 8);

    std::cerr << "Index built! Size=" << index.size() << "\n";

    auto result = index.search(data[0], 10, 200);
    std::cerr << "Search result size=" << result.size() << "\n";
    for (int id : result)
        std::cerr << id << " ";
    std::cerr << "\n";

    std::cerr << "PASS\n";
    return 0;
}
