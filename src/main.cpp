#include <iostream>
#include <random>
#include "hnsw.h"

int main() {
    const int N = 1000;
    const int DIM = 32;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> data(N, std::vector<float>(DIM));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < DIM; ++j)
            data[i][j] = dist(gen);

    HnswCPU index(16, 200);

    index.create(data);

    std::vector<float> query = data[123];

    auto result = index.search(query, 5);

    std::cout << "Top 5 neighbors:\n";
    for (auto id : result)
        std::cout << id << "\n";
}