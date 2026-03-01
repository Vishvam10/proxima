#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "hnsw.h"

using std::vector;

TEST(HnswCPU, BasicInsertAndSearch) {
    constexpr size_t N = 100;
    constexpr size_t DIM = 16;
    constexpr size_t K = 5;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    vector<vector<float>> data(N, vector<float>(DIM));
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < DIM; ++j)
            data[i][j] = dist(gen);

    HnswCPU index(8, 100);
    index.create(data);

    EXPECT_EQ(index.size(), N);

    for (size_t i = 0; i < 10; ++i) {
        auto result = index.search(data[i], K);
        // self ID should be present
        bool found_self = false;
        for (auto id : result)
            if (id == i) found_self = true;
        EXPECT_TRUE(found_self);
    }
}

TEST(HnswCPU, EmptyIndex) {
    HnswCPU index;
    vector<float> query(16, 0.5f);
    auto result = index.search(query, 5);
    EXPECT_TRUE(result.empty());
}

TEST(HnswCPU, AddSingleElement) {
    HnswCPU index;
    vector<float> v = {1.0f, 2.0f, 3.0f};
    index.add(v);
    EXPECT_EQ(index.size(), 1);

    auto result = index.search(v, 1);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 0);
}