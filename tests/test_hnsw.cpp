#include "hnsw.h"
#include <gtest/gtest.h>
#include <vector>

using std::vector;

/* ========================================
   Basic Construction
   ======================================== */

TEST(HnswBasicTest, EmptyIndex) {
    HnswCPU index;
    EXPECT_EQ(index.size(), 0);

    vector<float> q = {1.0f, 2.0f};
    auto result = index.search(q, 5);
    EXPECT_TRUE(result.empty());
}

TEST(HnswBasicTest, SinglePointSelfQuery) {
    HnswCPU index;
    vector<vector<float>> data = {{1.0f, 2.0f}};

    index.create(data);

    auto result = index.search(data[0], 1);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 0);
}

TEST(HnswBasicTest, TwoPointsNearest) {
    HnswCPU index;

    vector<vector<float>> data = {
        {0.0f, 0.0f},
        {10.0f, 10.0f}
    };

    index.create(data);

    auto result = index.search({0.1f, 0.1f}, 1);
    EXPECT_EQ(result[0], 0);
}

TEST(HnswEdgeTest, DuplicateVectors) {
    HnswCPU index;

    vector<vector<float>> data = {
        {1.0f, 1.0f},
        {1.0f, 1.0f},
        {1.0f, 1.0f}
    };

    index.create(data);

    auto result = index.search({1.0f, 1.0f}, 3);

    EXPECT_EQ(result.size(), 3);
}

TEST(HnswEdgeTest, HighDimLowN) {
    HnswCPU index;

    vector<vector<float>> data(5, vector<float>(512, 0.5f));

    index.create(data);

    auto result = index.search(data[0], 3);

    EXPECT_EQ(result.size(), 3);
}

TEST(HnswEdgeTest, SelfRecallRandom) {
    HnswCPU index(16, 200);

    int N = 500;
    int DIM = 32;

    vector<vector<float>> data(N, vector<float>(DIM));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < DIM; ++j)
            data[i][j] = dist(gen);

    index.create(data);

    auto result = index.search(data[123], 1);

    EXPECT_EQ(result[0], 123);
}