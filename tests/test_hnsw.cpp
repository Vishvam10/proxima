#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "hnsw.h"
#include "dist/l2.h"
#include "dist/inner_product.h"
#include "dist/cosine.h"

using std::vector;

namespace {

void expectDistFuncsClose(
    DistFunc a,
    DistFunc b,
    std::size_t dim,
    bool isCosine = false
) {
    std::mt19937 gen(777);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    vector<float> v1(dim), v2(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        v1[i] = dist(gen);
        v2[i] = dist(gen);
    }

    double ra = a(v1.data(), v2.data(), dim);
    double rb = b(v1.data(), v2.data(), dim);

    if (isCosine) {
        EXPECT_NEAR(ra, rb, 1e-5f);
    } else {
        EXPECT_FLOAT_EQ(ra, rb);
    }
}

} // namespace

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
        ASSERT_EQ(result.size(), K);
        for (auto id : result) {
            EXPECT_LT(id, N);
        }
    }
}

TEST(DistanceFunctions, BasicKernels) {
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float b[3] = {2.0f, 4.0f, 6.0f};

    EXPECT_FLOAT_EQ(l2_scalar(a, b, 3), 14.0f);
    EXPECT_FLOAT_EQ(ip_scalar(a, b, 3), 24.0f);
    EXPECT_NEAR(cosine_scalar(a, b, 3), 1.0f, 1e-5f);

    DistFunc l2 = getDistanceFunction(DistanceType::L2);
    DistFunc ip = getDistanceFunction(DistanceType::INNER_PRODUCT);
    DistFunc cos = getDistanceFunction(DistanceType::COSINE);

    EXPECT_FLOAT_EQ(l2(a, b, 3), l2_scalar(a, b, 3));
    EXPECT_FLOAT_EQ(ip(a, b, 3), ip_scalar(a, b, 3));
    EXPECT_NEAR(cos(a, b, 3), cosine_scalar(a, b, 3), 1e-5f);
}

TEST(DistanceFunctions, DispatchMatchesSimdImplementationsWhenAvailable) {
    constexpr std::size_t dim = 16;

#if defined(__AVX2__)
    DistFunc l2 = getDistanceFunction(DistanceType::L2);
    DistFunc l1 = getDistanceFunction(DistanceType::L1);
    DistFunc cos = getDistanceFunction(DistanceType::COSINE);

    expectDistFuncsClose(l2, &l2_avx, dim, false);
    expectDistFuncsClose(l1, &l1_avx, dim, false);
    expectDistFuncsClose(cos, &cosine_avx, dim, true);
#elif defined(__ARM_NEON__)
    DistFunc l2 = getDistanceFunction(DistanceType::L2);
    DistFunc l1 = getDistanceFunction(DistanceType::INNER_PRODUCT);
    DistFunc cos = getDistanceFunction(DistanceType::COSINE);

    expectDistFuncsClose(l2, &l2_neon, dim, false);
    expectDistFuncsClose(l1, &ip_neon, dim, false);
    expectDistFuncsClose(cos, &cosine_neon, dim, true);
#else
    // On scalar-only builds, just ensure dispatch matches scalar paths.
    DistFunc l2 = getDistanceFunction(DistanceType::L2);
    DistFunc l1 = getDistanceFunction(DistanceType::L1);
    DistFunc cos = getDistanceFunction(DistanceType::COSINE);

    expectDistFuncsClose(l2, &l2_scalar, dim, false);
    expectDistFuncsClose(l1, &l1_scalar, dim, false);
    expectDistFuncsClose(cos, &cosine_scalar, dim, true);
#endif
}

TEST(DistanceFunctions, PrintSimdInfoDoesNotCrash) {
    printSimdInfo();
}


TEST(HnswCPU, L1AndCosineModes) {
    constexpr size_t N = 50;
    constexpr size_t DIM = 8;
    constexpr size_t K = 3;

    std::mt19937 gen(321);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    vector<vector<float> > data(N, vector<float>(DIM));
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < DIM; ++j)
            data[i][j] = dist(gen);

    HnswCPU indexL1(8, 100, 123, DistanceType::INNER_PRODUCT);
    indexL1.create(data);

    HnswCPU indexCos(8, 100, 123, DistanceType::COSINE);
    indexCos.create(data);

    for (size_t i = 0; i < 5; ++i) {
        auto resL1 = indexL1.search(data[i], K);
        auto resCos = indexCos.search(data[i], K);

        ASSERT_EQ(resL1.size(), K);
        ASSERT_EQ(resCos.size(), K);

        for (auto id : resL1) {
            EXPECT_LT(id, N);
        }
        for (auto id : resCos) {
            EXPECT_LT(id, N);
        }
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