#include "hnsw.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using std::cout, std::vector;

/* ==============================
   Brute Force Baseline
   ============================== */

vector<uint64_t> bruteForce(
    const vector<vector<float>> &data,
    const vector<float> &query,
    size_t k
) {
    vector<std::pair<float, uint64_t>> dists;

    for (uint64_t i = 0; i < data.size(); ++i) {
        float dist = 0.0f;
        for (size_t j = 0; j < query.size(); ++j) {
            float diff = query[j] - data[i][j];
            dist += diff * diff;
        }
        dists.push_back({dist, i});
    }

    std::sort(dists.begin(), dists.end());

    vector<uint64_t> result;
    for (size_t i = 0; i < k; ++i)
        result.push_back(dists[i].second);

    return result;
}

/* ==============================
   Main Benchmark
   ============================== */

int main() {
    const int N = 10000;
    const int DIM = 64;
    const int K = 10;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    vector<vector<float>> data(N, vector<float>(DIM));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < DIM; ++j)
            data[i][j] = dist(gen);

    HnswCPU index(16, 200);

    /* Build Benchmark */
    auto t1 = std::chrono::high_resolution_clock::now();
    index.create(data);
    auto t2 = std::chrono::high_resolution_clock::now();

    cout << "Build time : " << std::chrono::duration<double>(t2 - t1).count()
         << " sec\n";

    /* Query Benchmark */
    vector<float> query = data[1234];

    auto q1 = std::chrono::high_resolution_clock::now();
    auto hnswResult = index.search(query, K);
    auto q2 = std::chrono::high_resolution_clock::now();

    cout << "Query time: "
         << std::chrono::duration<double, std::micro>(q2 - q1).count()
         << " us\n";

    /* Recall */
    auto bfResult = bruteForce(data, query, K);

    int correct = 0;
    for (auto id : hnswResult)
        if (std::find(bfResult.begin(), bfResult.end(), id) != bfResult.end())
            correct++;

    cout << "Recall @" << K << " = " << (float)correct / K << "\n";

    return 0;
}