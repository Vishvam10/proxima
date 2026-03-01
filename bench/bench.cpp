#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "hnsw.h"

using std::vector;
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

int main() {
    constexpr size_t N = 1000;
    constexpr size_t DIM = 128;
    constexpr size_t K = 10;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    vector<vector<float>> data(N, vector<float>(DIM));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < DIM; ++j) {
            data[i][j] = dist(gen);
        }
    }

    HnswCPU index(16, 200);
    auto t1 = high_resolution_clock::now();
    index.create(data);
    auto t2 = high_resolution_clock::now();
    auto build_time = duration_cast<microseconds>(t2 - t1).count() / 1e6;

    cout << "Build time : " << build_time << " sec" << endl;

    size_t total_correct = 0;
    auto t3 = high_resolution_clock::now();
    for (size_t i = 0; i < 100; ++i) {
        auto result = index.search(data[i], K);

        for (size_t id : result) {
            if (id == i) {
                ++total_correct;
                break;
            }
        }
    }
    auto t4 = high_resolution_clock::now();
    auto query_time = duration_cast<microseconds>(t4 - t3).count() / 100.0;

    cout << "Query time (us) : " << query_time << endl;
    cout << "Recall @" << K << " = "
         << static_cast<float>(total_correct) / 100 << endl;

    return 0;
}