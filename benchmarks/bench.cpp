#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "hnsw.h"

using std::cout;
using std::endl;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

struct Scenario {
    size_t N;
    size_t DIM;
    size_t K;
};

struct DistConfig {
    DistanceType type;
    const char* name;
};

struct SimdMode {
    bool forceScalar;
    const char* label;
    bool useMultithread;
};

vector<int> bruteForceKNN(const vector<vector<float>>& data,
                          const vector<float>& query, size_t k) {
    vector<std::pair<float, int>> dists(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        float d = 0;
        for (size_t j = 0; j < query.size(); ++j) {
            float diff = data[i][j] - query[j];
            d += diff * diff;
        }
        dists[i] = {d, static_cast<int>(i)};
    }
    std::partial_sort(dists.begin(), dists.begin() + static_cast<ptrdiff_t>(k), dists.end());
    vector<int> result(k);
    for (size_t i = 0; i < k; ++i) result[i] = dists[i].second;
    return result;
}

int main() {
    vector<Scenario> scenarios = {
        {1000, 128, 10},
        {5000, 64, 10},
        {10000, 32, 5},
        {50000, 64, 10},
        {100000, 32, 10}
    };

    vector<DistConfig> dists = {
        {DistanceType::L2, "l2"},
        {DistanceType::INNER_PRODUCT, "inner_product"},
        {DistanceType::COSINE, "cosine"}
    };

    vector<SimdMode> modes = {
        {true, "scalar", false},
        {false, "simd", false},
        {false, "simd_mt", true}
    };

    int numThreads = static_cast<int>(std::thread::hardware_concurrency());
    if (numThreads == 0) numThreads = 4;

    cout << "\nC++ Benchmarks\n\n";
    cout << "Hardware threads available: " << numThreads << "\n\n";

    cout << std::fixed << std::setprecision(4);
    cout << "Distance | Mode | Dataset | Dim | K | Build (s) | Query (us) | Brute (us) | Speedup | Recall\n";
    cout << "----------------------------------------------------------------------------------------------\n";

    std::ofstream csv("benchmarks/cpp_results.csv");
    csv << "distance,simd_mode,dataset,dim,k,build_s,query_us,brute_query_us,speedup,recall\n";

    std::mt19937 gen(42);

    for (const auto& dcfg : dists) {
        for (const auto& mode : modes) {
            for (const auto& s : scenarios) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);

                vector<vector<float>> data(s.N, vector<float>(s.DIM));
                for (size_t i = 0; i < s.N; ++i)
                    for (size_t j = 0; j < s.DIM; ++j)
                        data[i][j] = dist(gen);

                HnswCPU index(16, 200, 42, dcfg.type, mode.forceScalar);

                auto t1 = high_resolution_clock::now();
                if (mode.useMultithread) {
                    index.addParallel(data, numThreads);
                } else {
                    index.create(data);
                }
                auto t2 = high_resolution_clock::now();
                double build_time = duration_cast<microseconds>(t2 - t1).count() / 1e6;

                size_t total_correct = 0;
                size_t query_count = std::min(size_t(100), s.N);

                auto t3 = high_resolution_clock::now();
                for (size_t i = 0; i < query_count; ++i) {
                    auto result = index.search(data[i], static_cast<int>(s.K), 200);
                    for (int id : result)
                        if (id == static_cast<int>(i)) {
                            ++total_correct;
                            break;
                        }
                }
                auto t4 = high_resolution_clock::now();
                double query_time = duration_cast<microseconds>(t4 - t3).count() / double(query_count);

                double recall = static_cast<double>(total_correct) / query_count;

                auto t5 = high_resolution_clock::now();
                for (size_t i = 0; i < query_count; ++i) {
                    bruteForceKNN(data, data[i], s.K);
                }
                auto t6 = high_resolution_clock::now();
                double brute_query_time = duration_cast<microseconds>(t6 - t5).count() / double(query_count);

                double speedup = brute_query_time / query_time;

                cout << dcfg.name << " | " << mode.label << " | "
                     << s.N << " | " << s.DIM << " | " << s.K << " | "
                     << build_time << " | " << query_time << " | "
                     << brute_query_time << " | " << speedup << "x | " << recall << "\n";

                csv << dcfg.name << "," << mode.label << "," << s.N << "," << s.DIM << "," << s.K << ","
                    << build_time << "," << query_time << ","
                    << brute_query_time << "," << speedup << "," << recall << "\n";
            }
        }
    }

    cout << "\nResults saved to benchmarks/cpp_results.csv\n";

    return 0;
}
