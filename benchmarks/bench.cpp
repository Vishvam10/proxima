#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
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
    const char* label;
};

struct SimdMode {
    const char* label;
    bool forceScalar;
};

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
        {DistanceType::COSINE, "cosine"},
    };

    vector<SimdMode> modes = {
        {"simd", false},
        {"scalar", true},
    };

    std::ofstream csv("benchmarks/cpp_results.csv");
    csv << "simd_mode,distance,dataset,dim,k,build_s,query_us,recall\n";

    cout << "\nC++ Benchmarks\n";
    printSimdInfo();
    cout << std::fixed << std::setprecision(4);

    for (const auto& mode : modes) {
        cout << "\n[" << mode.label << "]\n";
        cout << "Distance        | Dataset | Dim | K | Build (s) | Query (us) | Recall\n";
        cout << "------------------------------------------------------------------------\n";

        std::mt19937 gen(42);

        for (const auto& dcfg : dists) {
            for (auto s : scenarios) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);

                vector<vector<float>> data(s.N, vector<float>(s.DIM));
                for (size_t i = 0; i < s.N; ++i)
                    for (size_t j = 0; j < s.DIM; ++j)
                        data[i][j] = dist(gen);

                HnswCPU index(16, 200, 42, dcfg.type, mode.forceScalar);

                auto t1 = high_resolution_clock::now();
                index.create(data);
                auto t2 = high_resolution_clock::now();
                auto build_time =
                    duration_cast<microseconds>(t2 - t1).count() / 1e6;

                size_t total_correct = 0;
                auto t3 = high_resolution_clock::now();
                size_t query_count = std::min(size_t(100), s.N);
                for (size_t i = 0; i < query_count; ++i) {
                    auto result = index.search(data[i], static_cast<int>(s.K), 200);
                    for (int id : result)
                        if (id == static_cast<int>(i)) {
                            ++total_correct;
                            break;
                        }
                }
                auto t4 = high_resolution_clock::now();
                auto query_time = duration_cast<microseconds>(t4 - t3).count()
                                  / double(query_count);

                double recall = static_cast<double>(total_correct) / query_count;

                cout << std::left << std::setw(15) << dcfg.label << " | "
                     << std::right << s.N << " | " << s.DIM << " | "
                     << s.K << " | " << build_time << " | " << query_time
                     << " | " << recall << "\n";

                csv << mode.label << "," << dcfg.label << "," << s.N << ","
                    << s.DIM << "," << s.K << "," << build_time << ","
                    << query_time << "," << recall << "\n";
            }
        }
        cout << "\n";
    }
}
