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

int main() {

    cout << "\n";
    vector<Scenario> scenarios = {
        {1000, 128, 10},
        {5000, 64, 10},
        {10000, 32, 5},
        {50000, 64, 10},
        {100000, 32, 10}
    };

    cout << "\nC++ Benchmarks\n\n";

    // Human-readable table
    cout << std::fixed << std::setprecision(4);
    cout << "Dataset | Dim | K | Build (s) | Query (us) | Recall\n";
    cout << "------------------------------------------------------\n";

    // CSV output for automated comparison
    std::ofstream csv("benchmarks/cpp_results.csv");
    csv << "dataset,dim,k,build_s,query_us,recall\n";

    std::mt19937 gen(42);

    for (auto s : scenarios) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        vector<vector<float>> data(s.N, vector<float>(s.DIM));
        for (size_t i = 0; i < s.N; ++i)
            for (size_t j = 0; j < s.DIM; ++j)
                data[i][j] = dist(gen);

        HnswCPU index(16, 200);

        auto t1 = high_resolution_clock::now();
        index.create(data);
        auto t2 = high_resolution_clock::now();
        auto build_time = duration_cast<microseconds>(t2 - t1).count() / 1e6;

        size_t total_correct = 0;
        auto t3 = high_resolution_clock::now();
        size_t query_count = std::min(size_t(100), s.N);
        for (size_t i = 0; i < query_count; ++i) {
            auto result = index.search(data[i], s.K, 200);
            for (size_t id : result)
                if (id == i) {
                    ++total_correct;
                    break;
                }
        }
        auto t4 = high_resolution_clock::now();
        auto query_time =
            duration_cast<microseconds>(t4 - t3).count() / double(query_count);

        double recall = static_cast<double>(total_correct) / query_count;

        cout << s.N << " | " << s.DIM << " | " << s.K << " | " << build_time
             << " | " << query_time << " | " << recall << "\n";

        csv << s.N << "," << s.DIM << "," << s.K << ","
            << build_time << "," << query_time << "," << recall << "\n";
    }
    cout << "\n";
}

