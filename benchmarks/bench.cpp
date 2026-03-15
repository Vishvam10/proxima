#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "hnsw.h"

using std::cout;
using std::vector;

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

struct Scenario {
    size_t N;
    size_t DIM;
    size_t K;
};

struct Mode {
    bool forceScalar;
    const char* name;
};

vector<int> bruteForceKNN(
    const vector<vector<float>>& data,
    const vector<float>& query,
    size_t k
) {

    vector<std::pair<float,int>> dists(data.size());

    for (size_t i = 0; i < data.size(); ++i) {

        float d = 0;

        for (size_t j = 0; j < query.size(); ++j) {

            float diff = data[i][j] - query[j];
            d += diff * diff;
        }

        dists[i] = {d, static_cast<int>(i)};
    }

    std::partial_sort(
        dists.begin(),
        dists.begin() + static_cast<ptrdiff_t>(k),
        dists.end()
    );

    vector<int> result(k);

    for (size_t i = 0; i < k; ++i)
        result[i] = dists[i].second;

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

    vector<Mode> modes = {
        {true,  "scalar"},
        {false, "simd"}
    };

    cout << "\nC++ Benchmarks\n\n";

    cout << std::fixed << std::setprecision(4);

    cout << std::left
         << std::setw(10) << "Mode"
         << std::setw(10) << "Dataset"
         << std::setw(6)  << "Dim"
         << std::setw(4)  << "K"
         << std::setw(12) << "Build(s)"
         << std::setw(12) << "Query(us)"
         << std::setw(12) << "Brute(us)"
         << std::setw(10) << "Speedup"
         << std::setw(8)  << "Recall"
         << "\n";

    cout << std::string(84, '-') << "\n";

    std::ofstream csv("benchmarks/cpp_results.csv");

    csv << "mode,dataset,dim,k,build_s,query_us,brute_query_us,speedup,recall\n";

    std::mt19937 gen(42);

    for (auto mode : modes) {

        for (auto s : scenarios) {

            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            vector<vector<float>> data(s.N, vector<float>(s.DIM));

            for (size_t i = 0; i < s.N; ++i)
                for (size_t j = 0; j < s.DIM; ++j)
                    data[i][j] = dist(gen);

            HnswCPU index(
                16,
                200,
                42,
                DistanceType::L2
            );

            auto t1 = high_resolution_clock::now();

            index.create(data);

            auto t2 = high_resolution_clock::now();

            double build_time =
                duration_cast<microseconds>(t2 - t1).count() / 1e6;

            size_t total_correct = 0;

            size_t query_count =
                std::min(size_t(100), s.N);

            auto t3 = high_resolution_clock::now();

            for (size_t i = 0; i < query_count; ++i) {

                auto result =
                    index.search(data[i], s.K, 200);

                for (size_t id : result)
                    if (id == i) {
                        ++total_correct;
                        break;
                    }
            }

            auto t4 = high_resolution_clock::now();

            double query_time =
                duration_cast<microseconds>(t4 - t3).count()
                / double(query_count);

            double recall =
                static_cast<double>(total_correct)
                / query_count;

            auto t5 = high_resolution_clock::now();

            for (size_t i = 0; i < query_count; ++i)
                bruteForceKNN(data, data[i], s.K);

            auto t6 = high_resolution_clock::now();

            double brute_query_time =
                duration_cast<microseconds>(t6 - t5).count()
                / double(query_count);

            double speedup =
                brute_query_time / query_time;

            cout << std::left
                 << std::setw(10) << mode.name
                 << std::setw(10) << s.N
                 << std::setw(6)  << s.DIM
                 << std::setw(4)  << s.K
                 << std::setw(12) << build_time
                 << std::setw(12) << query_time
                 << std::setw(12) << brute_query_time
                 << std::setw(10) << speedup
                 << std::setw(8)  << recall
                 << "\n";

            csv << mode.name << ","
                << s.N << ","
                << s.DIM << ","
                << s.K << ","
                << build_time << ","
                << query_time << ","
                << brute_query_time << ","
                << speedup << ","
                << recall << "\n";
        }
    }

    cout << "\nResults saved to benchmarks/cpp_results.csv\n\n";
}