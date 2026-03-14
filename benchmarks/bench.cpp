#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>
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

struct DistConfig {
    DistanceType type;
    const char *name;
};

struct SimdMode {
    bool forceScalar;
    const char *label;
};

vector<int> bruteForceKNN(
    const vector<vector<float>> &data,
    const vector<float> &query,
    size_t k,
    vector<std::pair<float, int>> &buffer
) {

    size_t N = data.size();
    size_t dim = query.size();

    k = std::min(k, N);

    for (size_t i = 0; i < N; ++i) {

        float dist = 0.0f;
        const auto &v = data[i];

        for (size_t j = 0; j < dim; ++j) {
            float diff = v[j] - query[j];
            dist += diff * diff;
        }

        buffer[i] = {dist, static_cast<int>(i)};
    }

    std::nth_element(buffer.begin(), buffer.begin() + k, buffer.end());

    vector<int> result(k);

    for (size_t i = 0; i < k; ++i)
        result[i] = buffer[i].second;

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
        {true, "scalar"},
        {false, "simd"},
    };

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0)
        numThreads = 4;

    const int W_DIST = 14;
    const int W_MODE = 10;
    const int W_DATA = 10;
    const int W_DIM = 6;
    const int W_K = 4;
    const int W_BUILD = 10;
    const int W_QUERY = 11;
    const int W_BRUTE = 11;
    const int W_SPEED = 9;
    const int W_RECALL = 8;

    cout << "\nC++ Benchmarks\n\n";
    cout << "Hardware threads available: " << numThreads << "\n\n";

    cout << std::left
         << std::setw(W_DIST) << "Distance"
         << std::setw(W_MODE) << "Mode"
         << std::setw(W_DATA) << "Dataset"
         << std::setw(W_DIM) << "Dim"
         << std::setw(W_K) << "K"
         << std::setw(W_BUILD) << "Build(s)"
         << std::setw(W_QUERY) << "Query(us)"
         << std::setw(W_BRUTE) << "Brute(us)"
         << std::setw(W_SPEED) << "Speedup"
         << std::setw(W_RECALL) << "Recall"
         << "\n";

    cout << std::string(
                W_DIST + W_MODE + W_DATA + W_DIM + W_K +
                W_BUILD + W_QUERY + W_BRUTE + W_SPEED + W_RECALL,
                '-'
            )
         << "\n";

    std::ofstream csv("benchmarks/cpp_results.csv");
    csv << "distance,mode,dataset,dim,k,build_s,query_us,brute_us,speedup,recall\n";

    std::mt19937 gen(42);

    for (const auto &dcfg : dists) {
        for (const auto &mode : modes) {
            for (const auto &s : scenarios) {

                std::uniform_real_distribution<float> dist(0.0f, 1.0f);

                vector<vector<float>> data(s.N, vector<float>(s.DIM));

                for (size_t i = 0; i < s.N; ++i)
                    for (size_t j = 0; j < s.DIM; ++j)
                        data[i][j] = dist(gen);

                HnswCPU index(16, 200, 42, dcfg.type, mode.forceScalar);

                auto t1 = high_resolution_clock::now();

                if (mode.useMultithread)
                    index.addParallel(data, numThreads);
                else
                    index.create(data);

                auto t2 = high_resolution_clock::now();

                double build_time =
                    duration_cast<microseconds>(t2 - t1).count() / 1e6;

                size_t query_count = std::min((size_t)100, s.N);

                vector<std::pair<float, int>> bruteBuffer(s.N);

                // -------- TRUE KNN (for recall) --------

                vector<vector<int>> groundTruth(query_count);

                for (size_t i = 0; i < query_count; ++i)
                    groundTruth[i] =
                        bruteForceKNN(data, data[i], s.K, bruteBuffer);

                // -------- HNSW QUERY TIMING --------

                size_t total_hits = 0;

                auto t3 = high_resolution_clock::now();

                for (size_t i = 0; i < query_count; ++i) {

                    auto approx = index.search(data[i], (int)s.K, 200);

                    std::unordered_set<int> truth(
                        groundTruth[i].begin(),
                        groundTruth[i].end());

                    for (int id : approx)
                        if (truth.count(id))
                            total_hits++;
                }

                auto t4 = high_resolution_clock::now();

                double query_time =
                    duration_cast<microseconds>(t4 - t3).count()
                    / double(query_count);

                double recall =
                    double(total_hits) / (query_count * s.K);

                // -------- BRUTE FORCE TIMING --------

                auto t5 = high_resolution_clock::now();

                for (size_t i = 0; i < query_count; ++i)
                    bruteForceKNN(data, data[i], s.K, bruteBuffer);

                auto t6 = high_resolution_clock::now();

                double brute_query_time =
                    duration_cast<microseconds>(t6 - t5).count()
                    / double(query_count);

                double speedup = brute_query_time / query_time;

                long build_i = std::llround(build_time);
                long query_i = std::llround(query_time);
                long brute_i = std::llround(brute_query_time);
                long speed_i = std::llround(speedup);
                long recall_i = std::llround(recall * 100);

                cout << std::left
                     << std::setw(W_DIST) << dcfg.name
                     << std::setw(W_MODE) << mode.label
                     << std::setw(W_DATA) << s.N
                     << std::setw(W_DIM) << s.DIM
                     << std::setw(W_K) << s.K
                     << std::setw(W_BUILD) << build_i
                     << std::setw(W_QUERY) << query_i
                     << std::setw(W_BRUTE) << brute_i
                     << std::setw(W_SPEED)
                     << (std::to_string(speed_i) + "x")
                     << std::setw(W_RECALL)
                     << recall_i
                     << "\n";

                csv << dcfg.name << ","
                    << mode.label << ","
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
    }

    cout << "\nResults saved to benchmarks/cpp_results.csv\n";
}