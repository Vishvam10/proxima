#include "hnsw.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace std::chrono;

struct Scenario {
    size_t N, DIM, K;
};
struct Mode {
    bool forceScalar;
    const char *name;
};

// Brute-force KNN
vector<int> bruteForceKNN(
    const vector<vector<float>> &data,
    const vector<float> &query,
    size_t k
) {
    vector<pair<float, int>> dists(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        float d = 0;
        for (size_t j = 0; j < query.size(); ++j)
            d += (data[i][j] - query[j]) * (data[i][j] - query[j]);
        dists[i] = {d, (int)i};
    }
    partial_sort(dists.begin(), dists.begin() + k, dists.end());
    vector<int> result(k);
    for (size_t i = 0; i < k; ++i)
        result[i] = dists[i].second;
    return result;
}

// Table helpers
void printTableHeader() {
    cout << "+------------+--------+--------+------+--------------+------------"
            "--+--------------+------------+----------+\n";
    cout << "| Mode       | N      | Dim    | K    | Build(us)    | Query(us)  "
            "  | Brute(us)    | Speedup    | Recall   |\n";
    cout << "+------------+--------+--------+------+--------------+------------"
            "--+--------------+------------+----------+\n";
}

void printTableRow(
    const string &mode,
    int N,
    int DIM,
    int K,
    double build_s,
    double query_us,
    double brute_us,
    double speedup,
    double recall
) {
    cout << "| " << left << setw(10) << mode << " | " << setw(6) << N << " | "
         << setw(6) << DIM << " | " << setw(4) << K << " | " << setw(12)
         << fixed << setprecision(2) << build_s << " | " << setw(12)
         << query_us << " | " << setw(12) << brute_us << " | " << setw(10)
         << speedup << " | " << setw(8) << recall << " |\n";
}

void printTableFooter() {
    cout << "+------------+--------+--------+------+--------------+------------"
            "--+--------------+------------+----------+\n";
}

int main() {

    cout << "\n\nC++ Benchmarks\n\n";

    vector<Scenario> scenarios = {
        {1000, 128, 10},
        {5000, 128, 10},
        {10000, 128, 10},
        {5000, 64, 10},
        {5000, 256, 10},
        {5000, 128, 5},
        {5000, 128, 50}
    };
    vector<Mode> modes = {{true, "cpp_scalar"}, {false, "cpp_simd"}};

    mt19937 gen(42);

    // CSV output
    ofstream csv("benchmarks/results/cpp_results.csv");
    csv << "impl,N,DIM,K,build_s,query_us,brute_us,speedup,recall\n";

    printTableHeader();

    for (auto mode : modes) {
        for (auto s : scenarios) {
            // Generate data
            uniform_real_distribution<float> dist(0.0, 1.0);
            vector<vector<float>> data(s.N, vector<float>(s.DIM));
            for (size_t i = 0; i < s.N; i++)
                for (size_t j = 0; j < s.DIM; j++)
                    data[i][j] = dist(gen);

            // Build HNSW
            HnswCPU index(16, 200, 42, DistanceType::L2, mode.forceScalar);
            auto t1 = high_resolution_clock::now();
            index.create(data);
            auto t2 = high_resolution_clock::now();
            double build_s = duration_cast<microseconds>(t2 - t1).count() / 1e6;

            // HNSW query
            size_t qcount = min((size_t)100, s.N);
            size_t correct = 0;
            auto t3 = high_resolution_clock::now();
            for (size_t i = 0; i < qcount; i++) {
                auto r = index.search(data[i], s.K, 200);
                for (auto id : r)
                    if (id == (int)i) {
                        correct++;
                        break;
                    }
            }
            auto t4 = high_resolution_clock::now();
            double query_us =
                duration_cast<microseconds>(t4 - t3).count() / (double)qcount;
            double recall = (double)correct / qcount;

            // Brute-force query
            auto t5 = high_resolution_clock::now();
            for (size_t i = 0; i < qcount; i++)
                bruteForceKNN(data, data[i], s.K);
            auto t6 = high_resolution_clock::now();
            double brute_us =
                duration_cast<microseconds>(t6 - t5).count() / (double)qcount;

            double speedup = brute_us / query_us;

            printTableRow(
                mode.name,
                s.N,
                s.DIM,
                s.K,
                build_s,
                query_us,
                brute_us,
                speedup,
                recall
            );
            csv << mode.name << "," << s.N << "," << s.DIM << "," << s.K << ","
                << build_s << "," << query_us << "," << brute_us << ","
                << speedup << "," << recall << "\n";
        }
    }

    printTableFooter();
    cout << "\nSaved results/cpp_results.csv\n";
    return 0;
}