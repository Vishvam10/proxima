#include <iostream>
#include <random>
#include <iomanip>

#include "hnsw.h"

using std::cout, std::vector;

int main() {
    const int N = 1000;
    const int DIM = 32;

    // Generate random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    vector<vector<float>> data(N, vector<float>(DIM));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < DIM; ++j)
            data[i][j] = dist(gen);

    HnswCPU index(16, 200);
    index.create(data);

    vector<float> query = data[123];

    size_t topK = 5, efSearch = 10;
    auto result = index.search(query, topK, efSearch);

    cout << "\n";
    cout << std::fixed << std::setprecision(4);
    cout << "Query vector (ID 123) : \n\n";
    for (float v : query) cout << v << " ";
    cout << "\n\n";

    cout << "Top 5 neighbors (ID, distance, vector) :\n\n";

    for (auto id : result) {
        double dist = 0.0;
        for (size_t j = 0; j < DIM; ++j) {
            double diff = query[j] - data[id][j];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);  // Euclidean distance

        cout << "ID " << id << ", distance : " << dist << "\nVector : ";
        for (float v : data[id]) cout << v << " ";
        cout << "\n\n";
    }

    return 0;
}