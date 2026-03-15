#pragma once

#include "dist/dispatch.h"
#include <cstdint>
#include <random>
#include <vector>

using std::vector;

struct Node {
    int id;
    int level;
    vector<vector<int>> neighbors;

    Node(int id_, int level_) :
        id(id_),
        level(level_) {
        neighbors.resize(static_cast<size_t>(level) + 1);
    }
};

class HnswIndex {
  public:
    virtual ~HnswIndex() = default;
    virtual void create(const vector<vector<float>> &data) = 0;
    virtual void add(const vector<float> &embedding) = 0;
    virtual vector<int>
    search(const vector<float> &query, int k, int efSearch = 50) = 0;
    virtual int size() const = 0;
};

class HnswCPU : public HnswIndex {
  private:
    int M;
    int M0;
    int efConstruction;
    double levelMultiplier;

    vector<Node> nodes;
    vector<float> embeddings;
    int entryPoint;
    int maxLevel;
    int currentId;

    std::mt19937 gen;
    std::uniform_real_distribution<float> uniform_dist;

    DistanceType distType;
    DistFunc distFunc;

    int sampleLevel();
    vector<int> searchLayer(const float *query, int entry, int ef, int level);
    vector<int> selectNeighbors(
        const float *query,
        const vector<int> &candidates,
        int max_neighbours
    );
    vector<int> selectNeighborsWithHeuristic(
        const float *query,
        const vector<int> &candidates,
        int max_neighbours,
        int layer,
        bool extendCandidates,
        bool keepPrunedConnections
    );

    const float *getEmbedding(int id) const {
        return embeddings.data() + id * dim;
    }
    size_t dim;

  public:
    HnswCPU(
        int M = 16,
        int efConstruction = 200,
        uint32_t seed = 42,
        DistanceType distType = DistanceType::L2,
        bool forceScalar = false
    );

    void create(const vector<vector<float>> &data) override;
    void add(const vector<float> &embedding) override;
    vector<int>
    search(const vector<float> &query, int k, int efSearch = 50) override;
    int size() const override;
    void printInfo();
};