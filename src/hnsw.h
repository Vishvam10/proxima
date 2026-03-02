#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "dist/dispatch.h"

using std::vector;

class Node {
public:
    int id;
    int level;
    vector<float> embedding;
    vector<vector<int>> neighbors;

    Node(int id, int level, const vector<float>& embedding);
};

class HnswIndex {
public:
    virtual ~HnswIndex() = default;
    virtual void create(const vector<vector<float>>& data) = 0;
    virtual void add(const vector<float>& embedding) = 0;
    virtual vector<int> search(const vector<float>& query, int k, int efSearch = 50) = 0;
    virtual int size() const = 0;
};

class HnswCPU : public HnswIndex {
public:
    HnswCPU(int M = 16, int efConstruction = 200, uint32_t seed = 42);

    void create(const vector<vector<float>>& data) override;
    void add(const vector<float>& embedding) override;
    vector<int> search(const vector<float>& query, int k, int efSearch = 50) override;
    int size() const override;

private:
    int M;
    int M0;
    int efConstruction;
    double levelMultiplier;

    vector<Node> nodes;
    int entryPoint;
    int maxLevel;
    int currentId;

    std::mt19937 gen;
    std::uniform_real_distribution<float> uniform_dist;

    int sampleLevel();
    double l2Distance(const vector<float>& a, const vector<float>& b) const;

    vector<int> searchLayer(
        const vector<float>& query,
        int entry,
        int ef,
        int level
    );

    vector<int> selectNeighbors(
        const vector<float>& query,
        const vector<int>& candidates,
        int max_neighbours
    );

    vector<int> selectNeighborsWithHeuristic(
        const vector<float>& query,
        const vector<int>& candidates,
        int max_neighbours,
        int layer,
        bool extendCandidates,
        bool keepPrunedConnections
    );

  public:
    HnswCPU(
        size_t M = 16,
        size_t efConstruction = 200,
        uint32_t seed = 42,
        DistanceType distType = DistanceType::L2
    );

    size_t size() const override;

    void create(const vector<vector<float> > &data) override;
    void add(const vector<float> &embedding) override;

    // void deleteNode(const size_t nodeId) override;

    vector<size_t>
    search(const vector<float> &query, size_t k, size_t efSearch = 50) override;
};