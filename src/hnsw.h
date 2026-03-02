#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "dist/dispatch.h"

using std::vector;

class Node {
  public:
    size_t id;
    size_t level;
    vector<float> embedding;
    vector<vector<size_t> > neighbors;

    Node(size_t id, size_t level, const vector<float> &embedding);
};

class HnswIndex {
  public:
    virtual ~HnswIndex() = default;

    virtual size_t size() const = 0;
    virtual void create(const vector<vector<float> > &data) = 0;

    // virtual void deleteNode(const size_t nodeId) = 0;

    // virtual void
    // update(const size_t nodeId, const vector<float> &embedding) = 0;

    virtual void add(const vector<float> &embedding) = 0;
    virtual vector<size_t>
    search(const vector<float> &query, size_t k, size_t efSearch = 50) = 0;
};

class HnswCPU : public HnswIndex {
  private:
    size_t M;
    size_t M0;
    size_t efConstruction;
    double levelMultiplier;

    vector<Node> nodes;
    size_t entryPoint;
    size_t maxLevel;
    size_t currentId;

    std::mt19937 gen;
    std::uniform_real_distribution<float> uniform_dist;

    DistanceType distType;
    DistFunc distFunc;

    float scoreForHeap(float raw) const;

    size_t sampleLevel();
    double l2Distance(const vector<float> &a, const vector<float> &b) const;

    vector<size_t> searchLayer(
        const vector<float> &query,
        size_t entry,
        size_t ef,
        size_t level
    );

    vector<size_t> selectNeighbors(
        const vector<float> &query,
        const vector<size_t> &candidates,
        const size_t max_neighbours
    );

    vector<size_t> selectNeighborsWithHeuristic(
        const vector<float> &query,
        const vector<size_t> &candidates,
        const size_t max_neighbours,
        const size_t layer,
        const bool extendCandidates,
        const bool keepPrunedConnections
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