#pragma once

#include <cstdint>
#include <random>
#include <vector>

using std::vector;

class Node {
public:
    uint64_t id;
    int level;
    vector<float> embedding;
    vector<vector<uint64_t>> neighbors;

    Node(uint64_t id, int level, const vector<float>& embedding);
};

class HnswIndex {
public:
    virtual ~HnswIndex() = default;
    virtual void create(const vector<vector<float>>& data) = 0;
    virtual void add(const vector<float>& embedding) = 0;
    virtual vector<uint64_t> search(const vector<float>& query, size_t k) = 0;
    virtual size_t size() const = 0;
};

class HnswCPU : public HnswIndex {
public:
    HnswCPU(size_t M = 16, size_t efConstruction = 200);

    void create(const vector<vector<float>>& data) override;
    void add(const vector<float>& embedding) override;
    vector<uint64_t> search(const vector<float>& query, size_t k) override;
    size_t size() const override;

private:
    size_t M;
    size_t M0;
    size_t efConstruction;
    float levelMultiplier;

    vector<Node> nodes;

    int entryPoint;
    int maxLevel;
    uint64_t currentId;

    std::mt19937 gen;
    std::uniform_real_distribution<float> uniform_dist;
    

    int sampleLevel();
    float l2Distance(const vector<float>& a, const vector<float>& b);

    vector<uint64_t> searchLayer(
        const vector<float>& query,
        uint64_t entry,
        size_t ef,
        int level
    );

    vector<uint64_t> selectNeighbors(
        const vector<float>& query,
        const vector<uint64_t>& candidates,
        size_t M
    );
};