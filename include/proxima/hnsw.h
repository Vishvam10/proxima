#pragma once

#include "proxima/dist/dispatch.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace proxima {

struct Node {
    int level;
    int neighbor_offset;

    Node(int level = 0, int offset = 0) :
        level(level),
        neighbor_offset(offset) {}
};

class HnswIndex {
  public:
    virtual ~HnswIndex() = default;

    virtual void create(const std::vector<std::vector<float>> &data) = 0;

    virtual void add(const std::vector<float> &embedding) = 0;

    virtual std::vector<int>
    search(const std::vector<float> &query, int k, int ef_search = 50) = 0;

    virtual int size() const = 0;
};

class HnswCPU final : public HnswIndex {
  private:
    int M_;
    int M0_;
    int ef_construction_;

    double level_multiplier_;

    std::vector<Node> nodes_;
    std::vector<int> neighbors_;
    std::vector<float> embeddings_;

    int entry_point_;
    int max_level_;
    int current_id_;

    std::mt19937 gen_;
    std::uniform_real_distribution<float> uniform_dist_;

    DistanceType dist_type_;
    std::size_t dim_;

    bool force_scalar_;

    std::vector<std::uint32_t> visited_;
    std::uint32_t visit_tag_;

    double distance(const float *a, const float *b) const;

    int sampleLevel();

    const float *getEmbedding(int id) const {
        return &embeddings_[static_cast<std::size_t>(id) * dim_];
    }

    int *getNeighborPtr(int id, int level);
    const int *getNeighborPtr(int id, int level) const;

    int getNeighborCount(int level) const;

    std::vector<int>
    searchLayer(const float *query, int entry, int ef, int level);

    std::vector<int> selectNeighbors(
        const float *query,
        const std::vector<int> &candidates,
        int max_neighbors
    );

    std::vector<int> selectNeighborsWithHeuristic(
        const float *query,
        const std::vector<int> &candidates,
        int max_neighbors,
        int layer
    );

    void reset();

    void validateEmbedding(const std::vector<float> &embedding) const;

  public:
    HnswCPU(
        int M = 16,
        int ef_construction = 200,
        std::uint32_t seed = 42,
        DistanceType dist_type = DistanceType::L2,
        bool force_scalar = false
    );

    void create(const std::vector<std::vector<float>> &data) override;

    void add(const std::vector<float> &embedding) override;

    std::vector<int>
    search(const std::vector<float> &query, int k, int ef_search = 50) override;

    int size() const override;

    std::size_t dimension() const { return dim_; }

    void printInfo();

    int M() const { return M_; }

    int efConstruction() const { return ef_construction_; }

    DistanceType distanceType() const { return dist_type_; }
};

} // namespace proxima