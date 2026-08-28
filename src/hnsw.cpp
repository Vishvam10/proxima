#include "proxima/hnsw.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <stdexcept>
#include <utility>

namespace proxima {

using DistanceIndexPair = std::pair<double, int>;

HnswCPU::HnswCPU(
    int M,
    int ef_construction,
    std::uint32_t seed,
    DistanceType dist_type,
    bool force_scalar
) :
    M_(M),
    M0_(2 * M),
    ef_construction_(ef_construction),
    level_multiplier_(1.0 / std::log(static_cast<double>(M))),
    entry_point_(-1),
    max_level_(0),
    current_id_(0),
    gen_(seed),
    uniform_dist_(0.0f, 1.0f),
    dist_type_(dist_type),
    dim_(0),
    force_scalar_(force_scalar),
    visit_tag_(1) {
    if (M <= 0) {
        throw std::invalid_argument("M must be positive");
    }

    if (ef_construction <= 0) {
        throw std::invalid_argument("ef_construction must be positive");
    }
}

double HnswCPU::distance(const float *a, const float *b) const {
    return computeDistance(dist_type_, a, b, dim_, force_scalar_);
}

int HnswCPU::sampleLevel() {
    const double u = 1.0 - static_cast<double>(uniform_dist_(gen_));

    return static_cast<int>(-std::log(u) * level_multiplier_);
}

int HnswCPU::getNeighborCount(int level) const { return level == 0 ? M0_ : M_; }

int *HnswCPU::getNeighborPtr(int id, int level) {
    Node &node = nodes_[id];

    const int offset = node.neighbor_offset;

    if (level == 0) {
        return &neighbors_[offset];
    }

    return &neighbors_[offset + M0_ + (level - 1) * M_];
}

const int *HnswCPU::getNeighborPtr(int id, int level) const {
    const Node &node = nodes_[id];

    const int offset = node.neighbor_offset;

    if (level == 0) {
        return &neighbors_[offset];
    }

    return &neighbors_[offset + M0_ + (level - 1) * M_];
}

std::vector<int>
HnswCPU::searchLayer(const float *query, int entry, int ef, int level) {
    if (visited_.size() < nodes_.size()) {
        visited_.resize(nodes_.size());
    }

    ++visit_tag_;

    if (visit_tag_ == 0) {
        std::fill(visited_.begin(), visited_.end(), 0);
        visit_tag_ = 1;
    }

    std::priority_queue<
        DistanceIndexPair,
        std::vector<DistanceIndexPair>,
        std::greater<>>
        candidates;

    std::priority_queue<DistanceIndexPair> top_results;

    const double dist = distance(query, getEmbedding(entry));

    candidates.push({dist, entry});
    top_results.push({dist, entry});

    visited_[entry] = visit_tag_;

    while (!candidates.empty()) {
        const auto [curr_dist, curr] = candidates.top();

        candidates.pop();

        if (curr_dist > top_results.top().first) {
            break;
        }

        const int *nbr = getNeighborPtr(curr, level);

        const int count = getNeighborCount(level);

        for (int i = 0; i < count; ++i) {
            const int nei = nbr[i];

            if (nei < 0) {
                continue;
            }

            if (visited_[nei] == visit_tag_) {
                continue;
            }

            visited_[nei] = visit_tag_;

            const double d = distance(query, getEmbedding(nei));

            if (static_cast<int>(top_results.size()) < ef ||
                d < top_results.top().first) {
                candidates.push({d, nei});
                top_results.push({d, nei});

                if (static_cast<int>(top_results.size()) > ef) {
                    top_results.pop();
                }
            }
        }
    }

    std::vector<int> result;
    result.reserve(top_results.size());

    while (!top_results.empty()) {
        result.push_back(top_results.top().second);

        top_results.pop();
    }

    return result;
}

std::vector<int> HnswCPU::selectNeighbors(
    const float *query,
    const std::vector<int> &candidates,
    int max_neighbors
) {
    std::vector<std::pair<double, int>> dist_list;

    dist_list.reserve(candidates.size());

    for (int id : candidates) {
        dist_list.emplace_back(distance(query, getEmbedding(id)), id);
    }

    std::sort(dist_list.begin(), dist_list.end());

    std::vector<int> result;

    const auto count =
        std::min(dist_list.size(), static_cast<std::size_t>(max_neighbors));

    result.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        result.push_back(dist_list[i].second);
    }

    return result;
}

std::vector<int> HnswCPU::selectNeighborsWithHeuristic(
    const float *query,
    const std::vector<int> &candidates,
    int max_neighbors,
    [[maybe_unused]] int layer
) {
    std::vector<std::pair<double, int>> dist_list;

    dist_list.reserve(candidates.size());

    for (int id : candidates) {
        dist_list.emplace_back(distance(query, getEmbedding(id)), id);
    }

    std::sort(dist_list.begin(), dist_list.end());

    std::vector<int> result;
    result.reserve(max_neighbors);

    for (const auto &[d, id] : dist_list) {
        bool good = true;

        for (int r : result) {
            const double dd = distance(getEmbedding(id), getEmbedding(r));

            if (dd < d) {
                good = false;
                break;
            }
        }

        if (good) {
            result.push_back(id);
        }

        if (static_cast<int>(result.size()) >= max_neighbors) {
            break;
        }
    }

    return result;
}

void HnswCPU::validateEmbedding(const std::vector<float> &embedding) const {
    if (embedding.empty()) {
        throw std::invalid_argument("embedding must not be empty");
    }

    if (dim_ != 0 && embedding.size() != dim_) {
        throw std::invalid_argument("embedding dimension does not match index");
    }
}

void HnswCPU::reset() {
    nodes_.clear();
    neighbors_.clear();
    embeddings_.clear();
    visited_.clear();

    entry_point_ = -1;
    max_level_ = 0;
    current_id_ = 0;
    dim_ = 0;
    visit_tag_ = 1;
}

void HnswCPU::create(const std::vector<std::vector<float>> &data) {
    reset();

    if (data.empty()) {
        return;
    }

    dim_ = data[0].size();

    if (dim_ == 0) {
        throw std::invalid_argument("embedding dimension must be positive");
    }

    embeddings_.reserve(data.size() * dim_);
    nodes_.reserve(data.size());

    for (const auto &embedding : data) {
        add(embedding);
    }
}

void HnswCPU::add(const std::vector<float> &embedding) {
    validateEmbedding(embedding);

    if (dim_ == 0) {
        dim_ = embedding.size();
    }

    const int level = sampleLevel();
    const int id = current_id_++;

    embeddings_.insert(embeddings_.end(), embedding.begin(), embedding.end());

    const int offset = static_cast<int>(neighbors_.size());

    neighbors_.resize(neighbors_.size() + M0_ + level * M_, -1);

    nodes_.emplace_back(level, offset);

    if (entry_point_ == -1) {
        entry_point_ = id;
        max_level_ = level;
        return;
    }

    int curr = entry_point_;

    for (int l = max_level_; l > level; --l) {
        curr = searchLayer(embedding.data(), curr, 1, l)[0];
    }

    for (int l = std::min(level, max_level_); l >= 0; --l) {
        const int max_neighbors = getNeighborCount(l);

        const auto candidates =
            searchLayer(embedding.data(), curr, ef_construction_, l);

        const auto selected = selectNeighborsWithHeuristic(
            embedding.data(), candidates, max_neighbors, l
        );

        int *nbr = getNeighborPtr(id, l);

        for (std::size_t i = 0; i < selected.size(); ++i) {
            nbr[i] = selected[i];
        }

        for (int other : selected) {
            int *other_neighbors = getNeighborPtr(other, l);

            const int count = getNeighborCount(l);

            for (int i = 0; i < count; ++i) {
                if (other_neighbors[i] < 0) {
                    other_neighbors[i] = id;
                    break;
                }
            }
        }
    }

    if (level > max_level_) {
        entry_point_ = id;
        max_level_ = level;
    }
}

std::vector<int>
HnswCPU::search(const std::vector<float> &query, int k, int ef_search) {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    if (ef_search <= 0) {
        throw std::invalid_argument("ef_search must be positive");
    }

    if (nodes_.empty()) {
        return {};
    }

    if (query.size() != dim_) {
        throw std::invalid_argument("query dimension does not match index");
    }

    int curr = entry_point_;

    for (int l = max_level_; l > 0; --l) {
        curr = searchLayer(query.data(), curr, 1, l)[0];
    }

    const auto candidates = searchLayer(query.data(), curr, ef_search, 0);

    return selectNeighbors(query.data(), candidates, k);
}

int HnswCPU::size() const { return static_cast<int>(nodes_.size()); }

void HnswCPU::printInfo() { printSimdInfo(); }

} // namespace proxima