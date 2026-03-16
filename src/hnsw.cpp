#include "hnsw.h"

#include "dist/cosine.h"
#include "dist/inner_product.h"
#include "dist/l2.h"

#include <algorithm>
#include <cmath>
#include <queue>

using std::pair;
using std::priority_queue;
using std::vector;

using DistanceIndexPair = pair<double, int>;

HnswCPU::HnswCPU(
    int M_,
    int efConstruction_,
    uint32_t seed,
    DistanceType distType_,
    bool forceScalar_
) :
    M(M_),
    M0(2 * M_),
    efConstruction(efConstruction_),
    levelMultiplier(1.0 / std::log((double)M_)),
    entryPoint(-1),
    maxLevel(0),
    currentId(0),
    gen(seed),
    uniform_dist(0.0f, 1.0f),
    distType(distType_),
    dim(0),
    forceScalar(forceScalar_),
    visitTag(1) {}

inline double HnswCPU::distance(const float *a, const float *b) const {

    if (forceScalar)
        return l2_scalar(a, b, dim);

    switch (distType) {

    case DistanceType::L2:
        return l2_scalar(a, b, dim);

    case DistanceType::INNER_PRODUCT:
        return ip_scalar(a, b, dim);

    case DistanceType::COSINE:
        return cosine_scalar(a, b, dim);
    }

    return l2_scalar(a, b, dim);
}

int HnswCPU::sampleLevel() {

    double u = 1.0 - (double)uniform_dist(gen);
    return (int)(-std::log(u) * levelMultiplier);
}

int HnswCPU::getNeighborCount(int level) const { return level == 0 ? M0 : M; }

int *HnswCPU::getNeighborPtr(int id, int level) {

    Node &n = nodes[id];

    int offset = n.neighbor_offset;

    if (level == 0)
        return &neighbors[offset];

    return &neighbors[offset + M0 + (level - 1) * M];
}

const int *HnswCPU::getNeighborPtr(int id, int level) const {

    const Node &n = nodes[id];

    int offset = n.neighbor_offset;

    if (level == 0)
        return &neighbors[offset];

    return &neighbors[offset + M0 + (level - 1) * M];
}

vector<int>
HnswCPU::searchLayer(const float *query, int entry, int ef, int level) {

    if (visited.size() < nodes.size())
        visited.resize(nodes.size());

    visitTag++;

    priority_queue<DistanceIndexPair, vector<DistanceIndexPair>, std::greater<>>
        candidates;

    priority_queue<DistanceIndexPair> topResults;

    double dist = distance(query, getEmbedding(entry));

    candidates.push({dist, entry});
    topResults.push({dist, entry});

    visited[entry] = visitTag;

    while (!candidates.empty()) {

        auto [currDist, curr] = candidates.top();
        candidates.pop();

        if (currDist > topResults.top().first)
            break;

        const int *nbr = getNeighborPtr(curr, level);
        int count = getNeighborCount(level);

        for (int i = 0; i < count; i++) {

            int nei = nbr[i];

            if (nei < 0)
                continue;

            if (visited[nei] == visitTag)
                continue;

            visited[nei] = visitTag;

            double d = distance(query, getEmbedding(nei));

            if ((int)topResults.size() < ef || d < topResults.top().first) {

                candidates.push({d, nei});
                topResults.push({d, nei});

                if ((int)topResults.size() > ef)
                    topResults.pop();
            }
        }
    }

    vector<int> result;

    while (!topResults.empty()) {
        result.push_back(topResults.top().second);
        topResults.pop();
    }

    return result;
}

vector<int> HnswCPU::selectNeighbors(
    const float *query,
    const vector<int> &candidates,
    int max_neighbors
) {

    vector<pair<double, int>> distList;

    for (int id : candidates)
        distList.emplace_back(distance(query, getEmbedding(id)), id);

    std::sort(distList.begin(), distList.end());

    vector<int> result;

    for (size_t i = 0; i < std::min(distList.size(), (size_t)max_neighbors);
         i++)
        result.push_back(distList[i].second);

    return result;
}

vector<int> HnswCPU::selectNeighborsWithHeuristic(
    const float *query,
    const vector<int> &candidates,
    int max_neighbors,
    int layer
) {

    vector<pair<double, int>> distList;

    for (int id : candidates)
        distList.emplace_back(distance(query, getEmbedding(id)), id);

    std::sort(distList.begin(), distList.end());

    vector<int> result;

    for (auto &[d, id] : distList) {

        bool good = true;

        for (int r : result) {

            double dd = distance(getEmbedding(id), getEmbedding(r));

            if (dd < d) {
                good = false;
                break;
            }
        }

        if (good)
            result.push_back(id);

        if ((int)result.size() >= max_neighbors)
            break;
    }

    return result;
}

void HnswCPU::create(const vector<vector<float>> &data) {

    if (data.empty())
        return;

    dim = data[0].size();

    embeddings.reserve(data.size() * dim);
    nodes.reserve(data.size());

    for (const auto &v : data)
        add(v);
}

void HnswCPU::add(const vector<float> &embedding) {

    int level = sampleLevel();
    int id = currentId++;

    embeddings.insert(embeddings.end(), embedding.begin(), embedding.end());

    int offset = neighbors.size();

    neighbors.resize(offset + M0 + level * M, -1);

    nodes.emplace_back(level, offset);

    if (entryPoint == -1) {

        entryPoint = id;
        maxLevel = level;
        return;
    }

    int curr = entryPoint;

    for (int l = maxLevel; l > level; l--)
        curr = searchLayer(embedding.data(), curr, 1, l)[0];

    for (int l = std::min(level, maxLevel); l >= 0; l--) {

        int maxN = getNeighborCount(l);

        auto candidates =
            searchLayer(embedding.data(), curr, efConstruction, l);

        auto selected =
            selectNeighborsWithHeuristic(embedding.data(), candidates, maxN, l);

        int *nbr = getNeighborPtr(id, l);

        for (size_t i = 0; i < selected.size(); i++)
            nbr[i] = selected[i];

        for (int other : selected) {

            int *onbr = getNeighborPtr(other, l);

            int count = getNeighborCount(l);

            for (int i = 0; i < count; i++) {

                if (onbr[i] < 0) {
                    onbr[i] = id;
                    break;
                }
            }
        }
    }

    if (level > maxLevel) {

        entryPoint = id;
        maxLevel = level;
    }
}

vector<int> HnswCPU::search(const vector<float> &query, int k, int efSearch) {

    if (nodes.empty())
        return {};

    int curr = entryPoint;

    for (int l = maxLevel; l > 0; l--)
        curr = searchLayer(query.data(), curr, 1, l)[0];

    auto candidates = searchLayer(query.data(), curr, efSearch, 0);

    return selectNeighbors(query.data(), candidates, k);
}

int HnswCPU::size() const { return nodes.size(); }
void HnswCPU::printInfo() { printSimdInfo(); }