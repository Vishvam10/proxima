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
    bool forceScalar
) :
    M(M_),
    M0(2 * M_),
    efConstruction(efConstruction_),
    levelMultiplier(1.0 / std::log(double(M_))),
    entryPoint(-1),
    maxLevel(0),
    currentId(0),
    gen(seed),
    uniform_dist(0.0f, 1.0f),
    distType(distType_),
    forceScalar(false),
    dim(0),
    visitTag(1) {}

inline double HnswCPU::distance(const float *a, const float *b) const {
    if(forceScalar) return l2_scalar(a, b, dim); 

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

    double u = 1.0 - static_cast<double>(uniform_dist(gen));

    return static_cast<int>(-std::log(u) * levelMultiplier);
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

    size_t lvl = static_cast<size_t>(level);

    while (!candidates.empty()) {

        auto [currDist, currId] = candidates.top();
        candidates.pop();

        if (currDist > topResults.top().first)
            break;

        for (int nei : nodes[currId].neighbors[lvl]) {

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
    result.reserve(topResults.size());

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
    distList.reserve(candidates.size());

    for (int id : candidates) {
        double d = distance(query, getEmbedding(id));
        distList.emplace_back(d, id);
    }

    std::sort(distList.begin(), distList.end());

    vector<int> result;
    result.reserve(max_neighbors);

    for (size_t i = 0; i < std::min(distList.size(), (size_t)max_neighbors);
         ++i)
        result.push_back(distList[i].second);

    return result;
}

vector<int> HnswCPU::selectNeighborsWithHeuristic(
    const float *query,
    const vector<int> &candidates,
    int max_neighbors,
    int layer,
    bool extendCandidates,
    bool keepPrunedConnections
) {

    if (visited.size() < nodes.size())
        visited.resize(nodes.size());

    visitTag++;

    vector<int> result;
    result.reserve(max_neighbors);

    priority_queue<DistanceIndexPair, vector<DistanceIndexPair>, std::greater<>>
        pq;

    priority_queue<DistanceIndexPair, vector<DistanceIndexPair>, std::greater<>>
        discarded;

    size_t layerIdx = static_cast<size_t>(layer);

    for (int cand : candidates) {

        if (visited[cand] != visitTag) {
            visited[cand] = visitTag;
            pq.push({distance(query, getEmbedding(cand)), cand});
        }

        if (extendCandidates) {
            for (int nei : nodes[cand].neighbors[layerIdx]) {
                if (visited[nei] != visitTag) {
                    visited[nei] = visitTag;
                    pq.push({distance(query, getEmbedding(nei)), nei});
                }
            }
        }
    }

    while (!pq.empty() && (int)result.size() < max_neighbors) {

        auto [d, id] = pq.top();
        pq.pop();

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
        else if (keepPrunedConnections)
            discarded.push({d, id});
    }

    if (keepPrunedConnections) {
        while (!discarded.empty() && (int)result.size() < max_neighbors) {
            result.push_back(discarded.top().second);
            discarded.pop();
        }
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

    int nodeLevel = sampleLevel();

    int id = currentId++;

    embeddings.insert(embeddings.end(), embedding.begin(), embedding.end());

    nodes.emplace_back(id, nodeLevel);

    if (entryPoint == -1) {
        entryPoint = 0;
        maxLevel = nodeLevel;
        return;
    }

    int curr = entryPoint;

    for (int level = maxLevel; level > nodeLevel; --level)
        curr = searchLayer(embedding.data(), curr, 1, level)[0];

    for (int level = std::min(nodeLevel, maxLevel); level >= 0; --level) {

        int maxNeighbors = (level == 0) ? M0 : M;
        auto layerNodes =
            searchLayer(embedding.data(), curr, efConstruction, level);

        if (layerNodes.size() > (size_t)efConstruction)
            layerNodes.resize(efConstruction);

        auto selected = selectNeighborsWithHeuristic(
            embedding.data(),
            layerNodes,
            maxNeighbors,
            level,
            false, // disable candidate expansion
            false  // disable pruned connections
        );

        nodes[id].neighbors[level] = selected;

        for (int nid : selected) {

            if (nodes[nid].level < level)
                continue;

            auto &neighList = nodes[nid].neighbors[level];
            neighList.push_back(id);

            if ((int)neighList.size() > maxNeighbors)
                neighList = selectNeighborsWithHeuristic(
                    getEmbedding(nid),
                    neighList,
                    maxNeighbors,
                    level,
                    false,
                    false
                );
        }
    }

    if (nodeLevel > maxLevel) {
        entryPoint = id;
        maxLevel = nodeLevel;
    }
}

vector<int> HnswCPU::search(const vector<float> &query, int k, int efSearch) {

    if (nodes.empty())
        return {};

    int curr = entryPoint;

    for (int level = maxLevel; level > 0; --level)
        curr = searchLayer(query.data(), curr, 1, level)[0];

    auto candidates = searchLayer(query.data(), curr, efSearch, 0);

    return selectNeighbors(query.data(), candidates, k);
}

int HnswCPU::size() const { return (int)nodes.size(); }

void HnswCPU::printInfo() { printSimdInfo(); }