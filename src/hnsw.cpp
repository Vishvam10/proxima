#include "hnsw.h"

#include <algorithm>
#include <cmath>
#include <queue>

using std::pair;
using std::priority_queue;
using std::vector;

using DistanceIndexPair = pair<double, int>;

Node::Node(int id_, int level_, const vector<float> &emb)
    : id(id_), level(level_), embedding(emb) {
    neighbors.resize(level + 1);
}

HnswCPU::HnswCPU(
    int M_,
    int efConstruction_,
    uint32_t seed,
    DistanceType distType,
    bool forceScalar
) :
    M(M_),
    M0(2 * M_),
    efConstruction(efConstruction_),
    levelMultiplier(1.0 / std::log((double)M_)),
    entryPoint(0),
    maxLevel(0),
    currentId(0),
    gen(seed),
    uniform_dist(0.0f, 1.0f),
    distType(distType) {
    distFunc = getDistanceFunction(distType, forceScalar);
}

int HnswCPU::sampleLevel() {
    double u = 1.0 - (double)uniform_dist(gen);
    return static_cast<int>(-std::log(u) * levelMultiplier);
}

vector<int>
HnswCPU::searchLayer(const vector<float> &query, int entry, int ef, int level) {

    priority_queue<DistanceIndexPair, vector<DistanceIndexPair>, std::greater<>>
        candidates;
    priority_queue<DistanceIndexPair> topResults;

    vector<uint8_t> visited(nodes.size(), 0);

    size_t queryLen = query.size();
    double dist =
        distFunc(query.data(), nodes[entry].embedding.data(), queryLen);

    candidates.push({dist, entry});
    topResults.push({dist, entry});
    visited[entry] = 1;

    while (!candidates.empty()) {

        auto [currDist, currId] = candidates.top();
        candidates.pop();

        if (currDist > topResults.top().first)
            break;

        for (int nei : nodes[currId].neighbors[level]) {

            if (visited[nei])
                continue;

            visited[nei] = 1;

            double d =
                distFunc(query.data(), nodes[nei].embedding.data(), queryLen);

            if (static_cast<int>(topResults.size()) < ef ||
                d < topResults.top().first) {

                candidates.push({d, nei});
                topResults.push({d, nei});

                if (static_cast<int>(topResults.size()) > ef)
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
    const vector<float> &query,
    const vector<int> &candidates,
    int max_neighbours
) {

    vector<pair<double, int>> distList;
    size_t queryLen = query.size();

    for (int id : candidates) {
        double d = distFunc(query.data(), nodes[id].embedding.data(), queryLen);
        distList.emplace_back(d, id);
    }

    std::sort(distList.begin(), distList.end());

    vector<int> result;
    for (int i = 0;
         i < std::min(max_neighbours, static_cast<int>(distList.size()));
         ++i) {
        result.push_back(distList[i].second);
    }

    return result;
}

vector<int> HnswCPU::selectNeighborsWithHeuristic(
    const vector<float> &query,
    const vector<int> &candidates,
    int max_neighbours,
    int layer,
    bool extendCandidates,
    bool keepPrunedConnections
) {

    vector<int> result;

    priority_queue<DistanceIndexPair, vector<DistanceIndexPair>, std::greater<>>
        pq;
    priority_queue<DistanceIndexPair, vector<DistanceIndexPair>, std::greater<>>
        discarded;

    vector<uint8_t> visited(nodes.size(), 0);

    for (int cand : candidates) {

        if (!visited[cand]) {
            visited[cand] = 1;
            double d =
                distFunc(query.data(), nodes[cand].embedding.data(), queryLen);
            pq.push({d, cand});
        }

        if (extendCandidates) {
            for (int nei : nodes[cand].neighbors[layer]) {
                if (!visited[nei]) {
                    visited[nei] = 1;
                    double d = distFunc(
                        query.data(), nodes[nei].embedding.data(), queryLen
                    );
                    pq.push({d, nei});
                }
            }
        }
    }

    while (!pq.empty() && static_cast<int>(result.size()) < max_neighbours) {

        auto [dist, id] = pq.top();
        pq.pop();

        bool good = true;

        for (int r : result) {
            double d = distFunc(
                nodes[id].embedding.data(), nodes[r].embedding.data(), queryLen
            );
            if (d < dist) {
                good = false;
                break;
            }
        }

        if (good) {
            result.push_back(id);
        } else {
            discarded.push({dist, id});
        }
    }

    if (keepPrunedConnections) {
        while (!discarded.empty() &&
               static_cast<int>(result.size()) < max_neighbours) {
            result.push_back(discarded.top().second);
            discarded.pop();
        }
    }

    return result;
}

void HnswCPU::create(const vector<vector<float>> &data) {
    for (const auto &v : data)
        add(v);
}

void HnswCPU::add(const vector<float> &embedding) {

    int nodeLevel = sampleLevel();
    int id = currentId++;

    nodes.emplace_back(id, nodeLevel, embedding);

    if (currentId == 1) {
        entryPoint = 0;
        maxLevel = nodeLevel;
        return;
    }

    int curr = entryPoint;

    for (int level = maxLevel; level > nodeLevel; --level) {
        curr = searchLayer(embedding, curr, 1, level)[0];
    }

    for (int level = std::min(nodeLevel, maxLevel); level >= 0; --level) {

        int maxNeighbors = (level == 0) ? M0 : M;

        auto layerNodes = searchLayer(embedding, curr, efConstruction, level);

        auto selected = selectNeighborsWithHeuristic(
            embedding, layerNodes, maxNeighbors, level, true, true
        );

        nodes[id].neighbors[level] = selected;

        for (int nid : selected) {

            if (nodes[nid].level < level)
                continue;

            auto &neighList = nodes[nid].neighbors[level];
            neighList.push_back(id);

            if (static_cast<int>(neighList.size()) > maxNeighbors) {
                neighList = selectNeighborsWithHeuristic(
                    nodes[nid].embedding,
                    neighList,
                    maxNeighbors,
                    level,
                    true,
                    true
                );
            }
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

    for (int level = maxLevel; level > 0; --level) {
        curr = searchLayer(query, curr, 1, level)[0];
    }

    auto candidates = searchLayer(query, curr, efSearch, 0);

    return selectNeighbors(query, candidates, k);
}

int HnswCPU::size() const { return static_cast<int>(nodes.size()); }

void HnswCPU::printInfo() {
    printSimdInfo();
} 