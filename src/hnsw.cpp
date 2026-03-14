#include "hnsw.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <thread>

using std::pair;
using std::priority_queue;
using std::vector;

using DistanceIndexPair = pair<double, int>;

Node::Node(int id_, int level_, const vector<float> &emb) :
    id(id_),
    level(level_),
    embedding(emb) {
    neighbors.resize(static_cast<size_t>(level) + 1);
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
    levelMultiplier(1.0 / std::log(static_cast<double>(M_))),
    entryPoint(0),
    maxLevel(0),
    currentId(0),
    gen(seed),
    uniform_dist(0.0f, 1.0f),
    distType(distType) {
    distFunc = getDistanceFunction(distType, forceScalar);
}

int HnswCPU::sampleLevel() {
    double u = 1.0 - static_cast<double>(uniform_dist(gen));
    return static_cast<int>(-std::log(u) * levelMultiplier);
}

vector<int>
HnswCPU::searchLayer(const vector<float> &query, int entry, int ef, int level) {

    priority_queue<DistanceIndexPair, vector<DistanceIndexPair>, std::greater<>>
        candidates;
    priority_queue<DistanceIndexPair> topResults;

    vector<uint8_t> visited(nodes.size(), 0);

    size_t queryLen = query.size();
    size_t entryIdx = static_cast<size_t>(entry);
    size_t lvl = static_cast<size_t>(level);
    double dist =
        distFunc(query.data(), nodes[entryIdx].embedding.data(), queryLen);

    candidates.push({dist, entry});
    topResults.push({dist, entry});
    visited[entryIdx] = 1;

    while (!candidates.empty()) {

        auto [currDist, currId] = candidates.top();
        candidates.pop();

        if (currDist > topResults.top().first)
            break;

        for (int nei : nodes[static_cast<size_t>(currId)].neighbors[lvl]) {
            size_t neiIdx = static_cast<size_t>(nei);

            if (visited[neiIdx])
                continue;

            visited[neiIdx] = 1;

            double d = distFunc(
                query.data(), nodes[neiIdx].embedding.data(), queryLen
            );

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
        double d = distFunc(
            query.data(),
            nodes[static_cast<size_t>(id)].embedding.data(),
            queryLen
        );
        distList.emplace_back(d, id);
    }

    std::sort(distList.begin(), distList.end());

    vector<int> result;
    size_t limit = static_cast<size_t>(
        std::min(max_neighbours, static_cast<int>(distList.size()))
    );
    for (size_t i = 0; i < limit; ++i) {
        result.push_back(distList[i].second);
    }

    return result;
}

vector<int> HnswCPU::selectNeighborsWithHeuristic(
    const vector<float> &query,
    const vector<int> candidates,
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

    size_t queryLen = query.size();
    size_t layerIdx = static_cast<size_t>(layer);

    for (int cand : candidates) {
        size_t candIdx = static_cast<size_t>(cand);

        if (!visited[candIdx]) {
            visited[candIdx] = 1;
            double d = distFunc(
                query.data(), nodes[candIdx].embedding.data(), queryLen
            );
            pq.push({d, cand});
        }

        if (extendCandidates) {
            for (int nei : nodes[candIdx].neighbors[layerIdx]) {
                size_t neiIdx = static_cast<size_t>(nei);
                if (!visited[neiIdx]) {
                    visited[neiIdx] = 1;
                    double d = distFunc(
                        query.data(), nodes[neiIdx].embedding.data(), queryLen
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
                nodes[static_cast<size_t>(id)].embedding.data(),
                nodes[static_cast<size_t>(r)].embedding.data(),
                queryLen
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

        size_t ind = static_cast<size_t>(id);
        size_t lvl = static_cast<size_t>(level);

        nodes[ind].neighbors[lvl] = selected;

        for (int nid : selected) {
            size_t nidIdx = static_cast<size_t>(nid);

            if (nodes[nidIdx].level < level)
                continue;

            auto &neighList = nodes[nidIdx].neighbors[lvl];
            neighList.push_back(id);

            if (static_cast<int>(neighList.size()) > maxNeighbors) {
                neighList = selectNeighborsWithHeuristic(
                    nodes[nidIdx].embedding,
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

void HnswCPU::printInfo() { printSimdInfo(); }