#include "hnsw.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

using std::vector, std::unordered_set, std::priority_queue, std::pair;

Node::Node(size_t id, size_t level, const vector<float> &embedding)
    : id(id), level(level), embedding(embedding) {
    neighbors.resize(level + 1);
}

HnswCPU::HnswCPU(size_t M_, size_t efConstruction_, uint32_t seed)
    : M(M_), M0(2 * M_), efConstruction(efConstruction_),
      levelMultiplier(1.0 / std::log(static_cast<double>(M_))), entryPoint(0),
      maxLevel(0), currentId(0), gen(seed), uniform_dist(0.0f, 1.0f) {}

size_t HnswCPU::sampleLevel() {
    double u = 1.0 - static_cast<double>(uniform_dist(gen));
    return static_cast<size_t>(-std::log(u) * levelMultiplier);
}

double
HnswCPU::l2Distance(const vector<float> &a, const vector<float> &b) const {
    double dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        dist += diff * diff;
    }
    return dist;
}

vector<size_t> HnswCPU::searchLayer(
    const vector<float> &query,
    size_t entry,
    size_t ef,
    size_t level
) {
    using DistIdPair = std::pair<double, size_t>;

    std::priority_queue<DistIdPair, vector<DistIdPair>, std::greater<>>
        candidates;
    std::priority_queue<DistIdPair> topResults;
    std::unordered_set<size_t> visited;

    double dist = l2Distance(query, nodes[entry].embedding);
    candidates.push({dist, entry});
    topResults.push({dist, entry});
    visited.insert(entry);

    while (!candidates.empty()) {
        auto [currDist, currId] = candidates.top();
        candidates.pop();

        double worstDist = topResults.top().first;
        if (currDist > worstDist)
            break;

        for (size_t nei : nodes[currId].neighbors[level]) {
            if (visited.count(nei))
                continue;
            visited.insert(nei);

            double d = l2Distance(query, nodes[nei].embedding);
            if (topResults.size() < ef || d < topResults.top().first) {
                candidates.push({d, nei});
                topResults.push({d, nei});
                if (topResults.size() > ef)
                    topResults.pop();
            }
        }
    }

    vector<size_t> result;
    while (!topResults.empty()) {
        result.push_back(topResults.top().second);
        topResults.pop();
    }
    return result;
}

vector<size_t> HnswCPU::selectNeighbors(
    const vector<float> &query,
    const vector<size_t> &candidates,
    size_t M_
) {
    vector<std::pair<double, size_t>> distList;
    for (size_t id : candidates) {
        distList.emplace_back(l2Distance(query, nodes[id].embedding), id);
    }

    std::sort(
        distList.begin(), distList.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
        }
    );

    vector<size_t> result;
    for (size_t i = 0; i < std::min(M_, distList.size()); ++i) {
        result.push_back(distList[i].second);
    }
    return result;
}

vector<size_t> HnswCPU::selectNeighborsWithHeuristic(
    const vector<float> &query,
    const vector<size_t> &candidates,
    const size_t max_neighbours,
    const size_t layer,
    const bool extendCandidates,
    const bool keepPrunedConnections
) {
    vector<size_t> ans;

    // (dist, ind)
    priority_queue<
        pair<double, size_t>,
        vector<pair<double, size_t>>,
        std::greater<>>
        pq;

    // (dist, ind)
    priority_queue<
        pair<double, size_t>,
        vector<pair<double, size_t>>,
        std::greater<>>
        discarded;

    unordered_set<size_t> vis;

    for (size_t cand : candidates) {
        if (vis.count(cand) == 0) {
            double dist = l2Distance(query, nodes[cand].embedding);
            pq.push({dist, cand});
            vis.insert(cand);
        }

        if (extendCandidates) {
            for (size_t nei : nodes[cand].neighbors[layer]) {
                if (vis.count(nei) == 0) {
                    double dist = l2Distance(query, nodes[nei].embedding);
                    pq.push({dist, nei});
                    vis.insert(nei);
                }
            }
        }
    }

    while (!pq.empty() && ans.size() < max_neighbours) {
        auto [dist, ind] = pq.top();
        pq.pop();

        bool good = true;

        for (size_t i : ans) {
            double d = l2Distance(query, nodes[i].embedding);
            if (d < dist) {
                // Existing element in ans is closer so reject ind
                good = false;
                break;
            }
        }

        if (good) {
            ans.push_back(ind);
        } else {
            discarded.push({dist, ind});
        }
    }

    if (keepPrunedConnections) {
        while (!discarded.empty() && ans.size() < max_neighbours) {
            auto [dist, ind] = discarded.top();
            discarded.pop();
            ans.push_back(ind);
        }
    }

    return ans;
}

void HnswCPU::create(const vector<vector<float>> &data) {
    for (const auto &v : data)
        add(v);
}

void HnswCPU::add(const vector<float> &embedding) {
    size_t nodeLevel = sampleLevel();
    size_t id = currentId++;
    nodes.emplace_back(id, nodeLevel, embedding);

    if (currentId == 1) { // first node
        entryPoint = 0;
        maxLevel = nodeLevel;
        return;
    }

    size_t curr = entryPoint;
    
    // Upper greedy routing (skip layers above new node)
    for (int level = (int)maxLevel; level > (int)nodeLevel; --level) {
        auto ans = searchLayer(embedding, curr, 1, level);
        curr = ans[0];
    }

    // Link layers including 0
    for (int level = (int)std::min(nodeLevel, maxLevel); level >= 0; --level) {
        size_t maxNeighbors = (level == 0) ? M0 : M;

        auto layerNodes = searchLayer(embedding, curr, efConstruction, level);

        auto selected = selectNeighborsWithHeuristic(
            embedding, layerNodes, maxNeighbors, level, true, true
        );

        nodes[id].neighbors[level] = selected;

        for (size_t nid : selected) {
            if (nodes[nid].level < (size_t)level)
                continue;

            auto &neighList = nodes[nid].neighbors[level];
            neighList.push_back(id);

            if (neighList.size() > maxNeighbors) {
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

vector<size_t>
HnswCPU::search(const vector<float> &query, size_t k, size_t efSearch) {

    if (nodes.empty())
        return {};

    size_t curr = entryPoint;

    // Greedy descent (skip level 0)
    for (int level = (int)maxLevel; level > 0; --level) {
        auto ans = searchLayer(query, curr, 1, level);
        curr = ans[0];
    }

    // Full search only at level 0
    auto candidates = searchLayer(query, curr, efSearch, 0);

    return selectNeighbors(query, candidates, k);
}

size_t HnswCPU::size() const { return nodes.size(); }
