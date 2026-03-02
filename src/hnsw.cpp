#include "hnsw.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

using std::vector, std::unordered_set, std::priority_queue, std::pair;

Node::Node(size_t id, size_t level, const vector<float> &embedding) :
    id(id),
    level(level),
    embedding(embedding) {
    neighbors.resize(level + 1);
}

HnswCPU::HnswCPU(
    size_t M_,
    size_t efConstruction_,
    uint32_t seed,
    DistanceType distType_
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
    distType(distType_),
    distFunc(getDistanceFunction(distType_)) {}

size_t HnswCPU::sampleLevel() {
    double u = 1.0 - static_cast<double>(uniform_dist(gen));
    return static_cast<size_t>(-std::log(u) * levelMultiplier);
}

float HnswCPU::scoreForHeap(float raw) const {
    if (distType == DistanceType::COSINE) {
        // For cosine we maximize similarity, so use negative for min-heap logic.
        return -raw;
    }
    // For true distances (L1/L2), smaller is better as-is.
    return raw;
}

vector<size_t> HnswCPU::searchLayer(
    const vector<float> &query,
    size_t entry,
    size_t ef,
    size_t level
) {

    priority_queue<
        pair<float, size_t>,
        vector<pair<float, size_t> >,
        std::greater<std::pair<float, size_t> > >
        candidates;
    priority_queue<pair<float, size_t>> topResults;
    unordered_set<size_t> vis;

    float dist =
        distFunc(query.data(), nodes[entry].embedding.data(), query.size());
    float score = scoreForHeap(dist);
    candidates.push({score, entry});
    topResults.push({score, entry});
    vis.insert(entry);

    while (!candidates.empty()) {
        auto [currScore, currId] = candidates.top();
        candidates.pop();

        float worstScore = topResults.top().first;
        if (currScore > worstScore)
            break;

        for (size_t nei : nodes[currId].neighbors[level]) {
            if (vis.count(nei))
                continue;
            vis.insert(nei);

            float d = distFunc(
                query.data(), nodes[nei].embedding.data(), query.size()
            );
            float dScore = scoreForHeap(d);
            if (topResults.size() < ef || dScore < topResults.top().first) {
                candidates.push({dScore, nei});
                topResults.push({dScore, nei});
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
    vector<std::pair<float, size_t>> distList;
    for (size_t id : candidates) {
        float dist =
            distFunc(query.data(), nodes[id].embedding.data(), query.size());
        float score = scoreForHeap(dist);
        distList.emplace_back(score, id);
    }

    std::sort(
        distList.begin(),
        distList.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; }
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
        pair<float, size_t>,
        vector<pair<float, size_t> >,
        std::greater<std::pair<float, size_t> > >
        pq;

    // (dist, ind)
    priority_queue<
        pair<float, size_t>,
        vector<pair<float, size_t> >,
        std::greater<std::pair<float, size_t> > >
        discarded;

    unordered_set<size_t> vis;

    for (size_t cand : candidates) {
        if (vis.count(cand) == 0) {
            float dist = distFunc(
                query.data(), nodes[cand].embedding.data(), query.size()
            );
            float score = scoreForHeap(dist);
            pq.push({score, cand});
            vis.insert(cand);
        }

        if (extendCandidates) {
            for (size_t nei : nodes[cand].neighbors[layer]) {
                if (vis.count(nei) == 0) {
                    float dist = distFunc(
                        query.data(), nodes[nei].embedding.data(), query.size()
                    );
                    float score = scoreForHeap(dist);
                    pq.push({score, nei});
                    vis.insert(nei);
                }
            }
        }
    }

    while (!pq.empty() && ans.size() < max_neighbours) {
        auto [score, ind] = pq.top();
        pq.pop();

        bool good = true;

        for (size_t i : ans) {
            float d = distFunc(
                query.data(), nodes[i].embedding.data(), query.size()
            );
            float dScore = scoreForHeap(d);
            if (dScore < score) {
                // Existing element in ans is closer so reject ind
                good = false;
                break;
            }
        }

        if (good) {
            ans.push_back(ind);
        } else {
            discarded.push({score, ind});
        }
    }

    if (keepPrunedConnections) {
        while (!discarded.empty() && ans.size() < max_neighbours) {
            auto [score, ind] = discarded.top();
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
    for (size_t level = maxLevel; level-- > nodeLevel;) {
        auto ans = searchLayer(embedding, curr, 1, level);
        curr = ans[0];
    }

    for (size_t level = std::min(nodeLevel, maxLevel); level-- > 0;) {
        size_t maxNeighbors = (level == 0) ? M0 : M;
        auto layerNodes = searchLayer(embedding, curr, efConstruction, level);
        auto selected = selectNeighborsWithHeuristic(
            embedding, layerNodes, maxNeighbors, level, true, true
        );

        nodes[id].neighbors[level] = selected;

        for (size_t nid : selected) {
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

    for (size_t level = maxLevel; level-- > 0;) {
        auto ans = searchLayer(query, curr, 1, level);
        curr = ans[0];
    }

    auto candidates = searchLayer(query, curr, efSearch, 0);

    return selectNeighbors(query, candidates, k);
}

size_t HnswCPU::size() const { return nodes.size(); }
