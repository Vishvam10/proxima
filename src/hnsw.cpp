#include "hnsw.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <unordered_set>

using std::pair;
using std::priority_queue;
using std::unordered_set;
using std::vector;

/* ============================
   Node Implementation
   ============================ */

Node::Node(uint64_t id, int level, const vector<float>& embedding)
    : id(id), level(level), embedding(embedding) {
    neighbors.resize(level + 1);
}

/* ============================
   HnswCPU Constructor
   ============================ */

HnswCPU::HnswCPU(size_t M, size_t efConstruction)
    : M(M),
      M0(2 * M),
      efConstruction(efConstruction),
      levelMultiplier(1.0f / std::log(M)),
      entryPoint(-1),
      maxLevel(-1),
      currentId(0),
      gen(std::random_device{}()),
      uniform_dist(0.0f, 1.0f) {}

/* ============================
   Utility Functions
   ============================ */

int HnswCPU::sampleLevel() {
    float u = 1.0f - uniform_dist(gen);
    return static_cast<int>(-std::log(u) * levelMultiplier);
}

float HnswCPU::l2Distance(
    const vector<float>& a,
    const vector<float>& b
) {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/* ============================
   SEARCH-LAYER (Algorithm 2)
   ============================ */

vector<uint64_t> HnswCPU::searchLayer(
    const vector<float>& query,
    uint64_t entry,
    size_t ef,
    int level
) {
    // Min-heap (C)
    priority_queue<
        pair<float, uint64_t>,
        vector<pair<float, uint64_t>>,
        std::greater<>
    > pq;

    // Max-heap (top ef results W)
    priority_queue<pair<float, uint64_t>> dists;

    unordered_set<uint64_t> visited;

    float dist = l2Distance(query, nodes[entry].embedding);

    pq.push({dist, entry});
    dists.push({dist, entry});
    visited.insert(entry);

    while (!pq.empty()) {
        auto [currDist, currId] = pq.top();
        pq.pop();

        float worstDist = dists.top().first;

        if (currDist > worstDist)
            break;

        for (uint64_t nei : nodes[currId].neighbors[level]) {
            if (visited.count(nei))
                continue;

            visited.insert(nei);

            float d = l2Distance(query, nodes[nei].embedding);

            if (dists.size() < ef || d < dists.top().first) {
                pq.push({d, nei});
                dists.push({d, nei});

                if (dists.size() > ef)
                    dists.pop();
            }
        }
    }

    vector<uint64_t> result;
    while (!dists.empty()) {
        result.push_back(dists.top().second);
        dists.pop();
    }

    return result;
}

/* ============================
   SELECT-NEIGHBORS (Simple)
   ============================ */

vector<uint64_t> HnswCPU::selectNeighbors(
    const vector<float>& query,
    const vector<uint64_t>& pq,
    size_t M
) {
    vector<pair<float, uint64_t>> distList;

    for (uint64_t id : pq) {
        float d = l2Distance(query, nodes[id].embedding);
        distList.push_back({d, id});
    }

    std::sort(
        distList.begin(),
        distList.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        }
    );

    vector<uint64_t> result;
    size_t limit = std::min(M, distList.size());

    for (size_t i = 0; i < limit; ++i)
        result.push_back(distList[i].second);

    return result;
}

/* ============================
   ADD (Insertion)
   ============================ */

void HnswCPU::add(const vector<float>& embedding) {
    int nodeLevel = sampleLevel();
    uint64_t id = currentId++;

    nodes.emplace_back(id, nodeLevel, embedding);

    if (entryPoint == -1) {
        entryPoint = id;
        maxLevel = nodeLevel;
        return;
    }

    uint64_t curr = entryPoint;

    // Navigation phase (greedy, ef=1)
    for (int level = maxLevel; level > nodeLevel; --level) {
        auto result = searchLayer(embedding, curr, 1, level);
        curr = result[0];
    }

    // Insertion phase
    for (int level = std::min(nodeLevel, maxLevel); level >= 0; --level) {
        size_t maxNeighbors = (level == 0) ? M0 : M;

        auto layerCandidates =
            searchLayer(embedding, curr, efConstruction, level);

        auto selected =
            selectNeighbors(embedding, layerCandidates, maxNeighbors);

        nodes[id].neighbors[level] = selected;

        for (uint64_t nid : selected) {
            auto& neighList = nodes[nid].neighbors[level];
            neighList.push_back(id);

            if (neighList.size() > maxNeighbors) {
                auto pruned =
                    selectNeighbors(nodes[nid].embedding,
                                    neighList,
                                    maxNeighbors);

                neighList = pruned;
            }
        }
    }

    if (nodeLevel > maxLevel) {
        entryPoint = id;
        maxLevel = nodeLevel;
    }
}

/* ============================
   SEARCH (Full Query)
   ============================ */

vector<uint64_t> HnswCPU::search(
    const vector<float>& query,
    size_t k
) {
    if (entryPoint == -1)
        return {};

    uint64_t curr = entryPoint;

    // Greedy descent
    for (int level = maxLevel; level > 0; --level) {
        auto result = searchLayer(query, curr, 1, level);
        curr = result[0];
    }

    auto pq = searchLayer(query, curr, efConstruction, 0);

    return selectNeighbors(query, pq, k);
}

/* ============================
   CREATE
   ============================ */

void HnswCPU::create(const vector<vector<float>>& data) {
    for (const auto& v : data)
        add(v);
}

/* ============================
   SIZE
   ============================ */

size_t HnswCPU::size() const {
    return nodes.size();
}