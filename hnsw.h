#include <iostream>
#include <vector>
#include <string>
#include <queue>

using std::vector, std::string;

class Node
{
public:
    uint64_t ind;
    uint32_t level = 0;
    vector<float> embedding;
    vector<Node *> neighbours;

    Node(uint32_t level, const vector<float> &embedding, const vector<Node *> &neighbours)
        : level(level), embedding(embedding), neighbours(neighbours) {}

    ~Node() = default;
};


class HnswIndex {
public:
    virtual ~HnswIndex() = default;

    virtual void create(const vector<vector<float>>& data) = 0;
    virtual void add(const vector<float>& embedding) = 0;

    virtual vector<Node*> search(const vector<float>& querylocale) = 0;

    virtual void remove(size_t id) = 0;

    virtual void save(const string& path) const = 0;
    virtual void load(const string& path) = 0;

    virtual size_t size() const = 0;
};


