#include "embedding_search.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace minibrain {

EmbeddingIndex::EmbeddingIndex(int dim) : dim_(dim), count_(0) {
    if (dim <= 0) throw std::invalid_argument("dim must be positive");
}

int EmbeddingIndex::add(const std::vector<double>& vec, const std::string& doc_id) {
    if ((int)vec.size() != dim_) throw std::invalid_argument("vector dim mismatch");

    double norm = 0.0;
    for (double v : vec) norm += v * v;
    norm = std::sqrt(norm);
    if (norm == 0.0) norm = 1e-12; // avoid div-by-zero for degenerate rows

    data_.reserve(data_.size() + dim_);
    for (double v : vec) data_.push_back(v / norm);

    norms_orig_.push_back(norm);
    doc_ids_.push_back(doc_id);

    return count_++;
}

std::vector<SearchResult> EmbeddingIndex::search(const std::vector<double>& query, int top_k) const {
    if ((int)query.size() != dim_) throw std::invalid_argument("query dim mismatch");
    if (top_k <= 0) return {};

    double qnorm = 0.0;
    for (double v : query) qnorm += v * v;
    qnorm = std::sqrt(qnorm);
    if (qnorm == 0.0) qnorm = 1e-12;

    std::vector<double> qn(dim_);
    for (int j = 0; j < dim_; ++j) qn[j] = query[j] / qnorm;

    std::vector<SearchResult> results;
    results.reserve(count_);

    for (int i = 0; i < count_; ++i) {
        const double* row = &data_[(size_t)i * dim_];
        double dot = 0.0;
        
        int j = 0;
        for (; j <= dim_ - 4; j += 4) {
            dot += row[j] * qn[j]
                 + row[j + 1] * qn[j + 1]
                 + row[j + 2] * qn[j + 2]
                 + row[j + 3] * qn[j + 3];
        }
        for (; j < dim_; ++j) {
            dot += row[j] * qn[j];
        }
        
        results.push_back({i, dot});
    }

    int k = std::min(top_k, count_);
    std::partial_sort(results.begin(), results.begin() + k, results.end(),
        [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
    results.resize(k);
    return results;
}

} // namespace minibrain
