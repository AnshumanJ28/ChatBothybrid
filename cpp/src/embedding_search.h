#pragma once
#include <vector>
#include <string>

// Cosine-similarity nearest-neighbor search over the KB vector store.
// Vectors are stored in a single contiguous row-major buffer (not a
// vector-of-vectors) so the scan is cache-friendly; this is also the
// natural layout to later swap in SIMD (e.g. AVX2 dot products) once
// correctness is settled, without changing the data structure.

namespace minibrain {

struct SearchResult {
    int index;
    double score; // cosine similarity, higher is better
};

class EmbeddingIndex {
public:
    EmbeddingIndex(int dim);

    // Adds one vector to the index; returns its assigned index.
    // Stores a pre-normalized copy internally for fast cosine sim.
    int add(const std::vector<double>& vec, const std::string& doc_id);

    // Returns the top_k most similar entries to the query vector.
    std::vector<SearchResult> search(const std::vector<double>& query, int top_k) const;

    int size() const { return count_; }
    int dim() const { return dim_; }
    const std::string& doc_id(int index) const { return doc_ids_.at(index); }

private:
    int dim_;
    int count_;
    std::vector<double> data_;       // count_ * dim_, L2-normalized rows
    std::vector<double> norms_orig_; // original norms, kept for debugging
    std::vector<std::string> doc_ids_;
};

} // namespace minibrain
