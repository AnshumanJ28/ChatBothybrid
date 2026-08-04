#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace minibrain {

class AttentionPooling {
public:
    // Computes self-attention weighted pooled vector across token embeddings (C++ SIMD unrolled loop)
    static std::vector<double> pool(const std::vector<std::vector<double>>& token_embeds) {
        if (token_embeds.empty()) return {};
        int dim = token_embeds[0].size();
        if (dim == 0) return {};

        int seq_len = token_embeds.size();
        std::vector<double> scores(seq_len, 0.0);

        // 1. Calculate centroid
        std::vector<double> centroid(dim, 0.0);
        for (const auto& vec : token_embeds) {
            for (int d = 0; d < dim; ++d) {
                centroid[d] += vec[d];
            }
        }
        for (int d = 0; d < dim; ++d) {
            centroid[d] /= seq_len;
        }

        // 2. Compute dot product similarity to centroid with 4x unrolling
        double max_score = -1e9;
        for (int i = 0; i < seq_len; ++i) {
            double dot = 0.0;
            int d = 0;
            for (; d + 3 < dim; d += 4) {
                dot += token_embeds[i][d] * centroid[d] +
                       token_embeds[i][d+1] * centroid[d+1] +
                       token_embeds[i][d+2] * centroid[d+2] +
                       token_embeds[i][d+3] * centroid[d+3];
            }
            for (; d < dim; ++d) {
                dot += token_embeds[i][d] * centroid[d];
            }
            scores[i] = dot / std::sqrt(static_cast<double>(dim));
            if (scores[i] > max_score) max_score = scores[i];
        }

        // 3. Softmax scaling
        double exp_sum = 0.0;
        std::vector<double> weights(seq_len, 0.0);
        for (int i = 0; i < seq_len; ++i) {
            weights[i] = std::exp(scores[i] - max_score);
            exp_sum += weights[i];
        }
        for (int i = 0; i < seq_len; ++i) {
            weights[i] /= (exp_sum > 0.0 ? exp_sum : 1.0);
        }

        // 4. Weighted sum
        std::vector<double> pooled(dim, 0.0);
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < dim; ++d) {
                pooled[d] += weights[i] * token_embeds[i][d];
            }
        }

        // L2 normalize
        double norm_sq = 0.0;
        for (int d = 0; d < dim; ++d) norm_sq += pooled[d] * pooled[d];
        double norm = std::sqrt(norm_sq);
        if (norm > 1e-12) {
            for (int d = 0; d < dim; ++d) pooled[d] /= norm;
        }

        return pooled;
    }
};

} // namespace minibrain
