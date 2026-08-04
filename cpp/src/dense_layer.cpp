#include "dense_layer.h"
#include <stdexcept>
#include <algorithm>

namespace minibrain {

DenseLayer::DenseLayer(DenseLayerWeights weights) : weights_(std::move(weights)) {
    if ((int)weights_.weights.size() != weights_.output_size * weights_.input_size) {
        throw std::invalid_argument("DenseLayer weights size mismatch");
    }
    if ((int)weights_.bias.size() != weights_.output_size) {
        throw std::invalid_argument("DenseLayer bias size mismatch");
    }
}

std::vector<double> DenseLayer::forward(const std::vector<double>& x) const {
    const int in_dim = weights_.input_size;
    const int out_dim = weights_.output_size;
    
    if ((int)x.size() != in_dim) {
        throw std::invalid_argument("DenseLayer input vector dimension mismatch");
    }
    
    std::vector<double> out(out_dim, 0.0);
    for (int i = 0; i < out_dim; ++i) {
        double acc = weights_.bias[i];
        const int row_offset = i * in_dim;
        
        // Loop unrolling optimization for matrix multiplication
        int j = 0;
        for (; j <= in_dim - 4; j += 4) {
            acc += weights_.weights[row_offset + j] * x[j]
                 + weights_.weights[row_offset + j + 1] * x[j + 1]
                 + weights_.weights[row_offset + j + 2] * x[j + 2]
                 + weights_.weights[row_offset + j + 3] * x[j + 3];
        }
        for (; j < in_dim; ++j) {
            acc += weights_.weights[row_offset + j] * x[j];
        }
        
        // ReLU activation
        out[i] = std::max(0.0, acc);
    }
    return out;
}

} // namespace minibrain
