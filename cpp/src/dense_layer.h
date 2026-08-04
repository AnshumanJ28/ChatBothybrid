#pragma once
#include <vector>

namespace minibrain {

struct DenseLayerWeights {
    std::vector<double> weights; // (output_size x input_size)
    std::vector<double> bias;    // (output_size)
    int input_size;
    int output_size;
};

class DenseLayer {
public:
    explicit DenseLayer(DenseLayerWeights weights);
    
    // Runs forward pass with ReLU activation: out = max(0.0, W * x + b)
    std::vector<double> forward(const std::vector<double>& x) const;
    
    int input_size() const { return weights_.input_size; }
    int output_size() const { return weights_.output_size; }

private:
    DenseLayerWeights weights_;
};

} // namespace minibrain
