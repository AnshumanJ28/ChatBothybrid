#pragma once
#include <vector>

// Hand-written LSTM cell forward pass (single timestep, single layer).
// Gate equations match the standard formulation used by torch.nn.LSTMCell:
//
//   i_t = sigmoid(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)   input gate
//   f_t = sigmoid(W_if x_t + b_if + W_hf h_{t-1} + b_hf)   forget gate
//   g_t = tanh   (W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)   cell candidate
//   o_t = sigmoid(W_io x_t + b_io + W_ho h_{t-1} + b_ho)   output gate
//   c_t = f_t * c_{t-1} + i_t * g_t
//   h_t = o_t * tanh(c_t)
//
// Weight layout mirrors PyTorch's packed convention: weight_ih is
// (4*hidden, input) and weight_hh is (4*hidden, hidden), with the four
// gates stacked in order [i, f, g, o] along the first dimension. This
// makes cross-checking against nn.LSTMCell weights a direct copy, no
// re-ordering needed.

namespace minibrain {

struct LSTMCellWeights {
    // Flattened row-major matrices.
    std::vector<double> weight_ih; // (4*hidden_size x input_size)
    std::vector<double> weight_hh; // (4*hidden_size x hidden_size)
    std::vector<double> bias_ih;   // (4*hidden_size)
    std::vector<double> bias_hh;   // (4*hidden_size)
    int input_size;
    int hidden_size;
};

struct LSTMCellState {
    std::vector<double> h; // (hidden_size)
    std::vector<double> c; // (hidden_size)
};

class LSTMCell {
public:
    explicit LSTMCell(LSTMCellWeights weights);

    // Runs one timestep. h_prev/c_prev may be empty to mean "zeros".
    LSTMCellState forward(const std::vector<double>& x,
                           const std::vector<double>& h_prev,
                           const std::vector<double>& c_prev) const;

    int input_size() const { return weights_.input_size; }
    int hidden_size() const { return weights_.hidden_size; }

private:
    LSTMCellWeights weights_;
};

// Stacked LSTM for deep architectures.
struct StackedLSTMState {
    std::vector<std::vector<double>> h; // State for each layer [layer_idx][hidden_val]
    std::vector<std::vector<double>> c; // Cell state for each layer [layer_idx][hidden_val]
};

class DeepLSTM {
public:
    explicit DeepLSTM(const std::vector<LSTMCellWeights>& layers_weights);

    // Runs stacked layers forward pass. Output h_t of layer L becomes input x_t of layer L+1.
    StackedLSTMState forward(const std::vector<double>& x,
                             const std::vector<std::vector<double>>& h_prev = std::vector<std::vector<double>>{},
                             const std::vector<std::vector<double>>& c_prev = std::vector<std::vector<double>>{}) const;

    int num_layers() const { return (int)layers_.size(); }
    const std::vector<LSTMCell>& layers() const { return layers_; }

private:
    std::vector<LSTMCell> layers_;
};

double sigmoid(double x);
double tanh_(double x);

} // namespace minibrain
