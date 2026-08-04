#include "lstm_cell.h"
#include <cmath>
#include <stdexcept>

namespace minibrain {

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double tanh_(double x) {
    return std::tanh(x);
}

LSTMCell::LSTMCell(LSTMCellWeights weights) : weights_(std::move(weights)) {
    const int h = weights_.hidden_size;
    const int in = weights_.input_size;
    if ((int)weights_.weight_ih.size() != 4 * h * in)
        throw std::invalid_argument("weight_ih size mismatch");
    if ((int)weights_.weight_hh.size() != 4 * h * h)
        throw std::invalid_argument("weight_hh size mismatch");
    if ((int)weights_.bias_ih.size() != 4 * h)
        throw std::invalid_argument("bias_ih size mismatch");
    if ((int)weights_.bias_hh.size() != 4 * h)
        throw std::invalid_argument("bias_hh size mismatch");
}

LSTMCellState LSTMCell::forward(const std::vector<double>& x,
                                 const std::vector<double>& h_prev_in,
                                 const std::vector<double>& c_prev_in) const {
    const int H = weights_.hidden_size;
    const int I = weights_.input_size;

    if ((int)x.size() != I)
        throw std::invalid_argument("input vector size mismatch");

    std::vector<double> h_prev = h_prev_in.empty() ? std::vector<double>(H, 0.0) : h_prev_in;
    std::vector<double> c_prev = c_prev_in.empty() ? std::vector<double>(H, 0.0) : c_prev_in;
    if ((int)h_prev.size() != H || (int)c_prev.size() != H)
        throw std::invalid_argument("hidden/cell state size mismatch");

    // gates[0..H) = i, [H..2H) = f, [2H..3H) = g, [3H..4H) = o
    std::vector<double> gates(4 * H, 0.0);

    for (int g = 0; g < 4 * H; ++g) {
        double acc = weights_.bias_ih[g] + weights_.bias_hh[g];
        const int row_ih = g * I;
        for (int j = 0; j < I; ++j) {
            acc += weights_.weight_ih[row_ih + j] * x[j];
        }
        const int row_hh = g * H;
        for (int j = 0; j < H; ++j) {
            acc += weights_.weight_hh[row_hh + j] * h_prev[j];
        }
        gates[g] = acc;
    }

    LSTMCellState out;
    out.h.resize(H);
    out.c.resize(H);

    for (int k = 0; k < H; ++k) {
        double i_t = sigmoid(gates[k]);
        double f_t = sigmoid(gates[H + k]);
        double g_t = tanh_(gates[2 * H + k]);
        double o_t = sigmoid(gates[3 * H + k]);

        double c_t = f_t * c_prev[k] + i_t * g_t;
        double h_t = o_t * tanh_(c_t);

        out.c[k] = c_t;
        out.h[k] = h_t;
    }

    return out;
}

DeepLSTM::DeepLSTM(const std::vector<LSTMCellWeights>& layers_weights) {
    if (layers_weights.empty()) {
        throw std::invalid_argument("DeepLSTM needs at least one layer");
    }
    for (const auto& w : layers_weights) {
        layers_.emplace_back(w);
    }
}

StackedLSTMState DeepLSTM::forward(const std::vector<double>& x,
                                   const std::vector<std::vector<double>>& h_prev,
                                   const std::vector<std::vector<double>>& c_prev) const {
    const int num_layers_val = (int)layers_.size();
    
    StackedLSTMState out;
    out.h.resize(num_layers_val);
    out.c.resize(num_layers_val);
    
    std::vector<double> current_input = x;
    
    for (int l = 0; l < num_layers_val; ++l) {
        const auto& cell = layers_[l];
        
        std::vector<double> hp = (l < (int)h_prev.size()) ? h_prev[l] : std::vector<double>{};
        std::vector<double> cp = (l < (int)c_prev.size()) ? c_prev[l] : std::vector<double>{};
        
        LSTMCellState state = cell.forward(current_input, hp, cp);
        out.h[l] = state.h;
        out.c[l] = state.c;
        
        // Output of current layer becomes input to next layer
        current_input = state.h;
    }
    
    return out;
}

} // namespace minibrain
