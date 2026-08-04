import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orchestrator import minibrain_cpp as mb


def random_weights(input_size, hidden_size, seed=0):
    rng = np.random.default_rng(seed)
    weight_ih = rng.normal(scale=0.5, size=(4 * hidden_size * input_size,))
    weight_hh = rng.normal(scale=0.5, size=(4 * hidden_size * hidden_size,))
    bias_ih = rng.normal(scale=0.1, size=(4 * hidden_size,))
    bias_hh = rng.normal(scale=0.1, size=(4 * hidden_size,))
    
    w = mb.LSTMCellWeights()
    w.input_size = input_size
    w.hidden_size = hidden_size
    w.weight_ih = weight_ih.tolist()
    w.weight_hh = weight_hh.tolist()
    w.bias_ih = bias_ih.tolist()
    w.bias_hh = bias_hh.tolist()
    return w


def test_stacked_lstm_correctness():
    print("Testing 3-layer C++ DeepLSTM stacked execution correctness...")
    
    dim = 8
    # 3 layers weights
    w1 = random_weights(dim, dim, seed=10)
    w2 = random_weights(dim, dim, seed=20)
    w3 = random_weights(dim, dim, seed=30)
    
    # 1. Test standard stacked deep forward pass
    deep_lstm = mb.DeepLSTM([w1, w2, w3])
    assert deep_lstm.num_layers == 3
    
    rng = np.random.default_rng(42)
    x = rng.normal(size=(dim,)).tolist()
    
    # Empty prev states default to zeros
    state = deep_lstm.forward(x, [], [])
    
    assert len(state.h) == 3
    assert len(state.c) == 3
    for l in range(3):
        assert len(state.h[l]) == dim
        assert len(state.c[l]) == dim
        
    # 2. Cross check each layer manually using single-layer LSTMCell
    cell1 = mb.LSTMCell(w1)
    cell2 = mb.LSTMCell(w2)
    cell3 = mb.LSTMCell(w3)
    
    s1 = cell1.forward(x, [], [])
    s2 = cell2.forward(s1.h, [], [])
    s3 = cell3.forward(s2.h, [], [])
    
    # Assert layer 1 output matches cell1
    assert np.allclose(state.h[0], s1.h, atol=1e-9)
    assert np.allclose(state.c[0], s1.c, atol=1e-9)
    
    # Assert layer 2 output matches cell2
    assert np.allclose(state.h[1], s2.h, atol=1e-9)
    assert np.allclose(state.c[1], s2.c, atol=1e-9)
    
    # Assert layer 3 output matches cell3
    assert np.allclose(state.h[2], s3.h, atol=1e-9)
    assert np.allclose(state.c[2], s3.c, atol=1e-9)
    
    print("PASS: C++ Stacked DeepLSTM cascades inputs through layers correctly and matches cell-by-cell reference.")


if __name__ == "__main__":
    test_stacked_lstm_correctness()
