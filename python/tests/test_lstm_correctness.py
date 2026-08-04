"""
Numerical sanity check for the hand-written C++ LSTM cell.

Reference: a plain-numpy implementation of the exact same equations
torch.nn.LSTMCell uses (gate order i, f, g, o; weight_ih shape
(4H, I); weight_hh shape (4H, H)). If torch is available in this
environment, the reference is cross-checked against nn.LSTMCell too
(see `test_against_torch_if_available`), but the primary correctness
test does not require torch to run.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orchestrator import minibrain_cpp as mb


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def numpy_lstm_cell(x, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size):
    gates = weight_ih @ x + bias_ih + weight_hh @ h_prev + bias_hh  # (4H,)
    i, f, g, o = np.split(gates, 4)
    i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
    g = np.tanh(g)
    c_t = f * c_prev + i * g
    h_t = o * np.tanh(c_t)
    return h_t, c_t


def random_weights(input_size, hidden_size, seed=0):
    rng = np.random.default_rng(seed)
    weight_ih = rng.normal(scale=0.5, size=(4 * hidden_size, input_size))
    weight_hh = rng.normal(scale=0.5, size=(4 * hidden_size, hidden_size))
    bias_ih = rng.normal(scale=0.1, size=(4 * hidden_size,))
    bias_hh = rng.normal(scale=0.1, size=(4 * hidden_size,))
    return weight_ih, weight_hh, bias_ih, bias_hh


def test_matches_numpy_reference():
    input_size, hidden_size = 8, 5
    weight_ih, weight_hh, bias_ih, bias_hh = random_weights(input_size, hidden_size, seed=42)

    rng = np.random.default_rng(7)
    x = rng.normal(size=(input_size,))
    h_prev = rng.normal(size=(hidden_size,))
    c_prev = rng.normal(size=(hidden_size,))

    # --- reference ---
    h_ref, c_ref = numpy_lstm_cell(x, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size)

    # --- C++ via pybind11 ---
    w = mb.LSTMCellWeights()
    w.input_size = input_size
    w.hidden_size = hidden_size
    w.weight_ih = weight_ih.flatten().tolist()
    w.weight_hh = weight_hh.flatten().tolist()
    w.bias_ih = bias_ih.tolist()
    w.bias_hh = bias_hh.tolist()

    cell = mb.LSTMCell(w)
    state = cell.forward(x.tolist(), h_prev.tolist(), c_prev.tolist())

    h_cpp = np.array(state.h)
    c_cpp = np.array(state.c)

    assert np.allclose(h_cpp, h_ref, atol=1e-9), f"h mismatch: {h_cpp} vs {h_ref}"
    assert np.allclose(c_cpp, c_ref, atol=1e-9), f"c mismatch: {c_cpp} vs {c_ref}"
    print("PASS: C++ LSTM cell matches numpy reference to 1e-9.")


def test_zero_initial_state_defaults():
    """h_prev/c_prev omitted (empty list) should behave as zeros."""
    input_size, hidden_size = 4, 3
    weight_ih, weight_hh, bias_ih, bias_hh = random_weights(input_size, hidden_size, seed=1)
    x = np.array([0.2, -0.1, 0.4, 0.0])
    zeros = np.zeros(hidden_size)

    h_ref, c_ref = numpy_lstm_cell(x, zeros, zeros, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size)

    w = mb.LSTMCellWeights()
    w.input_size, w.hidden_size = input_size, hidden_size
    w.weight_ih = weight_ih.flatten().tolist()
    w.weight_hh = weight_hh.flatten().tolist()
    w.bias_ih = bias_ih.tolist()
    w.bias_hh = bias_hh.tolist()
    cell = mb.LSTMCell(w)

    state = cell.forward(x.tolist(), [], [])
    assert np.allclose(state.h, h_ref, atol=1e-9)
    assert np.allclose(state.c, c_ref, atol=1e-9)
    print("PASS: default zero initial state matches reference.")


def test_against_torch_if_available():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("SKIP: torch not installed in this environment; "
              "numpy-reference test above already validates the same equations.")
        return

    input_size, hidden_size = 6, 4
    torch.manual_seed(0)
    cell_ref = nn.LSTMCell(input_size, hidden_size).double()

    x = torch.randn(1, input_size, dtype=torch.double)
    h0 = torch.randn(1, hidden_size, dtype=torch.double)
    c0 = torch.randn(1, hidden_size, dtype=torch.double)

    with torch.no_grad():
        h_ref, c_ref = cell_ref(x, (h0, c0))

    w = mb.LSTMCellWeights()
    w.input_size, w.hidden_size = input_size, hidden_size
    w.weight_ih = cell_ref.weight_ih.detach().numpy().flatten().tolist()
    w.weight_hh = cell_ref.weight_hh.detach().numpy().flatten().tolist()
    w.bias_ih = cell_ref.bias_ih.detach().numpy().tolist()
    w.bias_hh = cell_ref.bias_hh.detach().numpy().tolist()
    cpp_cell = mb.LSTMCell(w)

    state = cpp_cell.forward(x[0].tolist(), h0[0].tolist(), c0[0].tolist())

    assert np.allclose(state.h, h_ref[0].numpy(), atol=1e-9)
    assert np.allclose(state.c, c_ref[0].numpy(), atol=1e-9)
    print("PASS: C++ LSTM cell matches torch.nn.LSTMCell to 1e-9.")


if __name__ == "__main__":
    test_matches_numpy_reference()
    test_zero_initial_state_defaults()
    test_against_torch_if_available()
