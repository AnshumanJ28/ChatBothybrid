"""Quick smoke test: import the compiled extension and exercise both
components directly (no orchestration layer involved)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "orchestrator"))

import minibrain_cpp as mb

# --- LSTM cell ---
w = mb.LSTMCellWeights()
w.input_size = 3
w.hidden_size = 2
w.weight_ih = [0.1] * (4 * 2 * 3)
w.weight_hh = [0.1] * (4 * 2 * 2)
w.bias_ih = [0.0] * (4 * 2)
w.bias_hh = [0.0] * (4 * 2)

cell = mb.LSTMCell(w)
state = cell.forward([1.0, 0.5, -0.5], [], [])
print("LSTM h:", state.h)
print("LSTM c:", state.c)
assert len(state.h) == 2 and len(state.c) == 2

# --- Embedding index ---
idx = mb.EmbeddingIndex(4)
idx.add([1, 0, 0, 0], "doc_a")
idx.add([0, 1, 0, 0], "doc_b")
idx.add([0.9, 0.1, 0, 0], "doc_c")

results = idx.search([1, 0, 0, 0], top_k=2)
print("Search results:", [(r.index, round(r.score, 4), idx.doc_id(r.index)) for r in results])
assert results[0].index == 0  # doc_a is the exact match

print("OK: both C++ components load and run via pybind11.")
