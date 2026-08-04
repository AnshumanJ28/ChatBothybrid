"""Dialogue State Manager.

Holds per-session state: active flow instance, a bounded turn history
buffer, and a topic tag for continuity. Also owns explicit/implicit
reset conditions so stale context doesn't keep steering the router.
Includes a C++ DeepLSTM-backed stacked dialogue memory tracking system.
"""
from collections import deque
import math
import numpy as np
from . import minibrain_cpp as mb
from .embedding import EMBED_DIM

_RESET_PHRASES = {"start over", "new topic", "reset the conversation",
                   "reset conversation", "forget that", "never mind"}
_TOPIC_MISMATCH_LIMIT = 3  # consecutive off-topic turns before implicit reset


def get_default_lstm_weights(input_size: int, hidden_size: int) -> mb.LSTMCellWeights:
    """Helper to initialize C++ LSTM cell weights deterministically."""
    rng = np.random.default_rng(42)
    w = mb.LSTMCellWeights()
    w.input_size = input_size
    w.hidden_size = hidden_size
    w.weight_ih = rng.normal(scale=0.1, size=(4 * hidden_size * input_size,)).tolist()
    w.weight_hh = rng.normal(scale=0.1, size=(4 * hidden_size * hidden_size,)).tolist()
    w.bias_ih = rng.normal(scale=0.01, size=(4 * hidden_size,)).tolist()
    w.bias_hh = rng.normal(scale=0.01, size=(4 * hidden_size,)).tolist()
    return w


def get_default_dense_weights(input_size: int, output_size: int) -> mb.DenseLayerWeights:
    """Helper to initialize C++ DenseLayer weights deterministically close to Identity."""
    rng = np.random.default_rng(43)
    w = mb.DenseLayerWeights()
    w.input_size = input_size
    w.output_size = output_size
    # Initialize as Identity matrix + small random noise
    weights_matrix = np.eye(output_size, input_size) + rng.normal(scale=0.01, size=(output_size, input_size))
    w.weights = weights_matrix.flatten().tolist()
    w.bias = rng.normal(scale=0.001, size=(output_size,)).tolist()
    return w


class Turn:
    def __init__(self, message: str, subsystem: str, confidence: float, category: str | None):
        self.message = message
        self.subsystem = subsystem   # 'smalltalk' | 'kb' | 'flow' | 'web' | 'clarify' | 'conversation'
        self.confidence = confidence
        self.category = category     # KB category tag, if applicable


class DialogueState:
    def __init__(self, history_size: int = 8):
        self.turn_history: deque[Turn] = deque(maxlen=history_size)
        self.active_flow = None            # FlowInstance | None
        self.current_topic: str | None = None
        self._mismatch_streak = 0
        
        # Initialize C++ 3-Layer Stacked DeepLSTM cell memory tracking
        self.num_layers = 3
        self.layers_weights = [get_default_lstm_weights(EMBED_DIM, EMBED_DIM) for _ in range(self.num_layers)]
        self.lstm_cell = mb.DeepLSTM(self.layers_weights)
        self.h_states = [[0.0] * EMBED_DIM for _ in range(self.num_layers)]
        self.c_states = [[0.0] * EMBED_DIM for _ in range(self.num_layers)]
        self.persona = "default"  # "default" | "child"
        
        # Initialize C++ Dense Layer (Neuron Layer) on top of DeepLSTM
        self.dense_weights = get_default_dense_weights(EMBED_DIM, EMBED_DIM)
        self.dense_layer = mb.DenseLayer(self.dense_weights)

    def check_explicit_reset(self, normalized_text: str) -> bool:
        text = normalized_text.lower()
        if any(phrase in text for phrase in _RESET_PHRASES):
            self.reset()
            return True
        return False

    def update_lstm_memory(self, tokens: list[str]) -> list[float]:
        """Pass the token sequence sequentially through the stacked C++ LSTM, returning final L2 norms for each layer."""
        from . import embedding
        
        for tok in tokens:
            t_vec = embedding._token_vector(tok)
            state = self.lstm_cell.forward(t_vec, self.h_states, self.c_states)
            self.h_states = state.h
            self.c_states = state.c
        
        norms = []
        for l in range(self.num_layers):
            norm = math.sqrt(sum(v * v for v in self.h_states[l]))
            norms.append(norm)
        return norms

    def get_lstm_context(self) -> list[float]:
        """Runs the final C++ stacked LSTM hidden state through the C++ Dense Layer (ReLU neuron layer) for projection."""
        raw_context = self.h_states[-1]
        projected = self.dense_layer.forward(raw_context)
        return projected

    def record_turn(self, message: str, subsystem: str, confidence: float, category: str | None = None):
        self.turn_history.append(Turn(message, subsystem, confidence, category))

        if category is not None:
            if self.current_topic is not None and category != self.current_topic:
                self._mismatch_streak += 1
            else:
                self._mismatch_streak = 0
            self.current_topic = category

            if self._mismatch_streak >= _TOPIC_MISMATCH_LIMIT:
                self.reset(keep_topic=False)

    def topic_bias_category(self) -> str | None:
        """If the last turn was a confident KB match, bias the next
        low-confidence turn toward that category before falling back."""
        if not self.turn_history:
            return None
        last = self.turn_history[-1]
        if last.subsystem == "kb" and last.confidence >= 0.5:
            return last.category
        return None

    def reset(self, keep_topic: bool = False):
        self.active_flow = None
        self._mismatch_streak = 0
        self.persona = "default"
        if not keep_topic:
            self.current_topic = None
        # Reset C++ LSTM memory states for all layers
        self.h_states = [[0.0] * EMBED_DIM for _ in range(self.num_layers)]
        self.c_states = [[0.0] * EMBED_DIM for _ in range(self.num_layers)]
