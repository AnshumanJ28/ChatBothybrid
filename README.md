# Hybrid C++/LSTM "Mini Brain" Chatbot

A retrieval/template-based chatbot with **no LLM anywhere in the
pipeline**. Every reply is retrieved, computed, or template-filled.
Two performance-critical, compute-bound pieces are hand-written in
C++ and exposed to Python via pybind11; everything branching/logic-
bound (router, flow engine, dialogue state, composer) stays in Python.

```
User message
   -> Preprocessing (tokenize, spellcheck, normalize)
   -> Embedding layer (local encoder, NOT an LLM)
   -> Router (per-subsystem confidence thresholds)
        -> Small-talk (before KB, so greetings short-circuit)
        -> Flow Engine (multi-step slot-filling state machine)
        -> KB Search (C++ embedding similarity, top_k)   [tried first]
        -> Tool Dispatch -> web search (overflow, cached, templated)
        -> Fallback / Clarify
   -> Dialogue State Manager (slots, active flow, turn history, topic tag)
   -> Response Composer (template + slot-fill + rank — NOT a generator)
   -> Reply text
```

## Layout

```
cpp/
  src/lstm_cell.{h,cpp}          hand-written LSTM gate math (inference only)
  src/embedding_search.{h,cpp}   cosine-sim nearest-neighbor over KB vectors
  src/bindings.cpp               pybind11 bindings -> minibrain_cpp module
  Makefile                       builds the extension with g++ (no cmake needed)
  CMakeLists.txt                 alternative CMake build
  tests/test_bindings_smoke.py   imports the compiled module directly

python/
  orchestrator/
    preprocessing.py             tokenize / spellcheck / normalize
    embedding.py                 text -> vector (local encoder, swappable)
    kb_search.py                 wraps the C++ EmbeddingIndex
    smalltalk.py                 greeting/thanks/goodbye intent layer
    flow_engine.py                multi-step slot-filling state machine
    dialogue_state.py            turn history, active flow, topic continuity, resets
    router.py                    dispatches to the right subsystem
    composer.py                  template + slot-fill + rank -> reply text
    tool_dispatch/
      web_search.py              Google CSE call, cached, templated fallback
      cache.py                   TTL cache for tool-dispatch results
    main.py                      ChatSession + REPL entrypoint
    minibrain_cpp*.so            compiled extension (built by cpp/Makefile)
  data/kb_sample.json            sample knowledge base
  tests/
    test_lstm_correctness.py     C++ LSTM vs numpy reference (and torch, if installed)
    test_kb_search.py            embedding search sanity checks
    test_pipeline_integration.py end-to-end: smalltalk, KB, flow, web, reset, clarify
```

## Build

No CMake required — the Makefile calls `python3-config` and pybind11
directly:

```bash
pip install pybind11 numpy --break-system-packages   # or use a venv
cd cpp
make                      # builds minibrain_cpp*.so, copies it into python/orchestrator/
make test                 # builds + runs the binding smoke test
```

If you'd rather use CMake:

```bash
cd cpp
mkdir build && cd build
cmake .. && make
```

## Run

```bash
cd python
python3 -m orchestrator.main     # interactive REPL
```

```bash
# programmatic use
from orchestrator.main import ChatSession
session = ChatSession()
print(session.handle("how do I reset my password"))
print(session.handle("I want to return an item"))
print(session.handle("order 12345"))
print(session.handle("it arrived damaged"))
```

## Tests

```bash
cd python
python3 tests/test_lstm_correctness.py        # numerical correctness vs reference
python3 tests/test_kb_search.py
python3 tests/test_pipeline_integration.py    # full pipeline, all subsystems
```

`test_lstm_correctness.py`'s primary check is against a plain-numpy
implementation of the exact equations `torch.nn.LSTMCell` uses (same
gate order, same weight layout), so it runs with zero heavy deps. If
`torch` happens to be installed, a second test cross-checks directly
against `nn.LSTMCell` and is skipped otherwise.

## Wiring in a real embedding model

`orchestrator/embedding.py` ships a deterministic hashing-based
bag-of-words encoder so the whole project runs offline with no model
downloads. To use a real pretrained sentence encoder, replace
`encode()`'s body:

```python
from sentence_transformers import SentenceTransformer
_model = SentenceTransformer("all-MiniLM-L6-v2")
def encode(text: str) -> list[float]:
    return _model.encode(text).tolist()
```

Everything downstream (KB search, the C++ index, the router) is
dimension-agnostic, so nothing else changes — just update
`EMBED_DIM` to match the new model's output size.

## Wiring in live web search

`orchestrator/tool_dispatch/web_search.py` falls back to an offline
stub when credentials aren't set, so the router's overflow path is
exercisable without network access. To enable live results, set:

```bash
export GOOGLE_CSE_API_KEY=...
export GOOGLE_CSE_CX=...
```

## Non-negotiable constraints (preserved)

- No LLM calls, no text generation — every output is selected or
  template-filled.
- The embedding model maps text -> vector only; it does not generate text.
- Router thresholds are per-subsystem, not on one shared scale.
- The C++ LSTM is inference-only; training stays in Python/PyTorch
  and weights get exported/loaded, not trained in C++.
- pybind11 is the binding layer; Python is the orchestration entrypoint.

## Status / what's a stand-in vs. what's real

- **Real, tested, and numerically verified:** the C++ LSTM cell forward
  pass (matches the reference equations to 1e-9) and the C++ cosine
  similarity index, both bound via pybind11 and exercised by the full
  pipeline.
- **Stand-ins clearly marked in code**, swappable without touching
  anything downstream: the bag-of-words embedding encoder (real model:
  sentence-transformers), the spellchecker's tiny fix-up dict (real:
  symspellpy or similar), and the web search stub (real: live Google
  CSE calls once API keys are set).
- **Not yet implemented:** SIMD optimization of the embedding search
  (mentioned in the architecture doc as a later step, after
  correctness), and loading LSTM weights that were actually trained on
  a real task — the LSTM cell is verified numerically correct but
  isn't wired into the router yet (per the base doc, the router
  conditions on the embedding layer's vectors and turn history, and
  the LSTM hook point is `kb_search`/`router` if/when you want it to
  score turn-history-conditioned relevance rather than a static
  per-message embedding).
