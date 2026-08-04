# Hybrid C++/Python Chatbot (No LLM)

A high-performance **hybrid chatbot** built using **C++** and **Python** that performs conversational reasoning **without using any Large Language Models (LLMs)**.

Instead of generating text, the chatbot retrieves, computes, and composes responses using deterministic algorithms, knowledge-base retrieval, dialogue management, and handcrafted neural components.

---

## Overview

This project demonstrates how a modern conversational AI system can be built **without generative AI** while still supporting:

- Multi-turn conversations
- Dialogue memory
- Semantic knowledge retrieval
- Flow-based interactions
- Tool dispatching
- High-performance native inference

The architecture separates responsibilities:

- **C++** handles compute-intensive operations.
- **Python** orchestrates conversation flow and business logic.

---

# Features

- No LLMs or text generation
- Fully deterministic responses
- Offline-first architecture
- Hybrid C++ + Python implementation
- pybind11 integration
- Hand-written LSTM inference engine
- Fast cosine similarity search
- Dialogue state management
- Slot-filling conversation engine
- Knowledge-base retrieval
- Template-based response composition
- Optional web search fallback
- Modular architecture
- Unit and integration tests

---

# System Architecture

```text
                    User Message
                          │
                          ▼
                 Text Preprocessing
      (Tokenization • Normalization • Spellcheck)
                          │
                          ▼
                 Embedding Generation
              (Local Encoder • No LLM)
                          │
                          ▼
                    Intent Router
      ┌──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼
 Small Talk      Flow Engine     KB Search     Web Search
                                (C++)          (Optional)
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
                          │
                          ▼
              Dialogue State Manager
                          │
                          ▼
               Response Composer
        (Template + Slot Filling + Ranking)
                          │
                          ▼
                   Final Response
```

---

# Repository Structure

```text
HybridChatbot/
│
├── cpp/
│   │
│   ├── src/
│   │   ├── attention_pooling.cpp
│   │   ├── attention_pooling.h
│   │   ├── cognitive_engine.cpp
│   │   ├── cognitive_engine.h
│   │   ├── dense_layer.cpp
│   │   ├── dense_layer.h
│   │   ├── embedding_search.cpp
│   │   ├── embedding_search.h
│   │   ├── lstm_cell.cpp
│   │   ├── lstm_cell.h
│   │   ├── math_evaluator.cpp
│   │   ├── math_evaluator.h
│   │   ├── text_sanitizer.cpp
│   │   ├── text_sanitizer.h
│   │   └── bindings.cpp
│   │
│   ├── tests/
│   │   └── test_bindings_smoke.py
│   │
│   ├── CMakeLists.txt
│   └── Makefile
│
├── orchestrator/
│   │
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── kb_search.py
│   ├── router.py
│   ├── flow_engine.py
│   ├── dialogue_state.py
│   ├── conversation_brain.py
│   ├── generative_brain.py
│   ├── composer.py
│   ├── server.py
│   ├── main.py
│   │
│   ├── static/
│   │
│   ├── tool_dispatch/
│   │   ├── cache.py
│   │   └── web_search.py
│   │
│   └── __init__.py
│
├── tests/
│   ├── test_conversation.py
│   ├── test_deep_lstm.py
│   ├── test_kb_search.py
│   ├── test_math_and_cognitive.py
│   ├── test_pipeline_integration.py
│   └── test_summarizer.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Core Components

| Component | Description |
|------------|-------------|
| **Preprocessing** | Cleans, tokenizes, normalizes, and spell-corrects user input. |
| **Embedding Layer** | Converts text into dense numerical vectors using a local encoder. |
| **Router** | Determines which subsystem should process the request. |
| **Small Talk Engine** | Handles greetings, thanks, and casual conversations. |
| **Knowledge Base Search** | Retrieves semantically similar documents using the C++ similarity engine. |
| **Flow Engine** | Executes multi-turn slot-filling conversations. |
| **Dialogue State Manager** | Maintains conversation history and active dialogue state. |
| **Response Composer** | Selects and fills response templates. |
| **Tool Dispatcher** | Invokes optional external tools such as web search. |
| **Web Search** | Provides optional live search when local knowledge is insufficient. |

---

# Native C++ Components

Performance-critical modules are implemented entirely in C++.

## Implemented

- Custom LSTM Cell
- Dense Neural Layer
- Attention Pooling
- Embedding Similarity Search
- Mathematical Expression Evaluator
- Text Sanitizer
- Cognitive Engine
- pybind11 Python Bindings

Python remains responsible for orchestration while C++ performs numerical computation.

---

# Technologies

## Languages

- Python
- C++17

## Libraries

- pybind11
- NumPy
- Flask
- sentence-transformers (optional)
- Google Custom Search API (optional)

## Build Tools

- Make
- CMake
- GCC / Clang / MSVC

---

# Installation

## Clone Repository

```bash
git clone https://github.com/AnshumanJ28/ChatBothybrid.git
cd ChatBothybrid
```

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

# Build

## Using Make

```bash
cd cpp
make
```

## Using CMake

```bash
mkdir build
cd build

cmake ..
cmake --build .
```

---

# Running the Chatbot

```bash
python orchestrator/main.py
```

---

# Example Usage

```python
from orchestrator.main import ChatSession

chat = ChatSession()

print(chat.handle("Hello"))

print(chat.handle("How do I reset my password?"))

print(chat.handle("I want to return my order."))

print(chat.handle("Order number is 12345"))

print(chat.handle("The package arrived damaged."))
```

---

# Running Tests

```bash
python tests/test_conversation.py

python tests/test_deep_lstm.py

python tests/test_kb_search.py

python tests/test_math_and_cognitive.py

python tests/test_pipeline_integration.py

python tests/test_summarizer.py
```

---

# Replacing the Embedding Model

The default embedding implementation uses a deterministic hashing-based encoder so the project works entirely offline.

To use a production-quality embedding model:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def encode(text):
    return model.encode(text).tolist()
```

No other component requires modification.

---

# Optional Live Web Search

To enable Google Custom Search:

```bash
export GOOGLE_CSE_API_KEY=YOUR_API_KEY

export GOOGLE_CSE_CX=YOUR_SEARCH_ENGINE_ID
```

If these variables are absent, the chatbot automatically falls back to the offline implementation.

---

# Design Principles

- No Large Language Models
- No generated text
- Deterministic responses
- Offline-first execution
- High-performance native inference
- Modular architecture
- Easily replaceable embedding models
- Clear separation between orchestration and computation

---

# Current Status

## Completed

- Hybrid C++/Python architecture
- Custom LSTM implementation
- Embedding similarity search
- Dialogue state tracking
- Flow engine
- Template-based response generation
- Router
- Native Python bindings
- Unit testing
- Integration testing

## Planned Improvements

- SIMD optimization
- Approximate nearest-neighbor indexing
- Persistent vector database
- GPU acceleration
- Trained LSTM weight loading
- Voice interface
- REST API deployment
- Web dashboard

---

# Repository Highlights

- Hybrid C++ + Python architecture
- Zero dependency on LLMs
- High-performance native inference
- Offline-capable conversational pipeline
- Modular and extensible design
- Production-ready project organization
- Comprehensive testing suite

---

# License

This project is intended for educational and research purposes.
