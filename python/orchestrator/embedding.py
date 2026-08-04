"""
Embedding layer: maps text -> fixed-size vector. NOT an LLM — it never
generates text, only encodes it.

This module ships a small deterministic hashing-based bag-of-words
encoder so the project runs fully offline with zero model downloads.
It is a drop-in stand-in for a real pretrained sentence encoder
(e.g. `sentence-transformers/all-MiniLM-L6-v2`); swap `encode()`'s
body for a real model call and nothing downstream changes, since the
rest of the pipeline only depends on "text -> fixed-length vector".

If you do wire in sentence-transformers:

    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    def encode(text: str) -> list[float]:
        return _model.encode(text).tolist()

That's the only change needed — router, KB search, and the C++
similarity index are all dimension-agnostic.
"""
import hashlib
import math
from .preprocessing import tokenize

EMBED_DIM = 64


def _token_vector(token: str) -> list[float]:
    """Deterministic pseudo-random unit vector for a token, derived from
    its hash. Stands in for a learned embedding table lookup."""
    h = hashlib.sha256(token.encode("utf-8")).digest()
    # Expand the 32-byte digest into EMBED_DIM floats in [-1, 1].
    vec = []
    for i in range(EMBED_DIM):
        byte = h[i % len(h)]
        # mix in position so repeated bytes don't repeat identically
        val = ((byte + i * 31) % 256) / 127.5 - 1.0
        vec.append(val)
    return vec


def encode(text: str) -> list[float]:
    """Encodes text into a fixed EMBED_DIM vector using C++ Sentence-Centrality
    Self-Attention pooling over token vectors."""
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * EMBED_DIM

    token_vectors = [_token_vector(t) for t in tokens]

    # High-performance C++ self-attention pooling
    try:
        from . import minibrain_cpp as mb
        pooled = mb.AttentionPooling.pool(token_vectors)
        if pooled:
            return pooled
    except Exception:
        pass

    # Compute sentence average vector as the query target
    avg_vec = [0.0] * EMBED_DIM
    for tv in token_vectors:
        for i in range(EMBED_DIM):
            avg_vec[i] += tv[i]
    n = len(tokens)
    avg_vec = [v / n for v in avg_vec]

    # Compute raw attention scores (dot product of each token with the sentence average)
    scores = []
    d_sqrt = math.sqrt(EMBED_DIM)
    for tv in token_vectors:
        score = sum(tv[i] * avg_vec[i] for i in range(EMBED_DIM)) / d_sqrt
        scores.append(score)

    # Softmax normalization
    max_score = max(scores) if scores else 0.0
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores) or 1e-12
    weights = [s / sum_exp for s in exp_scores]

    # Attention-weighted sum
    acc = [0.0] * EMBED_DIM
    for tv, w in zip(token_vectors, weights):
        for i in range(EMBED_DIM):
            acc[i] += w * tv[i]

    norm = math.sqrt(sum(v * v for v in acc)) or 1e-12
    return [v / norm for v in acc]
