"""KB Search: embedding similarity over the local knowledge base,
backed by the C++ EmbeddingIndex (cache-friendly, SIMD-ready later)."""
import json
import os
from . import embedding
from . import minibrain_cpp as mb  # compiled extension, copied here by cpp/Makefile

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "kb_sample.json")


class KnowledgeBase:
    def __init__(self, entries: list[dict]):
        self.entries = entries  # [{id, category, question, answer}]
        self.index = mb.EmbeddingIndex(embedding.EMBED_DIM)
        for i, e in enumerate(entries):
            vec = embedding.encode(e["question"])
            self.index.add(vec, e["id"])

    @classmethod
    def from_file(cls, path: str = _DATA_PATH) -> "KnowledgeBase":
        with open(path, "r") as f:
            entries = json.load(f)
        return cls(entries)

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        qvec = embedding.encode(query_text)
        raw_results = self.index.search(qvec, top_k)
        out = []
        for r in raw_results:
            entry = self.entries[r.index]
            res_dict = {
                "id": entry["id"],
                "category": entry["category"],
                "question": entry["question"],
                "answer": entry["answer"],
                "score": r.score,
            }
            if "child_answer" in entry:
                res_dict["child_answer"] = entry["child_answer"]
            out.append(res_dict)
        return out
