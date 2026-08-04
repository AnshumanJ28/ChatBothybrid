"""Contextual Conversation Brain.

Uses a C++ EmbeddingIndex to query a conversational corpus using the
query embedding. Supports tag-based pooling: once a top match is found,
all entries sharing the same tags are included in the result set so that
variety selection can pick across the full semantic group (e.g., all jokes).
"""
import json
import os
from collections import deque
from . import embedding
from . import minibrain_cpp as mb

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "conversation_corpus.json")


class ConversationBrain:
    def __init__(self, entries: list[dict] = None):
        if entries is None:
            with open(_DATA_PATH, "r", encoding="utf-8") as f:
                entries = json.load(f)
        self.entries = entries
        self.index = mb.EmbeddingIndex(embedding.EMBED_DIM)
        for i, e in enumerate(entries):
            vec = embedding.encode(e["question"])
            self.index.add(vec, e["id"])

    def match_context(self, context_vector: list[float], top_k: int = 6) -> list[dict]:
        """Search the conversational database.

        After finding top-k results, expands using tag-based pooling: any
        corpus entry sharing a tag with the top result is force-included with
        the top score minus a tiny epsilon, ensuring all semantically grouped
        entries (e.g., all 4 joke entries) are always in the variety pool.
        """
        raw_results = self.index.search(context_vector, top_k)
        if not raw_results:
            return []

        out = []
        seen_ids = set()
        for r in raw_results:
            entry = self.entries[r.index]
            res_dict = {
                "id":       entry["id"],
                "category": entry["category"],
                "tags":     entry.get("tags", []),
                "question": entry["question"],
                "answer":   entry["answer"],
                "score":    r.score,
            }
            if "child_answer" in entry:
                res_dict["child_answer"] = entry["child_answer"]
            out.append(res_dict)
            seen_ids.add(entry["id"])

        # Tag-based score normalization: set all same-tag entries (already in out)
        # to top_score - 0.001 so they all land in the variety window.
        top_tags = set(out[0]["tags"]) if out else set()
        top_score = out[0]["score"] if out else 0.0
        if top_tags:
            for res in out[1:]:
                res_tags = set(res.get("tags", []))
                if res_tags & top_tags:
                    res["score"] = top_score - 0.001

            # Add any corpus entries NOT yet in out that share the tag
            for entry in self.entries:
                if entry["id"] in seen_ids:
                    continue
                entry_tags = set(entry.get("tags", []))
                if entry_tags & top_tags:
                    res_dict = {
                        "id":           entry["id"],
                        "category":     entry["category"],
                        "tags":         entry.get("tags", []),
                        "question":     entry["question"],
                        "answer":       entry["answer"],
                        "score":        top_score - 0.001,
                        "tag_expanded": True,
                    }
                    if "child_answer" in entry:
                        res_dict["child_answer"] = entry["child_answer"]
                    out.append(res_dict)
                    seen_ids.add(entry["id"])

        return out
