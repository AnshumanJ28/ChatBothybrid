import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.kb_search import KnowledgeBase


def test_exact_ish_match_ranks_first():
    kb = KnowledgeBase.from_file()
    results = kb.search("how do I reset my password", top_k=3)
    print(results[0])
    assert results[0]["id"] == "kb_001"
    assert results[0]["score"] > results[1]["score"]

def test_category_present():
    kb = KnowledgeBase.from_file()
    results = kb.search("where is my order", top_k=1)
    assert results[0]["category"] == "shipping"

if __name__ == "__main__":
    test_exact_ish_match_ranks_first()
    test_category_present()
    print("PASS: kb_search tests")
