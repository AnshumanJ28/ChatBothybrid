import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from orchestrator.tool_dispatch import summarizer


def test_basic_summarization():
    text = (
        "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints. "
        "The key features are: Fast: Very high performance, on par with NodeJS and Go. "
        "Robust: Get production-ready code. With automatic interactive documentation. "
        "Easy: Designed to be easy to use and learn. Less time reading docs. "
        "FastAPI is built on top of Starlette and Pydantic. It provides fast routing and serialization. "
        "When building a microservice with Python, developers often choose FastAPI for its outstanding execution speed."
    )
    query = "FastAPI speed performance"
    
    res = summarizer.summarize_text(text, query, num_sentences=2)
    summary = res["summary"]
    logs = res["logs"]
    
    print("Summary:")
    print(summary)
    print("\nLogs:")
    for log in logs:
        print(f" - {log}")
    
    assert summary
    assert len(logs) > 0
    assert "FastAPI" in summary
    assert "performance" in summary.lower() or "speed" in summary.lower()


def test_short_text():
    text = "Short text here. Another short sentence."
    res = summarizer.summarize_text(text, "query", num_sentences=3)
    assert res["summary"] == "Short text here. Another short sentence."


if __name__ == "__main__":
    test_basic_summarization()
    test_short_text()
    print("All summarizer tests passed.")
