"""
Quick test for the DuckDuckGo fallback search — no API key needed.

Run this from your project root (same folder as the `python/` directory):
    python test_ddg_search.py
"""
import sys
import os

# Make sure the project's python/ package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from orchestrator.tool_dispatch import web_search

if __name__ == "__main__":
    query = "latest news about artificial intelligence"
    print(f"Searching for: {query!r}\n")

    result = web_search.search(query, num=3)

    print("=== Logs ===")
    for line in result["logs"]:
        print(f"  {line}")

    print("\n=== Results ===")
    if result["results"]:
        for i, r in enumerate(result["results"], 1):
            print(f"{i}. {r['title']}")
            print(f"   {r['url']}")
            print(f"   {r['snippet'][:150]}...")
    else:
        print("  No results.")

    print(f"\n=== Summary ===\n{result['summary']}")

    if result["error"]:
        print(f"\nFAILED — error: {result['error']}")
        sys.exit(1)
    else:
        print("\nSUCCESS — DuckDuckGo fallback is working.")