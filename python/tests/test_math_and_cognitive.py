"""
Tests for C++ MathEvaluator and C++ CognitiveEngine
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.main import ChatSession
from orchestrator import minibrain_cpp as mb


def test_cpp_math_evaluator_direct():
    assert mb.MathEvaluator.is_math_query("what is 35+64")
    res = mb.MathEvaluator.evaluate("35+64")
    assert res.is_math
    assert res.value == 99.0
    assert "35+64 = 99" in res.formatted_result


def test_cpp_math_evaluator_complex():
    res = mb.MathEvaluator.evaluate("sqrt(144) + 2^3")
    assert res.is_math
    assert res.value == 20.0


def test_math_pipeline_integration():
    s = ChatSession()
    r = s.handle("so what is 35+64")
    print("math pipeline ->", r)
    assert "35+64 = 99" in r or "99" in r


def test_cpp_cognitive_engine():
    s = ChatSession()
    res = s.handle_structured("35+64")
    thoughts = res["thoughts"]
    print("cognitive thoughts ->", thoughts)
    assert any("Cognitive Engine" in t for t in thoughts)
    assert any("Math Engine" in t for t in thoughts)


if __name__ == "__main__":
    test_cpp_math_evaluator_direct()
    test_cpp_math_evaluator_complex()
    test_math_pipeline_integration()
    test_cpp_cognitive_engine()
    print("All C++ Math & Cognitive tests passed! 🎉")
