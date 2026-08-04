import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.main import ChatSession


def test_smalltalk():
    s = ChatSession()
    reply = s.handle("hey there")
    print("smalltalk ->", reply)
    assert reply

def test_kb_hit():
    s = ChatSession()
    reply = s.handle("how do I reset my password")
    print("kb ->", reply)
    assert "Reset Password" in reply or "password" in reply.lower()

def test_kb_topic_continuity():
    s = ChatSession()
    r1 = s.handle("how do I update my payment method")
    print("turn1 ->", r1)
    r2 = s.handle("what about canceling")  # low-confidence alone, same domain
    print("turn2 ->", r2)
    assert r2

def test_flow_multi_turn():
    s = ChatSession()
    r1 = s.handle("I want to return an item")
    print("flow1 ->", r1)
    assert "order ID" in r1
    r2 = s.handle("order 12345")
    print("flow2 ->", r2)
    assert "reason" in r2.lower()
    r3 = s.handle("it arrived damaged")
    print("flow3 ->", r3)
    assert "12345" in r3 and "damaged" in r3

def test_web_fallback_stub():
    s = ChatSession()
    reply = s.handle("what is the capital of nowhere-land-42")
    print("web ->", reply)
    assert reply  # stub path still produces a templated reply, not a crash

def test_explicit_reset():
    s = ChatSession()
    s.handle("I want to book a demo")
    r = s.handle("start over")
    print("reset ->", r)
    assert s.state.active_flow is None

def test_clarify_fallback():
    s = ChatSession()
    reply = s.handle("asdkjaslkdj qweoiqwe")
    print("clarify ->", reply)
    assert reply

def test_child_persona_mode():
    s = ChatSession()
    r1 = s.handle("talk like a 10 year old")
    print("persona switch ->", r1)
    assert "10-year-old" in r1 or "kid" in r1
    assert s.state.persona == "child"

    r2 = s.handle("tell me a joke")
    print("child joke ->", r2)
    # Any child joke is acceptable (variety pool)
    CHILD_JOKES = ["dino-snore", "byte", "Windows open", "fsh", "robot"]
    assert any(kw in r2 for kw in CHILD_JOKES), f"No child joke keyword found in: {r2!r}"

    r3 = s.handle("how do I reset my password")
    print("child kb ->", r3)
    assert "gear settings icon" in r3

    r4 = s.handle("talk normally")
    print("reset persona ->", r4)
    assert s.state.persona == "default"

    r5 = s.handle("tell me a joke")
    print("default joke ->", r5)
    # Any adult joke is acceptable (variety pool)
    ADULT_JOKES = ["C#", "Java", "SQL", "binary", "10 types"]
    assert any(kw in r5 for kw in ADULT_JOKES), f"No adult joke keyword found in: {r5!r}"



if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS: {name}\n")
    print("All integration tests passed.")
