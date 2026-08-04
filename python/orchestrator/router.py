"""Router: dispatches a preprocessed message to the right subsystem.

Order matters and mirrors the architecture diagram:
  1. Small-talk (before KB, so greetings don't burn a retrieval pass)
  2. Active flow continuation (if a flow is already in progress)
  3. Flow trigger (does this message start a new flow?)
  4. KB Search (local, fast, curated — tried first for real queries)
  4.5 Contextual Conversation Brain (Chitchat via DeepLSTM context matching)
  5. Tool Dispatch -> web search (only if KB is below threshold AND the
     query looks like a factual/current-info question)
  6. Fallback / Clarify

Thresholds are per-subsystem, not on a shared scale.
"""
import re
import math
from . import smalltalk, flow_engine, kb_search, generative_brain
from .tool_dispatch import web_search
from .conversation_brain import ConversationBrain

KB_CONFIDENCE_THRESHOLD = 0.35
WEB_LOOKUP_TRIGGER_WORDS = {"who", "what", "when", "where", "why", "how", "which"}

_QUESTION_WORD_RE = re.compile(
    r"^(who|what|when|where|why|how|which|is|are|does|do|did)\b"
)

_CONVERSATIONAL_EXCLUSIONS = {
    "you", "your", "yours", "yourself", "creator", "developer", "joke", "jokes",
    "funny", "name", "whoever", "feeling", "feel", "sad", "hurt", "happy", "depressed", "angry"
}


def _looks_like_web_query(tokens: list[str]) -> bool:
    """Lightweight rule: contains a question word, i.e. reads like a
    factual/current-info ask rather than an in-app support question or chitchat."""
    if not tokens:
        return False
    
    # Exclude queries containing conversational bot keywords
    if set(tokens) & _CONVERSATIONAL_EXCLUSIONS:
        return False
        
    return tokens[0] in WEB_LOOKUP_TRIGGER_WORDS or bool(_QUESTION_WORD_RE.match(" ".join(tokens)))


def route(preprocessed: dict, state, kb: kb_search.KnowledgeBase, conv_brain: ConversationBrain) -> dict:
    """Returns a dict describing which subsystem handled the message and
    the raw payload the Composer needs to build the final reply."""
    thoughts = []
    sources = []
    
    tokens = preprocessed["tokens"]
    normalized = preprocessed["normalized"]
    cleaned = preprocessed["cleaned_text"]

    # Log preprocessed input details
    thoughts.append(f"Input preprocessed. Tokens: {tokens}")
    thoughts.append(f"Normalized text: '{normalized}'")

    # Run C++ CognitiveEngine trace
    try:
        from . import minibrain_cpp as mb
        lstm_norms = [math.sqrt(sum(v * v for v in state.h_states[l])) for l in range(state.num_layers)]
        cog_trace = mb.CognitiveEngine.process(cleaned, tokens, lstm_norms)
        thoughts.extend(cog_trace.thoughts)
    except Exception:
        pass

    # 0. C++ High-Speed Math Engine check
    raw_text = preprocessed.get("raw", "")
    try:
        from . import minibrain_cpp as mb
        target_math = raw_text if mb.MathEvaluator.is_math_query(raw_text) else normalized
        if mb.MathEvaluator.is_math_query(target_math):
            math_res = mb.MathEvaluator.evaluate(target_math)
            if math_res.is_math and not math_res.error:
                thoughts.append(f"[C++ Math Engine] Calculated result: {math_res.formatted_result}")
                return {
                    "subsystem": "math",
                    "payload": {"result": math_res.formatted_result, "value": math_res.value},
                    "confidence": 1.0,
                    "category": "math",
                    "thoughts": thoughts,
                    "sources": sources,
                }
    except Exception:
        pass

    # 0. explicit reset check
    if state.check_explicit_reset(normalized):
        thoughts.append("Explicit reset command detected. Clearing state and resetting dialogue memory.")
        return {"subsystem": "reset", "payload": {}, "confidence": 1.0, "category": None, "thoughts": thoughts, "sources": sources}

    # 0.5 explicit persona switch check
    norm_lower = normalized.lower()
    if any(ph in norm_lower for ph in ["explain like i'm 10", "explain like im 10", "talk like a 10 year old", "talk like a kid", "child mode", "kid mode"]):
        state.persona = "child"
        thoughts.append("Explicit child persona switch detected. Setting state.persona to 'child'.")
        return {
            "subsystem": "persona_switch",
            "payload": {"persona": "child", "reply": "Okay! I will now talk to you like a 10-year-old. What would you like to know?"},
            "confidence": 1.0,
            "category": None,
            "thoughts": thoughts,
            "sources": sources
        }
    elif any(ph in norm_lower for ph in ["talk normally", "standard mode", "reset persona", "default mode"]):
        state.persona = "default"
        thoughts.append("Explicit default persona switch detected. Resetting state.persona to 'default'.")
        return {
            "subsystem": "persona_switch",
            "payload": {"persona": "default", "reply": "Persona reset. Back to standard mode. How can I help you?"},
            "confidence": 1.0,
            "category": None,
            "thoughts": thoughts,
            "sources": sources
        }

    # 1. active flow continuation takes priority over everything else
    if state.active_flow is not None:
        flow_name = state.active_flow.flow_name
        current_slot = state.active_flow.definition["slots"][state.active_flow.current_slot_index]
        thoughts.append(f"Active flow '{flow_name}' detected. Directing user input to fill slot '{current_slot}'.")
        
        prompt_or_completion = state.active_flow.fill_next_slot(cleaned)
        done = state.active_flow.is_complete
        
        thoughts.append(f"Slot '{current_slot}' filled with: '{cleaned}'. Flow complete: {done}")
        
        payload = {"text": prompt_or_completion, "flow_done": done}
        if done:
            state.active_flow = None
        return {"subsystem": "flow", "payload": payload, "confidence": 1.0, "category": "flow", "thoughts": thoughts, "sources": sources}

    # 2. small talk
    thoughts.append("Scanning for small-talk greetings or feedback intent...")
    st = smalltalk.detect(tokens)
    if st is not None:
        thoughts.append(f"Small-talk pattern matched: '{st['intent']}' (Confidence: {st['confidence']:.2f}). Short-circuiting directly.")
        return {"subsystem": "smalltalk", "payload": st, "confidence": st["confidence"], "category": None, "thoughts": thoughts, "sources": sources}

    # 3. new flow trigger
    thoughts.append("Checking for structured flow keywords...")
    flow_name = flow_engine.match_trigger(tokens)
    if flow_name is not None:
        thoughts.append(f"Flow trigger matched: '{flow_name}'. Initializing slot-filling state machine.")
        instance = flow_engine.FlowInstance(flow_name)
        state.active_flow = instance
        return {
            "subsystem": "flow",
            "payload": {"text": instance.next_prompt(), "flow_done": False},
            "confidence": 1.0,
            "category": "flow",
            "thoughts": thoughts,
            "sources": sources
        }

    # 4. KB search, with topic-continuity bias for weak queries
    thoughts.append("No active flows or small-talk detected. Querying local C++ vector similarity index...")
    kb_results = kb.search(cleaned, top_k=3)
    top = kb_results[0] if kb_results else None

    if top:
        thoughts.append(f"C++ similarity top candidate: '{top['question']}' in category '{top['category']}' (Similarity Score: {top['score']:.4f})")
        if top["score"] >= KB_CONFIDENCE_THRESHOLD:
            thoughts.append(f"High-confidence KB match found (Score {top['score']:.4f} >= threshold {KB_CONFIDENCE_THRESHOLD:.2f}). Routing to KB.")
            return {
                "subsystem": "kb",
                "payload": {"results": kb_results},
                "confidence": top["score"],
                "category": top["category"],
                "thoughts": thoughts,
                "sources": sources
            }
        else:
            thoughts.append(f"Top KB match score {top['score']:.4f} is below primary threshold {KB_CONFIDENCE_THRESHOLD:.2f}.")
    else:
        thoughts.append("No candidates found in KB.")

    # weak KB match: try topic continuity bias before giving up on KB
    biased_category = state.topic_bias_category()
    if biased_category and top:
        thoughts.append(f"Dialogue state has active topic bias: '{biased_category}'. Scanning candidate matches in this category...")
        same_domain = [r for r in kb_results if r["category"] == biased_category]
        biased_threshold = KB_CONFIDENCE_THRESHOLD * 0.6
        if same_domain and same_domain[0]["score"] >= biased_threshold:
            best = same_domain[0]
            thoughts.append(f"Bias match found: '{best['question']}' with score {best['score']:.4f} >= biased threshold {biased_threshold:.2f}. Overriding fallback.")
            return {
                "subsystem": "kb",
                "payload": {"results": same_domain, "biased": True},
                "confidence": best["score"],
                "category": best["category"],
                "thoughts": thoughts,
                "sources": sources
            }
        else:
            thoughts.append(f"No bias category match with sufficient confidence (need >= {biased_threshold:.2f}).")

    # 4.5. tool dispatch -> web search, only for factual/current-info-shaped queries
    thoughts.append("KB search failed to return confident candidate. Checking query syntax for web search...")
    if _looks_like_web_query(tokens) or web_search.extract_url(cleaned) is not None:
        thoughts.append("Query format matches factual query or contains URL. Dispatching Web Search scraper & summarizer...")
        result = web_search.search(cleaned)
        
        # Append web search logs to thoughts
        if "logs" in result:
            for l in result["logs"]:
                thoughts.append(f"[Web Scraper] {l}")
                
        sources = result.get("sources", [])
        
        thoughts.append("Web search summarization complete. Submitting summary response.")
        return {
            "subsystem": "web",
            "payload": result,
            "confidence": 0.6 if result["summary"] else 0.0,
            "category": None,
            "thoughts": thoughts,
            "sources": sources
        }
    else:
        thoughts.append("Query does not look like a factual web lookup (lacks question words/URLs).")

    # 5. Contextual Conversation Brain (Chitchat matching using blended LSTM context + query vector)
    thoughts.append("Blending query embedding with C++ DeepLSTM final-layer hidden state for context-sensitive chitchat routing...")
    from . import embedding
    q_vec = embedding.encode(cleaned)
    lstm_context = state.get_lstm_context()
    
    # Blending math: 60% query, 40% running context
    blended_vec = []
    for i in range(len(q_vec)):
        blended_vec.append(0.6 * q_vec[i] + 0.4 * lstm_context[i])
        
    # Re-normalize the blended unit vector
    norm = math.sqrt(sum(v * v for v in blended_vec)) or 1e-12
    blended_vec = [v / norm for v in blended_vec]
    
    # Use pure query vec for chitchat (avoids LSTM context biasing same entries)
    conv_results = conv_brain.match_context(q_vec, top_k=6)
    top_conv = conv_results[0] if conv_results else None

    if top_conv and top_conv["score"] >= 0.25:
        thoughts.append(f"Matched conversational template '{top_conv['question']}' (Score {top_conv['score']:.4f} >= threshold 0.25). Routing to Chitchat Brain.")
        return {
            "subsystem": "conversation",
            "payload": {"results": conv_results},
            "confidence": top_conv["score"],
            "category": None,
            "thoughts": thoughts,
            "sources": sources
        }
    else:
        if top_conv:
            thoughts.append(f"Chitchat candidate '{top_conv['question']}' score {top_conv['score']:.4f} is below threshold 0.25.")
        else:
            thoughts.append("No conversational candidate match found.")

    # 6. Fallback -> Generative Brain
    thoughts.append("No subsystem confident. Routing to Gemini Generative Brain...")
    
    lstm_norms = []
    for l in range(state.num_layers):
        h_vals = state.h_states[l]
        norm = math.sqrt(sum(v * v for v in h_vals))
        lstm_norms.append(norm)
        
    gen_result = generative_brain.generate_response(cleaned, lstm_norms=lstm_norms)
    for log_msg in gen_result.get("logs", []):
        thoughts.append(f"[Generative Brain] {log_msg}")
        
    return {
        "subsystem": "generative",
        "payload": {
            "text": gen_result["summary"],
            "from_api": gen_result["from_api"]
        },
        "confidence": 0.8,
        "category": None,
        "thoughts": thoughts,
        "sources": sources
    }
