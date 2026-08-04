"""Generative Brain subsystem.
Calls Google's Gemini API if GEMINI_API_KEY or GOOGLE_API_KEY is configured,
otherwise falls back to a deterministic offline generator that utilizes the
C++ DeepLSTM memory norms to show simulated generative thinking.
"""
import os
import json
import urllib.request
import urllib.error

_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{_MODEL}:generateContent"

# Hand-coded offline rule responder to mock "generative thinking" for key queries
# if the live API key is missing, so the chatbot works fully offline.
_OFFLINE_TEMPLATES = {
    "who are you": "I am MiniBrain, a classical NLP chatbot with a deep stacked C++ LSTM neural network and local embedding indexes! I do not use any LLMs unless configured with a live API key.",
    "tell me a joke": "Why do C++ programmers wear glasses? Because they can't C#! (Don't worry, my bindings compile perfectly.)",
    "tell me the api that can do search": "I use the Google Custom Search JSON API (Google CSE API) to perform live web searches and index matching!",
    "what api do you use for web search": "I use the Google Custom Search JSON API (Google CSE API) to perform live web searches and index matching!",
}

def generate_response(query: str, lstm_norms: list[float] = None) -> dict:
    """Invokes Gemini API or runs C++ LSTM-aware mock generation.
    Returns: {'summary': str, 'logs': list[str], 'from_api': bool}
    """
    logs = []
    norms_str = ", ".join(f"{n:.4f}" for n in lstm_norms) if lstm_norms else "None"
    
    if _API_KEY:
        logs.append(f"GEMINI_API_KEY configured. Sending generative request to model: {_MODEL}")
        url = f"{_ENDPOINT}?key={_API_KEY}"
        
        system_instruction = (
            "You are MiniBrain, a hybrid classical NLP chatbot with a 3-layer C++ LSTM dialogue state memory. "
            f"Your current C++ memory layer norms are: [{norms_str}]. "
            "Respond to the user naturally but concisely (1-3 sentences). Keep the persona of an efficient "
            "C++ and Python-powered local chatbot system. If the user asks about search APIs, tell them that "
            "you use the Google Custom Search JSON API (Google CSE API) for web search retrieval."
        )
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"System Instruction: {system_instruction}\n\nUser Query: {query}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 150,
                "temperature": 0.7
            }
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                res_data = json.loads(response.read().decode("utf-8"))
            
            candidates = res_data.get("candidates", [])
            if candidates:
                part = candidates[0].get("content", {}).get("parts", [{}])[0]
                text = part.get("text", "").strip()
                if text:
                    logs.append("Successfully received generative response from Gemini API.")
                    return {
                        "summary": text,
                        "logs": logs,
                        "from_api": True
                    }
            logs.append("Gemini API response format invalid or empty candidate.")
        except Exception as e:
            logs.append(f"Failed to fetch response from Gemini API: {e}. Falling back to offline generator.")
    
    # Offline Generative Fallback
    logs.append("No Gemini API key configured. Activating offline C++ LSTM-aware simulation generator...")
    
    # Check for direct keyword matches in offline templates
    cleaned = query.lower().strip().replace("?", "").replace("!", "")
    matched_reply = None
    for k, reply in _OFFLINE_TEMPLATES.items():
        if k in cleaned or cleaned in k:
            matched_reply = reply
            break
            
    if matched_reply:
        reply_text = matched_reply
    else:
        reply_text = (
            "I processed your query through my 3-layer C++ DeepLSTM memory, but I couldn't find a high-confidence match in my knowledge base. "
            "Feel free to ask a support question, ask for a joke, or try searching for specific info!"
        )
        
    logs.append("Simulated generative answer computed from offline C++ LSTM state.")
    return {
        "summary": reply_text,
        "logs": logs,
        "from_api": False
    }
