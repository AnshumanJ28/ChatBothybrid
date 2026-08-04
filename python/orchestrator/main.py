"""
Main orchestration entrypoint.

    User message
      -> Preprocessing
      -> Embedding layer (used inside KB Search / router)
      -> Router (smalltalk -> flow -> KB -> web -> clarify)
      -> Dialogue State Manager (slots, active flow, turn history, topic)
      -> Response Composer (template + slot-fill + rank)
      -> Reply text
"""

import os
from collections import deque

def load_dotenv():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(current_dir, "..", "..", ".env"),
        os.path.join(current_dir, "..", ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if val.startswith(('"', "'")) and val.endswith(('"', "'")):
                            val = val[1:-1]
                        os.environ[key] = val
            break

load_dotenv()

from . import preprocessing, router, composer, kb_search, embedding
from .dialogue_state import DialogueState
from .conversation_brain import ConversationBrain


class ChatSession:
    def __init__(self, kb: kb_search.KnowledgeBase | None = None):
        self.kb = kb or kb_search.KnowledgeBase.from_file()
        self.state = DialogueState()
        self.conv_brain = ConversationBrain()
        self.recent_conv_ids: deque[str] = deque(maxlen=3)  # circular buffer — blocks last 3 picks

    def handle(self, user_message: str) -> str:
        pre = preprocessing.preprocess(user_message)
        # Keep dialogue memory cell synced with sequential tokens
        self.state.update_lstm_memory(pre["tokens"])

        result = router.route(pre, self.state, self.kb, self.conv_brain)
        reply, chosen_id = composer.compose(result, self.state, self.recent_conv_ids)
        if chosen_id:
            self.recent_conv_ids.append(chosen_id)

        self.state.record_turn(
            message=user_message,
            subsystem=result["subsystem"],
            confidence=result.get("confidence", 0.0),
            category=result.get("category"),
        )
        return reply

    def handle_structured(self, user_message: str) -> dict:
        pre = preprocessing.preprocess(user_message)
        
        # Run C++ stacked LSTM update sequentially on preprocessed tokens
        memory_norms = self.state.update_lstm_memory(pre["tokens"])
        
        # Route and Compose
        result = router.route(pre, self.state, self.kb, self.conv_brain)
        reply, chosen_id = composer.compose(result, self.state, self.recent_conv_ids)
        if chosen_id:
            self.recent_conv_ids.append(chosen_id)
        
        # Encode for metadata logging
        vec = embedding.encode(user_message)
        
        # Log LSTM update as the first thought step
        thoughts = [
            f"Encoded message text into {len(vec)}-dimensional vector using Neural Self-Attention Pooling.",
            f"Fed token sequence sequentially to C++ DeepLSTM (3-layer stacked memory cell):",
        ]
        for l in range(len(memory_norms)):
            thoughts.append(f" - Layer {l+1} final hidden state L2 norm: {memory_norms[l]:.4f}")
        
        # Add the router thoughts
        if "thoughts" in result:
            thoughts.extend(result["thoughts"])
            
        self.state.record_turn(
            message=user_message,
            subsystem=result["subsystem"],
            confidence=result.get("confidence", 0.0),
            category=result.get("category"),
        )
        
        return {
            "answer": reply,
            "subsystem": result["subsystem"],
            "thoughts": thoughts,
            "sources": result.get("sources", [])
        }


def _repl():
    session = ChatSession()
    print("Hybrid classical-NLP chatbot (no LLM). Type 'quit' to exit.")
    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() in {"quit", "exit"}:
            break
        print("bot>", session.handle(user_input))


if __name__ == "__main__":
    _repl()
